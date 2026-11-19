# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Helpers for defining sharding for optimizer states based on existing sharding
for model parameters.
"""

import logging
import math
from copy import deepcopy
from dataclasses import replace
from typing import Dict, Iterable, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import torch

from megatron.core.utils import log_single_rank, to_local_if_dtensor

from .dict_utils import nested_values
from .gtp_layout import flat_range_chunks
from .mapping import (
    LocalNonpersistentObject,
    ReplicaId,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
    StateDict,
)
from .utils import extract_sharded_tensors_and_factories

KEEP_VARS_HINT = (
    " Make sure state dict contains original torch.nn.Parameters (not pure torch.Tensors)"
    " by passing `keep_vars=True` to `.state_dict()`. If any transformation of the original"
    " parameter is needed, use a ShardedTensorFactory."
)


def get_optim_param_to_id_map(optim_params_iter: Iterable[torch.nn.Parameter]) -> Dict[int, int]:
    """Generate mapping from optimizer param to optimizer state id."""
    param_mappings = {}
    for i, param in enumerate(optim_params_iter):
        param = to_local_if_dtensor(param)
        if id(param) not in param_mappings:
            param_mappings[id(param)] = i
    return param_mappings


def get_param_id_to_sharded_param_map(
    model_sharded_state_dict: ShardedStateDict, optim_params_iter: Iterable[torch.nn.Parameter]
) -> Dict[int, Union[ShardedTensor, ShardedTensorFactory]]:
    """Generate mapping from optimizer state ids to model sharded parameters.

    Args:
        model_sharded_state_dict: sharded state dict with all model sharded tensors
            (can have any structure)
        optim_params_iter: iterable which iterates over model parameters tracked by the optimizer.
            The iteration must be in the same order as in the optimizer parameters.

    Returns:
        Dict[int, Union[ShardedTensor, ShardedTensorFactory]]: mapping from optimizer state ids
            to model sharded parameters.
    """
    model_sharded_state_dict, _ = extract_sharded_tensors_and_factories(model_sharded_state_dict)
    id_to_sharded_param_map = {}
    param_to_id_map = get_optim_param_to_id_map(optim_params_iter)
    # If using PyTorch FSDP2 the values in model_sharded_state_dict would
    # have been converted to local tensors during initialization.
    # See the make_(tp)_sharded_tensor_for_checkpoint functions.
    for ten in nested_values(model_sharded_state_dict):
        if isinstance(ten, ShardedTensorFactory):
            # A factory may bind optimizer state to the live parameter through a companion
            # (e.g. GTP local fragments); match and shard the state through that companion.
            ten = ten.for_optimizer()
        if id(ten.data) in param_to_id_map:
            id_to_sharded_param_map[param_to_id_map[id(ten.data)]] = ten
        else:
            logger.debug('%s is not tracked by the optimizer', ten)

    if not id_to_sharded_param_map:
        log_single_rank(
            logger,
            logging.WARNING,
            "Sharded parameters mapping is empty. It means tensors in model state dict"
            " do not correspond to tensors in optimizer parameters map."
            " Make sure to call state_dict with `keep_vars=True`.",
        )
    return id_to_sharded_param_map


def make_sharded_optimizer_tensor(
    model_param: Union[ShardedTensor, ShardedTensorFactory], optim_param: torch.Tensor, prefix: str
) -> Union[ShardedTensor, ShardedTensorFactory]:
    """Build a ShardedTensor or ShardedTensorFactory for optimizer param based on model param

    Args:
        model_param (Union[ShardedTensor, ShardedTensorFactory]): model param
        optim_param (torch.Tensor): corresponding optimizer param
        prefix (str): optimizer prefix for the ShardedTensor or ShardedTensorFactory

    Returns:
        Union[ShardedTensor, ShardedTensorFactory]: wrapped optimizer parameter
    """
    optim_param = to_local_if_dtensor(optim_param)
    if isinstance(model_param, ShardedTensorFactory):
        return replace(
            model_param.for_optimizer(), key=f'{prefix}.{model_param.key}', data=optim_param
        )

    assert tuple(optim_param.shape) == model_param.local_shape, (
        f'Optimizer shape ({tuple(optim_param.shape)} does not match model shape '
        f'({model_param.local_shape})'
    )
    sh_ten = replace(
        model_param, key=f'{prefix}.{model_param.key}', data=optim_param, dtype=optim_param.dtype
    )
    sh_ten.validate_metadata_integrity()
    return sh_ten


def _make_sharded_optimizer_factory(
    model_parts: Iterable[ShardedTensor],
    optim_param: torch.Tensor,
    key: str,
    *,
    physical_numel: int,
    replica_id: ReplicaId,
    flattened_range: Optional[slice] = None,
) -> ShardedTensorFactory:
    """Map a physical optimizer tensor onto ordered logical model fragments without gathering.

    ``model_parts`` describe consecutive sections of this rank's local parameter, in flat
    order, as the model checkpoint saves them (e.g. GTP local fragments). The returned
    factory splits an optimizer tensor of the parameter's shape, or a flat slice of it given
    by ``flattened_range``, into boxes under the same keys and global coordinates (see
    :func:`flat_range_chunks`). Only trailing alignment padding may extend past the sections:
    it is not saved and loads as zeros.

    Args:
        model_parts: metadata of the logical fragments; data is ignored. Their keys must start
            with ``key`` so that later key rewrites of the factory carry over.
        optim_param: optimizer tensor, or slice, to bind.
        key: the model factory's key.
        physical_numel: element count of the local parameter, padding included.
        replica_id: replica ID of the optimizer state.
        flattened_range: flat slice of the parameter that ``optim_param`` holds.

    Raises:
        ValueError: on inconsistent keys or sizes, or an invalid flat slice.
    """
    templates = [part.without_data() for part in model_parts]
    if any(not part.key.startswith(key) for part in templates):
        raise ValueError(f'Optimizer section keys must share the model factory key prefix {key}')
    logical_numel = sum(math.prod(part.local_shape) for part in templates)
    if logical_numel > physical_numel:
        raise ValueError(
            f'{key}: logical sections hold {logical_numel} elements, more than the '
            f'{physical_numel} of the physical parameter'
        )
    base_key = key

    @torch.no_grad()
    def build(key, data, replica_id, flattened_range):
        if flattened_range is None:
            start, stop = 0, physical_numel
        else:
            start, stop = flattened_range.start, flattened_range.stop
            if (
                flattened_range.step not in (None, 1)
                or start is None
                or stop is None
                or not 0 <= start <= stop <= physical_numel
            ):
                raise ValueError(f'Invalid optimizer fragment {flattened_range} of {key}')
        if data.numel() != stop - start:
            raise ValueError(
                f'{key}: optimizer data has {data.numel()} elements, but its parameter slice '
                f'[{start}, {stop}) has {stop - start}'
            )
        flat = data.detach().reshape(-1)
        section_start, saved_numel = 0, 0
        chunks = []
        for template in templates:
            section_numel = math.prod(template.local_shape)
            lo, hi = max(start, section_start), min(stop, section_start + section_numel)
            if lo < hi:
                for chunk in flat_range_chunks(
                    template.local_shape, lo - section_start, hi - section_start
                ):
                    count = math.prod(chunk.shape)
                    offset = list(template.global_offset)
                    for axis, coordinate in enumerate(chunk.coordinates, template.prepend_axis_num):
                        offset[axis] += coordinate
                    # DCP's chunk extents and the tensor view must expose the same dimensions,
                    # so prepended (e.g. expert) axes become explicit singleton axes.
                    chunk_shape = (1,) * template.prepend_axis_num + chunk.shape
                    begin = section_start + chunk.start - start
                    chunks.append(
                        replace(
                            template,
                            key=key + template.key[len(base_key) :],
                            data=flat[begin : begin + count].view(chunk_shape),
                            dtype=data.dtype,
                            local_shape=chunk_shape,
                            global_offset=tuple(offset),
                            axis_fragmentations=None,
                            prepend_axis_num=0,
                            flattened_range=None,
                            replica_id=replica_id,
                        )
                    )
                    saved_numel += count
            section_start += section_numel
        state = {
            # Shape, unsaved trailing padding and a dtype/device prototype for the merge.
            'layout': LocalNonpersistentObject(
                (tuple(data.shape), data.numel() - saved_numel, data.new_empty(0))
            )
        }
        if chunks:
            # An empty list would end up in the common (rank-0) state, so leave it out; the
            # merge treats a missing entry as no chunks.
            state['chunks'] = chunks
        return state

    @torch.no_grad()
    def merge(state):
        output_shape, padding, prototype = state['layout']
        chunks = state.get('chunks', [])
        flat = torch.cat([chunk.reshape(-1) for chunk in chunks]) if chunks else prototype
        if padding:
            flat = torch.cat((flat, flat.new_zeros(padding)))
        return flat.reshape(output_shape)

    return ShardedTensorFactory(key, optim_param, build, merge, replica_id, flattened_range)


def optim_state_to_sharding_state(
    optim_state_dict: StateDict,
    id_to_sharded_param_map: Dict[int, ShardedTensor],
    exclude_keys: Tuple[str] = (),
):
    """Turn optimizer state dict to sharded state dict based on model state dict *in-place*.

    Can be used to add sharding information to most common optimizer state dict.
    Creates separate ShardedTensors for each key in `optim_state_dict['state']`
    (e.g. for torch.optim.Adam there will be separate tensors for `exp_avg` and `exp_avg_sq`)

    Args:
        optim_state_dict (StateDict): optimizer state dict with
            state parameters under `state` key and group hyperparameters under
            `param_groups` -> `params` key.
        id_to_sharded_param_map (Dict[int, ShardedTensor]): mapping from optimizer param ids
            to model sharded tensors. Can be generated with `get_param_id_to_sharded_param_map`
            function.
        exclude_keys (Tuple[str]): optimizer state keys to exclude from the final state dict.

    Returns:
        None: state dict is modified in place
    """
    sharded_state = {}
    for param_id, param_state in optim_state_dict['state'].items():
        sharded_state[param_id] = {}
        for state_key, param in param_state.items():
            if state_key in exclude_keys:
                continue
            if param_id in id_to_sharded_param_map:
                sharded_state[param_id][state_key] = make_sharded_optimizer_tensor(
                    id_to_sharded_param_map[param_id], param, prefix=f'optimizer.state.{state_key}'
                )
            else:
                raise ValueError(f'Param id {param_id} does not match any model sharded param')

    optim_state_dict['param_groups'] = deepcopy(optim_state_dict['param_groups'])
    for group in optim_state_dict['param_groups']:
        group['params'] = LocalNonpersistentObject(group['params'])
    optim_state_dict['state'] = sharded_state

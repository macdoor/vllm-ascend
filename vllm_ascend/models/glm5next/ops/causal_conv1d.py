# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AscendC short convolution for GLM prefill, decode and MTP verification."""

import torch
from fla_npu.ops.ascendc import causal_conv1d_fn, causal_conv1d_update
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend import envs
from vllm_ascend.ops.triton.kda.conv_state import copy_conv_state


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    *,
    run_mode: int,
    initial_state_mode: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    max_query_len: int = -1,
) -> torch.Tensor:
    """Consume GDN metadata and update the caller's [cache, state_len, dim] state."""
    # Padded requests can be skipped by the kernel; their output must stay zero.
    output = torch.zeros_like(x)
    if cache_indices.shape[0] == 0:
        return output
    if envs.VLLM_ASCEND_ENABLE_CAUSAL_CONV1D_V2:
        # aclnnCausalConv1dV2 fast path: dim-last x, host-side metadata,
        # spec-decode aware. GLM pads cache_indices with PAD_SLOT_ID (-1),
        # which matches the binding's hard-coded null_block_id=-1.
        indices = cache_indices[:, 0] if cache_indices.dim() == 2 else cache_indices
        kernel_indices = indices.to(torch.int32).contiguous()
        qsl = query_start_loc.to(torch.int32).contiguous()
        accepted = None
        if num_accepted_tokens is not None:
            accepted = num_accepted_tokens.to(torch.int32).contiguous()
        initial = None
        if initial_state_mode is not None:
            initial = (
                initial_state_mode
                if initial_state_mode.dtype == torch.bool
                else initial_state_mode.to(torch.int32)
            )
        # Stage non-contiguous state rows for this batch only, keeping page
        # strides and DS layouts; write mutations back after the kernel.
        kernel_state = conv_state
        staged = False
        if not conv_state.is_contiguous():
            requests = kernel_indices.shape[0]
            state_len, dim = conv_state.shape[1:]
            kernel_state = torch.empty((requests, state_len, dim), dtype=conv_state.dtype, device=conv_state.device)
            kernel_indices = torch.empty(requests, dtype=torch.int32, device=kernel_indices.device)
            copy_conv_state(conv_state, kernel_state, cache_indices, query_start_loc, kernel_indices, write_back=False)
            staged = True
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            output,
            x,
            weight,
            kernel_state,
            None,
            qsl,
            kernel_indices,
            initial,
            accepted,
            1,
            PAD_SLOT_ID,
            run_mode,
            x.shape[0] if run_mode == 0 else max_query_len,
        )
        if staged:
            copy_conv_state(conv_state, kernel_state, cache_indices, query_start_loc, kernel_indices, write_back=True)
        return output
    kernel_state = conv_state
    kernel_indices = cache_indices
    null_block_id = 0
    # aclnnCausalConv1d materializes a non-contiguous state without writing its
    # mutations back to the view. Stage only this batch's rows, retaining both
    # page strides and DS layouts; never copy the entire persistent cache.
    if not conv_state.is_contiguous():
        requests = cache_indices.shape[0]
        state_len, dim = conv_state.shape[1:]
        kernel_state = torch.empty((requests, state_len, dim), dtype=conv_state.dtype, device=conv_state.device)
        kernel_indices = torch.empty(requests, dtype=torch.int32, device=cache_indices.device)
        copy_conv_state(conv_state, kernel_state, cache_indices, query_start_loc, kernel_indices, write_back=False)
        # Packed rows start at zero, so zero is now a valid cache slot. Invalid
        # requests were mapped to PAD_SLOT_ID by the gather kernel.
        null_block_id = PAD_SLOT_ID
    kernel_indices = kernel_indices.contiguous()
    if run_mode == 0:
        result = causal_conv1d_fn(
            x,
            weight,
            None,
            conv_states=kernel_state,
            query_start_loc=query_start_loc,
            cache_indices=kernel_indices,
            has_initial_state=initial_state_mode,
            activation="silu",
            pad_slot_id=PAD_SLOT_ID,
            null_block_id=null_block_id,
        )
    else:
        result = causal_conv1d_update(
            x,
            kernel_state,
            weight,
            bias=None,
            activation="silu",
            conv_state_indices=kernel_indices,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=query_start_loc,
            null_block_id=null_block_id,
        )
    if not conv_state.is_contiguous():
        copy_conv_state(conv_state, kernel_state, cache_indices, query_start_loc, kernel_indices, write_back=True)
    return result

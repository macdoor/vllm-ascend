# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pooled-cache physical views and copy-on-write layout for GLM-Next."""

from collections.abc import Callable, Iterable, Sequence

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheSpec

from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    AscendMLAAttentionSpec,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
)


def _row_major_strides(shape: tuple[int, ...]) -> list[int]:
    strides = [1] * len(shape)
    for dim_idx in range(len(shape) - 2, -1, -1):
        strides[dim_idx] = strides[dim_idx + 1] * shape[dim_idx + 1]
    return strides


def _view_kpool_tail_cache(
    layer_name: str,
    kv_cache_spec: AscendIndexerKPoolTailSpec,
    raw_cache: torch.Tensor,
    num_blocks: int,
) -> list[torch.Tensor]:
    if not isinstance(raw_cache, torch.Tensor):
        raise ValueError(f"KPool tail cache for {layer_name} must use one raw tensor.")
    typed_slot = raw_cache.view(kv_cache_spec.dtype)
    dtype_size = get_dtype_size(kv_cache_spec.dtype)
    page_el = typed_slot.numel() // num_blocks if num_blocks else 0
    tail_block_el = kv_cache_spec.unpadded_page_size_bytes // dtype_size
    if num_blocks and tail_block_el > page_el:
        raise ValueError(
            f"KPool tail cache for {layer_name} does not fit one small page: "
            f"tail={tail_block_el} elements, page={page_el} elements."
        )
    return [
        torch.as_strided(
            typed_slot,
            size=(num_blocks, 2, kv_cache_spec.block_size, kv_cache_spec.head_size),
            stride=(page_el, kv_cache_spec.block_size * kv_cache_spec.head_size, kv_cache_spec.head_size, 1),
        )
    ]


def _view_compressed_indexer_cache(
    layer_name: str,
    kv_cache_spec: AscendMLAAttentionSpec,
    raw_cache: torch.Tensor | tuple[torch.Tensor, ...],
    attn_backend: AttentionBackend,
    kernel_block_size: int,
) -> tuple[torch.Tensor]:
    if isinstance(raw_cache, tuple):
        if len(raw_cache) != 1:
            raise ValueError(f"Compressed indexer cache for {layer_name} must be a single tensor.")
        raw_single = raw_cache[0]
    else:
        raw_single = raw_cache
    compression_ratio = get_kv_cache_compression_ratio(kv_cache_spec)
    indexer_kernel_block_size = kernel_block_size // compression_ratio
    num_blocks = raw_single.numel() // kv_cache_spec.page_size_bytes
    num_blocks_per_kv_block = get_storage_block_size(kv_cache_spec) // indexer_kernel_block_size
    shape = tuple(
        attn_backend.get_kv_cache_shape(
            num_blocks * num_blocks_per_kv_block,
            indexer_kernel_block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
        )
    )
    strides = _row_major_strides(shape)
    typed_slot = raw_single.view(kv_cache_spec.dtype)
    if strides[0] * shape[0] != typed_slot.numel():
        raise ValueError(
            f"Compressed indexer cache for {layer_name} does not exactly fill the small slot: "
            f"packed={strides[0] * shape[0]} elements, slot={typed_slot.numel()}."
        )
    return (torch.as_strided(typed_slot, size=shape, stride=tuple(strides)),)


def _view_nope_main_mla_cache(
    kv_cache_spec: AscendMLAAttentionSpec,
    raw_cache: torch.Tensor,
    attn_backend: AttentionBackend,
    kernel_block_size: int,
) -> list[torch.Tensor]:
    num_blocks = raw_cache.numel() // kv_cache_spec.page_size_bytes
    num_blocks_per_kv_block = get_storage_block_size(kv_cache_spec) // kernel_block_size
    shape = tuple(
        attn_backend.get_kv_cache_shape(
            num_blocks * num_blocks_per_kv_block,
            kernel_block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
        )
    )
    strides = _row_major_strides(shape)
    typed_slot = raw_cache.view(kv_cache_spec.dtype)
    k_cache = torch.as_strided(typed_slot, size=shape, stride=tuple(strides))
    rope_cache = torch.as_strided(
        typed_slot,
        size=(*shape[:-1], 0),
        stride=tuple(strides[:-1]) + (1,),
    )
    return [k_cache, rope_cache]


def view_glm5_next_cache(
    layer_name: str,
    kv_cache_spec: KVCacheSpec,
    raw_cache: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    attn_backend: AttentionBackend,
    kernel_block_size: int,
    num_blocks: int,
    get_kv_cache_dims: Callable[[str, AttentionSpec], tuple[int, int]],
) -> list[torch.Tensor] | tuple[torch.Tensor, ...] | None:
    """Build the pooled-cache views for one GLM-Next layer.

    Returns the view container the runner registers for the layer, or None
    when the spec is not owned by the GLM-Next pooled layout, so the caller
    falls through to its generic reshape paths.
    """
    if isinstance(kv_cache_spec, AscendIndexerKPoolTailSpec):
        return _view_kpool_tail_cache(layer_name, kv_cache_spec, raw_cache, num_blocks)
    if isinstance(kv_cache_spec, AscendMLAAttentionSpec) and getattr(
        kv_cache_spec, "indexes_kv_by_block_stride", False
    ):
        if get_kv_cache_compression_ratio(kv_cache_spec) > 1:
            return _view_compressed_indexer_cache(layer_name, kv_cache_spec, raw_cache, attn_backend, kernel_block_size)
        _k_dim, v_dim = get_kv_cache_dims(layer_name, kv_cache_spec)
        if v_dim == 0 and kv_cache_spec.page_size_bytes == kv_cache_spec.unpadded_page_size_bytes:
            return _view_nope_main_mla_cache(kv_cache_spec, raw_cache, attn_backend, kernel_block_size)
    return None


def build_kv_cache_copy_views(
    kv_cache_config: KVCacheConfig,
    get_layer_cache: Callable[[str], torch.Tensor | Sequence[torch.Tensor]],
    runner_caches: Iterable[torch.Tensor | Sequence[torch.Tensor]],
) -> list[torch.Tensor]:
    """Expose one raw page per shared slot to upstream block copies.

    ``copy_kv_cache_blocks_inplace`` copies the first tensor with a given
    ``data_ptr`` and skips the rest. Indexer rows and the tail ring share a
    small page but use different shapes, so the copy list has to be that
    page. Layer bindings are left unchanged. Runner views that start at a
    page already listed here are omitted; other caches are copied as
    themselves. Packed regions that do not share a page origin stay separate.
    """
    page_views: list[torch.Tensor] = []
    covered_origins: set[int] = set()
    num_blocks = kv_cache_config.num_blocks
    for descriptor in kv_cache_config.kv_cache_tensors:
        if (
            len(descriptor.layers) < 2
            or descriptor.layer_stride != 0
            or num_blocks <= 0
            or descriptor.block_stride <= 0
        ):
            continue
        caches = [get_layer_cache(name) for name in descriptor.layers]
        first_components = [cache if isinstance(cache, torch.Tensor) else cache[0] for cache in caches]
        first = first_components[0]
        if any(tensor.data_ptr() != first.data_ptr() for tensor in first_components[1:]):
            continue
        storage = first.untyped_storage()
        base = first.storage_offset() * first.element_size()
        page_bytes = num_blocks * descriptor.block_stride
        if base + page_bytes > storage.nbytes():
            raise ValueError(
                f"Shared KV page does not fit its storage: offset={base}, page={page_bytes}, storage={storage.nbytes()}."
            )
        page_views.append(
            torch.empty(0, dtype=torch.uint8, device=first.device).set_(
                storage, base, (num_blocks, descriptor.block_stride)
            )
        )
        covered_origins.add(first.data_ptr())

    copy_views = list(page_views)
    for cache in runner_caches:
        for tensor in (cache,) if isinstance(cache, torch.Tensor) else cache:
            if tensor.numel() == 0 or tensor.data_ptr() in covered_origins:
                continue
            copy_views.append(tensor)
    return copy_views

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import (
    AttentionGroup,
    prepare_kernel_block_sizes,
    select_common_block_size,
)

from vllm_ascend.utils import vllm_version_is


def build_kv_cache_copy_views(
    kv_cache_config: KVCacheConfig,
    get_layer_cache: Callable[[str], torch.Tensor | Sequence[torch.Tensor]],
    runner_caches: Iterable[torch.Tensor | Sequence[torch.Tensor]],
) -> list[torch.Tensor]:
    """Expose complete shared pages to COW without changing layer bindings.

    Layers can alias physical pages. Copy each shared page only once,
    preserving the logical views bound to layers.
    """
    copy_caches: list[torch.Tensor] = []
    shared_page_ranges: list[tuple[torch.device, int, int]] = []
    for descriptor in kv_cache_config.kv_cache_tensors:
        if len(descriptor.layers) < 2 or descriptor.layer_stride != 0:
            continue
        caches = [get_layer_cache(name) for name in descriptor.layers]
        first_components = [cache if isinstance(cache, torch.Tensor) else cache[0] for cache in caches]
        first = first_components[0]
        # A shared origin proves aliasing, but not block-major allocation:
        # K/V may use separate storages or separate segments of one storage.
        if any(tensor.data_ptr() != first.data_ptr() for tensor in first_components[1:]):
            continue
        storage = first.untyped_storage()
        base = first.storage_offset() * first.element_size()
        components: list[torch.Tensor] = []
        for cache in caches:
            if isinstance(cache, torch.Tensor):
                tensors = cache.unbind(0) if cache.ndim == 5 and cache.shape[0] == 2 else (cache,)
            else:
                tensors = cache
            components.extend(tensor for tensor in tensors if tensor.numel())
        if not components or base + descriptor.size > storage.nbytes():
            continue
        # Every component must advance by one physical page per scheduler
        # block and fit entirely within that page, including kernel splits.
        for tensor in components:
            if (
                tensor.device != first.device
                or tensor.untyped_storage().data_ptr() != storage.data_ptr()
                or tensor.shape[0] % kv_cache_config.num_blocks
            ):
                break
            blocks = tensor.unflatten(0, (kv_cache_config.num_blocks, tensor.shape[0] // kv_cache_config.num_blocks))
            page_end = (
                tensor.data_ptr()
                - first.data_ptr()
                + tensor.element_size()
                * (1 + sum((size - 1) * stride for size, stride in zip(blocks.shape[1:], blocks.stride()[1:])))
            )
            if (
                blocks.stride(0) * tensor.element_size() != descriptor.block_stride
                or tensor.data_ptr() < first.data_ptr()
                or page_end > descriptor.block_stride
            ):
                break
        else:
            pages = torch.empty(0, dtype=torch.uint8, device=first.device).set_(
                storage, base, (kv_cache_config.num_blocks, descriptor.block_stride)
            )
            copy_caches.append(pages)
            shared_page_ranges.append((pages.device, pages.data_ptr(), pages.data_ptr() + pages.numel()))

    for cache in runner_caches:
        for tensor in (cache,) if isinstance(cache, torch.Tensor) else cache:
            if tensor.numel() == 0:
                continue
            # Layer bindings and runner tensors can be distinct objects over
            # the same pages. Keep unrelated regions in a shared storage.
            if any(
                tensor.device == device and start <= tensor.data_ptr() < end
                for device, start, end in shared_page_ranges
            ):
                continue
            copy_caches.append(tensor)
    return copy_caches


def copy_kv_cache_blocks_inplace(
    kv_caches: Iterable[torch.Tensor | Sequence[torch.Tensor | None] | None],
    num_blocks: int,
    kv_cache_block_copies: Sequence[KVCacheBlockCopy],
) -> None:
    """Copy aliased pages and independent block-indexed cache segments.

    Equal origins do not imply equal views. Copy a shared physical page once
    only when every overlapping component satisfies its block-major layout;
    otherwise preserve component-wise copying, including segmented storage.
    A scheduler block may span multiple kernel-level blocks.
    """
    if not kv_cache_block_copies:
        return

    cache_tensors: list[torch.Tensor] = []
    seen_tensors: set[tuple[torch.device, int] | tuple[torch.device, int, tuple[int, ...], tuple[int, ...], int]] = (
        set()
    )
    aliased_origins: set[int] = set()
    seen_origins: set[int] = set()
    for entry in kv_caches:
        if entry is None:
            continue
        if isinstance(entry, torch.Tensor):
            tensors = entry.unbind(0) if entry.ndim == 5 and entry.shape[0] == 2 else (entry,)
        else:
            tensors = entry
        for tensor in tensors:
            if tensor is None or not tensor.numel():
                continue
            data_ptr = tensor.data_ptr()
            # Hybrid models can overlay layers of different kinds on one
            # allocation, so views at the same address can differ in geometry.
            # Main keys on that geometry; the release lane keys on (device, address).
            key = (
                (tensor.device, data_ptr)
                if vllm_version_is("0.33.0")
                else (tensor.device, data_ptr, tuple(tensor.shape), tuple(tensor.stride()), tensor.element_size())
            )
            if key in seen_tensors:
                continue
            seen_tensors.add(key)
            # Equal origins do not imply equal views: track addresses reached
            # through distinct geometries so aliased shared pages can be
            # collapsed into a single whole-page copy below.
            if data_ptr in seen_origins:
                aliased_origins.add(data_ptr)
            seen_origins.add(data_ptr)
            cache_tensors.append(tensor)

    if not cache_tensors:
        return

    # A paired hybrid slot exposes a contiguous main-page view and strided
    # recurrent component views of the same storage. Copying both can replay a
    # cyclic copy over already-updated source bytes. Keep the enclosing view
    # once; independent indexer key/scale planes still need their own copies.
    contiguous_regions = [
        (
            i,
            tensor.untyped_storage().data_ptr(),
            tensor.data_ptr(),
            tensor.data_ptr() + tensor.numel() * tensor.element_size(),
        )
        for i, tensor in enumerate(cache_tensors)
        if tensor.is_contiguous()
    ]
    selected = []
    for i, tensor in enumerate(cache_tensors):
        storage = tensor.untyped_storage().data_ptr()
        start = tensor.data_ptr()
        span = (
            1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
        ) * tensor.element_size()
        end = start + span
        enclosed = any(
            storage == parent_storage
            and parent_start <= start
            and end <= parent_end
            and (parent_end - parent_start > span or parent_index < i)
            for parent_index, parent_storage, parent_start, parent_end in contiguous_regions
            if parent_index != i
        )
        if not enclosed:
            selected.append(tensor)
    cache_tensors = selected

    device = cache_tensors[0].device
    for tensor in cache_tensors:
        assert tensor.device == device
        assert tensor.shape[0] % num_blocks == 0

    copy_pages: list[torch.Tensor] = []
    page_ranges: list[tuple[int, int]] = []
    for tensor in sorted(cache_tensors, key=torch.Tensor.data_ptr) if aliased_origins else ():
        start = tensor.data_ptr()
        if any(begin <= start < end for begin, end in page_ranges):
            continue
        blocks = tensor.unflatten(0, (num_blocks, tensor.shape[0] // num_blocks))
        page_bytes = blocks.stride(0) * tensor.element_size()
        storage = tensor.untyped_storage()
        base = tensor.storage_offset() * tensor.element_size()
        if (
            page_bytes <= 0
            or base + num_blocks * page_bytes > storage.nbytes()
            or not any(start <= origin < start + page_bytes for origin in aliased_origins)
        ):
            continue
        # Aliasing alone does not prove a page layout. All overlapping views
        # must advance by the same physical page and fit inside its first row.
        for component in cache_tensors:
            if component.untyped_storage().data_ptr() != storage.data_ptr():
                continue
            offset = component.data_ptr() - start
            if offset >= num_blocks * page_bytes:
                continue
            component_blocks = component.unflatten(0, (num_blocks, component.shape[0] // num_blocks))
            component_page_bytes = component_blocks.stride(0) * component.element_size()
            page_end = offset + component.element_size() * (
                1
                + sum(
                    (size - 1) * stride
                    for size, stride in zip(component_blocks.shape[1:], component_blocks.stride()[1:])
                )
            )
            if page_end + (num_blocks - 1) * component_page_bytes <= 0:
                continue
            if offset < 0 or component_page_bytes != page_bytes or page_end > page_bytes:
                break
        else:
            pages = torch.empty(0, dtype=torch.uint8, device=device).set_(storage, base, (num_blocks, page_bytes))
            copy_pages.append(pages)
            page_ranges.append((start, start + pages.numel()))

    cache_tensors = copy_pages + [
        tensor for tensor in cache_tensors if not any(start <= tensor.data_ptr() < end for start, end in page_ranges)
    ]
    for tensor in cache_tensors:
        # FP8 is storage here. NPU stack/index operators need byte views;
        # same-size bitcasts preserve the non-contiguous page geometry.
        if tensor.dtype == torch.float8_e4m3fn:
            tensor = tensor.view(torch.int8)
        kernel_blocks_per_block = tensor.shape[0] // num_blocks
        # Page-strided MLA views are non-contiguous, so a flat ``view`` would
        # either fail or materialize a copy. Split only dim 0 to preserve the
        # original strided storage while copying every kernel block in a page.
        blocks = torch.as_strided(
            tensor,
            size=(num_blocks, kernel_blocks_per_block, *tensor.shape[1:]),
            stride=(tensor.stride(0) * kernel_blocks_per_block, tensor.stride(0), *tensor.stride()[1:]),
            storage_offset=tensor.storage_offset(),
        )
        source_blocks = torch.stack([blocks[copy.src_block_id] for copy in kv_cache_block_copies])
        for index, copy in enumerate(kv_cache_block_copies):
            blocks[copy.dst_block_id].copy_(source_blocks[index])


def _group_tensor_strides(
    kv_cache_config: KVCacheConfig,
    layer_names: Sequence[str],
) -> list[tuple[int, int]]:
    """``(layer_stride, block_stride)`` of every tensor holding ``layer_names``."""
    return [
        (tensor.layer_stride, tensor.block_stride)
        for tensor in kv_cache_config.kv_cache_tensors
        if not set(tensor.layers).isdisjoint(layer_names)
    ]


def prepare_ascend_kernel_block_sizes(
    kv_cache_config: KVCacheConfig,
    attn_groups: list[list[AttentionGroup]],
) -> list[int]:
    """Upstream ``prepare_kernel_block_sizes`` with an alias-aware packing check.

    vLLM #57169 refuses to split a cache group whose KV tensors are
    block-outermost packed, detected as ``block_stride > layer_stride``. Ascend
    layerwise reuse aliases several layers inside one physical slot, so those
    tensors carry ``layer_stride == 0`` and trip that test although their pages
    stay dense and contiguous. The group then keeps the manager block size,
    leaving NoPE SFA addressing 256-token blocks while its backend, metadata
    builder and block table all address 128-token kernel blocks. Re-split the
    groups whose only evidence of packing is aliasing.

    Args:
        kv_cache_config: The KV cache configuration.
        attn_groups: Attention groups indexed by KV cache group id.

    Returns:
        List of kernel block sizes for each cache group.
    """
    kernel_block_sizes = prepare_kernel_block_sizes(kv_cache_config, attn_groups)

    # Upstream drops only encoder-only groups, so this index tracks its output.
    index = 0
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = next(iter(spec.kv_cache_specs.values()))
        if isinstance(spec, EncoderOnlyAttentionSpec):
            continue
        if index >= len(kernel_block_sizes):
            break
        manager_block_size = group.kv_cache_spec.block_size
        strides = _group_tensor_strides(kv_cache_config, group.layer_names)
        block_outermost_packed = any(0 < layer_stride < block_stride for layer_stride, block_stride in strides)
        aliases_layers = any(layer_stride == 0 and block_stride > 0 for layer_stride, block_stride in strides)
        if (
            spec.has_layer_views
            and isinstance(spec, AttentionSpec)
            and aliases_layers
            and not block_outermost_packed
            and kernel_block_sizes[index] == manager_block_size
            and group_id < len(attn_groups)
            and attn_groups[group_id]
        ):
            kernel_block_sizes[index] = select_common_block_size(
                manager_block_size,
                [attn_group.backend for attn_group in attn_groups[group_id]],
                [attn_group.kv_cache_spec for attn_group in attn_groups[group_id]],
            )
        index += 1
    return kernel_block_sizes


def get_single_raw_mla_backing(raw_cache: object) -> torch.Tensor | None:
    """Extract one raw MLA backing from either allocator representation.

    Pure MLA allocations use a one-element tuple so generic K/V unpacking does
    not mistake the backing for a `(key, value)` pair. Hybrid/shared pools can
    instead store a bare tensor slice for an MLA layer.
    """
    if isinstance(raw_cache, torch.Tensor):
        return raw_cache
    if isinstance(raw_cache, tuple) and len(raw_cache) == 1 and isinstance(raw_cache[0], torch.Tensor):
        return raw_cache[0]
    return None


def row_major_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """Return row-major strides for a shape without allocating tensor storage."""
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]
    return tuple(strides)


def make_page_strided_cache_view(
    raw_tensor: torch.Tensor,
    shape: Sequence[int],
    dtype: torch.dtype,
    page_size_bytes: int,
    offset_bytes: int = 0,
) -> torch.Tensor:
    """Create a first-axis page-strided view over a raw cache allocation."""
    dtype_size = get_dtype_size(dtype)
    strides = row_major_strides(tuple(shape))
    storage_offset_bytes = raw_tensor.storage_offset() * raw_tensor.element_size() + offset_bytes
    assert storage_offset_bytes % dtype_size == 0
    return torch.as_strided(
        raw_tensor.view(dtype),
        size=tuple(shape),
        stride=(page_size_bytes // dtype_size, *strides[1:]),
        storage_offset=storage_offset_bytes // dtype_size,
    )


@contextmanager
def disable_compilation(model: torch.nn.Module) -> Iterator[None]:
    """Temporarily disable compilation for all nested compiled modules."""
    # Multimodal wrappers can nest the compiled backbone below language_model.
    # Snapshot before changing flags; modules() also deduplicates shared modules.
    previous_states = [
        (module, module.do_not_compile) for module in model.modules() if hasattr(module, "do_not_compile")
    ]
    try:
        for module, _ in previous_states:
            module.do_not_compile = True
        yield
    finally:
        for module, previous in reversed(previous_states):
            module.do_not_compile = previous

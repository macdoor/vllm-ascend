# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for GLM-Next model-runner pooled cache views."""

from contextlib import nullcontext
from dataclasses import replace
from itertools import permutations
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import (
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    AscendMLAAttentionSpec,
    requires_padded_page_layout,
)
from vllm_ascend.models.glm5next.cache_config import (
    get_glm5_next_kv_cache_config,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.utils import get_kv_cache_tensor_layers
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.block_table import AscendBlockTables
from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2

MAIN = "model.layers.1.attn"
INDEXER = "model.layers.1.indexer.k_cache"
STATE = "model.layers.1.indexer.tail_cache"
MAMBA = "model.layers.0.linear_attn"


def _ratio_kwargs(ratio: int) -> dict[str, int]:
    return {"tokens_per_state": ratio}


class _AttentionBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        cache_dtype=None,
        **_kwargs,
    ):
        return num_blocks, block_size, num_kv_heads, head_size


class _StateBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks,
        block_size,
        _num_kv_heads,
        head_size,
        **_kwargs,
    ):
        return num_blocks, 2, block_size, head_size


def _make_config():
    return SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(max_model_len=64, dtype=torch.bfloat16),
        attention_config=SimpleNamespace(indexer_kv_dtype="auto"),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False, max_num_seqs=8),
        max_in_flight_tokens=8,
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            num_gpu_blocks_override=None,
            mamba_cache_mode="none",
            enable_prefix_caching=False,
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(
                is_layer_compact=True,
                is_block_compact=True,
            ),
        ),
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


def _make_specs(main_head_size=4):
    return {
        MAIN: AscendMLAAttentionSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=main_head_size,
            dtype=torch.bfloat16,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
        ),
        INDEXER: AscendMLAAttentionSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.bfloat16,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
            **_ratio_kwargs(2),
        ),
        STATE: AscendIndexerKPoolTailSpec(
            block_size=2,
            sliding_window=2,
            compress_ratio=2,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
        ),
        MAMBA: MambaSpec(
            block_size=8,
            shapes=((2, 2), (1, 2, 2)),
            dtypes=(torch.bfloat16, torch.float32),
        ),
    }


def _make_runner(config, main_cache_dims=(4, 0), runner_cls=NPUModelRunner):
    runner = runner_cls.__new__(runner_cls)
    runner.device = torch.device("cpu")
    runner.vllm_config = config
    runner.dcp_size = config.parallel_config.decode_context_parallel_size
    runner.ascend_config = SimpleNamespace(
        kvpp_config=SimpleNamespace(size=1),
        xlite_graph_config=SimpleNamespace(enabled=False),
    )
    runner.compilation_config = config.compilation_config
    runner.runner_only_attn_layers = set()
    runner.shared_kv_cache_layers = {}
    runner.kv_caches = []
    runner.use_sparse = False
    # GLM-Next exposes its layout through KV specs and does not define the
    # legacy ``compress_ratios`` config used to derive this runner flag.
    runner.use_compress = False
    runner.use_hybrid_blocks = True
    runner.sparse_kv_offload_enabled = False
    runner.sparse_kv_offload_config = SimpleNamespace(enabled=False)
    runner.tp_rank = 0
    runner.attn_backend = _AttentionBackend
    runner.kernel_block_sizes = [[8], [2], [0]]
    # The runner gates GLM-Next reshape views on the spec-carried
    # model_version marker, not on model_config.
    runner.model_config = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm5_next"))

    specs = _make_specs()
    attn_groups = [
        SimpleNamespace(
            backend=_AttentionBackend,
            kv_cache_spec=specs[MAIN],
            layer_names=[MAIN],
            kv_cache_group_id=0,
        ),
        SimpleNamespace(
            backend=_AttentionBackend,
            kv_cache_spec=specs[INDEXER],
            layer_names=[INDEXER],
            kv_cache_group_id=0,
        ),
        SimpleNamespace(
            backend=_StateBackend,
            kv_cache_spec=specs[STATE],
            layer_names=[STATE],
            kv_cache_group_id=1,
        ),
        SimpleNamespace(
            backend=None,
            kv_cache_spec=specs[MAMBA],
            layer_names=[MAMBA],
            kv_cache_group_id=2,
        ),
    ]
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(attn_groups)
    runner._get_attention_kv_cache_dims = lambda _name, _spec: main_cache_dims
    return runner


def _make_plan(num_blocks=3, main_head_size=4):
    # Match production: vLLM registers built-in specs before the Ascend hook.
    register_all_kvcache_specs(None)
    config = _make_config()
    specs = _make_specs(main_head_size)
    groups = get_glm5_next_kv_cache_groups(config, specs)
    bytes_per_block = get_glm5_next_pool_bytes_per_block(groups)
    plan = get_glm5_next_kv_cache_config(
        config,
        groups,
        num_blocks * bytes_per_block,
    )
    return config, groups, plan


@pytest.fixture(params=["v1", "v2"])
def runner_factory(request, monkeypatch):
    def make_runner(config, main_cache_dims=(4, 0), kernel_block_size=8):
        runner = _make_runner(config, main_cache_dims)
        runner.kernel_block_sizes[0] = [kernel_block_size]
        if request.param == "v2":
            layers = {name: SimpleNamespace() for name in _make_specs()}
            main_layer = MLAAttention.__new__(MLAAttention)
            torch.nn.Module.__init__(main_layer)
            main_layer.kv_lora_rank, main_layer.qk_rope_head_dim = main_cache_dims
            layers[MAIN] = main_layer
            monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
            monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_args: layers)
            monkeypatch.setattr(attn_utils, "enable_sfa", lambda _config: False)
            monkeypatch.setattr(attn_utils, "enable_fa_quant", lambda _config: False)
            runner._allocate_kv_cache_tensors = lambda plan: attn_utils._allocate_kv_cache(plan, {}, runner.device)

            def reshape(plan, raw):
                specs = attn_utils._get_layer_kv_cache_specs(plan)
                groups = [group for group in runner._kv_cache_spec_attn_group_iterator() if group.layer_names[0] in raw]
                for group in groups:
                    group.kv_cache_spec = specs[group.layer_names[0]]
                return attn_utils._reshape_kv_cache_v2(
                    groups,
                    raw,
                    "auto",
                    [sizes[0] for sizes in runner.kernel_block_sizes],
                    {},
                    plan,
                )

            runner._reshape_kv_cache_tensors = reshape
        return runner

    return make_runner


def test_glm5_next_runner_allocates_contiguous_slot_backings(runner_factory):
    config, _, plan = _make_plan()
    runner = runner_factory(config)

    raw_caches = runner._allocate_kv_cache_tensors(plan)
    assert raw_caches[MAIN] is raw_caches[MAMBA]
    assert raw_caches[INDEXER] is raw_caches[STATE]
    assert raw_caches[MAIN] is not raw_caches[INDEXER]

    caches = runner._reshape_kv_cache_tensors(plan, raw_caches)
    descriptors = {
        name: descriptor for descriptor in plan.kv_cache_tensors for name in get_kv_cache_tensor_layers(descriptor)
    }
    main_cache, main_rope_cache = caches[MAIN]
    (indexer_cache,) = caches[INDEXER]
    (tail_cache,) = caches[STATE]
    assert main_cache.shape == (3, 8, 1, 4)
    assert main_rope_cache.shape == (3, 8, 1, 0)
    assert main_cache.is_contiguous()
    assert indexer_cache.shape == (3, 4, 1, 4)
    assert tail_cache.shape == (3, 2, 2, 1)
    assert [cache.shape for cache in caches[MAMBA]] == [
        (3, 2, 2),
        (3, 1, 2, 2),
    ]

    assert indexer_cache.is_contiguous()
    assert indexer_cache.data_ptr() == raw_caches[INDEXER].data_ptr()
    assert tail_cache.data_ptr() == raw_caches[STATE].data_ptr()
    slot = raw_caches[STATE]
    page_bytes = descriptors[STATE].size // plan.num_blocks
    slot_fp32 = slot.view(torch.float32)
    page_els_fp32 = page_bytes // slot_fp32.element_size()
    tail_block_els = tail_cache[0].numel()
    for block_index in range(plan.num_blocks):
        assert tail_cache[block_index].data_ptr() == slot.data_ptr() + block_index * page_bytes
    tail_cache[2].fill_(7)
    assert torch.all(slot_fp32[2 * page_els_fp32 : 2 * page_els_fp32 + tail_block_els] == 7)
    assert torch.count_nonzero(slot_fp32[2 * page_els_fp32 + tail_block_els : 3 * page_els_fp32]) == 0
    assert torch.count_nonzero(slot_fp32[: 2 * page_els_fp32]) == 0
    for cache in caches[MAMBA]:
        assert cache.stride(0) * cache.element_size() == descriptors[MAMBA].size // plan.num_blocks

    mamba_second_offset = caches[MAMBA][0][0].numel() * caches[MAMBA][0].element_size()
    assert caches[MAMBA][1].data_ptr() - raw_caches[MAMBA].data_ptr() == mamba_second_offset
    mamba_payload_size = sum(cache.numel() * cache.element_size() for cache in caches[MAMBA])
    assert mamba_payload_size < descriptors[MAMBA].size


def test_glm5_next_runner_splits_main_mla_components_within_each_page(runner_factory):
    config, _, plan = _make_plan(main_head_size=6)
    runner = runner_factory(config, main_cache_dims=(4, 2))

    raw_caches = runner._allocate_kv_cache_tensors(plan)
    caches = runner._reshape_kv_cache_tensors(plan, raw_caches)

    kv_c_cache, k_pe_cache = caches[MAIN]
    assert kv_c_cache.shape == (3, 8, 1, 4)
    assert k_pe_cache.shape == (3, 8, 1, 2)
    page_size = next(
        descriptor.size // plan.num_blocks
        for descriptor in plan.kv_cache_tensors
        if MAIN in get_kv_cache_tensor_layers(descriptor)
    )
    assert kv_c_cache.stride(0) * kv_c_cache.element_size() == page_size
    assert k_pe_cache.stride(0) * k_pe_cache.element_size() == page_size
    assert k_pe_cache.data_ptr() - raw_caches[MAIN].data_ptr() == kv_c_cache[0].numel() * kv_c_cache.element_size()


@pytest.fixture
def copy_cache(monkeypatch):
    """Install COW through cache initialization and use the inherited request entry."""

    def copy(plan, bindings, caches, physical_views):
        config = _make_config()
        config.compilation_config.static_forward_context = {
            name: SimpleNamespace(kv_cache=cache) for name, cache in bindings.items()
        }
        runner = _make_runner(config, runner_cls=NPUModelRunnerV2)
        runner.ascend_config.sparse_kv_offload_config = runner.sparse_kv_offload_config
        runner.is_last_pp_rank = False
        runner.max_num_reqs = config.scheduler_config.max_num_seqs
        runner.model_config.enable_return_routed_experts = False
        runner.model_config.architecture = "Glm5NextForCausalLM"
        runner.model_state = SimpleNamespace()
        runner.model = torch.nn.Module()
        runner.pcp_manager = runner.speculator = None
        runner.attn_groups = []
        runner.req_states = SimpleNamespace(
            num_computed_tokens_np=np.zeros(0, dtype=np.int32),
            prefill_len=SimpleNamespace(np=np.zeros(0, dtype=np.int32)),
            num_computed_prefill_tokens=np.zeros(0, dtype=np.int32),
        )

        def initialize(self, cache_config, kv_cache_allocation_context=None):
            self.kv_cache_config = cache_config
            self.block_tables = MagicMock(spec=AscendBlockTables)
            self.kv_caches = caches

        monkeypatch.setattr(GPUModelRunner, "initialize_kv_cache", initialize)
        with (
            patch("vllm_ascend.worker.v2.model_runner.graph_manager_wrapper", return_value=nullcontext()),
            patch("vllm_ascend.worker.v2.model_runner.KVPPRuntime.create_from_kv_cache"),
        ):
            runner.initialize_kv_cache(plan)
        storages = {
            tensor.untyped_storage().data_ptr(): torch.empty(0, dtype=torch.uint8).set_(tensor.untyped_storage())
            for tensor in physical_views
        }
        for storage in storages.values():
            storage.fill_(99)
        for tensor in physical_views:
            for block_id in range(plan.num_blocks):
                tensor[block_id].fill_(11 * (block_id + 1))
        expected = {ptr: storage.clone() for ptr, storage in storages.items()}
        for tensor in physical_views:
            view = torch.empty(0, dtype=tensor.dtype).set_(
                expected[tensor.untyped_storage().data_ptr()].untyped_storage(),
                tensor.storage_offset(),
                tensor.shape,
                tensor.stride(),
            )
            view[1:].copy_(tensor[:-1])
        scheduler_output = SchedulerOutput.make_empty()
        scheduler_output.kv_cache_block_copies = [
            KVCacheBlockCopy(src_block_id=i, dst_block_id=i + 1) for i in range(plan.num_blocks - 1)
        ]
        runner.update_requests(scheduler_output)
        for ptr, storage in storages.items():
            torch.testing.assert_close(storage, expected[ptr])
        for name, cache in bindings.items():
            assert runner.compilation_config.static_forward_context[name].kv_cache is cache

    return copy


@pytest.mark.parametrize("reverse_bindings", [False, True])
@pytest.mark.parametrize("main_cache_dims", [(4, 0), (4, 2)])
@pytest.mark.parametrize("kernel_block_size", [4, 8])
def test_mrv2_copy_on_write_preserves_pooled_pages_and_layer_bindings(
    runner_factory, copy_cache, reverse_bindings, main_cache_dims, kernel_block_size
):
    config, _, plan = _make_plan(main_head_size=sum(main_cache_dims))
    config.kv_transfer_config = SimpleNamespace(kv_connector="MooncakeConnectorV2")
    allocator = runner_factory(config, main_cache_dims, kernel_block_size)
    raw_caches = allocator._allocate_kv_cache_tensors(plan)
    caches = allocator._reshape_kv_cache_tensors(plan, raw_caches)
    runner_caches = [tensor.view_as(tensor) for cache in caches.values() for tensor in cache]
    if reverse_bindings:
        runner_caches.reverse()
    copy_cache(plan, caches, runner_caches, [raw_caches[name].view(plan.num_blocks, -1) for name in (MAIN, INDEXER)])


@pytest.mark.parametrize(
    "layout,alignment_slack,layer_stride,layer_count",
    [
        ("split", 0, 0, 2),
        ("split", 64, 0, 2),
        ("segmented", 64, 0, 2),
        ("kv_first", 0, 0, 2),
        ("shared_page", 0, 0, 2),
        ("split", 0, 0, 1),
        ("split", 0, 24, 2),
    ],
)
def test_copy_inventory_preserves_aliased_component_layouts(
    copy_cache, layout, alignment_slack, layer_stride, layer_count
):
    num_blocks = 2 if layout == "kv_first" else 3
    key_block_bytes = 4 if layout == "kv_first" else 8
    binding: torch.Tensor | tuple[torch.Tensor, ...]
    if layout == "kv_first":
        binding = torch.empty((2, num_blocks, 1, 1, key_block_bytes), dtype=torch.int8)
        key_raw, value_raw = binding.unbind(0)
    elif layout == "shared_page":
        backing = torch.full((88,), 99, dtype=torch.uint8)
        pages, other = backing[8:56].view(3, 16), backing[56:80].view(3, 8)
        key_raw, value_raw = pages, other
    elif layout == "split":
        key_raw = torch.full((24 + alignment_slack,), 99, dtype=torch.int8)[:24]
        value_raw = torch.full((12 + alignment_slack,), 99, dtype=torch.int8)[:12]
    else:
        config = SimpleNamespace(kv_transfer_config=SimpleNamespace())
        with patch.object(attn_utils, "get_current_vllm_config", return_value=config):
            key_raw, value_raw = attn_utils._allocate_sparse_c8_indexer_tensors(
                24, 12, 64, torch.float32, torch.device("cpu")
            )
    if layout == "shared_page":
        components = (pages, other)
        bindings = {
            "first": (pages[:, :4], pages[:, 4:8]),
            "second": (pages[:, :4], pages[:, 4:12]),
        }
        page_bytes = 16
    else:
        components = (key_raw.view(num_blocks, key_block_bytes), value_raw.view(torch.float32).view(num_blocks, 1))
        if layout != "kv_first":
            binding = components
        bindings = {"first": binding, "second": binding}
        page_bytes = key_block_bytes + 4
    plan = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            SimpleNamespace(
                layers=list(bindings)[:layer_count],
                layer_stride=layer_stride,
                block_stride=page_bytes,
                size=num_blocks * page_bytes,
            )
        ],
        kv_cache_groups=[],
    )
    runner_caches = [
        tensor.view_as(tensor)
        for cache in bindings.values()
        for tensor in ((cache,) if isinstance(cache, torch.Tensor) else cache)
    ]
    runner_caches.append(other if layout == "shared_page" else torch.empty(num_blocks, 0))
    copy_cache(plan, bindings, runner_caches, components)


def test_standalone_mtp_uses_existing_compressed_cache_allocator(runner_factory):
    config = _make_config()
    specs = {name: spec for name, spec in _make_specs().items() if not isinstance(spec, MambaSpec)}
    groups = get_glm5_next_kv_cache_groups(config, specs)
    bytes_per_block = get_glm5_next_pool_bytes_per_block(groups)
    plan = get_glm5_next_kv_cache_config(config, groups, 3 * bytes_per_block)

    runner = runner_factory(config)
    raw_caches = runner._allocate_kv_cache_tensors(plan)

    assert set(raw_caches) == {MAIN, INDEXER, STATE}
    assert raw_caches[INDEXER] is raw_caches[STATE]
    runner.runner_only_attn_layers.add(MAMBA)
    caches = runner._reshape_kv_cache_tensors(plan, raw_caches)
    assert caches[INDEXER][0].shape == (3, 4, 1, 4)
    assert caches[STATE][0].shape == (3, 2, 2, 1)


def test_glm5_next_initialize_passes_all_pooled_views_to_cache_binding():
    config, _, plan = _make_plan()
    runner = _make_runner(config)
    runner.model_config = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm5_next"))

    with patch("vllm.v1.worker.utils.bind_kv_cache") as bind_kv_cache:
        caches = runner.initialize_kv_cache_tensors(plan)

    assert set(caches) == {MAIN, INDEXER, STATE, MAMBA}
    bind_kv_cache.assert_called_once_with(
        caches,
        runner.compilation_config.static_forward_context,
        runner.kv_caches,
        1,
    )


@pytest.mark.parametrize("offset", [0, 64])
def test_glm_shared_mla_mamba_pages_preserve_other_block_ids(offset, runner_factory):
    config, _, plan = _make_plan(num_blocks=4)
    runner = runner_factory(config)
    raw_tensors = runner._allocate_kv_cache_tensors(plan)
    assert raw_tensors[MAIN] is raw_tensors[MAMBA]
    page_bytes = raw_tensors[MAIN].numel() // plan.num_blocks
    backing = torch.zeros(offset + raw_tensors[MAIN].numel() + 64, dtype=torch.int8)
    raw = backing[offset : offset + plan.num_blocks * page_bytes]
    raw_tensors[MAIN] = raw_tensors[MAMBA] = raw
    caches = runner._reshape_kv_cache_tensors(plan, raw_tensors)
    latent, _ = caches[MAIN]
    conv, ssm = caches[MAMBA]
    for state in (conv, ssm):
        assert state.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
        assert state.stride(0) * state.element_size() == page_bytes
    assert ssm.data_ptr() - conv.data_ptr() == conv[0].numel() * conv.element_size()
    state_bytes = sum(state[0].numel() * state.element_size() for state in (conv, ssm))
    for mla_id, state_id in permutations(range(plan.num_blocks), 2):
        raw.zero_()
        latent[mla_id].fill_(7)
        conv[state_id].fill_(11)
        ssm[state_id].fill_(13)
        torch.testing.assert_close(latent[mla_id], torch.full_like(latent[mla_id], 7))
        latent[mla_id].fill_(17)
        torch.testing.assert_close(conv[state_id], torch.full_like(conv[state_id], 11))
        torch.testing.assert_close(ssm[state_id], torch.full_like(ssm[state_id], 13))
        assert torch.count_nonzero(raw.view(plan.num_blocks, page_bytes)[state_id, state_bytes:]) == 0
    assert torch.count_nonzero(backing[:offset]) == 0
    assert torch.count_nonzero(backing[offset + raw.numel() :]) == 0


def test_padded_page_layout_detected_for_shared_state_pages():
    # The pooled layout pads the state caches to the page size of the
    # block-stride addressed MLA/indexer caches they share physical pages with.
    # Probe the specs the runner itself derives from the KV cache config.
    config, _, plan = _make_plan()
    layer_specs = _make_runner(config)._get_layer_kv_cache_specs(plan)
    assert requires_padded_page_layout(layer_specs.values())


@pytest.mark.parametrize("page_size_padded", [None, 128])
def test_mrv2_block_stride_capability_preserves_generic_backing(monkeypatch, page_size_padded):
    config = _make_config()
    spec = replace(_make_specs()[MAIN], model_version=None, page_size_padded=page_size_padded)
    raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
    plan = KVCacheConfig(num_blocks=3, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec([MAIN], spec)])
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_args: {MAIN: layer})
    monkeypatch.setattr(attn_utils, "_get_attention_kv_cache_dims", lambda *_args: (4, 0))
    group = SimpleNamespace(kv_cache_group_id=0, kv_cache_spec=spec, layer_names=[MAIN], backend=_AttentionBackend)
    views = attn_utils._reshape_kv_cache_v2([group], {MAIN: raw}, "auto", [4], {}, plan)
    cache, rope = views[MAIN]
    assert rope.numel() == 0
    if page_size_padded is None:
        cache[2].fill_(7)
    else:
        cache[1, :4].fill_(7)
    physical_pages = raw.view(torch.bfloat16).view(3, -1)
    assert torch.all(physical_pages[1, :16] == 7)
    assert torch.count_nonzero(physical_pages[[0, 2]]) == 0
    assert torch.count_nonzero(physical_pages[1, 16:]) == 0


@pytest.mark.parametrize("geometry", ["offset", "layer_stride", "block_stride", "size"])
def test_mrv2_rejects_invalid_shared_slot_geometry(monkeypatch, geometry):
    config, _, plan = _make_plan()
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    descriptor = plan.kv_cache_tensors[-1]
    if geometry == "layer_stride":
        # A valid packed descriptor cannot be mixed with independent slots.
        descriptor = replace(descriptor, layer_stride=descriptor.size, size=descriptor.size * len(descriptor.layers))
    else:
        descriptor = replace(descriptor, **{geometry: 1})
    plan.kv_cache_tensors[-1] = descriptor
    # A bad later descriptor must fail before allocating earlier valid slots.
    with (
        patch.object(attn_utils, "_allocate_int8_cache_tensor") as allocate,
        pytest.raises(ValueError, match="shared-slot"),
    ):
        attn_utils._allocate_kv_cache(plan, {}, torch.device("cpu"))
    allocate.assert_not_called()


def test_padded_page_layout_rejected_without_tail_caches():
    # A standalone MTP runner has the same attention specs but no recurrent
    # state caches, so no state view needs the padded page stride.
    specs = {name: spec for name, spec in _make_specs().items() if not isinstance(spec, MambaSpec)}
    assert not requires_padded_page_layout(specs.values())


def test_padded_page_layout_rejected_for_packed_hybrid_pool():
    # Other hybrid models pad Mamba pages to the attention page size
    # (``cache_config.mamba_page_size_padded``) but keep the packed contiguous
    # state layout, so they must not take the shared padded page path.
    specs = [
        AscendMLAAttentionSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.bfloat16,
        ),
        MambaSpec(
            block_size=8,
            shapes=((2, 2), (1, 2, 2)),
            dtypes=(torch.bfloat16, torch.float32),
            page_size_padded=64,
        ),
    ]
    assert not requires_padded_page_layout(specs)

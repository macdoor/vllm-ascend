# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend.patch.worker  # noqa: F401
from vllm_ascend.attention.indexer_kpool import (
    AscendIndexerKPoolMetadataBuilder,
    AscendIndexerKPoolTailMetadataBuilder,
)
from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, AscendMLAAttentionSpec
from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool
from vllm_ascend.worker.v2.block_table import AscendBlockTables


@pytest.fixture
def tail_builder_factory(monkeypatch):
    torch.npu.set_device(0)
    monkeypatch.setattr("vllm_ascend.attention.indexer_kpool.enable_dsa_cp", lambda: False)

    def build(capacity, dim, max_tokens):
        spec = AscendIndexerKPoolTailSpec(
            block_size=capacity, sliding_window=4, compress_ratio=4, num_kv_heads=1, head_size=dim, dtype=torch.float32
        )
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
        )
        return spec, AscendIndexerKPoolTailMetadataBuilder(spec, ["layer.tail"], config, torch.device("npu"))

    return build


@pytest.mark.parametrize("capacity", [5, 7, 11])
@torch.inference_mode()
def test_circular_slot_mapping_wraparound_and_graph_replay(capacity: int, tail_builder_factory) -> None:
    device = torch.device("npu")
    lengths = [1025, 0, 17]
    count = sum(lengths)
    tail_spec, builder = tail_builder_factory(capacity, 128, count + 19)
    tables = AscendBlockTables(
        block_sizes=[32, capacity, 8],
        kernel_block_sizes=[16, capacity, 8],
        max_num_reqs=3,
        max_num_batched_tokens=count + 19,
        max_num_blocks_per_group=[256, 1, 1],
        device=device,
        slot_mapping_enabled=[True, tail_spec.uses_slot_mapping, False],
    )
    ordinary = torch.arange(100, 100 + 3 * 512, dtype=torch.int32).view(3, 512)
    ring = torch.tensor([[9], [7], [5]], dtype=torch.int32)
    tables.block_tables[0].gpu.copy_(ordinary.to(device))
    tables.block_tables[1].gpu.copy_(ring.to(device))
    tables.block_tables[2].gpu.fill_(-1)
    mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device=device)
    starts = torch.tensor([0, lengths[0], lengths[0], count], dtype=torch.int32, device=device)
    positions = torch.zeros(count, dtype=torch.int64, device=device)
    output = torch.full_like(tables.slot_mappings, 777)
    common = SimpleNamespace(
        num_reqs=3,
        num_input_tokens=count,
        num_actual_tokens=count,
        query_start_loc=starts,
        positions=positions,
        slot_mapping=output[1],
        block_table_tensor=ring[[2, 1, 0]].to(device),
    )

    def run():
        tables.compute_slot_mappings(mapping, starts, positions, count + 7, out=output)
        output[1, :count].copy_(builder.build(0, common).slot_mapping)

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()

    request_rows = torch.cat((torch.full((lengths[0],), 2), torch.zeros(lengths[2], dtype=torch.int64))).long()
    for offset in (capacity - 1, 4095, 4096):
        positions_cpu = torch.cat(
            (torch.arange(offset, offset + lengths[0]), torch.arange(offset + 2048, offset + 2048 + lengths[2]))
        )
        positions.copy_(positions_cpu.to(device))
        expected = torch.full(output.shape, -1, dtype=torch.int32)
        expected[0, :count] = ordinary[request_rows, positions_cpu // 16] * 16 + positions_cpu % 16
        expected[1, :count] = ring[request_rows, 0] * capacity + positions_cpu % capacity
        for replay in (False, True):
            output.fill_(777)
            if replay:
                graph.replay()
            else:
                run()
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    tables.slot_mapping_enabled[0] = False
    positions[0] = -1
    expected[0].fill_(-1)
    expected[1, 0] = -1
    graph.replay()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@torch.inference_mode()
def test_kpool_capture_replay_preserves_rejected_pool_history(tail_builder_factory) -> None:
    device = torch.device("npu")
    pool, capacity, dim, topk = 4, 7, 128, 2048
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4, max_num_seqs=1),
        model_config=SimpleNamespace(max_model_len=128),
    )
    builder = AscendIndexerKPoolMetadataBuilder(
        AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=dim,
            dtype=torch.bfloat16,
            tokens_per_state=pool,
            model_version="glm5_next",
        ),
        ["layer.indexer.k_cache"],
        config,
        device,
    )
    tail_spec, tail_builder = tail_builder_factory(capacity, dim, 4)
    tables = AscendBlockTables(
        [128, capacity], 1, 4, [1, 1], device, slot_mapping_enabled=[True, tail_spec.uses_slot_mapping]
    )
    tables.block_tables[0].gpu.fill_(3)
    tables.block_tables[1].gpu.fill_(2)
    mapping = torch.zeros(1, dtype=torch.int32, device=device)
    starts = torch.tensor([0, 1], dtype=torch.int32, device=device)
    positions = torch.zeros(4, dtype=torch.int64, device=device)
    slots = tables.compute_slot_mappings(mapping, starts, positions, 4)
    common = SimpleNamespace(
        num_reqs=1,
        num_input_tokens=4,
        num_actual_tokens=1,
        query_start_loc=starts,
        seq_lens=torch.ones(1, dtype=torch.int32, device=device),
        _seq_lens_cpu=torch.ones(1, dtype=torch.int32),
        seq_lens_cpu=None,
        positions=positions,
        slot_mapping=slots[0],
        block_table_tensor=tables.block_tables[0].gpu,
    )
    metadata = builder.build_for_cudagraph_capture(common)
    tail_common = SimpleNamespace(**vars(common))
    tail_common.slot_mapping = slots[1]
    tail_common.block_table_tensor = tables.block_tables[1].gpu
    tail_metadata = tail_builder.build(0, tail_common)
    tail = torch.zeros(4, 2, capacity, dim, device=device)
    cache = torch.zeros(4, 32, 1, dim, dtype=torch.bfloat16, device=device)
    keys = torch.zeros(4, dim, device=device)
    gates = torch.zeros_like(keys)
    ape = torch.zeros(pool, dim, device=device)
    query = torch.ones(4, 1, dim, dtype=torch.bfloat16, device=device)
    weights = torch.ones(4, 1, dtype=torch.bfloat16, device=device)
    output = torch.full((4, topk + pool - 1), 777, dtype=torch.int32, device=device)
    indexer = SparseAttnIndexerKpool(topk, dim)

    def forward():
        return indexer(
            keys[: metadata.num_tokens],
            query[: metadata.num_tokens],
            weights[: metadata.num_tokens],
            metadata.positions,
            cache,
            tail,
            metadata,
            tail_metadata,
            gate_score=gates,
            compress_ape=ape,
            index_kpool=pool,
            max_pool_seq_len=metadata.max_pool_seq_len,
            compute_topk=True,
            output_buffer=output,
            allow_cache_packing=False,
        )

    forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        forward()
    tail.zero_()
    cache.zero_()
    expected_tail = torch.zeros_like(tail, device="cpu")
    expected_cache = torch.zeros_like(cache, device="cpu")
    history: dict[int, torch.Tensor] = {}
    generator = torch.Generator().manual_seed(911)

    # Reject 7..9 after verifying 6..9 and resume at 7; repeat across pool 8..11.
    # Rejected lookahead stays in the physical ring until replay overwrites it.
    for begin, count in ((0, 4), (4, 2), (6, 4), (7, 1), (8, 1), (9, 1), (10, 4), (11, 1), (12, 1), (13, 1)):
        current_keys = torch.randn(count, dim, generator=generator)
        positions_cpu = torch.arange(begin, begin + count)
        positions.zero_()
        positions[:count].copy_(positions_cpu.to(device))
        keys.zero_()
        keys[:count].copy_(current_keys.to(device))
        starts[1] = count
        common.seq_lens.fill_(begin + count)
        common._seq_lens_cpu.fill_(begin + count)
        common.num_actual_tokens = count
        tail_common.num_actual_tokens = count
        tables.compute_slot_mappings(mapping, starts, positions, 4)
        tail_builder.build(0, tail_common)
        if count == 1:
            builder.build_for_drafting(common, draft_index=(begin - 7) % pool)
        else:
            builder.build(0, common)
        output.fill_(777)
        graph.replay()
        for row, pos in enumerate(positions_cpu.tolist()):
            history[pos] = current_keys[row]
            expected_tail[2, 0, pos % capacity] = current_keys[row]
            if (pos + 1) % pool == 0:
                expected_cache[3, pos // pool, 0] = (
                    torch.stack([history[p] for p in range(pos - pool + 1, pos + 1)]).mean(0).bfloat16()
                )
        torch.testing.assert_close(tail.cpu(), expected_tail, rtol=0, atol=0)
        torch.testing.assert_close(cache.cpu(), expected_cache, rtol=1e-2, atol=1e-2)
        actual = output.cpu()
        for row, pos in enumerate(positions_cpu.tolist()):
            expected_ids = torch.full((topk + pool - 1,), -1, dtype=torch.int32)
            expected_ids[: pos + 1] = torch.arange(pos + 1, dtype=torch.int32)
            torch.testing.assert_close(actual[row].sort().values, expected_ids.sort().values, rtol=0, atol=0)
        torch.testing.assert_close(actual[count:], torch.full_like(actual[count:], -1), rtol=0, atol=0)

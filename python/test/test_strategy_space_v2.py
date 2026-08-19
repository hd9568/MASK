from sparsene.op_gen.strategy_agent import (
    CompilerResourceStrategy,
    PipelineStrategy,
    RuntimeScheduleStrategy,
    StrategyDecision,
    TensorPlacementStrategy,
    StrategyFeedbackContext,
    _merge_strategy_patch,
    build_strategy_request,
    normalize_strategy_for_backend,
    strategy_field_catalog,
    strategy_search_space,
)
from sparsene.profiling.ncu import parse_ncu_csv


def test_strategy_space_size_matches_declared_axes():
    space = strategy_search_space()
    tensor = space["tensor_placement"]
    pipeline = space["pipeline"]
    compiler = space["compiler"]
    runtime = space["runtime_schedule"]
    product = 1
    for group in (tensor, pipeline, compiler, runtime):
        for choices in group.values():
            product *= len(choices)
    assert product == 1_327_104_000
    assert space["raw_cartesian_size"] == product


def test_field_catalog_covers_every_executable_decision_field():
    catalog = strategy_field_catalog()
    expected = {
        "tensor_placement": set(TensorPlacementStrategy.__dataclass_fields__),
        "pipeline": set(PipelineStrategy.__dataclass_fields__) - {"rationale"},
        "compiler": set(CompilerResourceStrategy.__dataclass_fields__),
        "runtime_schedule": set(RuntimeScheduleStrategy.__dataclass_fields__),
    }
    assert {group: set(fields) for group, fields in catalog.items()} == expected
    for fields in catalog.values():
        for metadata in fields.values():
            assert metadata["choices"]
            assert metadata["meaning"]
            assert metadata["applicability"]


def test_iterative_agent_uses_compact_patch_and_local_merge():
    baseline = StrategyDecision(
        provider="heuristic",
        rationale="baseline",
        tensor_placement=TensorPlacementStrategy(),
        pipeline=PipelineStrategy(),
    )
    context = StrategyFeedbackContext(
        iteration=1,
        max_iterations=12,
        best_iteration={"strategy": baseline.to_dict()},
    )
    request = build_strategy_request(
        format_name="SR_BCRS",
        gmem_inouts={},
        feedback_context=context,
    )
    assert "strategy_patch" in request.output_schema
    merged, mode, fields = _merge_strategy_patch(
        {
            "provider": "agent",
            "rationale": "test one-axis change",
            "strategy_patch": {"runtime_schedule": {"task_group_tiles": 128}},
        },
        request,
    )
    assert mode == "delta_patch"
    assert fields == ["runtime_schedule.task_group_tiles"]
    assert merged["runtime_schedule"]["task_group_tiles"] == 128
    assert merged["tensor_placement"]["tile_b"] == 64
    assert merged["pipeline"]["enable_async_pipeline"] is True


def test_strategy_normalization_enforces_cross_field_safety():
    decision = StrategyDecision.from_dict(
        {
            "provider": "test",
            "rationale": "exercise normalization",
            "tensor_placement": {
                "tile_b": 32,
                "materialize_b_array_ref_to_smem": False,
                "materialize_val_sidx_to_smem": False,
                "enable_b_array_ref_cp_async": True,
                "b_array_ref_mode": "cp_async_1x4",
                "coo_idx_output_mem": "rmem",
                "coo_val_output_mem": "smem",
                "coo_load_mode": "cp_async_x4_aligned_fallback",
            },
            "pipeline": {
                "enable_async_pipeline": False,
                "sync_threads_on_smem_input": False,
                "max_wait_prior": 2,
                "stage_shifts": [1, 1],
            },
            "compiler": {
                "launch_bounds_min_blocks_per_sm": 12,
                "max_registers_per_thread": 96,
            },
            "runtime_schedule": {
                "compile_max_tiles_per_segment": 64,
                "runtime_max_tiles_per_segment": 16,
                "execution_policy": "strict_lb",
                "task_group_tiles": 128,
            },
        }
    )
    tensor = decision.tensor_placement
    assert tensor.materialize_val_sidx_to_smem is True
    assert tensor.enable_b_array_ref_cp_async is False
    assert tensor.b_array_ref_mode == "sync_scalar"
    assert tensor.coo_idx_output_mem == "rmem"
    assert tensor.coo_val_output_mem == "rmem"
    assert tensor.coo_load_mode == "sync_scalar"
    assert decision.pipeline.stage_shifts == []
    assert decision.compiler.launch_bounds_min_blocks_per_sm == 12
    assert decision.runtime_schedule.execution_policy == "strict_lb"


def test_nondefault_three_stage_shifts_survive_normalization():
    decision = StrategyDecision.from_dict(
        {
            "provider": "test",
            "rationale": "pipeline distance",
            "tensor_placement": {},
            "pipeline": {"stage_shifts": [2, 3]},
        }
    )
    assert decision.pipeline.stage_shifts == [2, 3]


def test_mco_backend_normalizes_dense_only_task_mapping():
    decision = StrategyDecision.from_dict(
        {
            "provider": "test",
            "rationale": "backend normalization",
            "tensor_placement": {},
            "pipeline": {},
            "runtime_schedule": {
                "execution_policy": "multi_bind",
                "task_group_tiles": 256,
            },
        }
    )
    normalized = normalize_strategy_for_backend(
        decision, packing_mode="column_condensed"
    )
    assert normalized.runtime_schedule.execution_policy == "row"
    assert normalized.runtime_schedule.task_group_tiles == 64


def test_ncu_parser_uses_named_columns():
    text = (
        '"ID","Kernel Name","Section Name","Metric Name","Metric Unit","Metric Value"\n'
        '"0","generated_kernel<int>","Launch Statistics","Registers Per Thread","register/thread","96"\n'
        '"0","generated_kernel<int>","GPU Speed Of Light Throughput","Memory Throughput","%","71.5"\n'
        '"0","generated_kernel<int>","Memory Workload Analysis","Memory Throughput","byte/s","1,500,000,000,000"\n'
        '"0","generated_kernel<int>","Occupancy","Achieved Occupancy","%","50"\n'
    )
    parsed = parse_ncu_csv(text)
    assert parsed["available"] is True
    assert parsed["registers_per_thread"] == 96
    assert parsed["memory_throughput_pct"] == 71.5
    assert parsed["memory_bandwidth_bytes_per_s"] == 1_500_000_000_000
    assert parsed["achieved_occupancy_pct"] == 50
    assert parsed["kernels"] == ["generated_kernel<int>"]

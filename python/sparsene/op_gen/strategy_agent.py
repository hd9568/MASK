from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ast
from datetime import datetime
import json
import os
from pathlib import Path
from copy import deepcopy
import re
import subprocess
from typing import Any, Dict, List, Optional
import uuid

from sparsene.agent.runtime import AgentModelConfig, invoke_json_agent, make_zero_usage
from sparsene.logging import get_logger
from sparsene.performance_contract import KERNEL_ONLY_EVALUATION_CONTRACT


logger = get_logger(__name__)

DEFAULT_TOKEN_PLAN_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_TOKEN_PLAN_ANTHROPIC_URL = "https://token-plan-cn.xiaomimimo.com/anthropic"
DEFAULT_TOKEN_PLAN_MODEL = "mimo-v2.5-pro"
SAFE_B_SMEM_SWIZZLE = "swizzle_323"
SAFE_COO_RESTORE_SWIZZLE = "swizzle_123"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


def _shape_to_text(shape: Any) -> str:
    return str(shape)


def _safe_read_command_output(command: List[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


@dataclass
class HardwareProfile:
    gpu_name: str = "unknown"
    driver_version: str = "unknown"
    cuda_arch: str = "sm_80"
    total_memory_mb: Optional[int] = None
    sm_count: Optional[int] = None
    warp_size: int = 32
    max_threads_per_sm: int = 2048
    max_warps_per_sm: int = 64
    max_blocks_per_sm: int = 32
    register_file_32bit_per_sm: int = 65536
    shared_memory_per_sm_kb: int = 164
    source: str = "env_or_default"


@dataclass
class FormatTensorSummary:
    name: str
    dtype: str
    shape: str
    role: str


@dataclass
class SparseFormatProfile:
    format_name: str
    summary: str
    tensors: List[FormatTensorSummary]
    decision_targets: List[str]


@dataclass
class TensorPlacementStrategy:
    tile_b: int = 64
    materialize_b_array_ref_to_smem: bool = True
    materialize_val_sidx_to_smem: bool = True
    enable_b_array_ref_cp_async: bool = True
    b_array_ref_mode: str = "cp_async_1x4"
    b_smem_swizzle: str = SAFE_B_SMEM_SWIZZLE
    coo_idx_output_mem: str = "smem"
    coo_val_output_mem: str = "smem"
    coo_load_mode: str = "cp_async_x4_aligned_fallback"
    coo_restore_swizzle: str = SAFE_COO_RESTORE_SWIZZLE
    align_cp_async_smem_outputs: bool = True


@dataclass
class PipelineStrategy:
    enable_async_pipeline: bool = True
    sync_threads_on_smem_input: bool = True
    max_wait_prior: Optional[int] = None
    stage_shifts: List[int] = field(default_factory=list)
    rationale: str = ""


@dataclass
class CompilerResourceStrategy:
    """Compilation-time resource controls that alter emitted/compiled CUDA."""

    launch_bounds_min_blocks_per_sm: Optional[int] = None
    max_registers_per_thread: Optional[int] = None


@dataclass
class RuntimeScheduleStrategy:
    """Post-conversion execution mapping knobs owned by the host bridge."""

    compile_max_tiles_per_segment: int = 32
    runtime_max_tiles_per_segment: int = 0
    execution_policy: str = "auto"
    task_group_tiles: int = 64


@dataclass
class StrategyDecision:
    provider: str
    rationale: str
    tensor_placement: TensorPlacementStrategy
    pipeline: PipelineStrategy
    compiler: CompilerResourceStrategy = field(default_factory=CompilerResourceStrategy)
    runtime_schedule: RuntimeScheduleStrategy = field(default_factory=RuntimeScheduleStrategy)
    token_usage: Dict[str, Any] = field(default_factory=dict)
    agent_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StrategyDecision":
        tensor_payload = dict(payload.get("tensor_placement", {}) or {})
        pipeline_payload = dict(payload.get("pipeline", {}) or {})
        compiler_payload = dict(payload.get("compiler", {}) or {})
        runtime_payload = dict(payload.get("runtime_schedule", {}) or {})

        def choice_int(value: Any, allowed: set[int], default: int) -> int:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return default
            return parsed if parsed in allowed else default

        tile_b = choice_int(tensor_payload.get("tile_b", 64), {16, 32, 64}, 64)
        if tile_b not in {16, 32, 64}:
            tile_b = 64
        tensor_payload["tile_b"] = tile_b
        materialize_b = bool(tensor_payload.get("materialize_b_array_ref_to_smem", True))
        tensor_payload["materialize_b_array_ref_to_smem"] = materialize_b
        # val_sidx is consumed cooperatively by the current semantic-fusion
        # implementations.  RMEM/non-materialized variants need a different
        # lane-to-address ABI and are therefore not currently searchable.
        tensor_payload["materialize_val_sidx_to_smem"] = True
        b_mode = tensor_payload.get("b_array_ref_mode", "cp_async_1x4")
        b_async = bool(tensor_payload.get("enable_b_array_ref_cp_async", True))
        if not materialize_b or b_mode != "cp_async_1x4" or not b_async:
            tensor_payload["enable_b_array_ref_cp_async"] = False
            tensor_payload["b_array_ref_mode"] = "sync_scalar"
        else:
            tensor_payload["enable_b_array_ref_cp_async"] = True
            tensor_payload["b_array_ref_mode"] = "cp_async_1x4"
        coo_idx_mem = tensor_payload.get("coo_idx_output_mem", "smem")
        coo_val_mem = tensor_payload.get("coo_val_output_mem", "smem")
        if coo_idx_mem not in {"smem", "rmem"}:
            coo_idx_mem = "smem"
        if coo_val_mem not in {"smem", "rmem"}:
            coo_val_mem = "smem"
        # COO in rmem is only safe with synchronous scalar loads; cp.async
        # interacts badly with rmem-backed staging and produces illegal
        # memory accesses at runtime.
        if coo_idx_mem == "rmem" or coo_val_mem == "rmem":
            tensor_payload["coo_idx_output_mem"] = "rmem"
            tensor_payload["coo_val_output_mem"] = "rmem"
            tensor_payload["coo_load_mode"] = "sync_scalar"
        else:
            tensor_payload["coo_idx_output_mem"] = coo_idx_mem
            tensor_payload["coo_val_output_mem"] = coo_val_mem
        # The current ldmatrix S2R lowering hardcodes the canonical SMEM layouts
        # for A/B tiles. Until the lane-to-address mapping is made layout-aware,
        # non-default swizzles generate incorrect kernels even though codegen succeeds.
        tensor_payload["b_smem_swizzle"] = SAFE_B_SMEM_SWIZZLE
        tensor_payload["coo_restore_swizzle"] = SAFE_COO_RESTORE_SWIZZLE
        tensor_payload["align_cp_async_smem_outputs"] = bool(
            tensor_payload.get("align_cp_async_smem_outputs", True)
        )
        if not bool(pipeline_payload.get("enable_async_pipeline", True)):
            # Disabling the software pipeline is only safe once all async-copy
            # producers are downgraded to synchronous paths as well.
            tensor_payload["enable_b_array_ref_cp_async"] = False
            tensor_payload["b_array_ref_mode"] = "sync_scalar"
            if tensor_payload.get("coo_load_mode", "cp_async_x4_aligned_fallback") == "cp_async_x4_aligned_fallback":
                tensor_payload["coo_load_mode"] = "sync_scalar"
        max_wait_prior = pipeline_payload.get("max_wait_prior", None)
        if max_wait_prior is not None:
            try:
                max_wait_prior = int(max_wait_prior)
            except (TypeError, ValueError):
                max_wait_prior = None
        if max_wait_prior not in {None, 0, 1, 2, 3}:
            max_wait_prior = None
        pipeline_payload["max_wait_prior"] = max_wait_prior
        raw_stage_shifts = pipeline_payload.get("stage_shifts", [])
        normalized_stage_shifts: List[int] = []
        if isinstance(raw_stage_shifts, list) and len(raw_stage_shifts) == 2:
            try:
                parsed_shifts = [int(item) for item in raw_stage_shifts]
            except (TypeError, ValueError):
                parsed_shifts = []
            if parsed_shifts and all(shift in {1, 2, 3} for shift in parsed_shifts):
                # All current semantic backends partition into three stages, so
                # a strategy supplies the two inter-stage distances.  The empty
                # list is the canonical representation of the default [1, 1].
                if parsed_shifts != [1, 1]:
                    normalized_stage_shifts = parsed_shifts
        pipeline_payload["stage_shifts"] = normalized_stage_shifts
        pipeline_payload["enable_async_pipeline"] = bool(
            pipeline_payload.get("enable_async_pipeline", True)
        )
        pipeline_payload["sync_threads_on_smem_input"] = bool(
            pipeline_payload.get("sync_threads_on_smem_input", True)
        )
        pipeline_payload["rationale"] = str(pipeline_payload.get("rationale", ""))

        launch_min_blocks = compiler_payload.get("launch_bounds_min_blocks_per_sm")
        if launch_min_blocks is not None:
            try:
                launch_min_blocks = int(launch_min_blocks)
            except (TypeError, ValueError):
                launch_min_blocks = None
        if launch_min_blocks not in {None, 1, 2, 4, 8, 12, 16, 20}:
            launch_min_blocks = None
        compiler_payload["launch_bounds_min_blocks_per_sm"] = launch_min_blocks
        max_registers = compiler_payload.get("max_registers_per_thread")
        if max_registers is not None:
            try:
                max_registers = int(max_registers)
            except (TypeError, ValueError):
                max_registers = None
        if max_registers not in {None, 64, 80, 96, 112, 128}:
            max_registers = None
        compiler_payload["max_registers_per_thread"] = max_registers

        compile_segment = choice_int(
            runtime_payload.get("compile_max_tiles_per_segment", 32),
            {16, 32, 64, 128},
            32,
        )
        runtime_segment = choice_int(
            runtime_payload.get("runtime_max_tiles_per_segment", 0),
            {0, 16, 32, 64, 128},
            0,
        )
        execution_policy = str(runtime_payload.get("execution_policy", "auto"))
        if execution_policy not in {"auto", "row", "strict_lb", "multi_bind"}:
            execution_policy = "auto"
        task_group_tiles = choice_int(
            runtime_payload.get("task_group_tiles", 64),
            {16, 32, 64, 128, 256},
            64,
        )
        runtime_payload = {
            "compile_max_tiles_per_segment": compile_segment,
            "runtime_max_tiles_per_segment": runtime_segment,
            "execution_policy": execution_policy,
            "task_group_tiles": task_group_tiles,
        }
        tensor_fields = set(TensorPlacementStrategy.__dataclass_fields__)
        pipeline_fields = set(PipelineStrategy.__dataclass_fields__)
        compiler_fields = set(CompilerResourceStrategy.__dataclass_fields__)
        tensor_payload = {key: value for key, value in tensor_payload.items() if key in tensor_fields}
        pipeline_payload = {key: value for key, value in pipeline_payload.items() if key in pipeline_fields}
        compiler_payload = {key: value for key, value in compiler_payload.items() if key in compiler_fields}
        return cls(
            provider=payload.get("provider", "unknown"),
            rationale=payload.get("rationale", ""),
            tensor_placement=TensorPlacementStrategy(**tensor_payload),
            pipeline=PipelineStrategy(**pipeline_payload),
            compiler=CompilerResourceStrategy(**compiler_payload),
            runtime_schedule=RuntimeScheduleStrategy(**runtime_payload),
            token_usage=dict(payload.get("token_usage", {}) or {}),
            agent_metadata=dict(payload.get("agent_metadata", {}) or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyFeedbackContext:
    iteration: int = 1
    max_iterations: int = 1
    history: List[Dict[str, Any]] = field(default_factory=list)
    best_iteration: Optional[Dict[str, Any]] = None
    latest_validation: Dict[str, Any] = field(default_factory=dict)
    correctness_feedback: Dict[str, Any] = field(default_factory=dict)
    latest_performance: Dict[str, Any] = field(default_factory=dict)
    latest_profile: Dict[str, Any] = field(default_factory=dict)
    scoreboard: List[Dict[str, Any]] = field(default_factory=list)
    strategy_coverage: Dict[str, Any] = field(default_factory=dict)
    experiment_guidance: Dict[str, Any] = field(default_factory=dict)
    measurement_policy: Dict[str, Any] = field(default_factory=dict)
    current_kernel_path: Optional[str] = None
    current_kernel_code: str = ""
    include_full_kernel_code: bool = False
    additional_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyRequest:
    request_id: str
    format_name: str
    hardware: HardwareProfile
    sparse_format: SparseFormatProfile
    task: str
    output_schema: Dict[str, Any]
    constraints: List[str]
    observations: Dict[str, Any]
    design_context: Dict[str, Any] = field(default_factory=dict)
    search_space: Dict[str, Any] = field(default_factory=dict)
    evaluation_contract: Dict[str, Any] = field(
        default_factory=lambda: deepcopy(KERNEL_ONLY_EVALUATION_CONTRACT)
    )
    feedback_context: Dict[str, Any] = field(default_factory=dict)
    prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyConfig:
    provider: str = "heuristic"
    model_name: Optional[str] = DEFAULT_TOKEN_PLAN_MODEL
    manual_json_path: Optional[str] = None
    agent_command: Optional[str] = None
    log_dir: str = "docs/agent_runs"
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    anthropic_base_url: Optional[str] = None
    model: str = DEFAULT_TOKEN_PLAN_MODEL
    timeout_sec: int = 60
    temperature: float = 0.1
    max_tokens: int = 4096
    anthropic_version: str = DEFAULT_ANTHROPIC_VERSION
    model_config_path: Optional[str] = None
    skill_names: List[str] = field(
        default_factory=lambda: [
            "sparsene-framework",
            "strategy-optimization",
            "feedback-loop",
        ]
    )

    @classmethod
    def from_env(cls) -> "StrategyConfig":
        return cls(
            provider=os.environ.get("SPARSENE_STRATEGY_PROVIDER", "heuristic"),
            model_name=os.environ.get("SPARSENE_STRATEGY_MODEL_NAME", DEFAULT_TOKEN_PLAN_MODEL),
            manual_json_path=os.environ.get("SPARSENE_STRATEGY_MANUAL_JSON"),
            agent_command=os.environ.get("SPARSENE_STRATEGY_AGENT_COMMAND"),
            log_dir=os.environ.get("SPARSENE_STRATEGY_LOG_DIR", "docs/agent_runs"),
            api_key=os.environ.get("SPARSENE_STRATEGY_API_KEY"),
            api_base_url=os.environ.get("SPARSENE_STRATEGY_API_BASE_URL"),
            anthropic_base_url=os.environ.get("SPARSENE_STRATEGY_ANTHROPIC_BASE_URL"),
            model=os.environ.get("SPARSENE_STRATEGY_MODEL", DEFAULT_TOKEN_PLAN_MODEL),
            timeout_sec=int(os.environ.get("SPARSENE_STRATEGY_TIMEOUT_SEC", "60")),
            temperature=float(os.environ.get("SPARSENE_STRATEGY_TEMPERATURE", "0.1")),
            max_tokens=int(os.environ.get("SPARSENE_STRATEGY_MAX_TOKENS", "1200")),
            anthropic_version=os.environ.get(
                "SPARSENE_STRATEGY_ANTHROPIC_VERSION", DEFAULT_ANTHROPIC_VERSION
            ),
            model_config_path=os.environ.get("SPARSENE_MODEL_CONFIG_PATH"),
        )


def discover_hardware_profile() -> HardwareProfile:
    cuda_arch = os.environ.get("CUDA_ARCH", "sm_80")
    query = _safe_read_command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,multiprocessor_count",
            "--format=csv,noheader,nounits",
        ]
    )
    if not query:
        return HardwareProfile(cuda_arch=cuda_arch)

    first_line = query.splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    total_memory_mb: Optional[int] = None
    if len(parts) >= 3:
        try:
            total_memory_mb = int(parts[2])
        except ValueError:
            total_memory_mb = None
    sm_count: Optional[int] = None
    if len(parts) >= 4:
        try:
            sm_count = int(parts[3])
        except ValueError:
            sm_count = None
    return HardwareProfile(
        gpu_name=parts[0] if len(parts) >= 1 else "unknown",
        driver_version=parts[1] if len(parts) >= 2 else "unknown",
        total_memory_mb=total_memory_mb,
        sm_count=sm_count,
        cuda_arch=cuda_arch,
        source="nvidia-smi",
    )


_STRATEGY_FIELD_CATALOG: Dict[str, Dict[str, Dict[str, Any]]] = {
    "tensor_placement": {
        "tile_b": {
            "choices": [16, 32, 64],
            "meaning": "N 方向的 B/C 输出 tile 宽度，同时决定每个 row window 的 N-grid 数量。",
            "applicability": "所有当前 SM80 semantic backend。N=512 时分别产生 32/16/8 个 N tiles。",
        },
        "materialize_b_array_ref_to_smem": {
            "choices": [True, False],
            "meaning": "是否把间接 gather 得到的 B tile 物化到 shared memory 后再由 ldmatrix/MMA 消费。",
            "applicability": "所有含 B gather 的 backend；false 会强制关闭 B cp.async。",
        },
        "materialize_val_sidx_to_smem": {
            "choices": [True],
            "meaning": "是否把稀疏列索引 val_sidx 放到 shared memory 供一个 warp 协作消费。",
            "applicability": "当前 semantic-fusion ABI 固定为 true，不是有效调优轴。",
        },
        "enable_b_array_ref_cp_async": {
            "choices": [True, False],
            "meaning": "B gather producer 是否使用 SM80 cp.async 进行 global-to-shared 搬运。",
            "applicability": "仅当 B 被物化到 SMEM、b_array_ref_mode=cp_async_1x4 且异步 pipeline 开启。",
        },
        "b_array_ref_mode": {
            "choices": ["cp_async_1x4", "sync_scalar"],
            "meaning": "B gather 使用 16-byte 异步向量搬运，或使用同步标量加载。",
            "applicability": "所有 B gather backend；与物化和 enable_b_array_ref_cp_async 联动规范化。",
        },
        "b_smem_swizzle": {
            "choices": [SAFE_B_SMEM_SWIZZLE],
            "meaning": "B shared-memory tile 的 CUTE swizzle/layout。",
            "applicability": "当前 ldmatrix lane mapping 只验证了 swizzle_323，固定字段。",
        },
        "coo_idx_output_mem": {
            "choices": ["smem", "rmem"],
            "meaning": "COO 恢复路径的 index staging 位于 shared memory 或 registers。",
            "applicability": "仅 COO leaf；SR-BCRS dense 与 MCO backend 中无效。任一 COO tensor 进 RMEM 会使 idx/val 同时进 RMEM。",
        },
        "coo_val_output_mem": {
            "choices": ["smem", "rmem"],
            "meaning": "COO 恢复路径的 value staging 位于 shared memory 或 registers。",
            "applicability": "仅 COO leaf；与 coo_idx_output_mem 共同规范化。",
        },
        "coo_load_mode": {
            "choices": ["cp_async_x4_aligned_fallback", "sync_scalar"],
            "meaning": "COO idx/value 流使用带对齐 fallback 的 x4 cp.async，或同步标量加载。",
            "applicability": "仅 COO leaf；RMEM staging 或关闭异步 pipeline 时强制 sync_scalar。",
        },
        "coo_restore_swizzle": {
            "choices": [SAFE_COO_RESTORE_SWIZZLE],
            "meaning": "COO restore shared-memory tile 的 CUTE swizzle/layout。",
            "applicability": "当前 restore/ldmatrix ABI 只验证了 swizzle_123，固定字段。",
        },
        "align_cp_async_smem_outputs": {
            "choices": [True, False],
            "meaning": "是否为 cp.async producer 的 SMEM 输出强制 16-byte 对齐。",
            "applicability": "存在异步 producer 时有效；false 属于实验候选，必须通过正确性验证。",
        },
    },
    "pipeline": {
        "enable_async_pipeline": {
            "choices": [True, False],
            "meaning": "是否允许阶段流水中的异步 producer；false 把 producer 降级为同步加载。",
            "applicability": "当前多缓冲阶段循环仍保留，因为 buf_idx 是生成 ABI 的组成部分。",
        },
        "sync_threads_on_smem_input": {
            "choices": [True, False],
            "meaning": "shared-memory consumer 前是否发出同步屏障；单 warp backend 最终降低为 __syncwarp。",
            "applicability": "所有使用 SMEM producer/consumer 的 backend；false 必须通过全输出正确性。",
        },
        "max_wait_prior": {
            "choices": [None, 0, 1, 2, 3],
            "meaning": "cp.async wait_prior 允许保留的先前异步 group 上限；null 使用依赖计划计算值。",
            "applicability": "存在 cp.async producer 时有效；越小等待越保守，可能减少 overlap。",
        },
        "stage_shifts": {
            "choices": [[], [1, 2], [2, 1], [2, 2], [1, 3], [3, 1], [2, 3], [3, 2], [3, 3]],
            "meaning": "当前三阶段 pipeline 的两个相邻阶段迭代距离；空列表是默认 [1,1]。",
            "applicability": "当前 ME-TCF/BIT-TCF/BIT-BSR/SR-BCRS semantic fusion 均形成三阶段。",
        },
    },
    "compiler": {
        "launch_bounds_min_blocks_per_sm": {
            "choices": [None, 1, 2, 4, 8, 12, 16, 20],
            "meaning": "生成 __launch_bounds__(32, minBlocks)，向 NVCC 提供每个 SM 最少 resident blocks 目标。",
            "applicability": "所有单 warp kernel；不改变固定的 32 threads/block，只影响资源分配/编译可行性。",
        },
        "max_registers_per_thread": {
            "choices": [None, 64, 80, 96, 112, 128],
            "meaning": "NVCC --maxrregcount 的每线程寄存器上限。",
            "applicability": "所有 kernel；较低上限可能提高 occupancy，也可能 spill 并降低性能。",
        },
    },
    "runtime_schedule": {
        "compile_max_tiles_per_segment": {
            "choices": [16, 32, 64, 128],
            "meaning": "编译/host ABI 的默认 row-window 分段 tile 上限，并进入 backend plan/build identity。",
            "applicability": "所有 backend；运行时非零 override 可覆盖它。",
        },
        "runtime_max_tiles_per_segment": {
            "choices": [0, 16, 32, 64, 128],
            "meaning": "本轮执行的分段 tile 上限；0 由 row-tile 分布自动选择。",
            "applicability": "所有 backend；实际解析值必须从 e2e_result.packed 回馈，而不能只看请求值。",
        },
        "execution_policy": {
            "choices": ["auto", "row", "strict_lb", "multi_bind"],
            "meaning": "row-window 到 CUDA tasks 的绑定策略：自动、自然 row、严格负载均衡或重行多绑定。",
            "applicability": "strict_lb/multi_bind 适用于 column_condensed_dense、active_fixed_blocks_dense 与 sparse_row_column_condensed_dense；MCO backend 规范化为 row。auto 可能实际解析成 strict_lb。",
        },
        "task_group_tiles": {
            "choices": [16, 32, 64, 128, 256],
            "meaning": "task-mapped 调度中每个 task group 的目标 sparse tiles 数。",
            "applicability": "strict_lb/multi_bind 时有效；越小负载更细但 task/atomic/metadata 开销更高。",
        },
    },
}


def strategy_field_catalog() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return the authoritative documented meaning of every strategy field."""

    return deepcopy(_STRATEGY_FIELD_CATALOG)


def strategy_search_space() -> Dict[str, Any]:
    """Return the exact prompt-visible space implemented by deterministic code."""

    groups = {
        group: {
            field_name: deepcopy(metadata["choices"])
            for field_name, metadata in fields.items()
        }
        for group, fields in _STRATEGY_FIELD_CATALOG.items()
    }
    raw_size = 1
    for fields in groups.values():
        for choices in fields.values():
            raw_size *= len(choices)
    return {
        "schema_version": "sparsene.strategy_space.v2",
        "raw_cartesian_size": raw_size,
        "note": (
            "The raw product includes invalid, equivalent, and format-inapplicable "
            "combinations; StrategyDecision.from_dict and the selected backend "
            "canonicalize them before code generation."
        ),
        "field_catalog": {
            group: {
                field_name: {
                    "meaning": metadata["meaning"],
                    "applicability": metadata["applicability"],
                }
                for field_name, metadata in fields.items()
            }
            for group, fields in _STRATEGY_FIELD_CATALOG.items()
        },
        **groups,
    }


def normalize_strategy_for_backend(
    decision: StrategyDecision,
    *,
    packing_mode: str,
) -> StrategyDecision:
    """Remove knobs that the selected semantic backend cannot execute."""

    payload = decision.to_dict()
    dense_task_mapping_modes = {
        "column_condensed_dense",
        "active_fixed_blocks_dense",
        "sparse_row_column_condensed_dense",
    }
    if packing_mode not in dense_task_mapping_modes:
        # MCO backends currently use row/segment binding; dense task-mapped
        # strict_lb/multi_bind kernels have a distinct ABI.
        payload["runtime_schedule"]["execution_policy"] = "row"
        payload["runtime_schedule"]["task_group_tiles"] = 64
    normalized = StrategyDecision.from_dict(payload)
    normalized.agent_metadata = dict(decision.agent_metadata)
    normalized.agent_metadata["backend_normalization"] = {
        "packing_mode": packing_mode,
        "execution_policy": normalized.runtime_schedule.execution_policy,
    }
    return normalized


def build_sparse_format_profile(
    *,
    format_name: str,
    gmem_inouts: Dict[str, Any],
) -> SparseFormatProfile:
    format_summary = {
        "ME_TCF": (
            "ME_TCF is a hierarchical mixed sparse SpMM format. "
            "It uses length/offset metadata, sparse row indices (sidx), "
            "COO idx/value streams, B-side row gathers, and Tensor Core MMA tiles. "
            "The main performance-sensitive decisions are tensor placement, "
            "cp.async usage, and software-pipeline synchronization."
        ),
    }.get(
        format_name,
        f"{format_name} is a sparse format lowered through OPIR -> NVIR -> CUDA codegen.",
    )

    tensor_roles: Dict[str, str] = {
        "val_len": "sparse length metadata",
        "val_coo_idx": "COO sparse A indices",
        "val_coo_val": "COO sparse A values",
        "val_sidx": "row remap / sparse index for B gather",
        "B_val": "dense B input matrix",
        "C_val": "dense C output matrix",
    }
    tensor_summaries = []
    for name, gmem in gmem_inouts.items():
        if isinstance(gmem, dict):
            dtype = str(gmem.get("dtype", "unknown"))
            shape = str(gmem.get("shape", "unknown"))
        else:
            dtype = str(gmem.dtype)
            shape = _shape_to_text(gmem.shape)
        tensor_summaries.append(
            FormatTensorSummary(
                name=name,
                dtype=dtype,
                shape=shape,
                role=tensor_roles.get(name, "external tensor"),
            )
        )
    decision_targets = [
        "N-direction tile size (TILE_B) selection among supported variants",
        "tensor placement for gathered/intermediate tensors",
        "whether B-side gather should materialize to smem",
        "whether COO idx/value should reside in smem or rmem",
        "whether COO idx/value loads should use cp.async or sync_scalar",
        "pipeline wait/commit/sync behavior and max_wait_prior depth",
        "stage_shifts for pipeline sync placement",
        "shared-memory alignment requirements for async paths",
        "launch-bounds and register-cap compilation constraints",
        "execution segmentation, row/strict load balance, and task-group size",
    ]
    return SparseFormatProfile(
        format_name=format_name,
        summary=format_summary,
        tensors=tensor_summaries,
        decision_targets=decision_targets,
    )


def _default_output_schema() -> Dict[str, Any]:
    return {
        "provider": "string",
        "rationale": "string",
        "tensor_placement": {
            "tile_b": "enum(16|32|64)",
            "materialize_b_array_ref_to_smem": "bool",
            "materialize_val_sidx_to_smem": "bool",
            "enable_b_array_ref_cp_async": "bool",
            "b_array_ref_mode": "enum(cp_async_1x4|sync_scalar)",
            "b_smem_swizzle": "enum(swizzle_323)",
            "coo_idx_output_mem": "enum(smem|rmem)",
            "coo_val_output_mem": "enum(smem|rmem)",
            "coo_load_mode": "enum(cp_async_x4_aligned_fallback|sync_scalar)",
            "coo_restore_swizzle": "enum(swizzle_123)",
            "align_cp_async_smem_outputs": "bool",
        },
        "pipeline": {
            "enable_async_pipeline": "bool",
            "sync_threads_on_smem_input": "bool",
            "max_wait_prior": "int|null",
            "stage_shifts": "list[int]",
            "rationale": "string",
        },
        "compiler": {
            "launch_bounds_min_blocks_per_sm": "enum(null|1|2|4|8|12|16|20)",
            "max_registers_per_thread": "enum(null|64|80|96|112|128)",
        },
        "runtime_schedule": {
            "compile_max_tiles_per_segment": "enum(16|32|64|128)",
            "runtime_max_tiles_per_segment": "enum(0|16|32|64|128)",
            "execution_policy": "enum(auto|row|strict_lb|multi_bind)",
            "task_group_tiles": "enum(16|32|64|128|256)",
        },
    }


def _delta_output_schema() -> Dict[str, Any]:
    """Compact iterative schema: return only fields changed from best-correct."""

    return {
        "provider": "string",
        "rationale": "string",
        "strategy_patch": {
            "tensor_placement": "object containing only changed fields",
            "pipeline": "object containing only changed fields",
            "compiler": "object containing only changed fields",
            "runtime_schedule": "object containing only changed fields",
        },
    }


def build_strategy_request(
    *,
    format_name: str,
    gmem_inouts: Dict[str, Any],
    feedback_context: Optional[StrategyFeedbackContext] = None,
    design_context: Optional[Dict[str, Any]] = None,
    search_space: Optional[Dict[str, Any]] = None,
) -> StrategyRequest:
    hardware = discover_hardware_profile()
    sparse_format = build_sparse_format_profile(
        format_name=format_name,
        gmem_inouts=gmem_inouts,
    )
    is_feedback_loop = (
        feedback_context is not None and feedback_context.max_iterations > 1
    )
    constraints = [
        "Must preserve correctness for the selected run-owned format and semantic backend.",
        "Should prefer strategies already known to be safe on SM80/A100-like hardware.",
        "Should only choose from the documented structured schema.",
        "Do not invent new schema fields or unsupported codegen knobs.",
        "Current lowering only supports canonical swizzles: B staging uses swizzle_323 and COO restore uses swizzle_123.",
        "When COO idx/val are placed in rmem, coo_load_mode must also be sync_scalar; cp.async is incompatible with rmem-based staging.",
        "If async pipeline is disabled, all cp.async-based producers must also be downgraded to synchronous loads.",
        "The generated kernel has an explicit one-warp/32-thread ABI; launch_bounds may constrain minimum resident blocks but may not change threads per block.",
        "Runtime strict_lb/multi_bind is executable for dense SR-BCRS ABI packing modes (column_condensed_dense, active_fixed_blocks_dense, and sparse_row_column_condensed_dense); deterministic backend normalization forces MCO backends to row binding.",
        "Need to leave logs of request, prompt, response, performance, and profile for later analysis.",
        "Optimize only the warmed GPU SpMM compute-kernel time after format conversion/reordering; never add or penalize offline conversion/reorder latency.",
        "Only propose fields that are active for the selected backend and that change the normalized strategy, compiled binary, or effective runtime execution; avoid semantically duplicate experiments.",
        "Classify one dominant bottleneck and state the expected benefit, counter-cost, and falsifying observation before choosing a patch.",
    ]
    if is_feedback_loop:
        constraints.append(
            "This is an iterative autotuning loop; use prior performance and profile feedback to refine the next strategy."
        )
    else:
        constraints.append(
            "This is a single-shot strategy decision, not an iterative autotuning loop."
        )
    observations = {
        "known_good_baseline": {
            "tile_b": 64,
            "coo_range_preserved": True,
            "coo_load_cp_async": True,
            "b_side_cp_async": True,
            "cp_async_shared_align_16": True,
            "b_smem_swizzle": SAFE_B_SMEM_SWIZZLE,
            "coo_restore_swizzle": SAFE_COO_RESTORE_SWIZZLE,
        },
        "current_goal": (
            "drive codegen decisions for tensor placement and pipeline strategy"
            + (" with iterative performance feedback" if is_feedback_loop else " in one pass")
        ),
    }
    if feedback_context is not None:
        observations["feedback_loop"] = {
            "iteration": feedback_context.iteration,
            "max_iterations": feedback_context.max_iterations,
            "history_length": len(feedback_context.history),
            "has_strategy_history": bool(feedback_context.history),
            "has_profile_feedback": bool(feedback_context.latest_profile),
        }
    request = StrategyRequest(
        request_id=str(uuid.uuid4()),
        format_name=format_name,
        hardware=hardware,
        sparse_format=sparse_format,
        task=(
            "Choose code-generation and execution strategy knobs for the selected sparse format. "
            "Cover tensor placement, cp.async, software pipeline, compiler resources, and runtime task mapping. "
            "Return only structured data."
        ),
        output_schema=(_delta_output_schema() if is_feedback_loop else _default_output_schema()),
        constraints=constraints,
        observations=observations,
        design_context=deepcopy(design_context or {}),
        search_space=deepcopy(search_space or strategy_search_space()),
        feedback_context=(
            feedback_context.to_dict() if feedback_context is not None else {}
        ),
    )
    request.prompt = render_strategy_prompt(request)
    return request


def render_strategy_prompt(request: StrategyRequest) -> str:
    tensor_lines = "\n".join(
        f"- {tensor.name}: dtype={tensor.dtype}, shape={tensor.shape}, role={tensor.role}"
        for tensor in request.sparse_format.tensors
    )
    constraint_lines = "\n".join(f"- {item}" for item in request.constraints)
    target_lines = "\n".join(
        f"- {item}" for item in request.sparse_format.decision_targets
    )
    feedback = request.feedback_context or {}
    history = feedback.get("history", [])
    best_iteration = feedback.get("best_iteration")
    latest_validation = feedback.get("latest_validation", {})
    correctness_feedback = feedback.get("correctness_feedback", {})
    latest_performance = feedback.get("latest_performance", {})
    latest_profile = feedback.get("latest_profile", {})
    scoreboard = feedback.get("scoreboard", [])
    strategy_coverage = feedback.get("strategy_coverage", {})
    experiment_guidance = feedback.get("experiment_guidance", {})
    measurement_policy = feedback.get("measurement_policy", {})
    additional_notes = feedback.get("additional_notes", [])

    optimization_guidance = "\n".join(
        [
            "- On SM80/A100/A800-like GPUs, cp.async usually helps only when data is aligned, reuse exists, and the producer-consumer distance can hide latency.",
            "- Turning off cp.async or moving tensors from smem to rmem may improve safety, but often reduces bandwidth amortization and Tensor Core feed efficiency.",
            "- High registers/thread or low waves-per-SM indicates occupancy pressure; if the grid is also small, aggressive pipelining may not help.",
            "- If Memory Throughput is much higher than Compute Throughput, prefer memory-traffic reductions and stable async copies over extra synchronization.",
            "- If Compute Throughput is low because the grid is too small, avoid strategies that add overhead without increasing arithmetic intensity.",
            "- For B-side gather and COO load paths, preserve alignment and conservative fallback logic when enabling async movement.",
            "- TILE_B changes the N-direction tile size. Larger TILE_B can improve Tensor Core feed reuse but may reduce flexibility or increase wasted work when N is small.",
            "- The current ldmatrix lowering assumes fixed canonical swizzles. Do not vary B staging or COO restore swizzles until the S2R path becomes layout-aware.",
            "- COO staging in SMEM with aligned cp.async is the current high-throughput baseline, not a universal winner; RMEM forces sync_scalar and is justified only by measured register/synchronization trade-offs on a COO-capable backend.",
            "- pipeline max_wait_prior: None (unbounded), 0-3 (bounded depth). Lower values reduce occupancy pressure but limit async overlap. Values 1-3 are all valid.",
            "- enable_async_pipeline=false downgrades async producers but retains the staged loop because generated buffer indices use that ABI.",
            "- stage_shifts has two entries because current semantic kernels have three dependency stages. [] is canonical [1,1]; other legal values use distances 1-3, such as [1,2], [2,1], or [2,2].",
            "- sync_threads_on_smem_input can be toggled independently; disabling it removes a __syncthreads before B/COO consumption.",
            "- align_cp_async_smem_outputs can be toggled; disabling it may save shared memory but risks misaligned cp.async on some problem sizes.",
            "- Change only a small number of knobs between adjacent iterations unless profile evidence strongly suggests a larger change.",
            "- If actual physical tile/MMA, padding, restore, or B-gather work dominates and no active strategy field changes it, report a representation-bound diagnosis instead of random strategy diversity.",
        ]
    )

    diagnosis_procedure = "\n".join(
        [
            "1. Establish the best-correct base, measurement noise threshold, and normalized/effective signatures already tested.",
            "2. Remove inactive, backend-normalized, and semantically duplicate fields from consideration.",
            "3. Classify one dominant hypothesis: task/grid imbalance, memory/feed, register/shared-resource pressure, pipeline/synchronization, or representation-bound physical work.",
            "4. Choose normally one to three effective changed fields and predict both benefit and counter-cost.",
            "5. Name the next timing/profile observation that would falsify the hypothesis; normal repeated kernel time remains the ranking metric.",
        ]
    )

    if history:
        history_lines = []
        for item in history[-5:]:
            perf = item.get("performance", {})
            profile = item.get("profile_summary", {})
            validation = item.get("validation", {})
            history_lines.append(
                "- iter {iteration}: correct={correct} mykernel_ms={my_ms} cusparse_ms={cu_ms} "
                "speedup_vs_baseline={baseline_speedup} delta_vs_best_pct={delta_best} "
                "effective_policy={effective_policy} tasks={tasks} grid={grid} "
                "registers={registers} occupancy_pct={occupancy} dram_tp_pct={dram_tp} "
                "compute_tp_pct={compute_tp} strategy={strategy}".format(
                    iteration=item.get("iteration", "?"),
                    correct=validation.get("passed"),
                    my_ms=perf.get("mykernel_ms"),
                    cu_ms=perf.get("cusparse_ms"),
                    baseline_speedup=perf.get("speedup_vs_baseline"),
                    delta_best=perf.get("delta_pct_vs_best"),
                    effective_policy=item.get("effective_execution", {}).get("execution_policy"),
                    tasks=item.get("effective_execution", {}).get("tasks"),
                    grid=profile.get("grid_size"),
                    registers=profile.get("registers_per_thread"),
                    occupancy=profile.get("achieved_occupancy_pct"),
                    dram_tp=profile.get("dram_throughput_pct"),
                    compute_tp=profile.get("compute_throughput_pct"),
                    strategy=json.dumps(item.get("strategy", {}), ensure_ascii=False),
                )
            )
        history_text = "\n".join(history_lines)
    else:
        history_text = "- no previous iterations"

    best_iteration_text = "- none"
    if best_iteration:
        perf = best_iteration.get("performance", {})
        best_iteration_text = (
            f"- best_iter={best_iteration.get('iteration')} "
            f"mykernel_ms={perf.get('mykernel_ms')} "
            f"ratio_vs_cusparse={perf.get('ratio_vs_cusparse')} "
            f"correct={best_iteration.get('validation', {}).get('passed')}\n"
            f"- best_strategy={json.dumps(best_iteration.get('strategy', {}), ensure_ascii=False)}\n"
            f"- best_effective_execution={json.dumps(best_iteration.get('effective_execution', {}), ensure_ascii=False)}\n"
            f"- best_profile={json.dumps(best_iteration.get('profile_summary', {}), ensure_ascii=False)}"
        )

    base_strategy_text = "- none (use the conservative baseline)"
    if best_iteration and isinstance(best_iteration.get("strategy"), dict):
        base_strategy_text = json.dumps(
            best_iteration["strategy"], indent=2, ensure_ascii=False
        )

    latest_validation_text = json.dumps(latest_validation, indent=2, ensure_ascii=False) if latest_validation else "{}"
    correctness_feedback_text = json.dumps(correctness_feedback, indent=2, ensure_ascii=False) if correctness_feedback else "{}"
    latest_performance_text = json.dumps(latest_performance, indent=2, ensure_ascii=False) if latest_performance else "{}"
    latest_profile_text = json.dumps(latest_profile, indent=2, ensure_ascii=False) if latest_profile else "{}"
    additional_notes_text = "\n".join(f"- {item}" for item in additional_notes) if additional_notes else "- none"
    design_context_text = json.dumps(request.design_context, indent=2, ensure_ascii=False)
    search_space_text = json.dumps(request.search_space, indent=2, ensure_ascii=False)
    is_iterative_response = "strategy_patch" in request.output_schema
    if is_iterative_response:
        base_strategy_heading = "Base strategy to preserve unless a field is explicitly patched"
        output_mode_requirements = """- This is an iterative feedback round. Return a compact `strategy_patch` containing
  only fields changed relative to the base strategy. Omit unchanged groups and
  fields; never repeat the complete strategy JSON. An empty patch is allowed
  when the evidence supports retaining the best, the active search is saturated,
  or the remaining bottleneck is representation-bound.
- The local driver merges the patch with the base strategy, then applies the
  deterministic safety normalizer. Do not put `token_usage`, `agent_metadata`,
  or generated code in the patch."""
    else:
        base_strategy_heading = "Conservative reference baseline (return a complete strategy)"
        output_mode_requirements = """- This is a single-shot decision. Return the complete structured strategy using
  every required top-level group in the schema. Do not emit `strategy_patch`."""

    return f"""You are a kernel strategy agent for sparse Tensor Core code generation.
You are also an expert in CUDA kernel optimization, Tensor Core scheduling, cp.async, shared-memory staging,
occupancy trade-offs, and profile-guided performance tuning.

Hardware:
- gpu_name: {request.hardware.gpu_name}
- driver_version: {request.hardware.driver_version}
- cuda_arch: {request.hardware.cuda_arch}
- total_memory_mb: {request.hardware.total_memory_mb}
- sm_count: {request.hardware.sm_count}
- warp_size: {request.hardware.warp_size}
- max_threads_per_sm: {request.hardware.max_threads_per_sm}
- max_warps_per_sm: {request.hardware.max_warps_per_sm}
- max_blocks_per_sm: {request.hardware.max_blocks_per_sm}
- register_file_32bit_per_sm: {request.hardware.register_file_32bit_per_sm}
- shared_memory_per_sm_kb: {request.hardware.shared_memory_per_sm_kb}

Sparse format:
- format_name: {request.sparse_format.format_name}
- summary: {request.sparse_format.summary}

Run-owned design and backend context:
{design_context_text}

Fixed performance evaluation contract:
{json.dumps(request.evaluation_contract, indent=2, ensure_ascii=False)}

Visible external tensors:
{tensor_lines}

What you need to do:
{target_lines}

Constraints:
{constraint_lines}

High-level CUDA optimization guidance:
{optimization_guidance}

Required bottleneck-diagnosis procedure:
{diagnosis_procedure}

Exact deterministic search space:
{search_space_text}

Feedback loop state:
- iteration: {feedback.get("iteration", 1)}
- max_iterations: {feedback.get("max_iterations", 1)}

Recent strategy history and measured feedback:
{history_text}

Best-known iteration:
{best_iteration_text}

{base_strategy_heading}:
{base_strategy_text}

Latest validation result:
{latest_validation_text}

Structured correctness feedback:
{correctness_feedback_text}

Latest performance result:
{latest_performance_text}

Latest ncu profile summary:
{latest_profile_text}

All-iteration compact scoreboard:
{json.dumps(scoreboard, indent=2, ensure_ascii=False)}

Search coverage and untested effective values:
{json.dumps(strategy_coverage, indent=2, ensure_ascii=False)}

Controller experiment guidance:
{json.dumps(experiment_guidance, indent=2, ensure_ascii=False)}

Measurement/noise policy:
{json.dumps(measurement_policy, indent=2, ensure_ascii=False)}

Additional notes:
{additional_notes_text}

Output requirements:
- Return only structured JSON compatible with the provided schema.
{output_mode_requirements}
- Do not return token usage, agent metadata, kernel source, or invented fields.
- Keep the strategy conservative: prefer correctness-preserving choices.
- For cp.async paths, ensure alignment and safe fallback logic are considered.
- For pipeline decisions, avoid aggressive waits/commits unless justified.
- Use correctness feedback to avoid repeating strategies that fail systematically on the same synthetic shapes or sparsity levels.
- Explore TILE_B and placement/pipeline choices when the current strategies cluster too tightly; do not keep repeating the same memory/pipeline recipe.
- Consider compiler resource limits and runtime task mapping together with NCU register/occupancy and matrix heavy-tail evidence.
- Every selected value must come from the exact search space. Deterministic normalization makes the final backend-specific legality decision.
- Use the history and profile feedback to explain why the next strategy should improve or stabilize performance.
- Distinguish requested runtime policy from e2e_result.packed effective execution. In particular, auto may already resolve to strict_lb.
- Treat changes below the measurement noise threshold as inconclusive unless confirmed by repeated samples; do not infer that higher occupancy alone means lower time.
- Select an untested effective value from the controller guidance when reasonable, while changing as few unrelated fields as possible.
- In iterative mode, normally change one to three effective fields. The concise
  rationale must name the bottleneck class, evidence, intended benefit,
  counter-cost, and a falsifying timing/profile observation.
- Rank strategies only by the recorded kernel-only metric. Format conversion,
  metadata construction, reorder/inverse-reorder, allocation/copies, compilation,
  and verification are outside this objective and must not be introduced as penalties.
- Prefer incremental changes from the best correct iteration unless the profile strongly indicates a different bottleneck.
- Focus on historical strategy outputs and measured feedback rather than reconstructing the full generated kernel source.

Schema:
{json.dumps(request.output_schema, indent=2)}
"""


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
    def _attempt_parse(candidate: str) -> Dict[str, Any]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            sanitized = candidate
            sanitized = sanitized.replace("\r", "")
            sanitized = sanitized.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")
            sanitized = sanitized.replace(":True", ":true").replace(":False", ":false").replace(":None", ":null")
            sanitized = sanitized.replace(",}", "}").replace(",]", "]")
            try:
                return json.loads(sanitized)
            except json.JSONDecodeError:
                python_candidate = re.sub(r"\btrue\b", "True", sanitized, flags=re.IGNORECASE)
                python_candidate = re.sub(r"\bfalse\b", "False", python_candidate, flags=re.IGNORECASE)
                python_candidate = re.sub(r"\bnull\b", "None", python_candidate, flags=re.IGNORECASE)
                parsed = ast.literal_eval(python_candidate)
                if not isinstance(parsed, dict):
                    raise
                return parsed
    try:
        return _attempt_parse(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return _attempt_parse(stripped[start : end + 1])


def _base_strategy_payload(request: StrategyRequest) -> Dict[str, Any]:
    feedback = request.feedback_context or {}
    best = feedback.get("best_iteration")
    if isinstance(best, dict) and isinstance(best.get("strategy"), dict):
        return deepcopy(best["strategy"])
    return _heuristic_strategy_response(request).to_dict()


def _merge_strategy_patch(
    payload: Dict[str, Any],
    request: StrategyRequest,
) -> tuple[Dict[str, Any], str, List[str]]:
    """Merge a compact Agent patch into best-correct without trusting omissions."""

    patch = payload.get("strategy_patch")
    if not isinstance(patch, dict):
        return dict(payload), "full", []
    merged = _base_strategy_payload(request)
    changed: List[str] = []
    for group in ("tensor_placement", "pipeline", "compiler", "runtime_schedule"):
        group_patch = patch.get(group, {})
        if not isinstance(group_patch, dict):
            continue
        target = merged.setdefault(group, {})
        for field_name, value in group_patch.items():
            target[field_name] = value
            changed.append(f"{group}.{field_name}")
    if "provider" in payload:
        merged["provider"] = payload["provider"]
    if "rationale" in payload:
        merged["rationale"] = payload["rationale"]
    return merged, "delta_patch", changed


def _heuristic_strategy_response(request: StrategyRequest) -> StrategyDecision:
    return StrategyDecision(
        provider="heuristic",
        rationale=(
            f"Use the fixed conservative SM80 baseline for {request.format_name}: keep "
            "gathered B tiles in shared memory, use aligned cp.async producers, and "
            "preserve the format-specific semantic-fusion implementation."
        ),
        tensor_placement=TensorPlacementStrategy(),
        pipeline=PipelineStrategy(
            enable_async_pipeline=True,
            sync_threads_on_smem_input=True,
            max_wait_prior=None,
            stage_shifts=[],
            rationale=(
                "Keep the existing dynamic stage partitioning and default shift=1 "
                "between adjacent stages."
            ),
        ),
    )


def _load_manual_strategy(config: StrategyConfig) -> StrategyDecision:
    if not config.manual_json_path:
        raise ValueError("manual_json provider requires manual_json_path")
    payload = json.loads(Path(config.manual_json_path).read_text())
    payload.setdefault("token_usage", make_zero_usage().to_dict())
    payload.setdefault("agent_metadata", {"source": "manual_json", "path": config.manual_json_path})
    return StrategyDecision.from_dict(payload)


def _run_llm_strategy(
    *,
    config: StrategyConfig,
    request: StrategyRequest,
) -> tuple[StrategyDecision, Path]:
    provider = "anthropic_compatible" if config.provider == "anthropic_http" else "openai_compatible"
    result = invoke_json_agent(
        agent_name="strategy_agent",
        subject=request.format_name.lower(),
        log_dir=config.log_dir,
        model_config=AgentModelConfig(
            provider=provider,
            model_name=config.model_name or config.model,
            model=None if config.model_name else config.model,
            api_key=config.api_key,
            api_base_url=config.api_base_url,
            anthropic_base_url=config.anthropic_base_url,
            anthropic_version=config.anthropic_version,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_sec=config.timeout_sec,
            model_config_path=config.model_config_path,
        ),
        system_prompt=(
            "You are a conservative sparse kernel strategy planner and CUDA/Tensor Core optimization expert. "
            "Optimize warmed GPU SpMM compute-kernel time after all format conversion and reordering; "
            "offline conversion/reorder latency is excluded and must never be a ranking penalty. "
            "Diagnose one dominant bottleneck, change only active effective fields, and explain the "
            "expected benefit, counter-cost, and falsifying observation; return an empty iterative "
            "patch when the fixed representation is the limiting layer or the search is saturated. "
            "Return only valid JSON matching the requested schema. "
            "Do not include markdown fences, prose, or extra keys."
        ),
        user_prompt=request.prompt,
        request_payload=request.to_dict(),
        response_parser=_extract_json_object,
        skill_names=config.skill_names,
        json_output=True,
    )
    decision_payload, response_mode, patch_fields = _merge_strategy_patch(
        dict(result.parsed_payload), request
    )
    if "provider" not in decision_payload:
        decision_payload["provider"] = config.provider
    decision_payload["token_usage"] = result.usage.to_dict()
    decision_payload["agent_metadata"] = {
        "run_dir": str(result.run_dir),
        "resolved_model": result.resolved_model.to_dict(),
        "response_mode": response_mode,
        "patch_fields": patch_fields,
    }
    return StrategyDecision.from_dict(decision_payload), result.run_dir


def _run_subprocess_strategy(
    *,
    config: StrategyConfig,
    request: StrategyRequest,
) -> StrategyDecision:
    if not config.agent_command:
        raise ValueError("subprocess_json provider requires agent_command")
    payload = {
        "request": request.to_dict(),
        "prompt": request.prompt,
    }
    result = subprocess.run(
        config.agent_command,
        input=json.dumps(payload),
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    response = json.loads(result.stdout)
    response.setdefault("token_usage", make_zero_usage().to_dict())
    response.setdefault("agent_metadata", {"source": "subprocess_json"})
    return StrategyDecision.from_dict(response)


def _ensure_log_dir(base_dir: str) -> Path:
    path = Path(base_dir)
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        path = repo_root / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _log_strategy_run(
    *,
    config: StrategyConfig,
    request: StrategyRequest,
    response: StrategyDecision,
) -> Path:
    base_dir = _ensure_log_dir(config.log_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{stamp}_{request.format_name.lower()}_{response.provider}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "request.json").write_text(
        json.dumps(request.to_dict(), indent=2, ensure_ascii=False)
    )
    (run_dir / "prompt.txt").write_text(request.prompt)
    (run_dir / "response.json").write_text(
        json.dumps(response.to_dict(), indent=2, ensure_ascii=False)
    )
    return run_dir


def run_strategy_agent(
    *,
    format_name: str,
    gmem_inouts: Dict[str, Any],
    config: Optional[StrategyConfig] = None,
    feedback_context: Optional[StrategyFeedbackContext] = None,
    design_context: Optional[Dict[str, Any]] = None,
    search_space: Optional[Dict[str, Any]] = None,
) -> tuple[StrategyDecision, Path]:
    if config is None:
        config = StrategyConfig.from_env()
    request = build_strategy_request(
        format_name=format_name,
        gmem_inouts=gmem_inouts,
        feedback_context=feedback_context,
        design_context=design_context,
        search_space=search_space,
    )
    if config.provider == "heuristic":
        response = _heuristic_strategy_response(request)
        response.token_usage = make_zero_usage().to_dict()
        response.agent_metadata = {"source": "heuristic"}
    elif config.provider == "manual_json":
        response = _load_manual_strategy(config)
    elif config.provider == "openai_http":
        response, run_dir = _run_llm_strategy(config=config, request=request)
        logger.info(
            "Strategy agent provider=%s logged request/response to %s",
            response.provider,
            run_dir,
        )
        return response, run_dir
    elif config.provider == "anthropic_http":
        response, run_dir = _run_llm_strategy(config=config, request=request)
        logger.info(
            "Strategy agent provider=%s logged request/response to %s",
            response.provider,
            run_dir,
        )
        return response, run_dir
    elif config.provider == "subprocess_json":
        response = _run_subprocess_strategy(config=config, request=request)
    else:
        raise ValueError(
            f"Unsupported strategy provider: {config.provider}. "
            "Expected one of heuristic|manual_json|openai_http|anthropic_http|subprocess_json."
        )

    run_dir = _log_strategy_run(
        config=config,
        request=request,
        response=response,
    )
    logger.info(
        "Strategy agent provider=%s logged request/response to %s",
        response.provider,
        run_dir,
    )
    return response, run_dir

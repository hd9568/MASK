from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional
import uuid

from sparsene.agent.runtime import AgentModelConfig, invoke_json_agent, make_zero_usage
from sparsene.logging import get_logger
from sparsene.op_gen.nvir.nvop import GmemInout


logger = get_logger(__name__)

DEFAULT_TOKEN_PLAN_API_KEY = "tp-ckk8svdzeha2h3dly4ec5984bq914cpty4shelmntgg8c83p"
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
class StrategyDecision:
    provider: str
    rationale: str
    tensor_placement: TensorPlacementStrategy
    pipeline: PipelineStrategy
    token_usage: Dict[str, Any] = field(default_factory=dict)
    agent_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StrategyDecision":
        tensor_payload = payload.get("tensor_placement", {})
        pipeline_payload = payload.get("pipeline", {})
        tile_b = int(tensor_payload.get("tile_b", 64))
        if tile_b not in {16, 32, 64}:
            tile_b = 64
        tensor_payload["tile_b"] = tile_b
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
        if not isinstance(raw_stage_shifts, list):
            raw_stage_shifts = []
        normalized_stage_shifts: List[int] = []
        for item in raw_stage_shifts[:1]:
            try:
                shift = int(item)
            except (TypeError, ValueError):
                continue
            if shift == 1:
                # `[]` already means the default shift=1. Keep the canonical form
                # to avoid wasting tuning iterations on an equivalent configuration.
                continue
            if shift in {0, 2}:
                normalized_stage_shifts = [shift]
        pipeline_payload["stage_shifts"] = normalized_stage_shifts
        return cls(
            provider=payload.get("provider", "unknown"),
            rationale=payload.get("rationale", ""),
            tensor_placement=TensorPlacementStrategy(**tensor_payload),
            pipeline=PipelineStrategy(**pipeline_payload),
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
    api_key: Optional[str] = DEFAULT_TOKEN_PLAN_API_KEY
    api_base_url: Optional[str] = DEFAULT_TOKEN_PLAN_BASE_URL
    anthropic_base_url: Optional[str] = DEFAULT_TOKEN_PLAN_ANTHROPIC_URL
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
            api_key=os.environ.get("SPARSENE_STRATEGY_API_KEY", DEFAULT_TOKEN_PLAN_API_KEY),
            api_base_url=os.environ.get(
                "SPARSENE_STRATEGY_API_BASE_URL", DEFAULT_TOKEN_PLAN_BASE_URL
            ),
            anthropic_base_url=os.environ.get(
                "SPARSENE_STRATEGY_ANTHROPIC_BASE_URL",
                DEFAULT_TOKEN_PLAN_ANTHROPIC_URL,
            ),
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
            "--query-gpu=name,driver_version,memory.total",
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
    return HardwareProfile(
        gpu_name=parts[0] if len(parts) >= 1 else "unknown",
        driver_version=parts[1] if len(parts) >= 2 else "unknown",
        total_memory_mb=total_memory_mb,
        cuda_arch=cuda_arch,
        source="nvidia-smi",
    )


def build_sparse_format_profile(
    *,
    format_name: str,
    gmem_inouts: Dict[str, GmemInout],
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
    tensor_summaries = [
        FormatTensorSummary(
            name=name,
            dtype=gmem.dtype,
            shape=_shape_to_text(gmem.shape),
            role=tensor_roles.get(name, "external tensor"),
        )
        for name, gmem in gmem_inouts.items()
    ]
    decision_targets = [
        "N-direction tile size (TILE_B) selection among supported variants",
        "tensor placement for gathered/intermediate tensors",
        "whether B-side gather should materialize to smem",
        "whether COO idx/value should reside in smem or rmem",
        "whether COO idx/value loads should use cp.async or sync_scalar",
        "pipeline wait/commit/sync behavior and max_wait_prior depth",
        "stage_shifts for pipeline sync placement",
        "shared-memory alignment requirements for async paths",
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
    }


def build_strategy_request(
    *,
    format_name: str,
    gmem_inouts: Dict[str, GmemInout],
    feedback_context: Optional[StrategyFeedbackContext] = None,
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
        "Must preserve correctness for the current ME_TCF lowering path.",
        "Should prefer strategies already known to be safe on SM80/A100-like hardware.",
        "Should only choose from the documented structured schema.",
        "Do not invent new schema fields or unsupported codegen knobs.",
        "Current lowering only supports canonical swizzles: B staging uses swizzle_323 and COO restore uses swizzle_123.",
        "When COO idx/val are placed in rmem, coo_load_mode must also be sync_scalar; cp.async is incompatible with rmem-based staging.",
        "If async pipeline is disabled, all cp.async-based producers must also be downgraded to synchronous loads.",
        "Need to leave logs of request, prompt, response, performance, and profile for later analysis.",
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
            "Choose code-generation strategy knobs for ME_TCF. "
            "Focus on tensor placement, cp.async usage, and pipeline behavior. "
            "Return only structured data."
        ),
        output_schema=_default_output_schema(),
        constraints=constraints,
        observations=observations,
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
            "- COO staging: smem with cp.async gives best throughput; rmem is valid but forces sync_scalar loads which trade bandwidth for lower register/sync pressure.",
            "- pipeline max_wait_prior: None (unbounded), 0-3 (bounded depth). Lower values reduce occupancy pressure but limit async overlap. Values 1-3 are all valid.",
            "- stage_shifts: [] (default shift 1), [0] (no shift), [2] (double shift). Do not emit [1]; it is equivalent to [] and wastes an iteration.",
            "- sync_threads_on_smem_input can be toggled independently; disabling it removes a __syncthreads before B/COO consumption.",
            "- align_cp_async_smem_outputs can be toggled; disabling it may save shared memory but risks misaligned cp.async on some problem sizes.",
            "- Change only a small number of knobs between adjacent iterations unless profile evidence strongly suggests a larger change.",
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
                "ratio_vs_cusparse={ratio} waves_per_sm={waves} mem_tp_pct={mem_tp} "
                "compute_tp_pct={compute_tp} strategy={strategy}".format(
                    iteration=item.get("iteration", "?"),
                    correct=validation.get("passed"),
                    my_ms=perf.get("mykernel_ms"),
                    cu_ms=perf.get("cusparse_ms"),
                    ratio=perf.get("ratio_vs_cusparse"),
                    waves=profile.get("waves_per_sm"),
                    mem_tp=profile.get("memory_throughput_pct"),
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
            f"correct={best_iteration.get('validation', {}).get('passed')}"
        )

    latest_validation_text = json.dumps(latest_validation, indent=2, ensure_ascii=False) if latest_validation else "{}"
    correctness_feedback_text = json.dumps(correctness_feedback, indent=2, ensure_ascii=False) if correctness_feedback else "{}"
    latest_performance_text = json.dumps(latest_performance, indent=2, ensure_ascii=False) if latest_performance else "{}"
    latest_profile_text = json.dumps(latest_profile, indent=2, ensure_ascii=False) if latest_profile else "{}"
    additional_notes_text = "\n".join(f"- {item}" for item in additional_notes) if additional_notes else "- none"

    return f"""You are a kernel strategy agent for sparse Tensor Core code generation.
You are also an expert in CUDA kernel optimization, Tensor Core scheduling, cp.async, shared-memory staging,
occupancy trade-offs, and profile-guided performance tuning.

Hardware:
- gpu_name: {request.hardware.gpu_name}
- driver_version: {request.hardware.driver_version}
- cuda_arch: {request.hardware.cuda_arch}
- total_memory_mb: {request.hardware.total_memory_mb}
- warp_size: {request.hardware.warp_size}

Sparse format:
- format_name: {request.sparse_format.format_name}
- summary: {request.sparse_format.summary}

Visible external tensors:
{tensor_lines}

What you need to do:
{target_lines}

Constraints:
{constraint_lines}

High-level CUDA optimization guidance:
{optimization_guidance}

Feedback loop state:
- iteration: {feedback.get("iteration", 1)}
- max_iterations: {feedback.get("max_iterations", 1)}

Recent strategy history and measured feedback:
{history_text}

Best-known iteration:
{best_iteration_text}

Latest validation result:
{latest_validation_text}

Structured correctness feedback:
{correctness_feedback_text}

Latest performance result:
{latest_performance_text}

Latest ncu profile summary:
{latest_profile_text}

Additional notes:
{additional_notes_text}

Output requirements:
- Return only structured JSON compatible with the provided schema.
- Keep the strategy conservative: prefer correctness-preserving choices.
- For cp.async paths, ensure alignment and safe fallback logic are considered.
- For pipeline decisions, avoid aggressive waits/commits unless justified.
- Use correctness feedback to avoid repeating strategies that fail systematically on the same synthetic shapes or sparsity levels.
- Explore TILE_B and placement/pipeline choices when the current strategies cluster too tightly; do not keep repeating the same memory/pipeline recipe.
- Use the history and profile feedback to explain why the next strategy should improve or stabilize performance.
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


def _heuristic_strategy_response(request: StrategyRequest) -> StrategyDecision:
    return StrategyDecision(
        provider="heuristic",
        rationale=(
            "Use the current best-known ME_TCF alignment recipe: preserve COO range, "
            "keep gathered B tiles in smem, enable cp.async for aligned COO/B-side loads, "
            "and keep the pipeline conservative with sync-on-smem-consumption."
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
            model=config.model,
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
            "Return only valid JSON matching the requested schema. "
            "Do not include markdown fences, prose, or extra keys."
        ),
        user_prompt=request.prompt,
        request_payload=request.to_dict(),
        response_parser=_extract_json_object,
        skill_names=config.skill_names,
    )
    decision_payload = dict(result.parsed_payload)
    if "provider" not in decision_payload:
        decision_payload["provider"] = config.provider
    decision_payload["token_usage"] = result.usage.to_dict()
    decision_payload["agent_metadata"] = {
        "run_dir": str(result.run_dir),
        "resolved_model": result.resolved_model.to_dict(),
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
    gmem_inouts: Dict[str, GmemInout],
    config: Optional[StrategyConfig] = None,
    feedback_context: Optional[StrategyFeedbackContext] = None,
) -> tuple[StrategyDecision, Path]:
    if config is None:
        config = StrategyConfig.from_env()
    request = build_strategy_request(
        format_name=format_name,
        gmem_inouts=gmem_inouts,
        feedback_context=feedback_context,
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

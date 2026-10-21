#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="/mnt/lustre-client/huoda/venv/llm-sog"
MODEL="${MODEL:?set MODEL through a model-specific launcher}"
KEY_ENV="${KEY_ENV:?set KEY_ENV through a model-specific launcher}"
EXPECTED_ENDPOINT="${EXPECTED_ENDPOINT:?set EXPECTED_ENDPOINT through a model-specific launcher}"
MODEL_CONFIG="${MODEL_CONFIG:-${ROOT}/llm_models.yaml}"
FEATURE_DIR="${FEATURE_DIR:-/mnt/lustre-client/huoda/selected_mtx}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/logs/e2e_runs}"
FORMAT_MAX_TOKENS="${FORMAT_MAX_TOKENS:-32768}"
FORMAT_MAX_ATTEMPTS="${FORMAT_MAX_ATTEMPTS:-2}"
FORMAT_TIMEOUT_SEC="${FORMAT_TIMEOUT_SEC:-600}"
STRATEGY_MAX_TOKENS="${STRATEGY_MAX_TOKENS:-16384}"
STRATEGY_TIMEOUT_SEC="${STRATEGY_TIMEOUT_SEC:-600}"
STRATEGY_ITERATIONS="${STRATEGY_ITERATIONS:-3}"
DATASET_WORKERS="${DATASET_WORKERS:-2}"
NCU_BIN="${NCU_BIN:-/home/huoda/local/cuda/nsight-compute-2025.1.0/ncu}"
TUNING_NAME="${TUNING_NAME:-strategy_${MODEL//-/_}_${STRATEGY_ITERATIONS}iter_w3_r32}"
DRIVER_PREFIX="${DRIVER_PREFIX:-format_strategy_driver}"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
DATASETS=(
    ASIC_680k DD Ga41As41H72 OVCAR-8H Si41Ge41H72 Yeast YeastH cant
    conf5_4-8x8-10 consph cop20k_A dc2 ddi eu-2005 mac_econ_fwd500 mc2depi
    mip1 pdb1HYS protein pwtk reddit rma10 scircuit shipsec1 web-BerkStan
    webbase-1M
)

usage() {
    cat <<EOF
Usage: $0 {check|formats|operators-check|start|resume|operators|status|tail|stop} [RUN_DIR]

check: local validation only. formats: connectivity probe and 26 format designs
only. start: connectivity probe, 26 format designs, then fixed-format strategy
campaigns. resume: continue the combined campaign in the same run directory.
operators: use only accepted, executable formats already in RUN_DIR; skip the
format Agent and report rejected/unsupported designs before starting.
operators-check: inspect that selection locally without starting a campaign.
status/tail/stop: inspect or stop only this campaign. start prints RUN_DIR.

Model=${MODEL}; key variable=${KEY_ENV}; format attempts=${FORMAT_MAX_ATTEMPTS};
strategy iterations=${STRATEGY_ITERATIONS} plus heuristic baseline;
N=512 FP32 TF32, warmup=3, repeat=32, measurement runs=3.
GPU measurements are serialized. Set RUN_PROBE=0 to skip the paid probe;
SKIP_NCU=1 to disable NCU. Existing accepted formats are reused on resume.
EOF
}

managed_pid_is_live() {
    local pid="${1:-}" run_dir="$2" pgid cmdline
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ')"
    [[ "${pgid}" == "${pid}" ]] || return 1
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
    [[ "${cmdline}" == *"${SCRIPT_PATH}"* && "${cmdline}" == *"_worker"* && "${cmdline}" == *"${run_dir}"* ]]
}

operator_preflight() {
    local run_dir="$1"
    local report_path="${run_dir}/${TUNING_NAME}_preflight.json"
    [[ -f "${run_dir}/format_design_summary.json" ]] || {
        echo "Missing format summary: ${run_dir}/format_design_summary.json" >&2
        return 1
    }
    cd "${ROOT}"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" python - \
        "${run_dir}" "${MODEL}" "${report_path}" "${OPERATOR_DATASETS_CSV:-}" <<'PY'
import json
import sys
from pathlib import Path

from sparsene.formats.custom.loader import load_run_format
from sparsene.op_gen.backend_registry import select_backend

run_dir, model, report_path, requested_csv = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
summary = json.loads((run_dir / "format_design_summary.json").read_text(encoding="utf-8"))
if summary.get("completed") != 26 or summary.get("interrupted") or summary.get("pending_datasets"):
    raise SystemExit("format-design batch is not complete; finish/resume it before operators")
if manifest.get("model", {}).get("model") != model:
    raise SystemExit("format run belongs to a different model")
by_dataset = {row["dataset"]: row for row in summary.get("results", [])}
repair_path = run_dir / "manual_repair_manifest.json"
repairs = json.loads(repair_path.read_text(encoding="utf-8")).get("repairs", {}) if repair_path.is_file() else {}
datasets = manifest.get("datasets", [])
if requested_csv:
    requested = [item.strip() for item in requested_csv.split(",") if item.strip()]
    if not requested or len(requested) != len(set(requested)) or not set(requested).issubset(datasets):
        raise SystemExit("OPERATOR_DATASETS_CSV must contain unique datasets from the run manifest")
    datasets = requested
selected, skipped = [], []
for dataset in datasets:
    row = by_dataset.get(dataset)
    if (not row or row.get("status") != "ok") and dataset not in repairs:
        skipped.append({"dataset": dataset, "reason": "format_not_accepted"})
        continue
    try:
        spec, _, artifact_path, artifact_sha = load_run_format(run_dir, dataset)
        plan = select_backend(spec, target="sm80", tile_n=64,
                              max_tiles_per_execution_segment=32)
        selected.append({"dataset": dataset, "backend": plan.backend,
                         "format_path": str(artifact_path),
                         "format_sha256": artifact_sha,
                         "source": "manual_repair" if dataset in repairs else "agent_accepted"})
    except Exception as exc:
        skipped.append({"dataset": dataset,
                        "reason": f"{type(exc).__name__}: {exc}"})
if not selected:
    raise SystemExit("no accepted format has an executable SM80 backend")
report = {"schema_version": "sparsekg.existing_format_operator_preflight.v1",
          "model": model, "run_dir": str(run_dir),
          "requested_datasets": datasets,
          "accepted_executable_count": len(selected), "skipped_count": len(skipped),
          "selected": selected, "skipped": skipped}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Operator preflight: {len(selected)} executable formats; "
      f"{len(skipped)} skipped. Report: {report_path}")
for item in skipped:
    print(f"  [{item['dataset']}] {item['reason']}")
PY
}

run_operator_phase() {
    local run_dir="$1"
    cd "${ROOT}"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    export PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
    operator_preflight "${run_dir}"
    mapfile -t RUN_DATASETS < <(
        python -c 'import json,sys; print("\n".join(x["dataset"] for x in json.load(open(sys.argv[1]))["selected"]))' \
            "${run_dir}/${TUNING_NAME}_preflight.json"
    )
    [[ "${#RUN_DATASETS[@]}" -gt 0 ]] || { echo "No executable datasets" >&2; return 1; }
    echo "Operator phase: ${#RUN_DATASETS[@]} datasets; ${STRATEGY_ITERATIONS} Agent rounds + iteration 0"
    NCU_ARGS=()
    [[ "${SKIP_NCU:-0}" == "1" ]] && NCU_ARGS+=(--skip-ncu)
    python -u "${ROOT}/python/test/run_v7_strategy_feedback.py" \
        "${run_dir}" --datasets "${RUN_DATASETS[@]}" --tuning-name "${TUNING_NAME}" \
        --agent-iterations "${STRATEGY_ITERATIONS}" \
        --n 512 --warmup 3 --repeat 32 --measurement-runs 3 \
        --jobs 1 --dataset-workers "${DATASET_WORKERS}" \
        --strategy-provider openai_http --strategy-model "${MODEL}" \
        --strategy-max-tokens "${STRATEGY_MAX_TOKENS}" \
        --strategy-timeout-sec "${STRATEGY_TIMEOUT_SEC}" \
        --model-config "${MODEL_CONFIG}" --ncu-bin "${NCU_BIN}" \
        --resume "${NCU_ARGS[@]}"
}

local_check() {
    local run_dir="${1:-}"
    [[ -x "${VENV}/bin/python" ]] || { echo "Missing project venv: ${VENV}" >&2; return 1; }
    [[ -f "${MODEL_CONFIG}" ]] || { echo "Missing model config: ${MODEL_CONFIG}" >&2; return 1; }
    [[ -d "${FEATURE_DIR}" ]] || { echo "Missing feature directory: ${FEATURE_DIR}" >&2; return 1; }
    if [[ "${SKIP_NCU:-0}" != "1" ]]; then
        [[ -x "${NCU_BIN}" ]] || { echo "Missing NCU: ${NCU_BIN} (or set SKIP_NCU=1)" >&2; return 1; }
    fi
    cd "${ROOT}"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" python - \
        "${MODEL_CONFIG}" "${FEATURE_DIR}" "${MODEL}" "${KEY_ENV}" \
        "${EXPECTED_ENDPOINT}" "${run_dir}" "${DATASETS[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

from sparsene.agent.config import resolve_model_config

config_path, feature_dir, name, key_env, endpoint, run_dir = sys.argv[1:7]
datasets = sys.argv[7:]
model = resolve_model_config(model_name=name, yaml_path=config_path)
if (model.provider, model.model, model.api_base_url.rstrip("/")) != (
    "openai_compatible", name, endpoint.rstrip("/")):
    raise SystemExit("model/provider/endpoint mismatch in llm_models.yaml")
if not os.environ.get(key_env) or not model.api_key:
    raise SystemExit(f"export {key_env} before running this campaign")
if len(datasets) != 26 or len(set(datasets)) != 26:
    raise SystemExit("expected 26 unique dataset names")
missing = [d for d in datasets if not (Path(feature_dir) / f"{d}.features.v3.json").is_file()]
if missing:
    raise SystemExit("missing feature files: " + ", ".join(missing))
if run_dir:
    manifest_path = Path(run_dir) / "run_manifest.json"
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_model = old.get("model", {})
        if old_model.get("model") != name or set(old.get("datasets", [])) != set(datasets):
            raise SystemExit("run directory belongs to another model or dataset set")
print(f"Local preflight OK: {name}; 26/26 feature files; key present=True; "
      f"reasoning_effort={model.reasoning_effort}; thinking_mode={model.thinking_mode}; "
      f"stream={model.stream}")
PY
}

probe_model() {
    local run_dir="$1"
    mkdir -p "${run_dir}/connectivity"
    cd "${ROOT}"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" python - \
        "${MODEL_CONFIG}" "${MODEL}" "${run_dir}/connectivity/model_probe_$(date +%Y%m%d_%H%M%S_%N).json" <<'PY'
import json
import sys
from pathlib import Path

from sparsene.agent.client import call_openai_compatible
from sparsene.agent.config import resolve_model_config

config_path, name, output_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
cfg = resolve_model_config(model_name=name, yaml_path=config_path, timeout_sec=120)
system_prompt = "Connectivity check. Reply with OK."
user_prompt = "Reply with OK."
record = {
    "model": cfg.to_dict(),
    "request": {"system_prompt": system_prompt, "user_prompt": user_prompt,
                "max_tokens": 2048, "reasoning_effort": cfg.reasoning_effort,
                "thinking_mode": cfg.thinking_mode, "stream": cfg.stream},
    "status": "failed",
}
try:
    response = call_openai_compatible(
        model=cfg.model, api_key=cfg.api_key, api_base_url=cfg.api_base_url,
        system_prompt=system_prompt, user_prompt=user_prompt,
        temperature=cfg.temperature, max_tokens=2048, timeout_sec=cfg.timeout_sec,
        reasoning_effort=cfg.reasoning_effort, thinking_mode=cfg.thinking_mode,
        stream=cfg.stream,
    )
    message = response.raw_payload.get("choices", [{}])[0].get("message", {})
    if not str(message.get("content") or "").strip() or response.finish_reason == "length":
        raise RuntimeError("probe returned no final answer or exhausted its output budget")
    record.update(status="ok", response=response.to_dict())
except Exception as exc:
    record.update(error_type=type(exc).__name__,
                  error=str(exc).replace(cfg.api_key, "[REDACTED]")[:500])
output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Connectivity: {record['status']}; audit: {output_path}")
if record["status"] != "ok":
    raise SystemExit(record.get("error", "empty model response"))
PY
}

run_format_phase() {
    local run_dir="$1"
    cd "${ROOT}"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    export PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
    RESUME_ARGS=()
    [[ -f "${run_dir}/run_manifest.json" ]] && RESUME_ARGS+=(--resume)
    python -u "${ROOT}/python/test/run_format_design_batch.py" \
        --feature-dir "${FEATURE_DIR}" --run-root "${RUN_ROOT}" \
        --run-dir "${run_dir}" --datasets "${DATASETS[@]}" \
        --model "${MODEL}" --model-config "${MODEL_CONFIG}" \
        --max-tokens "${FORMAT_MAX_TOKENS}" --max-attempts "${FORMAT_MAX_ATTEMPTS}" \
        --timeout-sec "${FORMAT_TIMEOUT_SEC}" "${RESUME_ARGS[@]}"
}

ACTION="${1:-help}"
shift || true
RUN_DIR="${1:-${RUN_DIR:-}}"
if [[ -n "${RUN_DIR}" ]]; then
    RUN_DIR="$(realpath -m "${RUN_DIR}")"
fi

case "${ACTION}" in
    check)
        local_check "${RUN_DIR}"
        ;;
    operators-check)
        [[ -n "${RUN_DIR}" ]] || { echo "operators-check requires RUN_DIR" >&2; exit 2; }
        local_check "${RUN_DIR}"
        operator_preflight "${RUN_DIR}"
        ;;
    start|resume|formats|operators)
        if [[ "${ACTION}" == "resume" && -z "${RUN_DIR}" ]]; then
            echo "resume requires RUN_DIR" >&2
            exit 2
        fi
        if [[ "${ACTION}" == "operators" && -z "${RUN_DIR}" ]]; then
            echo "operators requires RUN_DIR" >&2
            exit 2
        fi
        if [[ -z "${RUN_DIR}" ]]; then
            RUN_DIR="${RUN_ROOT}/$(date +%Y%m%d_%H%M%S)_${MODEL//-/_}_format_strategy26"
        fi
        local_check "${RUN_DIR}"
        mkdir -p "${RUN_DIR}"
        PID_FILE="${RUN_DIR}/${DRIVER_PREFIX}.pgid"
        LOG_FILE="${RUN_DIR}/${DRIVER_PREFIX}.log"
        if [[ -f "${PID_FILE}" ]] && managed_pid_is_live "$(<"${PID_FILE}")" "${RUN_DIR}"; then
            echo "Campaign already running: PID/PGID $(<"${PID_FILE}")"
            exit 0
        fi
        echo "Run directory: ${RUN_DIR}"
        if [[ "${ACTION}" == "operators" ]]; then
            operator_preflight "${RUN_DIR}"
        elif [[ "${RUN_PROBE:-1}" == "1" ]]; then
            echo "Checking ${MODEL} connectivity (timeout 120 s)..."
            probe_model "${RUN_DIR}"
        fi
        rm -f "${RUN_DIR}/stop.requested"
        WORKER_ACTION=_worker
        [[ "${ACTION}" == "operators" ]] && WORKER_ACTION=_operator_worker
        [[ "${ACTION}" == "formats" ]] && WORKER_ACTION=_format_worker
        nohup setsid bash "${SCRIPT_PATH}" "${WORKER_ACTION}" "${RUN_DIR}" >>"${LOG_FILE}" 2>&1 </dev/null &
        DRIVER_PID=$!
        printf '%s\n' "${DRIVER_PID}" > "${PID_FILE}"
        for _ in $(seq 1 10); do
            if managed_pid_is_live "${DRIVER_PID}" "${RUN_DIR}"; then
                echo "Started PID/PGID ${DRIVER_PID}; run directory: ${RUN_DIR}"
                echo "Log: ${LOG_FILE}"
                echo "Follow: $0 tail '${RUN_DIR}'"
                echo "Stop:   $0 stop '${RUN_DIR}'"
                exit 0
            fi
            if ! kill -0 "${DRIVER_PID}" 2>/dev/null; then
                echo "Worker exited during startup:" >&2
                tail -n 40 "${LOG_FILE}" >&2 || true
                exit 1
            fi
            sleep 1
        done
        echo "Worker did not become ready; inspect ${LOG_FILE}" >&2
        exit 1
        ;;
    _worker)
        [[ -n "${RUN_DIR}" ]] || exit 2
        echo "Phase 1/2: format design for ${MODEL}"
        run_format_phase "${RUN_DIR}"
        [[ ! -f "${RUN_DIR}/stop.requested" ]] || { echo "Stop requested after format phase"; exit 143; }
        python - "${RUN_DIR}/format_design_summary.json" <<'PY'
import json
import sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if s.get("ok") != 26 or s.get("failed") != 0 or s.get("interrupted"):
    raise SystemExit(f"Format phase incomplete: ok={s.get('ok')} failed={s.get('failed')} "
                     f"interrupted={s.get('interrupted')}; inspect logs and resume")
print("Format quality gate: 26/26 accepted")
PY
        echo "Phase 2/2: strategy codegen, correctness, timing and NCU"
        run_operator_phase "${RUN_DIR}"
        ;;
    _format_worker)
        [[ -n "${RUN_DIR}" ]] || exit 2
        echo "Format-only phase: 26 datasets, model=${MODEL}"
        run_format_phase "${RUN_DIR}"
        echo "Format-only phase finished. Review format_design_summary.json before starting operators."
        ;;
    _operator_worker)
        [[ -n "${RUN_DIR}" ]] || exit 2
        run_operator_phase "${RUN_DIR}"
        ;;
    status|tail|stop)
        [[ -n "${RUN_DIR}" ]] || { echo "${ACTION} requires RUN_DIR" >&2; exit 2; }
        PID_FILE="${RUN_DIR}/${DRIVER_PREFIX}.pgid"
        LOG_FILE="${RUN_DIR}/${DRIVER_PREFIX}.log"
        case "${ACTION}" in
            status)
                if [[ -f "${PID_FILE}" ]] && managed_pid_is_live "$(<"${PID_FILE}")" "${RUN_DIR}"; then
                    echo "RUNNING PID/PGID $(<"${PID_FILE}")"
                else
                    echo "NOT RUNNING"
                fi
                if [[ -f "${RUN_DIR}/format_design_summary.json" ]]; then
                    python - "${RUN_DIR}/format_design_summary.json" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
print(f"Formats: accepted={s.get('ok', 0)}/26 failed={s.get('failed', 0)}")
PY
                fi
                if [[ -f "${RUN_DIR}/${TUNING_NAME}_summary.json" ]]; then
                    python - "${RUN_DIR}/${TUNING_NAME}_summary.json" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
print("Strategies:", s.get("counts", {}))
PY
                fi
                ;;
            tail)
                [[ -f "${LOG_FILE}" ]] || { echo "Log not found: ${LOG_FILE}" >&2; exit 1; }
                tail -n 40 -f "${LOG_FILE}"
                ;;
            stop)
                [[ -f "${PID_FILE}" ]] || { echo "No campaign PID recorded"; exit 0; }
                DRIVER_PID="$(<"${PID_FILE}")"
                managed_pid_is_live "${DRIVER_PID}" "${RUN_DIR}" || { echo "Campaign is not running"; exit 0; }
                touch "${RUN_DIR}/stop.requested"
                kill -TERM -- "-${DRIVER_PID}"
                echo "Stop signal sent to campaign process group ${DRIVER_PID}"
                ;;
        esac
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

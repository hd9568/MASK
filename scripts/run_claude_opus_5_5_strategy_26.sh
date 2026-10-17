#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="/mnt/lustre-client/huoda/venv/llm-sog"
FORMAT_RUN_DIR="${FORMAT_RUN_DIR:-${ROOT}/logs/e2e_runs/20261014_002307_claude_opus_5_5_format26}"
MODEL_CONFIG="${MODEL_CONFIG:-${ROOT}/llm_models.yaml}"
MODEL="claude-opus-5-5"
TUNING_NAME="${TUNING_NAME:-strategy_claude_opus_5_5_3iter_w2_r16}"
WARMUP="${WARMUP:-2}"
REPEAT="${REPEAT:-16}"
AGENT_ITERATIONS="${AGENT_ITERATIONS:-3}"
MEASUREMENT_RUNS="${MEASUREMENT_RUNS:-3}"
DATASET_WORKERS="${DATASET_WORKERS:-2}"
STRATEGY_MAX_TOKENS="${STRATEGY_MAX_TOKENS:-8192}"
STRATEGY_TIMEOUT_SEC="${STRATEGY_TIMEOUT_SEC:-600}"
NCU_BIN="${NCU_BIN:-/home/huoda/local/cuda/nsight-compute-2025.1.0/ncu}"

usage() {
    cat <<EOF
Usage:
  $0 check [FORMAT_RUN_DIR]   Local preflight only; no API/GPU calls
  $0 start [FORMAT_RUN_DIR]   Start all 26 formats (GPU measurements are serialized)
  FAILED_ONLY=1 FAILED_SOURCE_SUMMARY=<summary.json> $0 start [FORMAT_RUN_DIR]
                              Rerun datasets not marked `completed` in a prior summary
  $0 status [FORMAT_RUN_DIR]  Show process and dataset summary
  $0 tail [FORMAT_RUN_DIR]    Follow the live log (Ctrl-C exits tail only)
  $0 stop [FORMAT_RUN_DIR]    Stop only this campaign process group

The fixed Opus 5.5 format run is:
  ${FORMAT_RUN_DIR}

Defaults: strategy model=${MODEL}, agent iterations=${AGENT_ITERATIONS} plus
heuristic iteration 0, N=512, warmup=${WARMUP}, repeat=${REPEAT},
${MEASUREMENT_RUNS} independent measurements per candidate. Set
FORMAT_RUN_DIR, MODEL_CONFIG, WARMUP, REPEAT, AGENT_ITERATIONS,
MEASUREMENT_RUNS, DATASET_WORKERS, STRATEGY_MAX_TOKENS, STRATEGY_TIMEOUT_SEC,
or NCU_BIN in the environment to override.
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
    [[ "${cmdline}" == *"run_v7_strategy_feedback.py"* && "${cmdline}" == *"${run_dir}"* ]]
}

local_preflight() {
    local format_run_dir="$1"
    [[ -x "${VENV}/bin/python" ]] || { echo "ERROR: project venv missing: ${VENV}" >&2; return 1; }
    [[ -f "${MODEL_CONFIG}" ]] || { echo "ERROR: missing model config: ${MODEL_CONFIG}" >&2; return 1; }
    [[ -f "${format_run_dir}/run_manifest.json" ]] || { echo "ERROR: missing format run manifest: ${format_run_dir}/run_manifest.json" >&2; return 1; }
    [[ -f "${format_run_dir}/format_design_summary.json" ]] || { echo "ERROR: missing format summary: ${format_run_dir}/format_design_summary.json" >&2; return 1; }

    cd "${ROOT}"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" python - "${format_run_dir}" "${MODEL_CONFIG}" "${TUNING_NAME}" <<'PY'
import json
import sys
from pathlib import Path

from sparsene.agent.config import resolve_model_config
from sparsene.formats.custom.loader import load_run_format
from sparsene.op_gen.backend_registry import select_backend

run_dir = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
tuning_name = sys.argv[3]
manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
summary = json.loads((run_dir / "format_design_summary.json").read_text(encoding="utf-8"))
datasets = manifest.get("datasets", [])
if len(datasets) != 26 or len(set(datasets)) != 26:
    raise SystemExit(f"expected 26 unique datasets in manifest, found {len(datasets)}")
if summary.get("ok") != 26 or summary.get("failed") != 0 or summary.get("interrupted"):
    raise SystemExit(
        "format design is not a clean 26/26 completion: "
        f"ok={summary.get('ok')} failed={summary.get('failed')} interrupted={summary.get('interrupted')}"
    )
missing = []
for dataset in datasets:
    if not (run_dir / "formats" / f"{dataset}.format.json").is_file():
        missing.append(f"format:{dataset}")
    if not (run_dir / "datasets" / dataset / "matrix_features.json").is_file():
        missing.append(f"features:{dataset}")
if missing:
    raise SystemExit("missing required artifacts: " + ", ".join(missing))

model = resolve_model_config(model_name="claude-opus-5-5", yaml_path=config_path)
if model.provider != "openai_compatible" or model.model != "claude-opus-5-5":
    raise SystemExit(f"unexpected model resolution: {model.model_name}/{model.model}/{model.provider}")
if model.api_base_url.rstrip("/") != "https://api.openlux.ai/v1":
    raise SystemExit(f"expected OpenLux /v1 endpoint, got {model.api_base_url}")
if not model.api_key:
    raise SystemExit("Claude Opus 5.5 API key is unavailable; import OPENLUX_API_KEY in this shell")

supported, unsupported = [], []
for dataset in datasets:
    try:
        spec, _, artifact_path, artifact_sha = load_run_format(run_dir, dataset)
        plan = select_backend(spec, target="sm80", tile_n=64, max_tiles_per_execution_segment=32)
        supported.append({"dataset": dataset, "backend": plan.backend, "codegen_format": plan.codegen_format_name,
                          "format_path": str(artifact_path), "format_sha256": artifact_sha})
    except Exception as exc:
        unsupported.append({"dataset": dataset, "error": f"{type(exc).__name__}: {exc}"})

report = {
    "schema_version": "sparsekg.strategy_preflight.v1",
    "format_run_dir": str(run_dir),
    "tuning_name": tuning_name,
    "model": {"name": model.model, "provider": model.provider, "endpoint": model.api_base_url,
              "api_key_present": True},
    "datasets": datasets,
    "supported_count": len(supported),
    "unsupported_count": len(unsupported),
    "supported": supported,
    "unsupported": unsupported,
}
report_path = run_dir / f"{tuning_name}_preflight.json"
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Local preflight OK: format designs 26/26; executable backend {len(supported)}/26.")
print(f"Strategy model: {model.model} via OpenLux; iterations={int(__import__('os').environ.get('AGENT_ITERATIONS', '3'))} + heuristic baseline.")
print(f"Evaluation: N=512, warmup={int(__import__('os').environ.get('WARMUP', '2'))}, repeat={int(__import__('os').environ.get('REPEAT', '16'))}; GPU calls are serialized.")
print(f"Preflight report: {report_path}")
for item in unsupported:
    print(f"  [backend_unsupported] {item['dataset']}: {item['error']}")
if unsupported:
    print("WARNING: unsupported formats are retained and reported; they will not produce operators.")
PY
}

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
ACTION="${1:-start}"
shift || true
FORMAT_RUN_DIR="$(realpath -m "${1:-${FORMAT_RUN_DIR}}")"

case "${ACTION}" in
    check)
        local_preflight "${FORMAT_RUN_DIR}"
        ;;
    start)
        local_preflight "${FORMAT_RUN_DIR}"
        PID_FILE="${FORMAT_RUN_DIR}/${TUNING_NAME}.pgid"
        LOG_FILE="${FORMAT_RUN_DIR}/${TUNING_NAME}_driver.log"
        if [[ -f "${PID_FILE}" ]] && managed_pid_is_live "$(<"${PID_FILE}")" "${FORMAT_RUN_DIR}"; then
            echo "Campaign already running (PID/PGID $(<"${PID_FILE}"))."
            echo "Use: $0 tail ${FORMAT_RUN_DIR}"
            exit 0
        fi
        mkdir -p "${FORMAT_RUN_DIR}"
        # This launcher is commonly invoked as `bash script.sh`; it need not
        # carry the executable bit. Start its worker through bash explicitly.
        nohup setsid bash "${SCRIPT_PATH}" _worker "${FORMAT_RUN_DIR}" >>"${LOG_FILE}" 2>&1 </dev/null &
        DRIVER_PID=$!
        printf '%s\n' "${DRIVER_PID}" > "${PID_FILE}"
        for _ in $(seq 1 10); do
            if managed_pid_is_live "${DRIVER_PID}" "${FORMAT_RUN_DIR}"; then
                echo "Started PID/PGID ${DRIVER_PID}."
                echo "Live log: ${LOG_FILE}"
                echo "Monitor: $0 tail ${FORMAT_RUN_DIR}"
                echo "Status:  $0 status ${FORMAT_RUN_DIR}"
                echo "Stop:    $0 stop ${FORMAT_RUN_DIR}"
                exit 0
            fi
            if ! kill -0 "${DRIVER_PID}" 2>/dev/null; then
                echo "ERROR: worker exited during startup; recent log:" >&2
                tail -n 60 "${LOG_FILE}" >&2 || true
                exit 1
            fi
            sleep 1
        done
        echo "ERROR: worker did not become ready; recent log:" >&2
        tail -n 60 "${LOG_FILE}" >&2 || true
        exit 1
        ;;
    _worker)
        cd "${ROOT}"
        # shellcheck disable=SC1091
        source "${VENV}/bin/activate"
        export PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
        if [[ "${FAILED_ONLY:-0}" == "1" ]]; then
            SOURCE_SUMMARY="${FAILED_SOURCE_SUMMARY:-${FORMAT_RUN_DIR}/${TUNING_NAME}_summary.json}"
            [[ -f "${SOURCE_SUMMARY}" ]] || { echo "ERROR: missing failed-case source summary: ${SOURCE_SUMMARY}" >&2; exit 1; }
            mapfile -t DATASETS < <(
                python - "${SOURCE_SUMMARY}" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
results = payload.get("results", [])
selected = [item["dataset"] for item in results
            if item.get("status") in {"completed_with_failures", "failed", "unsupported"}]
if not selected:
    raise SystemExit("No failed/unsupported datasets found in source summary")
print("\n".join(selected))
PY
            )
            echo "Selected ${#DATASETS[@]} failed/unsupported datasets from ${SOURCE_SUMMARY}: ${DATASETS[*]}"
        else
            mapfile -t DATASETS < <(
                python -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["datasets"]))' \
                    "${FORMAT_RUN_DIR}/run_manifest.json"
            )
        fi
        exec python -u "${ROOT}/python/test/run_v7_strategy_feedback.py" \
            "${FORMAT_RUN_DIR}" \
            --datasets "${DATASETS[@]}" \
            --tuning-name "${TUNING_NAME}" \
            --agent-iterations "${AGENT_ITERATIONS}" \
            --n 512 --warmup "${WARMUP}" --repeat "${REPEAT}" \
            --measurement-runs "${MEASUREMENT_RUNS}" \
            --jobs 1 --dataset-workers "${DATASET_WORKERS}" \
            --strategy-provider openai_http --strategy-model "${MODEL}" \
            --strategy-max-tokens "${STRATEGY_MAX_TOKENS}" \
            --strategy-timeout-sec "${STRATEGY_TIMEOUT_SEC}" \
            --model-config "${MODEL_CONFIG}" \
            --ncu-bin "${NCU_BIN}" --resume
        ;;
    status|tail|stop)
        PID_FILE="${FORMAT_RUN_DIR}/${TUNING_NAME}.pgid"
        LOG_FILE="${FORMAT_RUN_DIR}/${TUNING_NAME}_driver.log"
        case "${ACTION}" in
            status)
                if [[ -f "${PID_FILE}" ]] && managed_pid_is_live "$(<"${PID_FILE}")" "${FORMAT_RUN_DIR}"; then
                    echo "RUNNING PID/PGID $(<"${PID_FILE}")"
                else
                    echo "NOT RUNNING"
                fi
                SUMMARY="${FORMAT_RUN_DIR}/${TUNING_NAME}_summary.json"
                if [[ -f "${SUMMARY}" ]]; then
                    python - "${SUMMARY}" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
c=d.get("counts", {})
print(f"Datasets={c.get('datasets', 0)} completed={c.get('completed', 0)} with_failures={c.get('completed_with_failures', 0)} failed={c.get('failed', 0)} unsupported={c.get('unsupported', 0)}")
PY
                elif [[ -f "${LOG_FILE}" ]]; then
                    tail -n 8 "${LOG_FILE}"
                fi
                ;;
            tail)
                [[ -f "${LOG_FILE}" ]] || { echo "Log not created yet: ${LOG_FILE}"; exit 1; }
                tail -n 40 -f "${LOG_FILE}"
                ;;
            stop)
                [[ -f "${PID_FILE}" ]] || { echo "No campaign PID recorded."; exit 0; }
                DRIVER_PID="$(<"${PID_FILE}")"
                managed_pid_is_live "${DRIVER_PID}" "${FORMAT_RUN_DIR}" || { echo "Campaign is not running (stale PID record)."; exit 0; }
                echo "Sending SIGTERM only to campaign process group ${DRIVER_PID}..."
                kill -TERM -- "-${DRIVER_PID}"
                for _ in $(seq 1 30); do
                    kill -0 -- "-${DRIVER_PID}" 2>/dev/null || { echo "Campaign stopped."; exit 0; }
                    sleep 1
                done
                echo "Stop signal sent; inspect status/log for current iteration completion."
                ;;
        esac
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

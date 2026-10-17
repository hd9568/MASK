#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="/mnt/lustre-client/huoda/venv/llm-sog"
MODEL="${MODEL:-claude-opus-5}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/logs/e2e_runs}"
MODEL_CONFIG="${MODEL_CONFIG:-${ROOT}/llm_models.yaml}"
FEATURE_DIR="${FEATURE_DIR:-/mnt/lustre-client/huoda/selected_mtx}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"
DATASETS=(
    ASIC_680k DD Ga41As41H72 OVCAR-8H Si41Ge41H72 Yeast YeastH cant
    conf5_4-8x8-10 consph cop20k_A dc2 ddi eu-2005 mac_econ_fwd500 mc2depi
    mip1 pdb1HYS protein pwtk reddit rma10 scircuit shipsec1 web-BerkStan
    webbase-1M
)
if [[ -n "${DATASETS_CSV:-}" ]]; then
    IFS=, read -r -a DATASETS <<< "${DATASETS_CSV}"
fi

usage() {
    cat <<EOF
Usage:
  $0 check [RUN_DIR]       Validate local config and selected feature files; no API call
  $0 start [RUN_DIR]       Probe OpenLux connectivity, then start the format batch
  $0 status RUN_DIR        Show process and latest progress
  $0 tail RUN_DIR          Follow the live batch log (Ctrl-C exits tail only)
  $0 stop RUN_DIR          Stop only this campaign process group

By default, start creates a timestamped directory below RUN_ROOT. Pass that
returned path to status/tail/stop. RUN_ROOT, MODEL_CONFIG, FEATURE_DIR,
MAX_ATTEMPTS, MAX_TOKENS and TIMEOUT_SEC can be overridden in the environment.
The default dataset set is all 26 matrices. DATASETS_CSV may select a subset.
The model is selected by MODEL (default claude-opus-5) through OpenLux's endpoint;
OPENLUX_API_KEY must already be imported and is never printed or stored.
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
    [[ "${cmdline}" == *"run_format_design_batch.py"* && "${cmdline}" == *"${run_dir}"* ]]
}

local_check() {
    local run_dir="${1:-}"
    [[ -f "${MODEL_CONFIG}" ]] || { echo "ERROR: missing model config: ${MODEL_CONFIG}" >&2; return 1; }
    [[ -d "${FEATURE_DIR}" ]] || { echo "ERROR: missing feature directory: ${FEATURE_DIR}" >&2; return 1; }
    [[ -x "${VENV}/bin/python" ]] || { echo "ERROR: project venv is unavailable: ${VENV}" >&2; return 1; }

    cd "${ROOT}"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" python - "${MODEL_CONFIG}" "${FEATURE_DIR}" "${run_dir}" "${DATASETS[@]}" "${MODEL}" <<'PY'
import json
import sys
from pathlib import Path

from sparsene.agent.config import resolve_model_config

config_path, feature_root, run_dir = sys.argv[1:4]
model_name = sys.argv[-1]
datasets = sys.argv[4:-1]
model = resolve_model_config(model_name=model_name, yaml_path=config_path)
if model.provider != "openai_compatible" or model.model != model_name:
    raise SystemExit(f"unexpected model/provider resolution: {model.model_name}/{model.model} via {model.provider}")
if model.api_base_url.rstrip("/") != "https://api.openlux.ai/v1":
    raise SystemExit(f"{model_name} must resolve to the configured OpenLux /v1 endpoint")
if not model.api_key:
    raise SystemExit(
        f"{model_name}: OPENLUX_API_KEY is not visible to Python. Export it in this shell "
        "before running the script (a non-exported shell variable is not inherited)."
    )
feature_root = Path(feature_root)
missing = [name for name in datasets if not (feature_root / f"{name}.features.v3.json").is_file()]
if missing:
    raise SystemExit(f"missing v3 feature files for: {', '.join(missing)}")
if not datasets or len(set(datasets)) != len(datasets):
    raise SystemExit(f"internal dataset list error: expected unique nonempty datasets, got {len(datasets)}")
if run_dir:
    manifest_path = Path(run_dir) / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous = manifest.get("model", {})
        if previous.get("model") != model_name or previous.get("api_base_url", "").rstrip("/") != "https://api.openlux.ai/v1":
            raise SystemExit("refusing to resume a run directory created with another model or endpoint")
        prior_datasets = set(manifest.get("datasets", []))
        if prior_datasets and prior_datasets != set(datasets):
            raise SystemExit("refusing to resume a run directory with a different dataset set")
print(f"Local preflight OK: {model_name} via OpenLux; {len(datasets)}/{len(datasets)} selected feature files found; streaming={model.stream}; API key present=True.")
if run_dir:
    print(f"Run directory: {run_dir}")
PY
}

probe_openlux() {
    local run_dir="$1"
    cd "${ROOT}"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    mkdir -p "${run_dir}/connectivity"
    PROBE_NAME="${MODEL//-/_}_probe_$(date +%Y%m%d_%H%M%S).json"
    PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" python - "${MODEL_CONFIG}" "${run_dir}/connectivity/${PROBE_NAME}" "${TIMEOUT_SEC}" "${MODEL}" <<'PY'
import json
import sys
import time
from pathlib import Path

from sparsene.agent.config import resolve_model_config
from sparsene.agent.client import call_openai_compatible

config_path, output_path, timeout, model_name = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
cfg = resolve_model_config(model_name=model_name, yaml_path=config_path, max_tokens=32, timeout_sec=timeout)
started = time.time()
system_prompt = "You are a connectivity check. Reply with exactly OK."
user_prompt = "Reply with exactly OK."
record = {
    "model_name": cfg.model_name,
    "model": cfg.model,
    "provider": cfg.provider,
    "api_base_url": cfg.api_base_url,
    "api_key_present": bool(cfg.api_key),
    "status": "failed",
    "request_payload": {
        "model": cfg.model,
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": bool(cfg.stream),
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    },
    "system_prompt": system_prompt,
    "user_prompt": user_prompt,
}
try:
    response = call_openai_compatible(
        model=cfg.model,
        api_key=cfg.api_key,
        api_base_url=cfg.api_base_url,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=32,
        timeout_sec=cfg.timeout_sec,
        stream=cfg.stream,
    )
    record.update({
        "status": "ok",
        "elapsed_s": response.elapsed_s,
        "finish_reason": response.finish_reason,
        "usage": response.usage.to_dict(),
        "response": response.content[:200],
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_response": response.raw_payload,
    })
except Exception as exc:
    record.update({
        "elapsed_s": round(time.time() - started, 3),
        "error_type": type(exc).__name__,
        "error": str(exc)[:600],
    })
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"OpenLux connectivity: {record['status']} (model={cfg.model}, endpoint={cfg.api_base_url})")
print(f"Probe record: {output_path}")
if record["status"] != "ok":
    print(f"Probe error: {record.get('error', 'empty response')}", file=sys.stderr)
    raise SystemExit(1)
PY
}

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
ACTION="${1:-start}"
shift || true

case "${ACTION}" in
    check)
        RUN_DIR="${1:-${RUN_DIR:-}}"
        local_check "${RUN_DIR}"
        ;;
    start)
        if [[ -n "${1:-}" ]]; then
            RUN_DIR="$(realpath -m "$1")"
        elif [[ -n "${RUN_DIR:-}" ]]; then
            RUN_DIR="$(realpath -m "${RUN_DIR}")"
        else
            STAMP="$(date +%Y%m%d_%H%M%S)"
            RUN_DIR="${RUN_ROOT}/${STAMP}_${MODEL//-/_}_format${#DATASETS[@]}"
        fi
        [[ -n "${OPENLUX_API_KEY:-}" ]] || {
            echo "ERROR: OPENLUX_API_KEY is not set; import it before start." >&2
            exit 1
        }
        local_check "${RUN_DIR}"
        mkdir -p "${RUN_DIR}"
        PID_FILE="${RUN_DIR}/format_design_batch.pgid"
        if [[ -f "${PID_FILE}" ]] && managed_pid_is_live "$(<"${PID_FILE}")" "${RUN_DIR}"; then
            echo "Campaign already running (PID/PGID $(<"${PID_FILE}"))."
            echo "Run directory: ${RUN_DIR}"
            exit 0
        fi
        rm -f "${RUN_DIR}/stop.requested"
        probe_openlux "${RUN_DIR}"
        echo "Starting 26-dataset format design. Live log: ${RUN_DIR}/format_design_batch.log"
        nohup setsid "${SCRIPT_PATH}" _worker "${RUN_DIR}" >>"${RUN_DIR}/format_design_batch.log" 2>&1 </dev/null &
        DRIVER_PID=$!
        printf '%s\n' "${DRIVER_PID}" > "${PID_FILE}"
        for _ in $(seq 1 10); do
            if managed_pid_is_live "${DRIVER_PID}" "${RUN_DIR}"; then
                echo "Started PID/PGID ${DRIVER_PID}."
                echo "Run directory: ${RUN_DIR}"
                echo "Monitor: $0 tail '${RUN_DIR}'"
                echo "Stop this campaign: $0 stop '${RUN_DIR}'"
                exit 0
            fi
            if ! kill -0 "${DRIVER_PID}" 2>/dev/null; then
                echo "ERROR: batch worker exited during startup. Recent log:" >&2
                tail -n 50 "${RUN_DIR}/format_design_batch.log" >&2 || true
                exit 1
            fi
            sleep 1
        done
        echo "ERROR: batch worker did not become ready. Recent log:" >&2
        tail -n 50 "${RUN_DIR}/format_design_batch.log" >&2 || true
        exit 1
        ;;
    _worker)
        RUN_DIR="${1:?internal worker requires RUN_DIR}"
        cd "${ROOT}"
        # shellcheck disable=SC1091
        source "${VENV}/bin/activate"
        export PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
        RESUME_ARGS=()
        if [[ -f "${RUN_DIR}/run_manifest.json" ]]; then
            RESUME_ARGS+=(--resume)
        fi
        exec python -u "${ROOT}/python/test/run_format_design_batch.py" \
            --feature-dir "${FEATURE_DIR}" \
            --run-root "${RUN_ROOT}" \
            --run-dir "${RUN_DIR}" \
            --datasets "${DATASETS[@]}" \
            --model "${MODEL}" \
            --model-config "${MODEL_CONFIG}" \
            --max-tokens "${MAX_TOKENS}" \
            --max-attempts "${MAX_ATTEMPTS}" \
            --timeout-sec "${TIMEOUT_SEC}" \
            "${RESUME_ARGS[@]}"
        ;;
    status|tail|stop)
        RUN_DIR="${1:-${RUN_DIR:-}}"
        if [[ -z "${RUN_DIR}" ]]; then
            echo "ERROR: provide RUN_DIR as an argument or environment variable." >&2
            usage >&2
            exit 2
        fi
        RUN_DIR="$(realpath -m "${RUN_DIR}")"
        PID_FILE="${RUN_DIR}/format_design_batch.pgid"
        LOG_FILE="${RUN_DIR}/format_design_batch.log"
        case "${ACTION}" in
            status)
                if [[ -f "${PID_FILE}" ]] && managed_pid_is_live "$(<"${PID_FILE}")" "${RUN_DIR}"; then
                    if [[ -f "${RUN_DIR}/stop.requested" ]]; then
                        echo "STOPPING PID/PGID $(<"${PID_FILE}") (finishing current request before exit)"
                    else
                        echo "RUNNING PID/PGID $(<"${PID_FILE}")"
                    fi
                else
                    echo "NOT RUNNING"
                fi
                if [[ -f "${RUN_DIR}/format_design_summary.json" ]]; then
                    python - "${RUN_DIR}/format_design_summary.json" <<'PY'
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
print(f"Progress summary: {d.get('completed', d.get('ok', 0) + d.get('failed', 0))}/{d.get('dataset_count', 0)} completed; {d.get('ok', 0)} accepted; {d.get('failed', 0)} failed; interrupted={d.get('interrupted', False)}")
totals=d.get('token_totals', {}).get('all_recorded_requests', {})
if totals:
    print(f"Recorded/estimated tokens: input={totals.get('prompt_tokens', 0)} output={totals.get('completion_tokens', 0)} total={totals.get('total_tokens', 0)}")
PY
                elif [[ -f "${LOG_FILE}" ]]; then
                    tail -n 8 "${LOG_FILE}"
                fi
                python - "${RUN_DIR}" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
audits=list(root.glob("datasets/*/format_agent/*/audit.json"))
probe_files=list(root.glob("connectivity/*_probe.json"))
if audits or probe_files:
    totals={"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
    estimated=api=est_prompt=est_completion=0
    for path in audits + probe_files:
        try:
            audit=json.loads(path.read_text())
            usage=audit.get("usage",{})
        except (OSError, ValueError):
            continue
        totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        totals["total_tokens"] += int(usage.get("total_tokens") or 0)
        if audit.get("token_source") == "api" or usage.get("raw_usage"): api += 1
        else:
            estimated += 1
            est_prompt += int(usage.get("prompt_tokens") or 0)
            est_completion += int(usage.get("completion_tokens") or 0)
    print(f"Audit token rollup ({len(audits) + len(probe_files)} requests; API-reported={api}, estimated={estimated}): input={totals['prompt_tokens']} output={totals['completion_tokens']} total={totals['total_tokens']}")
    if estimated:
        print(f"  Estimated-only subset: input={est_prompt}, output={est_completion}; do not treat as provider billing totals.")
PY
                ;;
            tail)
                [[ -f "${LOG_FILE}" ]] || { echo "Log not created yet: ${LOG_FILE}"; exit 1; }
                tail -n 40 -f "${LOG_FILE}"
                ;;
            stop)
                [[ -f "${PID_FILE}" ]] || { echo "No campaign PID recorded."; exit 0; }
                DRIVER_PID="$(<"${PID_FILE}")"
                managed_pid_is_live "${DRIVER_PID}" "${RUN_DIR}" || {
                    echo "Campaign is not running (stale PID record)."
                    exit 0
                }
                touch "${RUN_DIR}/stop.requested"
                echo "Sending SIGTERM only to campaign process group ${DRIVER_PID}..."
                kill -TERM -- "-${DRIVER_PID}"
                for _ in $(seq 1 30); do
                    kill -0 -- "-${DRIVER_PID}" 2>/dev/null || { echo "Campaign stopped."; exit 0; }
                    sleep 1
                done
                echo "Stop requested. The current OpenLux request is being allowed to finish so its usage and partial summary can be saved."
                echo "Check progress with: $0 status '${RUN_DIR}'"
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

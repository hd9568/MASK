#!/usr/bin/env bash
# Rerun currently below-Sparsene DeepSeek V4 Pro cases using fresh Agent format
# designs (3 outer formats) and heuristic + 5 Agent strategies per format.
# Target selection is frozen from the current six-model CSV on the first run.
set -euo pipefail

ROOT=/home/huoda/sparsene
export RUN_DIR="${RUN_DIR:-${ROOT}/logs/e2e_runs/20261107_ds_v4pro_current_below_sparsene_f3_s5}"
export TABLE="${TABLE:-${ROOT}/logs/e2e_runs/incumbents/a100_a800_n512_fp32/six_model_speedup_comparison_data.csv}"
export FORMAT_ITERATIONS=3
export INITIAL_FORMAT=agent
export STRATEGY_AGENT_ITERATIONS=5
export MAX_PARALLEL_DATASETS="${MAX_PARALLEL_DATASETS:-7}"

exec bash "${ROOT}/scripts/run_ds_v4pro_below_sparsene_f3_s5.sh" "$@"

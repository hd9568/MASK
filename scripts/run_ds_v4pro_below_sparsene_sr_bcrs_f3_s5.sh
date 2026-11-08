#!/usr/bin/env bash
# Three formats: fixed repository SR-BCRS 16x8 at format 0, then two Agent designs.
# Each format: heuristic strategy iter 0 plus Agent iterations 1..5 (six total).
set -euo pipefail

ROOT=/home/huoda/sparsene
export RUN_DIR="${RUN_DIR:-${ROOT}/logs/e2e_runs/20261103_ds_v4pro_below_sparsene_sr_bcrs_f3_s5}"
export FORMAT_ITERATIONS=3
export INITIAL_FORMAT=sr_bcrs_16x8
export STRATEGY_AGENT_ITERATIONS=5
exec bash "${ROOT}/scripts/run_ds_v4pro_below_sparsene_f3_s5.sh" "$@"

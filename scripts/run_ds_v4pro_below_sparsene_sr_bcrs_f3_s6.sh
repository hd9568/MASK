#!/usr/bin/env bash
# Compatibility alias. The corrected experiment has six total strategy points (iter0..5).
set -euo pipefail

ROOT=/home/huoda/sparsene
echo "Note: corrected strategy budget is iter0..5; using the f3_s5 script." >&2
exec bash "${ROOT}/scripts/run_ds_v4pro_below_sparsene_sr_bcrs_f3_s5.sh" "$@"

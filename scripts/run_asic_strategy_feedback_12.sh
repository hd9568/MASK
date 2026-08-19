#!/usr/bin/env bash
set -eu

REPO_ROOT=/home/huoda/sparsene
RUN_DIR="$REPO_ROOT/logs/e2e_runs/20260722_232100_mask_deepseek_v4_pro_format26_v7_empirical_feedback"

cd "$REPO_ROOT"
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
export PYTHONPATH=python

# A fresh, isolated ASIC-only campaign: iteration 0 heuristic plus 12 Agent
# decisions.  Each compiled strategy is independently measured three times and
# ranked by median generated-kernel time. GPU execution remains strictly serial.
exec python python/test/run_v7_strategy_feedback.py \
  "$RUN_DIR" \
  --datasets ASIC_680k \
  --tuning-name strategy_tuning_asic_v2_12iter \
  --agent-iterations 12 \
  --measurement-runs 3 \
  --n 512 \
  --warmup 3 \
  --repeat 32 \
  --jobs 1 \
  --strategy-provider openai_http \
  --strategy-model deepseek-v4-pro \
  --strategy-max-tokens 16384 \
  --model-config llm_models.yaml \
  --ncu-bin /home/huoda/local/cuda/bin/ncu \
  --resume \
  "$@"

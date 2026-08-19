#!/usr/bin/env bash
set -eu

REPO_ROOT=/home/huoda/sparsene
RUN_DIR="$REPO_ROOT/logs/e2e_runs/20260722_232100_mask_deepseek_v4_pro_format26_v7_empirical_feedback"

cd "$REPO_ROOT"
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
export PYTHONPATH=python

exec python python/test/run_v7_strategy_feedback.py \
  "$RUN_DIR" \
  --agent-iterations 5 \
  --n 512 \
  --warmup 3 \
  --repeat 32 \
  --measurement-runs 3 \
  --jobs 1 \
  --strategy-provider openai_http \
  --strategy-model deepseek-v4-pro \
  --strategy-max-tokens 16384 \
  --model-config llm_models.yaml \
  --ncu-bin /home/huoda/local/cuda/bin/ncu \
  --resume \
  "$@"

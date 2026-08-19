#!/usr/bin/env bash
set -eu

REPO_ROOT=/home/huoda/sparsene
RUN_DIR="$REPO_ROOT/logs/e2e_runs/20260722_232100_mask_deepseek_v4_pro_format26_v7_empirical_feedback"

cd "$REPO_ROOT"
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
export PYTHONPATH=python

# Dataset pipelines overlap two at a time. Agent/network waits and GPU samples
# can be interleaved; the driver protects the non-thread-safe lowering/compiler
# with a small process lock and serializes GPU correctness, timing and NCU to
# keep single-card measurements uncontended. ASIC_680k is intentionally excluded
# because it has its own completed 12-iteration campaign.
exec python python/test/run_v7_strategy_feedback.py \
  "$RUN_DIR" \
  --exclude-datasets ASIC_680k \
  --tuning-name strategy_tuning_remaining_v2_5iter \
  --agent-iterations 5 \
  --measurement-runs 3 \
  --dataset-workers 2 \
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

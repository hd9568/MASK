#!/usr/bin/env bash
set -eu

REPO_ROOT=/home/huoda/sparsene
RUN_DIR="$REPO_ROOT/logs/e2e_runs/20260722_232100_mask_deepseek_v4_pro_format26_v7_empirical_feedback"

cd "$REPO_ROOT"
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
export PYTHONPATH=python

# The two dataset pipelines may overlap Agent/API waiting and CPU preparation.
# Code generation is protected by the driver's compiler lock, while correctness,
# timing, and NCU collection are serialized by its GPU lock, so the single-card
# measurements do not contend with each other.
exec python python/test/run_v7_strategy_feedback.py \
  "$RUN_DIR" \
  --datasets YeastH eu-2005 \
  --tuning-name strategy_tuning_yeasth_eu2005_v1_5iter \
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

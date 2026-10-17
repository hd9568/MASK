#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/huoda/sparsene
RUN_DIR=${RUN_DIR:-${REPO_ROOT}/logs/e2e_runs/20260902_sparsekg_joint_format3_strategy5_3datasets_v1}
SEED_RUN_DIR=${SEED_RUN_DIR:-${REPO_ROOT}/logs/e2e_runs/20260722_232100_mask_deepseek_v4_pro_format26_v7_empirical_feedback}
INCUMBENT_DIR=${INCUMBENT_DIR:-${REPO_ROOT}/logs/e2e_runs/incumbents/a100_a800_n512_fp32}

cd "${REPO_ROOT}"
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RUN_DIR}"

python python/test/run_joint_format_strategy_feedback.py \
  --run-dir "${RUN_DIR}" \
  --feature-dir /mnt/lustre-client/huoda/selected_mtx \
  --datasets ASIC_680k ddi OVCAR-8H \
  --format-iterations 3 \
  --strategy-agent-iterations 5 \
  --initial-feedback-run-dir "${SEED_RUN_DIR}" \
  --incumbent-dir "${INCUMBENT_DIR}" \
  --format-model deepseek-v4-pro \
  --format-max-tokens 32768 \
  --strategy-provider openai_http \
  --strategy-model deepseek-v4-pro \
  --strategy-max-tokens 16384 \
  --model-config "${REPO_ROOT}/llm_models.yaml" \
  --n 512 \
  --warmup 3 \
  --repeat 32 \
  --measurement-runs 3 \
  --resume \
  "$@"

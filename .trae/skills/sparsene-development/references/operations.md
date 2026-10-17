# Environment and Operations

## Contents

- Required environment
- Data and artifact paths
- Current entry points
- Cache controls
- Long-run discipline

## Required environment

Activate the project virtual environment in the same shell as every project
Python, Agent, generation, or validation command:

```bash
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
```

Use `PYTHONPATH=python` for repository entry points that import `sparsene`.
Current defaults target `sm_80` and GCC 13 through
`/opt/rh/gcc-toolset-13/root/usr/bin/g++`; verify the actual GPU, NVCC, driver,
host compiler, and NCU before a long run.

Do not interpret an Agent/IDE sandbox without `/dev/nvidia*` as proof that the
host node has no GPU. Report the execution environment precisely and rerun GPU
validation in the intended GPU shell when needed.

## Data and artifact paths

Keep these path roles separate:

| Role | Current path |
|---|---|
| Raw selected MatrixMarket inputs | `/mnt/lustre-client/zhangkaige/selected_mtx` |
| Persistent schema-v3 feature store | `/mnt/lustre-client/huoda/selected_mtx` |
| Matrix checkpoint cache | `/mnt/lustre-client/huoda/mtx_cache` |
| Coordinated Agent runs | `/home/huoda/sparsene/logs/e2e_runs/<run_id>` |

Persistent feature files are named `<dataset>.features.v3.json`; validate
`features_manifest.csv` status and SHA256 before reuse. Raw `.mtx` files and
derived feature JSON are not interchangeable even though both directories are
named `selected_mtx`.

## Current entry points

Always inspect `--help` or `parse_args()` before composing a command. The root
README contains useful architecture context but some older flag names.

```bash
PYTHONPATH=python python3 python/test/end2end_pipeline.py --help
PYTHONPATH=python python3 python/test/run_format_design_batch.py --help
PYTHONPATH=python python3 python/test/run_strategy_feedback_multidataset.py --help
PYTHONPATH=python python3 python/test/test_generate.py --help
PYTHONPATH=python python3 python/test/generate_sparse_formats.py --help
PYTHONPATH=python python3 python/test/generate_custom_format_operators.py --help
PYTHONPATH=python python3 python/test/run_custom_format_e2e.py --help
```

Use these workflows according to purpose:

- Extract persistent v3 features:
  `bash scripts/extract_matrix_features.sh <raw-mtx-dir> <feature-dir>`.
- Run audited 26-matrix research-format design in the background:
  `bash scripts/run_format_design_26_background.sh [run-dir]`.
- Generate declared format kernels into `python/test/results`:
  `PYTHONPATH=python python3 python/test/generate_sparse_formats.py`.
- Export an audited format run into offline modules:
  `PYTHONPATH=python python3 python/test/export_custom_formats.py <run-dir>`.
- Generate and validate matching saved custom formats in their original run:
  `PYTHONPATH=python python3 python/test/generate_custom_format_operators.py <run-dir>`
  followed by `PYTHONPATH=python python3 python/test/run_custom_format_e2e.py <run-dir>`.
- Run the current integrated single-matrix loop:
  `PYTHONPATH=python python3 python/test/end2end_pipeline.py` with current flags.
- Sweep generated ME-TCF `TC_BLOCK_K={8,16,32}` with full provenance and
  correctness gating: `bash scripts/sweep_tc_block_k.sh`.

The v6 research-format batch accepts valid frontend-only designs. The legacy
single-matrix DTC route remains ME-TCF/COO-specific. Saved designs matching the
semantic TCF-MCO SM80 contract use the custom-format generator/runner; do not
send unmatched research designs into that backend.

## Cache controls

`main.cu` uses `/mnt/lustre-client/huoda/mtx_cache` unless
`SPARSENE_MTX_CACHE_DIR` overrides it. The host uses checkpoint caching when
`SPARSENE_DISABLE_MTX_CACHE` is absent or `0`; current end-to-end and sweep
drivers explicitly disable it for safety. `SPARSENE_MTX_CACHE_REFRESH=1`
forces refresh when the host cache is enabled.

Checkpoint identity currently includes matrix name/shape, `TC_BLOCK_K`,
`ROW_WINDOW_M`, and synthetic sparsity. When adding a format, reorder,
preprocessing rule, or metadata semantic, extend the key before enabling cache
reuse.

`end2end_pipeline.py` keeps compiled binaries below
`<log-root>/.kernel_binary_cache`. Its key currently includes the generated
`.inc` digest, BLK_K, BLK_M, and N. Extend the key for every new compile-time or
host ABI semantic. Prefer an isolated log root/cache while investigating a new
configuration rather than trusting historical binaries.

`SPARSENE_TC_BLOCK_K_AGGRESSIVE=1` currently changes heuristic BLK_K from 8 to
16 only when the extracted mean NNZ per row is positive and below 6. Treat this
as a dated heuristic in source, not a general format-design law.

## Long-run discipline

- Use a safe baseline before paid Agent iterations.
- Keep one GPU evaluation process at a time unless multiple GPUs are explicitly
  assigned; parallel tests on one GPU corrupt timing.
- Record PID, run directory, logs, revision/dirty state, model configuration
  identity, and resume policy.
- Prefer `--resume` for accepted paid format results; never mix incompatible
  format-design schemas in one run directory.
- Inspect a saved parse/API failure before authorizing a paid retry.
- Do not put credentials on command lines, in logs, or in documentation.

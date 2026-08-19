---
name: "sparsene-framework"
description: "Explains the Sparsene compilation stack and data model. Invoke when an agent must reason about format -> RTS -> OPIR -> NVIR -> CUDA generation."
---

# Sparsene Framework

## Performance Timing Contract

The primary project metric is `sparsene.kernel_only_post_conversion.v1`:
warmed repeated GPU SpMM compute-kernel time after format conversion, metadata
construction, padding, and row/column reorder. Conversion/reorder and
correctness-only inverse permutation, allocation/copies, compilation,
checkpoint I/O, and verification are outside the primary timer and must not
penalize format or strategy ranking. Record their correctness, storage, cache,
and implementation requirements separately. Operations actually executed in
the timed kernel remain included.

## Mission

Use this skill whenever a task touches sparse format semantics, code generation, or runtime integration inside Sparsene.

For the v7 strategy campaign, the format is a run-owned artifact below
`logs/e2e_runs/<run_id>/formats`. Load it by artifact path and SHA256, select a
semantic backend from hierarchy/axis/atomic semantics, and write every strategy
iteration below the same dataset tree. Do not substitute the legacy ME-TCF
format or a global exported custom module.

## Core Pipeline

The canonical pipeline is:

`Format -> derive_rts() -> Computent -> OPIR -> VarlenLoweringPass -> CValFlattenPass -> NVIR -> software pipeline/codegen -> .inc -> nvcc binary`

## What Each Layer Means

- `Format`: hierarchical sparse layout description with recursive `Format` nodes and an atomic leaf.
- `RTS`: flattened transformation sequence derived from the format tree.
- `Computent`: schedule plus array definitions inferred from RTS.
- `OPIR`: structured sparse operator IR with loops, loads, array refs, COO restore, and MMA.
- `NVIR`: CUDA-oriented lowering with explicit memory spaces and software-pipeline decisions.
- `.inc`: final generated CUDA/CUTLASS-style kernel implementation consumed by `host_program_generated.cu`.

## Key Format Concepts

- Axis factories: `df_axis`, `dv_axis`, `sf_axis`, `sv_axis`.
- `sv_axis` can represent a per-parent variable-length compressed coordinate,
  including but not limited to the K/column condensation used by ME-TCF.
- The research format agent may design with any declared atomic family or a new
  hierarchy. Deterministic capability classification records whether that
  proposal is integrated, standalone-validated, codegen-declared, or currently
  frontend-only. Backend support must not constrain research design and must
  never be overstated.

## Files That Matter

- Format definitions: `python/sparsene/formats/*.py`
- Format data model: `python/sparsene/format/format.py`
- Format design agent: `python/sparsene/op_gen/format_design_agent.py`
- RTS derivation: `python/sparsene/transform/rts.py`
- Computent: `python/sparsene/op_gen/computent/computent.py`
- OPIR generation: `python/sparsene/op_gen/opir/generate.py`
- NVIR lowering: `python/sparsene/op_gen/nvir/generate.py`
- Software pipeline/codegen: `python/sparsene/op_gen/nvir/compiler_driver.py`, `codegen.py`, `templates.py`
- Runtime host path: `examples/src_fp32/dtc/testbed/host_program_generated.cu`, `main.cu`
- v7 semantic host path: `examples/src_fp32/acc/testbed/custom_format_e2e.cu`
- v7 strategy loop: `python/test/run_v7_strategy_feedback.py`
- Reference hand-tuned kernel: `examples/src_fp32/dtc/testbed/kernel.inc` (2134 lines, used by `host_program.cu`)
- Benchmark script: `scripts/benchmark_all_me_tcf.py` (heuristic mode, all datasets)
- Batch testing: `scripts/run_end2end_all_datasets.sh` (agent mode, all datasets)
- Results summarizer: `scripts/summarize_e2e_results.py`

## Key Workflows

### Running the Pipeline

```bash
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate

# Single dataset, heuristic mode (no LLM agent):
python python/test/end2end_pipeline.py --mode heuristic -m cant.mtx -N 512 -i 10

# Single dataset, agent mode (LLM decides format + strategy):
python python/test/end2end_pipeline.py --mode agent -m cant.mtx -N 512 -i 10

# All datasets, heuristic mode, produce comparison markdown:
python scripts/benchmark_all_me_tcf.py

# All datasets, agent mode, produce log directory:
bash scripts/run_end2end_all_datasets.sh > log.log 2>&1 &

# Summarize batch results:
python scripts/summarize_e2e_results.py logs/logs_all_datasets_YYYYMMDD_HHMMSS
```

### Binary Kernel Cache

`_build_binary()` in `end2end_pipeline.py` implements a kernel binary cache to avoid
redundant nvcc compilations.  The cache key includes:

- **md5 of `.inc` content** (first 12 hex chars) — detects any codegen change
- **BLK_K, BLK_M, N** nvcc flags

Cache location: `{log_root}/.kernel_binary_cache/`.

**Important**: The `.inc` content digest ensures that different strategy decisions
(swizzle, pipeline config) produce different cache keys.  Always clear the cache
when switching between major codegen versions:

```bash
rm -rf logs/.kernel_binary_cache
```

### Dataset Status Conventions

| Status | Meaning |
|--------|---------|
| `ok` | Generated kernel matches cuSPARSE within tolerance |
| `FAILED` | Pipeline crashed or produced no output |
| `FAILED_NUMERICAL` | Kernel ran and produced valid timing, but element-wise comparison differs from cuSPARSE.  Timing is still usable for performance analysis. |
| `NO SUMMARY` | Pipeline did not produce a `summary.json` (incomplete run) |

**Historical legacy-DTC FAILED_NUMERICAL datasets** (do not apply this list to
the current v7 semantic backend without fresh evidence):

| Dataset | avg NNZ/row | Root Cause |
|---------|------------|------------|
| mc2depi | 4.0 | Extremely sparse TC blocks (1–2 NNZ each) |
| OVCAR-8H | 2.1 | Same |
| webbase-1M | 3.1 | Same |
| YeastH | 2.1 | Same |
| Yeast | 2.1 | Same |

These datasets produce correct results with the reference hand-tuned kernel (`kernel.inc`)
but have element-wise errors with the generated kernel (`me_tcf_kernel3.inc`).
The bug is in the COO atomic format sparse-to-dense restore logic for near-empty blocks.

## Agent Guardrails

- Do not free-form generate CUDA first; make structured decisions first.
- In research format design, prefer feature-backed physical layouts regardless
  of current implementation. In execution/strategy mode, require a matched and
  validated backend profile before code generation.
- Preserve correctness on varlen sparse axes before chasing performance.
- **Current integrated DTC route only — BLK_M must be ≥ 16 and a multiple of
  16** because that route uses the `m16n8k8` tensor-core atom. This is not a
  restriction on a research DSL proposal for a future backend.
  - `BLK_M=8` → `BLK_MMA_MNK.m = 0` → zero-sized register arrays → compile error.
  - `_adaptive_blk_m()` pins current heuristic execution to 16; capability
    validation admits current agent execution only for a consistent 16-row tile.
- **Current integrated DTC route only — BLK_K controls `TC_BLOCK_K`**:
  heuristic mode defaults to 8 and its opt-in adaptive layer selects 16 only
  when average NNZ/row is below 6. An integrated agent result uses `16x8` or
  `16x16`; research proposals may select other parameters and then require a
  new/extended backend profile.
  - Codegen generates K-iteration loops via `get<2>(BLK_MMA_MNK{})`.
  - Swizzle S parameter auto-adapts: `S = log2(BLK_K)` (8→3, 16→4).
  - Checkpoint filenames include `_TBK` and `_RWM` suffixes.
- **Swizzle contract**: host (`main.cu`) and kernel (`generate.py`) must agree on the same
  `Swizzle<1,2,S>` parameter.  `end2end_pipeline.py` overrides `coo_restore_swizzle` to
  `swizzle_124` when `BLK_K ≠ 8`.
- **Format design agent**: v5 designs a general hierarchical DSL and validates
  it without rewriting the proposal to ME-TCF. The response records a separate
  capability level. The current integrated DTC strategy/codegen loop may proceed
  only when the proposal exactly matches the integrated ME-TCF profile and uses
  `16x8` or `16x16`; other valid proposals remain design artifacts until their
  backend/host route is implemented.
- **Historical legacy-DTC codegen limitation**: extremely sparse matrices (avg NNZ/row < 4) may produce
  numerical errors (status `FAILED_NUMERICAL`).  These are matrices like Yeast, YeastH,
  OVCAR-8H, webbase-1M, mc2depi.  The generated kernel runs and produces valid timing
  (summarized in the stats), but individual element comparisons against cuSPARSE show
  discrepancies.  This is a pre-existing codegen pipeline issue, not introduced by
  the adaptive BLK_K/BLK_M changes.
- **Generated vs Reference kernel performance gap**: the generated kernel
  (`me_tcf_kernel3.inc`) is structurally different from the hand-tuned reference
  (`kernel.inc`).  Key differences:
  - Reference uses single large loop with fine-grained `__pipeline_commit/wait`
    (62 commit / 54 wait vs Generated's 49 / 40)
  - Reference has loop fusion (13 for-loops vs Generated's 19)
  - Reference has adaptive pipeline depth; Generated uses fixed 3-stage pipeline
  - **Impact**: Reference is faster on 25/26 datasets; only `shipsec1` is faster
    with Generated.  Gap ranges from +11% to +84%.
  - **Closing this gap** requires codegen architecture changes (loop fusion,
    adaptive pipeline depth), not simple parameter tuning.

## Shell Script Robustness

The batch runner (`scripts/run_end2end_all_datasets.sh`) includes:
- PID logging for OOM/debug diagnostics
- EXIT trap to log unexpected termination codes
- `set -eu` (not `set -euo pipefail` — pipefail was removed after causing silent exits when
  Python subprocess tee/pipes produced non-zero exit codes)

To debug unexpected exits:
```bash
dmesg | grep -i "killed process\|out of memory" | tail -10  # check OOM kills
grep "SCRIPT EXITED WITH CODE" log.log                        # check trap output
```

## Runtime Configuration

### Python Environment

All sparsene scripts rely on the venv at `/mnt/lustre-client/huoda/venv/llm-sog`.
Activate it before running any pipeline:

```bash
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
```

The venv lives on Lustre to avoid consuming home-directory disk quota.

### Environment Variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `SPARSENE_MTX_CACHE_DIR` | `/mnt/lustre-client/huoda/mtx_cache` | Directory for sparse-format-layer checkpoint files. `main.cu` writes/reads `.checkpoint` files here. |
| `SPARSENE_DISABLE_MTX_CACHE` | `"1"` (cache disabled) | **Default is `1`** in `end2end_pipeline.py` (set via `os.environ.setdefault`). Set to `0` to re-enable checkpoint cache. |
| `SPARSENE_TC_BLOCK_K_AGGRESSIVE` | `"0"` | Heuristic-mode opt-in for adaptive BLK_K; agent mode applies the normalized format-agent candidate. |
| `DATASET_DIR` | `/mnt/lustre-client/zhangkaige/selected_mtx` | Root directory containing `.mtx` files. |
| `NVCC_CCBIN` | `/opt/rh/gcc-toolset-13/root/usr/bin/g++` | Host compiler for nvcc. |
| `CUDA_ARCH` | `sm_80` | CUDA compute capability for nvcc. |

### Checkpoint Cache

Sparse format checkpoint files are persisted under the path set by `SPARSENE_MTX_CACHE_DIR`.
By default, `end2end_pipeline.py` sets `SPARSENE_DISABLE_MTX_CACHE=1` to avoid stale
checkpoints when switching BLK_K/BLK_M configurations.  When enabled (`=0`), `main.cu`
defaults to `/mnt/lustre-client/huoda/mtx_cache`.  The checkpoint filename now includes
`_TBK` and `_RWM` suffixes for per-configuration isolation.

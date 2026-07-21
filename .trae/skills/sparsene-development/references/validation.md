# Validation and Performance Evidence

## Contents

- Provenance first
- Validation ladder
- Correctness interpretation
- Performance methodology
- Cache and environment controls
- Focused commands

## Provenance first

Before testing, record which kernel is under test:

- freshly generated from current Python code;
- generated then modified by a fix/patch step;
- hand-tuned reference;
- historical checked-in artifact.

Record the `.inc` path and content digest. Confirm the host program includes or copies the same file. A fast binary is meaningless if it was built from a different artifact.

## Validation ladder

### 1. Python and structure

- Compile/import changed Python modules.
- Run focused unit tests.
- Print or inspect Format, RTS, Computent arrays/schedule, OPIR, and NVIR at the first failing boundary.
- Verify expected operation counts, logical kinds, shapes, memory spaces, and global-memory interfaces.

### 2. Generation determinism

- Generate twice from the same normalized input.
- Compare `.inc` digests.
- Confirm changed knobs alter expected code or metadata.
- Inspect template constants, launch bounds, SMEM layouts, `cp.async`, waits/commits, `ldmatrix`, MMA loops, and stores relevant to the hypothesis.

### 3. Compilation

- Use the architecture, host compiler, defines, and include paths expected by the testbed.
- Save the full build command and log.
- Treat compile success only as syntax/type/template validation.

### 4. Numerical correctness

Start with small synthetic cases that cover:

- dimensions smaller than and not divisible by tiles;
- empty and nearly empty rows/blocks;
- very low and moderate density;
- boundary offsets and partial tiles;
- each changed `BLK_M`/`BLK_K`/atomic-format configuration.

Then test representative real matrices against cuSPARSE. Require no reported element errors and consistent NaN/check values. Inspect output parsing code because executables may cap printed errors.

The current DTC `main.cu` emits an uncapped summary line:
`VERIFICATION errors=<n> nonfinite=<n> ...` and exits with code 2 when either
count is nonzero. Prefer this summary plus the process exit code over counting
printed `Error(...)` samples. A missing verification summary is a failure, not
evidence of correctness.

### 5. Performance

Only benchmark correct kernels. Record:

- GPU, driver, CUDA/NVCC, architecture, precision;
- matrix, `M/K/N`, NNZ, format and tile parameters;
- warmup/repeat counts;
- evaluation contract `sparsene.kernel_only_post_conversion.v1` and confirm
  that conversion/reorder/preprocessing are excluded;
- checkpoint and binary-cache state;
- generated/reference provenance;
- kernel time, cuSPARSE time, and the direction of the ratio.

Use distributions or repeated measurements for close results. Differences of only a few percent need noise control and reruns.

### 6. Profiling

Use NCU to answer a specific hypothesis. Relevant metrics may include registers/thread, static SMEM, grid/block size, waves/SM, memory and compute throughput, cache hit rates, memory-pipe utilization, and duration. Confirm the profiled launch is the generated kernel and that warmup/launch filters are correct.

## Correctness interpretation

- `mykernel_time` plus `cusparse_time` does not imply correctness.
- A matching aggregate or NaN check does not replace element-wise comparison.
- `FAILED_NUMERICAL` results may still help diagnose performance, but must never enter best-correct selection.
- A failure concentrated at first rows/columns often suggests layout, offset, or initialization issues; failures in near-empty blocks suggest atomic restore or padding/boundary semantics. Treat these as hypotheses, not automatic diagnoses.

## Performance methodology

The project-standard primary comparison is kernel-only after preprocessing:

- build/convert the sparse format, construct metadata, apply row/column
  reordering, and perform correctness-only inverse permutation outside the
  timed region;
- exclude allocation, H2D/D2H, checkpoint I/O, compilation, verification, and
  cuSPARSE `cusparseSpMM_preprocess`;
- time only the warmed repeated GPU SpMM compute launch(es). Every operation
  actually executed inside those launches—including metadata reads, gathers,
  restore, padding work, synchronization, MMA and stores—is included;
- never add preprocessing/reorder latency to the primary result or penalize a
  format's kernel ranking for it. Report implementation feasibility, metadata
  footprint, and optional preprocessing measurements separately.

Keep these comparisons separate:

- generated vs cuSPARSE: end-to-end goal for the generator;
- generated vs reference: codegen quality gap;
- candidate vs heuristic generated baseline: strategy-agent value;
- format A vs format B: rank by the same kernel-only metric; keep storage and
  optional preprocessing measurements as separate secondary columns.

Do not compare numbers across different GPUs, `N`, precisions, caches, or timer
boundaries without explicit normalization. Never label the current kernel-only
result as end-to-end application latency.

Keep ratio direction explicit:

- sweep `speedup_vs_cusparse = cusparse_ms / mykernel_ms`; greater than 1 wins;
- integrated `ratio_vs_cusparse = mykernel_ms / cusparse_ms`; less than 1 wins.

Never average or rank a mixed column without converting it to one convention.

## Cache and environment controls

- `end2end_pipeline.py` disables matrix checkpoint caching by default to avoid stale format/tile data.
- Its binary-cache key includes an `.inc` digest and selected compile parameters; verify the key whenever adding a semantic knob or build define.
- Clear or isolate caches when diagnosing unexpected invariance.
- Check dataset paths, CUDA visibility, `nvidia-smi`, NVCC, host compiler, and NCU availability before a long run.
- Avoid parallel final benchmarks on a single GPU because interference corrupts measurements.

## Current TC_BLOCK_K sweep

`scripts/sweep_tc_block_k.sh` is the current generated ME-TCF sweep entry. It:

- activates the project venv and generates/reuses
  `python/test/results/dtc.inc`;
- records the generated include and per-binary SHA256 digests;
- compiles `TC_BLOCK_K={8,16,32}` with BLK_M/ROW_WINDOW_M fixed to 16;
- disables matrix checkpoints for each benchmark;
- waits for an idle GPU, uses per-run timeouts, and records full logs;
- accepts a combination only when timing is positive, the process succeeds,
  and full verification reports zero errors/nonfinite values;
- selects the fastest correct K per dataset;
- reports the count, fraction, average, maximum, and dataset list for results
  faster than cuSPARSE.

When changing this sweep, preserve include-path provenance, atomic binary
publication, full verification parsing, correct-only best selection, and the
separate faster-than-cuSPARSE section. Do not infer K quality from a failed or
missing-verification run.

## Focused commands

Use current CLI help instead of copying dated commands:

```bash
PYTHONPATH=python python3 python/test/end2end_pipeline.py --help
PYTHONPATH=python python3 python/test/test_generate.py --help
PYTHONPATH=python python3 python/test/run_strategy_feedback_multidataset.py --help
PYTHONPATH=python python3 python/test/run_format_design_batch.py --help
PYTHONPATH=python python3 python/test/generate_sparse_formats.py --help
```

For non-GPU validation, run focused Python compilation/import and repository tests that touch the changed layer. For GPU validation, derive the exact build/run invocation from the current testbed scripts and save it with the results.

# SparseKG SM80 snapshot and H100 porting handoff

This file is intentionally tracked so a fresh clone of `sparsekg_on_h100`
retains the A800 conclusions. The H100 branch starts from the working SM80
implementation; it does **not** claim to contain an optimized or validated
SM90 backend yet. Keep the SM80 route operational while adding an explicitly
selected SM90 route.

## Current system and evidence boundaries

- The compiler path is Format DSL → RTS → Computent → OPIR → NVIR → CUDA,
  followed by compilation, full-output correctness, repeated kernel timing,
  optional NCU profiling, and two Agent feedback loops. The current default
  target is SM80/A100/A800, `N=512`, FP32 inputs/outputs with TF32 tensor-core
  execution. The primary score is warmed GPU SpMM compute time after format
  conversion (`sparsene.kernel_only_post_conversion.v1`).
- The format Agent owns hierarchy, axes, atomic leaf, grouping, logical tile
  sizes, and reorder semantics. The strategy Agent owns executable tensor
  placement, pipeline/compiler options, and runtime task mapping. A valid
  format proposal is not automatically executable; capability, host packing,
  codegen, and correctness are separate gates.
- Historical hand-tuned Sparsene, generated SparseKG, and cuSPARSE are distinct
  provenance classes. Do not label a newly generated heuristic kernel as
  the historical Sparsene 1.0× point without measuring that same kernel.

## Preserved A100/A800 conclusions

1. `sota_results/summary/paper_a100_n512_fp32_sparsene_per_dataset.csv`
   contains Sparsene speedups digitized from the paper's A100 figure.
   `sota_results/summary/sparsene_a100_n512.csv` instead selects the fastest
   of nine historical raw-log variants for each matrix. They are different
   baselines. The paper figure does not disclose the per-matrix format label.
2. Of the 26 historical fastest variants, 19 are SR-BCRS-16x8, three
   SR-BCRS-16x8-strict-lb, three SR-BCRS-16x8-multi-bind, and one
   ACC-BIT-TCF (`webbase-1M`). The first three share a dense format family
   but differ in execution policy; the ACC variant uses an MCO leaf.
3. The completed DeepSeek V4 Pro A800 joint run contains 370 validated
   generated points. Rebasing on the historical-best Sparsene reference
   reuses 364 points and excludes six old `webbase-1M` format0 SR-BCRS
   points because they are not ACC. In the historical-best comparison,
   DeepSeek best-of-history is numerically ahead on 22/26 matrices; the
   non-wins are `eu-2005`, `scircuit`, `web-BerkStan`, and `webbase-1M`.
   These cross-GPU ratios are exploratory, not same-GPU paper claims.
4. `webbase-1M` demonstrates why matching a Format DSL does not reproduce
   a hand-tuned kernel. The paper's bar is about 1.232× over cuSPARSE;
   the historical ACC log is 3.42285 ms (about 1.251×), while historical
   SR-BCRS-16x8 is 3.49842 ms (about 1.224×). The old generated SR-BCRS
   candidate reached 3.5008 ms (about 1.229×). The new generated
   ACC/MCO candidate's best correct iteration was 6.93114 ms (0.618×).
   Both historical and current cuSPARSE are about 4.282 ms, so the ~2×
   generated-ACC deficit is primarily a backend execution difference,
   not an A100/A800 cuSPARSE discrepancy.
5. For that MCO backend, `webbase-1M` has 62,501 row windows, 230,471
   active tiles, p95 6 tiles/window, but maximum 605. With a 16-tile
   segment cap it creates 63,598 execution segments. The current host
   emits two chunked main launches and a full-output reduction, allocating
   about 2.08 GB of partial output; the original ACC host directly writes
   output without that global reduction. The added traffic and launch
   topology are a source-grounded explanation for the gap, but their exact
   individual time costs have not yet been isolated by profiling.
   Current MCO strategy normalization forces row binding and its exposed
   segment caps top out at 128, so more Agent iterations alone cannot
   remove the reduction path.
6. The A100 historical log only retained a NaN sentinel, whereas the
   current generated-candidate ranking requires full-output correctness.
   Preserve this caveat when comparing historical and current times.

The original run directories, raw `.mtx` inputs, feature snapshots,
checkpoint caches, NCU reports, and generated `.inc` artifacts are under
ignored/external paths; **they are not in this Git branch**. Key local A800
records are:

- `logs/e2e_runs/20261107_ds_v4pro_joint_all26_f3_s5_backendgate_v1/`
- `logs/e2e_runs/20261108_webbase_acc_format0_s5/`
- `logs/e2e_runs/incumbents/a100_a800_n512_fp32/`
- `docs/codex/sparsene_per_dataset_format0_reference_20261108.md`
- `docs/codex/ds_v4pro_historical_reference_rebase_20261108.md`

Copy those artifacts separately if the H100 team needs raw trajectories;
never mix A800 binary caches or generated checkpoints into H100 scoring.

## Cache migration and portability

The A800 checkpoint directory now intended for use is
`/mnt/lustre-client/huoda/sparsekg_tmp/mtx_cache`; this is external data and
is not committed. Set `SPARSEKG_TMP_DIR` to a writable H100 scratch root and
`SPARSENE_MTX_CACHE_DIR` to its `mtx_cache` child. ACC/DTC/CSR legacy hosts
honor the explicit cache variable, then `SPARSEKG_TMP_DIR/mtx_cache`, while
keeping their old defaults if neither variable is set. Existing A800 SM80
builds therefore remain compatible. Disable or isolate checkpoint caching
while changing format semantics, architecture, tile shape, or host ABI.

## H100 development boundary

Before performance work, inventory the H100 node's GPU SKU, driver, CUDA
toolkit, host compiler, and Nsight Compute versions. Establish a same-GPU
cuSPARSE and original-reference baseline and run full numerical validation.
Then introduce an architecture selector and an SM90-specific backend plan,
codegen/host ABI, compilation flags, and hardware profile. Do not globally
replace SM80 tensor layouts, instruction selection, launch contracts, or
strategy normalization. The format Agent should see the new hardware
capabilities and physical-work costs; the strategy Agent should search only
knobs executable by the selected backend. Keep separate SM80 and SM90
tests, caches, run manifests, and performance tables. Check the exact
Format → host packing → generated code → verification contract for each
new instruction/dataflow path before ranking performance.

The first high-value backend investigation is the MCO long-tail path above:
avoid making a few extreme windows force a full-output partial buffer and
reduction. This is an SM80 finding to re-evaluate on H100, not permission
to assume one format or scheduling policy will win on the new GPU.

Relevant source entry points: `python/sparsene/op_gen/backend_registry.py`,
`python/sparsene/op_gen/strategy_agent.py`,
`python/sparsene/op_gen/format_design_agent.py`,
`python/test/run_joint_format_strategy_feedback.py`,
`python/test/run_v7_strategy_feedback.py`, and
`examples/src_fp32/acc/testbed/custom_format_e2e.cu`.

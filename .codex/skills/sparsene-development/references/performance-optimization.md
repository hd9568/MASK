# Sparse SpMM Performance Optimization

## Contents

- Evidence from the current project
- Kernel-work decomposition
- Cross-layer static optimization
- Format-Agent decision rules
- Strategy-Agent decision rules
- Feedback-loop and reporting rules

## Evidence from the current project

The durable lesson is that performance comes from coordinating representation,
lowering, physical layout, data movement, pipeline, and task mapping.  Neither a
smaller format nor a larger search space is sufficient by itself.

- Sparsene 2.0's optimized fixed SR-BCRS route reached a historical 26-matrix
  geometric mean of 1.9791x over cuSPARSE.  Its reusable contribution is
  semantic-preserving cross-layer specialization: a dense-leaf format selects
  a matching backend, aligned/vectorized A/B staging, canonical ldmatrix/MMA
  layouts, a proven single-warp synchronization contract, and heavy-tail task
  mapping.  It is not evidence that dense SR-BCRS is universally best.
- The first fixed-format two-Agent campaign reached 1.8472x geometric mean,
  beating Sparsene 2.0 on 11/26 matrices but losing overall.  This proves that
  useful per-matrix wins coexist with costly format mistakes; an Agent result
  must retain a fixed baseline/incumbent rather than replacing it by policy.
- In the comparable 18-matrix Strategy-Agent subset, best-of-0..5 reduced
  per-matrix time by 5.27% on average but only 0.10% at the median; six matrices
  exceeded the 1.5% noise threshold and averaged 15.66% improvement.  Strategy
  search is mainly valuable for outliers, not as a guaranteed uniform gain.
- ASIC_680k showed why proxy metrics are dangerous.  Capping registers raised
  occupancy but did not improve time; changing tile_b from 64 to 32 doubled
  the N grid and hurt reuse; multi-bind exploded tasks/grid.  Coarsening the
  already-effective strict load-balance group from 64 to 128 reduced overhead
  and gave a stable 3.74% improvement.
- The current cross-run correctness-gated incumbent portfolio combines 16
  winners from the fixed-format run and 10 from the joint Format/Strategy run,
  reaching 1.9652x geometric mean and 24/26 wins over cuSPARSE.  Portfolio
  selection and regression protection are part of the optimizer, not merely
  result postprocessing.

These are historical measurements under the recorded A100/A800 SM80
comparability assumption.  Use them as engineering evidence and hypotheses;
new paper claims still require same-machine reproduction.

## Kernel-work decomposition

Before changing code or asking an Agent for a proposal, build a physical-work
ledger for the timed kernel:

`T = T_metadata + T_A/B_move + T_restore + T_MMA + T_store/reduce + T_imbalance/schedule + T_sync`

Track at least:

1. active logical tiles and actual physical MMA tiles;
2. physical padding in M/K/N and zero work executed by MMA/store;
3. unique B rows, B gather count/bytes, locality, and reuse across N tiles;
4. A payload bytes plus mask/offset/rank/restore work;
5. metadata loads and indirect-address dependencies;
6. row-window/task distribution, maximum/p95 work, segments, task groups, grid,
   atomic boundaries, and partial-C reductions;
7. shared memory, registers, pipeline buffers, barriers, and spills;
8. output stores and any timed inverse/reduction work.

Do not substitute global density, storage bytes, occupancy, or one NCU
percentage for this ledger.  A compressed payload can still execute the same
full physical MMA and B gathers.  Higher occupancy can lose when it destroys
reuse or introduces spills and scheduling work.

## Cross-layer static optimization

Treat optimization as a semantic contract spanning:

`Format -> BackendPlan -> RTS/IR -> semantic fusion -> CUDA layout -> host packing -> launch/task mapping`

The general Sparsene 2.0 pattern is **semantic-preserving cross-layer
specialization**:

- match a backend from hierarchy/axis/atomic/reorder semantics, never a format
  name or dataset name;
- carry logical and physical tile sizes separately through every layer;
- replace generic per-element work with a legal format-aware tile dataflow;
- align host packing with vectorized cp.async, shared-memory swizzle, ldmatrix,
  K-fragment iteration, MMA, and C-store layout;
- narrow synchronization only under an explicit launch ABI (for example, a
  proven 32-thread block), never by textual barrier replacement;
- separate the stable MMA body from distribution-aware row/strict/multi-bind
  task mapping so heavy tails can be addressed without rebuilding arithmetic;
- include padding, split launches, atomics, and partial-C reduction in timing;
- repair the generator/ABI when a generated artifact is wrong; a local `.inc`
  patch is not a reusable optimization.

Check correctness first.  Historical K-fragment, M=32 store, misaligned copy,
grid mapping, and long-window scheduling bugs demonstrated that an apparently
fast path can be incomplete or can benchmark a different physical operation.

## Format-Agent decision rules

The Format Agent owns representation-level choices: hierarchy, axes, grouping,
atomic leaf, reorder, and logical tile parameters.  For every proposal require
the causal chain:

`exact matrix feature -> representation mechanism -> physical-work change -> expected timed-kernel effect -> falsifying measurement`

Use a kernel-work ledger to compare at least three complete formats.  In
particular:

- compare physical tile/MMA count, padding, B gathers/reuse, A payload,
  metadata/restore, task-distribution tail, and timed reduction/store work;
- distinguish storage compression from computation reduction;
- compare dense and MCO siblings for the same hierarchy when close: current
  MCO saves A payload but does not skip B gather or full MMA and adds restore;
- account for logical-to-physical alignment, such as logical M=8 padded to the
  SM80 m16 physical tile;
- use row/column skew, overlap, locality, tile occupancy, and long-tail features
  together; global density alone is insufficient;
- treat candidate statistics as exact only for the hierarchy that produced
  them and as proxies otherwise;
- regard two DSLs with the same backend, packing mode, physical tile, reorder,
  and atomic behavior as performance-equivalent until a real physical-work
  difference is identified;
- use the incoming incumbent as a performance floor and calibration record.
  Challenge it with one concrete representation mechanism; do not merely
  rename or syntactically deepen it;
- keep research freedom separate from executable capability.  Unsupported
  designs are valuable research artifacts but cannot enter a performance
  winner table until their host/backend path passes full correctness.

Raw NCU counters stay in the inner fixed-format Strategy loop.  The outer
Format loop should consume only representation-level aggregates and measured
best/safe-baseline performance.

## Strategy-Agent decision rules

The Strategy Agent owns active knobs for one fixed executable format.  Before
emitting a patch:

1. verify the best-correct base, noise threshold, normalized signatures, and
   which fields are active for this backend;
2. inspect effective execution (`packed` policy, segment, tasks, groups, grid),
   not only the requested values; `auto` may already be `strict_lb`;
3. classify the dominant hypothesis as task/grid imbalance, memory/feed,
   register/shared-resource pressure, pipeline/synchronization, or
   representation-bound physical work;
4. choose one primary hypothesis and normally one to three effective field
   changes; state what observation would disprove it;
5. predict both the intended benefit and the likely counter-cost (for example,
   fewer tasks versus coarser imbalance, or higher occupancy versus spills);
6. prefer an untested effective value and avoid normalized duplicates,
   inapplicable COO fields on dense/MCO paths, or source-only changes that do
   not alter binary/runtime behavior;
7. allow an empty patch when the best is stable, all relevant axes are covered,
   or the bottleneck is representation-bound.  Random diversity after
   saturation wastes compilation, GPU, and tokens.

Use NCU to test a hypothesis, not to rank candidates.  Normal repeated kernel
time remains the objective.  Treat throughput names with their units, combine
registers with spill/occupancy/duration, and combine grid/tasks with matrix
tail evidence.  Do not assume cp.async, smaller tiles, higher occupancy,
strict_lb, or multi-bind is universally beneficial.

## Feedback-loop and reporting rules

- Preserve deterministic iteration 0 and the cross-run incumbent.  Select the
  fastest compatible full-output-correct candidate across both.
- Use warmup=3, repeat=32, three independent measurements, and median ranking
  for the current N=512 FP32 protocol.  Changes below 1.5% are inconclusive
  unless repeated evidence supports them.
- Keep GPU performance runs serial.  Parallelize CPU/model/code generation
  only where it cannot overlap the final GPU benchmark.
- Record requested, normalized, backend-adjusted, generated, compiled, and
  effective runtime strategies separately.
- Hash the format, generated include, binary inputs, matrix, and source
  summaries.  Isolate or invalidate caches when any semantic identity changes.
- Keep failed, incorrect, unsupported, slower, and duplicate candidates as
  negative evidence, but never allow them to replace the best correct entry.
- Report distributions: wins, losses, median, geometric mean, noise-qualified
  gains, and regression tails.  An average driven by a few matrices is not a
  general improvement.
- Separate generated-current, hand-tuned reference, historical Sparsene,
  Sparsene 2.0, and paper values.  Do not mix absolute milliseconds across
  GPUs/protocols; normalized cross-GPU comparisons must state the assumption.

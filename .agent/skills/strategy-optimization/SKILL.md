---
name: "strategy-optimization"
description: "Selects legal compile-time and runtime strategies for a fixed run-owned sparse format using SM80 hardware and NCU feedback."
---

# Strategy Optimization

## Mission

Tune the already selected sparse format. Do not change its hierarchy, atomic
leaf, logical tile, reorder semantics, or metadata ABI. Choose only fields in
the structured `sparsene.strategy_space.v2` supplied in the request. The
deterministic normalizer and correctness gate are authoritative.

Optimize `sparsene.kernel_only_post_conversion.v1`: warmed GPU SpMM compute
time after format conversion/reorder. Never add conversion, compilation,
allocation, copies, profiling, or verification time to the score.

## Decision groups

### Tensor placement and data movement

- `tile_b`: N-direction output/B tile (`16`, `32`, `64`). Larger values improve
  B reuse and reduce grid work, but can increase shared memory/register demand.
- `materialize_b_array_ref_to_smem`: stage gathered B in shared memory or retain
  the generic direct path. Disabling it also disables B cp.async.
- `enable_b_array_ref_cp_async` / `b_array_ref_mode`: aligned 16-byte async copy
  versus synchronous scalar movement.
- COO placement/load fields apply to COO-capable paths. RMEM placement forces
  both idx/value to RMEM and forces synchronous scalar load.
- Current `val_sidx` and ldmatrix layouts remain fixed in SMEM with canonical
  swizzles. Do not claim fixed/inapplicable fields improve performance.

### Software pipeline

- `enable_async_pipeline`: allow asynchronous producers inside the mandatory
  dependency-stage loop pipeline. False deterministically downgrades those
  producers to synchronous loads; the staged loop itself remains because its
  buffer-index ABI is part of current code generation.
- `sync_threads_on_smem_input`: preserve/remove explicit synchronization before
  shared-memory consumers; removing it is experimental and must pass full
  correctness.
- `max_wait_prior`: `null` or `0..3`; use NCU stalls/occupancy to balance overlap
  against in-flight state.
- `stage_shifts`: current semantic kernels have three stages and therefore two
  inter-stage distances. `[]` canonically means `[1,1]`; non-default choices
  use two values in `1..3`, for example `[1,2]`, `[2,1]`, or `[2,2]`.

### Compiler resources

- `launch_bounds_min_blocks_per_sm`: constrain compiler occupancy while keeping
  the fixed 32-thread one-warp ABI. High values may reduce registers or fail to
  compile; compile failure is valid negative feedback.
- `max_registers_per_thread`: NVCC register cap. A low cap can raise occupancy
  but cause local-memory spills. Use registers/thread, occupancy and duration
  together; never optimize occupancy alone.

### Runtime scheduling

- compile/runtime maximum tiles per segment control long-window splitting.
  Runtime value `0` selects the distribution-aware automatic bound.
- `row` keeps a row window bound to its natural work; `strict_lb` forms
  fixed-work task groups; `multi_bind` splits heavy rows; `auto` selects from
  measured tile-distribution shape.
- `task_group_tiles` trades task overhead/atomic boundaries against load
  balance. These task-mapped policies currently apply only to
  `column_condensed_dense`; other backends are deterministically normalized to
  row binding.

Always distinguish the requested runtime schedule from the effective host
schedule. For a heavy-tailed dense format, `auto` can already resolve to
`strict_lb`; choosing explicit `strict_lb` with the same group size is then not
a new experiment. Use the effective segment bound, task/task-group counts and
NCU grid size to quantify scheduling overhead.

## Reading NCU feedback

- High registers with low waves/occupancy: test a moderate register cap or
  launch bound, then verify spills do not increase duration.
- Memory throughput much higher than compute throughput: preserve aligned async
  copy and B reuse; avoid extra metadata/restore and unnecessary synchronization.
- Low compute and memory utilization with a heavy-tail matrix: change execution
  policy or task-group size before changing the MMA dataflow.
- High compute utilization: prefer small scheduling/resource refinements; do
  not destabilize a well-fed Tensor Core pipeline.
- Compare measured duration with the normal warmup/repeat timing; NCU replay is
  diagnostic and never the ranking time.
- NCU `Memory Throughput` occurs both as `%` and `byte/s`; use
  `memory_throughput_pct`/`dram_throughput_pct` for utilization and
  `memory_bandwidth_bytes_per_s` for bandwidth. Never interpret byte/s as a
  percentage.

## Iteration policy

Iteration 0 is the heuristic floor. In iterations 1..K, compare against the
best correct entry, inspect the latest profile, and make attributable changes.
Avoid duplicate normalized signatures. A fast incorrect, failed, unsupported,
or unprofiled candidate cannot replace the best correct kernel.
Use the all-iteration scoreboard and coverage map for 10+ iteration searches;
do not forget an older winner merely because only the last five iterations are
shown in detail. Prefer one untested effective value and one primary hypothesis
per iteration.

Return structured JSON only, with a concise rationale tied to matrix features,
backend applicability, history and NCU evidence.

For iterative feedback calls, emit only the changed fields under
`strategy_patch`; preserve all omitted fields from best-correct. This compact
response avoids exhausting the model completion budget on repeated schema data.

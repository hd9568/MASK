# Research Context

## Contents

- Paper thesis
- Hierarchical format model
- Automatic pipeline construction
- Format-aware load balancing
- Mapping the paper to SparseKG
- Research claims and engineering evidence

## Paper thesis

The SC26 paper presents Sparsene as a format-driven automatic SpMM optimization framework. Its central claim is that a hierarchical sparse-format specification can serve as the semantic source from which kernel structure, fine-grained operations, pipeline schedules, and load-balancing behavior are derived, avoiding a dedicated hand-written kernel for every format.

The paper targets the mismatch between iteration-centric sparse compiler abstractions and GPU execution. Iteration descriptions explain traversal but do not directly encode block/warp decomposition, Tensor Core tile alignment, memory placement, or asynchronous pipeline structure.

## Hierarchical format model

The paper separates two ideas:

- Declarative hierarchy: `Format`, axes, nested child formats, and atomic leaves.
- Dynamic transformations: split, swap, sparsify, spawn, and atomicize operations that lower a flat tensor into a hardware-oriented physical layout.

ME-TCF is the representative hierarchy: row windows, condensed/sparse columns, Tensor Core blocks, and an atomic encoding. Other families change the atomic encoding or hierarchy while retaining the multi-level principle.

The Python implementation represents this with recursive `Format`/`AtomicFormat` objects and derives a transformation sequence through `derive_rts()`.

## Automatic pipeline construction

The frontend lowers format and computation semantics into fine-grained operations and dependencies. The paper calls an executable unit a `pTask` and the dependency DAG a `pGraph`.

A pipeline plan contains:

- a partition of tasks into stages;
- in-stage/order constraints;
- shift offsets controlling cross-iteration overlap;
- enough buffers for the maximum producer-consumer distance.

The paper divides execution into fill, full-stage, remainder, and drain/empty phases.

Dependency-aware sandwich profiling measures:

- issue time `Ti`: scheduler time before the next task can issue;
- execution time `Te`: operation completion latency after issue.

The hardware-aware simulator serializes issue phases, overlaps asynchronous execution, and delays consumers until their producers complete. Because exhaustive plan compilation is expensive, the paper uses simulator-guided structural clustering/pruning followed by a measured exploration/exploitation search.

## Format-aware load balancing

The paper compares:

- normal binding: one thread block per row window;
- multi-binding: split a heavy row window across blocks and use atomic aggregation where needed;
- strict load balance: give blocks fixed work quotas that may span row windows.

The intended design keeps the mainloop and task graph stable so that load-balance selection does not require rebuilding the pipeline plan.

## Mapping the paper to SparseKG

SparseKG adds intelligence at decision boundaries while retaining deterministic lowering:

- The format agent uses matrix/hardware features to propose a general research
  hierarchy, then deterministic code classifies current implementation support
  separately.
- The strategy agent chooses exposed placement, async-copy, tile, and pipeline knobs.
- The compiler remains responsible for generating legal IR and CUDA.
- Correctness, performance, and profile measurements return as structured feedback.

This division is important: use the LLM to search constrained design spaces and interpret evidence, not as an unchecked CUDA source generator.

## Research claims and engineering evidence

The paper reports broad speedups and simulator/search quality across A100, H100, and RTX 4090. Treat those as paper results. For code changes in this workspace, make narrower claims backed by fresh logs from the exact generated kernel, dataset, `N`, precision, GPU, CUDA toolchain, warmup/repeat counts, and cache policy.

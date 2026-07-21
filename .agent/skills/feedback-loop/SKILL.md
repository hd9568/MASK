---
name: "feedback-loop"
description: "Explains the compile-run-profile optimization loop. Invoke when an agent must improve a generated sparse kernel using correctness and performance feedback."
---

# Feedback Loop

## Mission

Use this skill when running iterative autotuning over generated kernels.

Use performance contract `sparsene.kernel_only_post_conversion.v1`: feedback,
scores, and best-entry selection use warmed repeated GPU SpMM compute-kernel
time after format conversion/reorder. Do not add or penalize offline format
conversion, metadata construction, reorder/inverse-reorder, allocation/copies,
compilation, or verification latency. Keep those as separate engineering
records. All work actually executed inside the timed kernel remains included.

## Canonical Loop

1. Extract matrix features.
2. Choose or validate sparse format parameters.
3. Generate kernel code with a candidate strategy.
4. Compile the generated kernel.
5. Run correctness and timing evaluation.
6. Profile with NCU when available.
7. Feed structured results back into the next strategy decision.

## What Feedback Should Influence

- correctness failures and error bias
- kernel time vs cuSPARSE
- occupancy and waves per SM
- memory vs compute throughput
- register pressure
- repeated strategy signatures and stagnation

## Iteration Policy

- Start from a safe heuristic baseline.
- Prefer incremental changes from the best correct iteration.
- If multiple rounds stagnate, force more diversity across at least two knobs.
- Never prioritize speed over correctness regressions.

## Logging Requirements

Each iteration should persist:

- request payload
- prompts and skills used
- structured agent response
- token usage
- generated strategy JSON
- compile log
- run log
- aggregate metrics
- profile summary

## Best-Entry Selection

Choose the best iteration from correct runs only. Rank by the kernel-only ratio
vs cuSPARSE, then by failure penalties and absolute kernel-only time.

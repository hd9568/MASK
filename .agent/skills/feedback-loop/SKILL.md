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
2. Load and validate the fixed run-owned sparse format and backend plan.
3. Generate kernel code with a candidate strategy.
4. Compile the generated kernel.
5. Run independent correctness/timing measurements and rank by their median.
6. Profile one generated-kernel launch with NCU when correctness passes.
7. Feed structured results back into the next strategy decision.

## What Feedback Should Influence

- correctness failures and error bias
- kernel time vs cuSPARSE
- occupancy and waves per SM
- memory vs compute throughput
- register pressure
- repeated strategy signatures and stagnation
- compile-time launch/register limits and spill/occupancy trade-offs
- row-window heavy tail, task mapping, and atomic-boundary overhead
- requested scheduling knobs versus the effective policy, segment size, task
  count, and grid recorded by the host result
- explored/untested field values and whether the generated source or only the
  compile/runtime behavior actually changed

## Iteration Policy

- Start from a safe heuristic baseline.
- Load a compatible cross-run incumbent when available and preserve it as a
  correctness-gated performance floor; a slower new campaign must not erase it.
- Treat that baseline as iteration 0 and run the requested number of Agent
  iterations after it.
- Prefer incremental changes from the best correct iteration.
- Treat a sub-noise-threshold change as inconclusive; do not declare a register
  cap or occupancy change beneficial/harmful from one close timing sample.
- `auto` is a request, not an effective policy. Inspect `e2e_result.packed`
  before proposing `strict_lb` or `multi_bind` as a supposedly new strategy.
- If multiple rounds stagnate, first determine whether a different active
  hypothesis remains untested. Explore targeted diversity only when it changes
  effective execution; never require a multi-knob change merely for novelty.
- If the remaining bottleneck is representation-bound or the active strategy
  axes are exhausted, stop or return an empty patch instead of compiling random
  variants.
- Never prioritize speed over correctness regressions.
- In iterative rounds, return a compact `strategy_patch` containing only fields
  changed from the best-correct strategy. The local driver merges and
  normalizes it; do not repeat the full 21-field strategy JSON.

## Logging Requirements

Each iteration should persist:

- request payload
- prompts and skills used
- structured agent response
- token usage
- generated strategy JSON
- compile log
- run log
- every raw measurement plus median/spread aggregate metrics
- profile summary
- exact format/operator hashes, backend plan, runtime schedule, and NCU command

The patch response must remain structured JSON and should include one primary
hypothesis plus a short rationale. A missing field means “preserve the base
strategy,” not “reset to a default.”

## Best-Entry Selection

Choose the best iteration from correct runs only. For one fixed dataset and
format, rank by warmed generated-kernel milliseconds; report cuSPARSE speedup
separately. Across runs, compare the fixed-format winner with the compatible
registry incumbent and replace only on a strict measured improvement. NCU
replay duration is diagnostic and must never replace the normal warmup/repeat
timing.

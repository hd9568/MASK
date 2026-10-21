---
name: sparsekg-literature-guided-search
description: Use literature-grounded reasoning to design, critique, or evaluate SparseKG's iterative format/strategy search, including candidate selection, budget allocation, feedback interpretation, memory, stopping, and fair ablations. Use for research planning or implementation of the search controller; not for ordinary CUDA syntax fixes or unrelated paper summaries.
---

# Literature-guided SparseKG search

When available locally, use the [per-paper survey](../../../paper/reference/Agent生成GPU算子与稀疏编译逐篇综述_20261021.md) and [paper index](../../../paper/reference/近三年Agent生成GPU算子与稀疏编译论文索引_20261021.md) for deeper evidence. The slim remote branch intentionally omits `paper/`; this skill remains usable without those files. The survey's `方法/摘要/入口` labels are evidence levels, not quality rankings. When a specific paper claim matters, verify the original current paper before quoting a number, algorithm detail, or novelty claim.

## Model the actual search object

Represent a trial as `(matrix features, hardware, format semantics, backend capability, normalized active strategy, generated code identity, validation, kernel timing, optional profile)`. Keep these layers distinct:

1. Format semantics/legal DSL.
2. Backend lowering and host support.
3. Candidate implementation correctness.
4. Kernel-only performance and uncertainty.
5. Search cost and diagnostic profile.

A failed code generation does not prove a format idea is bad; an unsupported backend does not make a legal research format invalid. A correct but slow implementation is not proof of a representation's performance ceiling. A profile collection failure is missing evidence, not a zero metric. These separations are motivated especially by K-Search and AdaExplore.

## Choose experiments, not just iteration counts

- Preserve a correctness-gated heuristic baseline and the best measured candidate. Track parent/action/result so one can attribute improvements.
- Before allocating additional strategy rounds, compare formats' **physical work**: MMA tile count and padding, B gathers/reuse, A/metadata bytes, restore/reduction, and long-tail task balance. Compare semantically distinct candidates, not merely differently named DSLs.
- Allocate some budget to plausible underexplored format families and some to the current best; consider bandit/UCB or successive halving only after checking assumptions and noise. KernelBand's cluster/bandit policy is inspiration, **not** proof that a format-level bandit will work.
- Use local strategy patches when the current format has headroom and the bottleneck maps to an active strategy field. Use a structural format redesign when repeated effective strategy changes leave physical tile/padding/metadata work unchanged. Switch only with a falsifiable hypothesis.
- De-duplicate by canonical format/strategy/codegen semantics and effective host execution; a changed JSON field that normalizes away is not a new experiment.
- Maintain more than one promising family when evidence justifies it. Search trees and diversity archives can preserve non-monotonic paths; they must not rank incorrect kernels.

## Turn feedback into causal evidence

For each attempt, record: predicted bottleneck, one to three effective changes, expected benefit and counter-cost, correctness/latency result, measurement spread, and whether the emitted kernel actually changed. Classify failures as format quality, backend capability, generation/compile, numerical, timing/profiling infrastructure, or measured regression. Feed only relevant compact history to the Agent.

Use NCU on representative candidates when a decision requires it; translate raw metrics into a bottleneck hypothesis and a small diagnostic experiment. Never infer speedup from an NCU percentage or include NCU timing in the score. CudaForge, KernelPro, PRAGMA, and Feedback-to-Plan motivate this boundary. The physical-work ledger is essential for irregular SpMM; generic roofline classification alone is insufficient.

Promote cross-dataset memory only from auditable contrasts: same target hardware/precision/objective, a successful or failed transformation, active normalized fields, observed physical work/profile, and numerical validation. Distinguish transferable constraint knowledge from performance heuristics; transfer claims need held-out datasets or models. AdaExplore, AccelOpt, KernelBlaster, and daVinci-kernel motivate this distinction.

## Evaluate a smarter controller fairly

Keep the project objective `sparsene.kernel_only_post_conversion.v1` unchanged: rank only correct candidates by warmed post-conversion GPU SpMM compute-kernel time. Report conversion/storage/compile/API/NCU cost separately as search-efficiency measures. Use the same hardware, datasets, `N`, precision, candidate correctness tests, and total budget when comparing against fixed `F×(S+1)`, random allocation, and greedy best-only search.

Report **anytime best-correct curves** plus per-round validity, compile/correctness rate, unique candidate count, token/GPU time, and uncertainty for small latency gaps. Ablate candidate selection, format-vs-strategy switch, memory, and profile feedback independently. Freeze candidate history or use controlled feedback injection when claiming that NCU changed a decision; a final best-of-more result alone is not evidence of intelligent iteration. KernelBenchX warns that more refinement can improve correctness without improving performance.

## Research boundary

The cited works already cover bandit search, tree search, multi-agent strategy representations, profile-guided repair, and programmable sparse formats. Do not present any of those alone as SparseKG's invention. A defensible new contribution must be specific to the **coupled sparse representation + deterministic compiler + hardware-aware strategy** setting, implemented in the search controller and supported by ablation on correct generated kernels. Treat the survey's suggestions as hypotheses, not as features already present in the repository.

For implementation work, also apply `sparsene-development` for repository contracts, validation, and development-log requirements. This skill does not authorize GPU runs, paid Agent calls, or source edits beyond the user's request.

---
name: sparsene-development
description: Develop, debug, validate, and optimize SparseKG, a Matrix- and Architecture-Aware Agentic Compiler for SpMM Kernel Generation built on the Sparsene compiler foundation, including hierarchical sparse-format design, Format-to-RTS-to-Computent-to-OPIR-to-NVIR-to-CUDA lowering, format and strategy agents, compile-run-profile feedback loops, atomic-format extensions, generated-kernel correctness, and GPU performance comparison. Use when working anywhere in this repository on sparse formats, agent prompts or schemas, code generation, CUDA kernels, host bridges, benchmarks, NCU analysis, or paper-to-code reasoning.
---

# Sparsene Development

## Establish the current state

1. Work from the repository root.
2. Before running project Python, Agent, generation, or end-to-end commands,
   activate the project environment with
   `source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate` in the same shell.
3. Inspect `git status --short` before editing. Preserve unrelated tracked and untracked work.
4. Use `README.md` for orientation, then read
   [project-map.md](references/project-map.md). Do not copy command-line flags
   from the README without checking the current CLI.
5. Treat source code and current CLI `--help` output as authoritative. Treat
   dated files under `docs/`, `.trae/skills`, and copied onboarding material as
   experiment records that may describe an older implementation.
6. Read only the references relevant to the task:
   - Paper concepts or compiler architecture: [research-context.md](references/research-context.md)
   - Formats, lowering, atomic-format work, or codegen: [format-codegen.md](references/format-codegen.md)
   - Format agent, strategy agent, prompts, schemas, or feedback loops: [agent-loop.md](references/agent-loop.md)
   - Correctness, compilation, benchmarking, or profiling: [validation.md](references/validation.md)
   - Cross-layer performance optimization, Format/Strategy decisions, or
     regression interpretation:
     [performance-optimization.md](references/performance-optimization.md)
   - Environments, datasets, caches, and current entry points:
     [operations.md](references/operations.md)
   - Conflicting historical documents or old performance claims:
     [knowledge-provenance.md](references/knowledge-provenance.md)

## Trace changes end to end

Before changing a format or kernel path, identify every affected contract:

`Format -> RTS -> Computent -> OPIR -> lowering passes -> NVIR -> semantic fusion/pipeline -> .inc -> host bridge -> preprocessing/cache -> correctness/performance logs`

Do not assume that adding a Python format class makes a format executable. Verify the metadata arrays, global-memory interface, OPIR/NVIR operations, generated template parameters, host conversion, runtime dispatch, and checkpoint identity.

For agent changes, trace this contract:

`matrix features + hardware + supported search space -> structured decision -> normalization/guardrails -> deterministic codegen -> build -> correctness -> timing/profile -> next feedback context`

Keep LLM output constrained to an explicit schema. Put correctness constraints in deterministic validation and normalization code, not only in the prompt.

## Respect the two skill systems

- `.codex/skills/sparsene-development/` is this Codex development skill.
- `.agent/skills/*/SKILL.md` is loaded by `python/sparsene/agent/skills.py` and injected into the project's runtime LLM agents.

Changing one system does not update the other. Modify `.agent/skills` only when the requested change concerns runtime agent behavior.

## Apply project guardrails

- Use the fixed performance objective
  `sparsene.kernel_only_post_conversion.v1`: rank correct formats and
  strategies by warmed GPU SpMM compute-kernel time after all format
  conversion, metadata construction, padding, and row/column reordering.
  Conversion/reorder/inverse-permutation latency, allocation/copy, compilation,
  checkpoint I/O, and correctness verification are excluded from this primary
  metric. Record their semantics/storage/implementation separately, but never
  add them to or use them as a penalty in the kernel-only ranking. All work
  actually executed inside the timed compute kernel remains included.
- Distinguish generated kernels from hand-tuned reference kernels. Never report a patched or reference `.inc` result as generated-codegen performance.
- Require correctness before ranking performance. A kernel that compiles or reports a plausible time is not necessarily correct.
- Start tuning from the heuristic baseline and prefer small, attributable strategy changes.
- Keep host and generated-kernel layout contracts synchronized, especially `BLK_M`, `BLK_K`, swizzles, tensor shapes, and metadata order.
- Disable or isolate checkpoint and binary caches while changing formats, tile sizes, preprocessing, or codegen. Confirm cache keys cover all semantics being varied.
- Treat grid-partitioned and mixed-format files as experimental until imports, preprocessing, host dispatch, accumulation, and numerical tests prove the full path.
- Never print, copy into documentation, or commit API keys. Refer to environment-variable names or redacted placeholders only.
- Some legacy `.trae` material contains literal credentials. Never copy or use
  them; resolve credentials through protected environment/model configuration.
- Do not infer GPU success from code inspection. If the required CUDA/GPU environment is unavailable, state exactly which validation layer remains unexecuted.

## Optimize physical work across layers

Before tuning, decompose timed work into metadata, A/B movement, restore,
physical MMA/padding, store/reduction, synchronization, and task/grid imbalance.
Trace the proposed benefit from an exact matrix feature or measured profile to
a concrete change in this ledger.  Storage bytes, occupancy, global density,
or one NCU percentage are evidence, not objectives.

Preserve the general Sparsene 2.0 lesson: performance is often unlocked by
semantic-preserving cross-layer specialization that keeps Format semantics,
BackendPlan, IR fusion, CUDA layouts, host packing, vectorized movement,
single-/multi-warp synchronization contracts, and runtime task mapping
consistent.  Do not misstate this as a universal SR-BCRS or dense-format win.

For Format-Agent work, compare complete formats by physical tile count,
padding, B gathers/reuse, A payload, metadata/restore, long-tail distribution,
and timed reduction/store.  Treat physically equivalent DSLs as equivalent
until a real work change is identified, and challenge the current incumbent
with a falsifiable representation mechanism.

For Strategy-Agent work, inspect normalized active fields and effective host
execution before patching.  Classify one bottleneck, change normally one to
three effective fields, predict benefit and counter-cost, avoid duplicates and
inactive axes, and allow an empty patch when the format is the limiting layer
or measurements are saturated.  Rank only by repeated correct kernel time;
use NCU to test the hypothesis, never as the score.

Read [performance-optimization.md](references/performance-optimization.md) for
the evidence, detailed work ledger, Agent role boundary, and reporting rules.

## Use a proportional validation ladder

Run the cheapest relevant checks first, then advance only after they pass:

1. Python syntax/import and focused unit tests.
2. Format/RTS/IR structural inspection.
3. Deterministic kernel generation and source inspection.
4. NVCC compilation with the intended architecture and host bridge.
5. Small synthetic correctness cases, including boundary and low-density cases.
6. Representative `.mtx` correctness against cuSPARSE.
7. Warmed repeated timing with cache state recorded.
8. NCU profiling when a performance hypothesis requires it.
9. Multi-dataset regression before claiming generality.

Read [validation.md](references/validation.md) before running GPU commands or interpreting benchmark output.

## Finish with evidence

Report:

- files and contracts changed;
- validation commands actually run and their outcomes;
- generated/reference/baseline provenance;
- correctness and performance evidence separately;
- unresolved limitations, environment gaps, and the next highest-value check.

Do not promote historical numbers from `docs/` to current results without reproducing them or clearly labeling them as historical.

## Keep a persistent development record

For every material repository development task, create or update a focused
record under `docs/codex/` and append a concise entry to
`docs/codex/development_log.md`. Record the design rationale, contracts changed,
commands actually run, evidence, output locations, and unresolved limitations.
Do this before declaring the task complete so future Codex sessions do not rely
only on conversation history.

Matrix feature extraction uses the v3 contract documented in
`docs/codex/matrix_feature_extraction_v3.md`. The persistent feature store for
the 26 selected matrices is `/mnt/lustre-client/huoda/selected_mtx`, using
`<dataset>.features.v3.json` plus `features_manifest.csv`. Prefer these files
over historical per-run `matrix_features.json` when their manifest status is
`ok`; regenerate them after changing feature semantics or schema version.
The shared MatrixMarket reader must continue to support `real`, `integer`,
`pattern`, and `complex` coordinate inputs; include at least one non-`real`
matrix when validating extractor changes.

Format-agent development follows the current research-design contract documented in
`docs/codex/format_design_agent_v6.md`. The agent may freely propose valid
hierarchies, axes, atomics, parameters, and reordering based on matrix features;
current backend support must not restrict the proposal. Deterministic validation
must preserve the proposed DSL, reject invalid structure, and classify support
separately as frontend research, codegen-declared, standalone-validated, or
integrated e2e. Never silently normalize a novel proposal to ME-TCF and never
report a valid research design as executable without its full host/backend and
numerical-validation path.

Saved research formats can be exported offline with
`python/test/export_custom_formats.py` into
`python/sparsene/formats/custom`. Keep the public `formats/dsl.py` construction
layer Agent/API-independent. Select executable capability through
`op_gen/backend_registry.py` from hierarchy/axis/atomic/reorder/params
semantics, never from a dataset name or descriptive format name. The first
generalized backend is `tcf_mco_tensorcore_v1`; its audited first-three v6
results and limitations are in
`docs/codex/custom_format_operator_e2e_v6_first3.md`.
Every format-agent prompt, result artifact, and later strategy/performance phase
must carry `sparsene.kernel_only_post_conversion.v1`. The Agent must optimize
post-conversion/reorder GPU compute time, not end-to-end conversion latency.
The source-derived design vocabulary is maintained in
`python/sparsene/formats/design_catalog.py` and explained in
`python/sparsene/formats/README.md`; keep them synchronized with concrete
format declarations and the runtime `format-design` skill. Format-agent JSON
calls use DeepSeek JSON Output, unquoted integer levels, the restricted length
grammar, concrete integer params, and explicit `finish_reason=length`
rejection. The 26-dataset wrapper uses 32768 output tokens by default because
thinking and final JSON share the budget.

For coordinated multi-phase Agent experiments, use
`/home/huoda/sparsene/logs/e2e_runs/<run_id>` as the common run directory for
feature snapshots, format-agent conversations, strategy iterations, generated
kernels, builds, correctness, timing, and NCU. Every paid Agent invocation must
retain exact system/user content, exact response content, raw and parsed
payloads, API token usage, and an audit/status record. For the 26-dataset format
phase, use `python/test/run_format_design_batch.py` or the venv-activating
background wrapper `scripts/run_format_design_26_background.sh`; prefer
`--resume` over repeating already accepted paid results. Do not spend paid API
calls on prompt smoke tests when local schema/normalizer tests suffice. Default
to one attempt per dataset and inspect the recorded failure before authorizing
a retry.

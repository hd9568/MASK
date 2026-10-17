# Agent System and Feedback Loop

## Contents

- Runtime agent infrastructure
- Format agent
- Strategy agent
- Feedback loop
- Design rules
- Failure modes

## Runtime agent infrastructure

`python/sparsene/agent/` resolves model configuration, invokes OpenAI-compatible or Anthropic-compatible endpoints, parses structured responses, records token usage, and injects runtime skill documents.

Runtime skills are discovered only as `.agent/skills/*/SKILL.md`. They are separate from the Codex skill under `.codex/skills`.

Keep credentials in environment variables or a protected local configuration. Logs and summaries should record only whether a key was present, never the key value.

Credential resolution may fall back from an explicit argument to an
environment-variable name and then to a model-config entry. Treat all resolved
credentials as secrets regardless of their source. Never copy literal keys from
legacy `.trae` examples or expose them while inspecting configuration.

Each invocation through `invoke_json_agent()` persists the exact composed
messages and an audit trail. Preserve these files when changing Agent behavior:

- `request.json` and loaded runtime-skill identities;
- `system_prompt.txt`, `user_prompt.txt`, and `messages.json`;
- `response_text.txt` and `raw_response.json`;
- `parsed.json`, `response.json`, and `usage.json`;
- `audit.json`, including status, content digests, token source, and timing;
- parse/API error artifacts when a call fails.

Token totals must include failed and superseded attempts, not only the final
accepted decision.

## Format agent

`format_design_agent.py` receives matrix features and a hardware profile,
renders general DSL education plus design rules, requests JSON, validates the
proposal, and classifies current implementation capability.

Important current behavior:

- The current contract is `sparsene.format_design.v6` /
  `hierarchical_sparse_format_research_v2`. It uses provider JSON Output and a
  type-correct example; `level` is an unquoted integer and params are concrete
  positive integers.
- The fixed performance contract is
  `sparsene.kernel_only_post_conversion.v1`: compare formats by warmed GPU SpMM
  compute-kernel time after format conversion, metadata construction, and
  reorder. Those transformations and correctness-only inverse permutations
  are untimed and must not penalize ranking. Their semantics, storage, cache,
  and implementation requirements remain mandatory and are reported
  separately. Work executed inside the timed kernel always counts.
- Length is a logical bound/split factor using only M/K/N, declared params,
  parentheses, and `+ - * /`. Varlen cardinality belongs in metadata semantics,
  not prose or undeclared symbols in the length field.
- The prompt injects every declaration under `python/sparsene/formats` through
  `formats/design_catalog.py`, including HME and row-sparse/reorder caveats, as
  composable references rather than a closed menu.
- The agent may design one to six levels with df/dv/sf/sv axes, optional reorder,
  and COO/MCO/CSR/ELL/DIA/dense leaves. It may also propose a custom research
  leaf when it fully specifies storage arrays, metadata, logical mapping, and
  restore semantics; such leaves require backend extension.
- Structural normalization is forbidden: invalid proposals are reported and
  retried/rejected rather than replaced with ME-TCF.
- Exact dotted-path feature evidence is grounded against the compact v3 digest.
- Known profiles are context rather than a closed menu. Valid unmatched formats
  are classified `frontend_research` and retain implementation requirements.
- Invocation or JSON failure produces an explicit failed design, never a hidden
  ME-TCF fallback.
- `finish_reason=length` is an explicit truncated failure even if a partial
  object can be parsed. Raw responses and usage remain audited.
- The integrated strategy/codegen driver stops clearly when a valid proposal is
  not `integrated_e2e`; format-only batch design still accepts it.
- Format-only quality gates require a valid DSL, grounded feature evidence,
  complete alternatives/semantics/implementation requirements, and no fallback
  source. They do not require the design to be executable today.

When changing format-agent capability, update together:

- feature schema and semantics;
- prompt and output schema;
- response validator/canonicalizer;
- `build_format_from_response()`;
- capability registry and known-profile signatures;
- backend and host support when execution is being added (not merely design);
- fallback behavior and logs;
- end-to-end correctness cases.

## Strategy agent

`strategy_agent.py` constructs a request from hardware, format/tensor summaries, constraints, prior strategies, correctness, timing, and NCU summaries. Providers include heuristic, manual JSON, supported HTTP variants, and subprocess JSON.

The decision is normalized by `StrategyDecision.from_dict()`. Current examples of deterministic guardrails include:

- supported `tile_b` values only;
- canonical memory placement values;
- RMEM COO staging forces scalar synchronous loads;
- canonical swizzles are enforced in the strategy model for the default lowering;
- disabling the async pipeline downgrades async producers;
- wait depth and stage shifts are canonicalized.

After normalization, the end-to-end driver can apply additional tile-dependent overrides. Log raw response, normalized decision, and actual generated metadata separately when debugging agent effectiveness.

Current strategy decisions cover TILE_B, B/COO placement and async-copy modes,
alignment, pipeline enablement, synchronization, wait depth, and one canonical
stage shift. They do not make a general format executable and should not be
used to mask a format/host/backend mismatch.

## Feedback loop

The integrated loop follows:

1. extract features;
2. decide/build the format;
3. lower it to reusable IR artifacts;
4. generate and evaluate a heuristic strategy baseline;
5. build a feedback context from recent and best correct iterations;
6. request and normalize the next strategy;
7. generate the `.inc`;
8. compile, optionally apply a narrowly scoped fix agent, and recompile;
9. run numerical comparison and timing;
10. collect NCU metrics when available;
11. persist iteration artifacts;
12. select the lowest-score correct result.

The multi-dataset driver adds synthetic cases, diversity mutations, aggregate correctness/profile summaries, final dataset evaluation, and curve/report generation.

For coordinated experiments, place future strategy, codegen, build,
correctness, timing, and NCU artifacts under the same
`logs/e2e_runs/<run_id>/datasets/<dataset>/` tree established by the format
batch. Older feedback drivers default to separate historical directories, so
pass explicit roots when reproducible cross-phase provenance is required.

## Design rules

- Use a safe heuristic iteration 0 so agent proposals have a real baseline.
- Feed compact structured history rather than unbounded kernel source.
- Make one or a few attributable changes per round unless evidence supports a larger move.
- Detect duplicate normalized strategy signatures; raw textual diversity does not imply search diversity.
- Compare a candidate to the best correct generated baseline, not to an
  incorrect fast iteration or to a hand-tuned reference presented as generated.
- Penalize or exclude build failures and numerical errors. Never reward a fast incorrect kernel.
- Preserve enough provenance to reproduce every candidate: revision, matrix, features, format, strategy, generated source digest, compile flags, hardware, cache policy, and metrics.
- Separate decision quality from backend capability. Repeated failure across reasonable decisions may indicate a lowering or host bug, not a weak model.

## Failure modes

- Prompt says a structure is forbidden, but parser accepts it.
- Raw response differs from normalized/applied strategy.
- Changed strategy does not alter generated code.
- Stale binary/checkpoint makes two candidates appear identical.
- Correctness parser samples only printed errors while the executable truncates output.
- NCU is unavailable or profiles a different kernel name/launch.
- Agent repeats semantically equivalent variants such as canonical/default shift forms.
- Old runtime skills encode constraints that conflict with current source.
- Fix agent patches a generated artifact without fixing the generator, creating non-reproducible success.
- A paid retry is launched automatically even though the previous raw response
  and parse error have not been inspected.
- Results from incompatible format-design schema versions are resumed into one
  run and corrupt the meaning of the experiment.

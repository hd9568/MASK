# Sparsene Codex Skill Knowledge Consolidation — 2026-07-18

## Goal

Consolidate the correct, durable project knowledge in `docs/`, `.agent/`, and
`.trae/` into a legal Codex skill at
`.codex/skills/sparsene-development/`. The result must help future Codex
sessions reason from the current compiler and Agent contracts without treating
dated experiment notes as current truth.

## Source inventory and deduplication

The review covered:

- 26 files under `docs/`;
- 5 runtime Agent skills under `.agent/skills/*/SKILL.md`;
- 18 files under `.trae/skills`;
- the existing Codex skill and all of its references.

SHA256 comparison showed that 10 `.trae` files are byte-identical copies of
documents under `docs/`. The remaining eight are older indexes, runtime-skill
summaries, or earlier versions of ME-TCF/ref-vs-generated notes. Identical
files were read once while retaining the source relationship.

The material was grouped into:

1. project/compiler architecture;
2. hierarchical Format DSL and atomic-format extension;
3. feature, format-Agent, strategy-Agent, and feedback-loop contracts;
4. host preprocessing, caches, correctness, performance, and profiling;
5. historical performance/debugging evidence.

## Source verification

Dated statements were checked against current source and CLI behavior. The
main verification points were:

- atomic factories in `python/sparsene/format/format.py`;
- atomic RTS/Computent/OPIR routes and `SPARSE_FORMAT_REGISTRY`;
- v5 format capability profiles and validation behavior;
- current strategy schema and deterministic normalization;
- runtime Agent audit artifacts and token accounting;
- `end2end_pipeline.py` argument names, BLK_K heuristic, binary cache key, and
  integrated-format gate;
- `main.cu` checkpoint identity and full `VERIFICATION` summary;
- `scripts/sweep_tc_block_k.sh` provenance, correctness gating, best selection,
  and faster-than-cuSPARSE statistics;
- current `--help` output for five primary Python entry points.

This exposed several important corrections to legacy material:

- the current aggressive heuristic selects BLK_K=16 only when mean NNZ/row is
  positive and below 6, not below 16;
- checkpoint caching is enabled by the host when no disable variable is set,
  while current end-to-end/sweep drivers explicitly disable it;
- root README command examples contain older Agent flag names, so current CLI
  help is authoritative;
- printed `Error(...)` lines are capped, whereas `VERIFICATION errors/nonfinite`
  is the full correctness result;
- ME-TCF row-window column condensation/sorting is not a global matrix row or
  column reorder;
- an atomic factory or NVIR registry entry is not evidence of a complete host
  and numerically validated execution path;
- v5 general research DSL design and current integrated DTC execution are
  intentionally separate.

## Skill structure after consolidation

The legal skill keeps a concise routing and guardrail file in `SKILL.md` and
uses focused references:

- `project-map.md`: authoritative paths and current execution routes;
- `research-context.md`: paper concepts and their implementation mapping;
- `format-codegen.md`: DSL, lowering, preprocessing/reorder semantics, atomic
  capability, and extension checklist;
- `agent-loop.md`: runtime audit, v5 format design, strategy normalization, and
  feedback-loop rules;
- `validation.md`: generated/reference provenance, full correctness, ratio
  direction, caches, NCU, and the current K sweep;
- `operations.md`: required venv, raw/derived data paths, entry points, cache
  variables, and long-run discipline;
- `knowledge-provenance.md`: conflict precedence, surveyed legacy sources, and
  information deliberately excluded from current operational rules.

The main skill also retains the persistent project conventions established in
earlier work:

- activate `/mnt/lustre-client/huoda/venv/llm-sog`;
- store schema-v3 features in `/mnt/lustre-client/huoda/selected_mtx`;
- use `logs/e2e_runs/<run_id>` for coordinated Agent phases;
- record every material development task in `docs/codex`;
- preserve exact paid-Agent prompts, responses, raw/parsed payloads, token
  usage, and audit status;
- never silently rewrite a valid research format to ME-TCF.

## Information deliberately not promoted

The skill does not promote the following as current truth:

- literal credentials found in legacy material;
- dated performance tables, speedup counts, generated/reference gaps, or
  operation counts;
- historical no-GPU, compile-failure, or fixed numerical-failure states;
- old fixed-format prompt/fallback behavior;
- model-generated descriptive format names as proof of distinct execution;
- format support inferred from declaration alone.

These records remain useful as hypotheses and experiment provenance, but a
current claim requires fresh source/log evidence from the same configuration.

## Security observation

The review found literal credential material in a legacy `.trae` Agent guide
and a source-level default credential in the strategy-Agent module. No secret
was copied into the Codex skill or this record, and no credential was tested.
This consolidation did not change runtime authentication because the requested
scope was the Codex skill. A follow-up security cleanup should migrate all
runtime credentials to protected environment/model configuration and rotate any
credential that may have been exposed historically.

## Validation performed

The following non-GPU, non-API checks passed:

```text
python skill-creator/scripts/quick_validate.py .codex/skills/sparsene-development
  -> Skill is valid!

reference-link and agents/openai.yaml presence checks
  -> passed

end2end_pipeline.py --help
run_format_design_batch.py --help
run_strategy_feedback_multidataset.py --help
test_generate.py --help
generate_sparse_formats.py --help
  -> all imported and printed current CLI help in the required venv
```

The strategy feedback import emitted only a Matplotlib warning that the home
cache directory was unwritable and used a temporary directory. No repository
artifact, paid model call, kernel generation, compilation, or GPU benchmark was
needed for this knowledge-only change.

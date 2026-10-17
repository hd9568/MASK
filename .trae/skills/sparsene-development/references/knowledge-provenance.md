# Knowledge Provenance and Conflict Handling

## Source precedence

When two claims conflict, use this order:

1. currently executed source, current CLI, and focused tests;
2. fresh generated artifacts and logs from the same revision/configuration;
3. current `docs/codex/` contracts;
4. root README for architecture orientation;
5. dated `docs/`, `.agent`, and `.trae` notes as historical evidence.

Do not resolve a conflict by choosing the newest date alone. Verify that the
record used the same generated artifact, host bridge, matrix, N, tile shape,
precision, GPU, cache policy, and correctness criterion.

## Surveyed legacy material

The Codex skill was consolidated from all files under `docs/`, the five runtime
skills under `.agent/skills`, and all files under `.trae/skills` on 2026-07-18.
Ten `.trae` files were byte-identical copies of `docs`; the remaining `.trae`
files were older skills, indexes, or earlier versions of optimization notes.

Retain these stable conclusions:

- The format is the semantic root of the complete lowering chain.
- Atomic-format identity requires real metadata/payload/restore semantics, not
  merely a new class or name.
- Variable-length ranges, tensor layouts, swizzles, host metadata order, and
  cache identity are correctness contracts.
- Preserve structured decisions and deterministic validators between an LLM
  and codegen.
- Require compile and numerical correctness before performance ranking.
- Generated, patched, and hand-tuned reference kernels are different
  provenance classes.
- Pipeline topology and redundant dependencies can dominate performance even
  when DRAM traffic or register counts look similar.
- Failed optimization attempts are useful hypotheses only when their exact
  revision and validation evidence remain attached.

## Do not internalize as current truth

The following legacy content is deliberately excluded from operational rules:

- literal API keys or tokens embedded in old `.trae` examples;
- historical generated/reference/cuSPARSE timings and speedup counts;
- fixed lists of matrices said to fail numerically;
- old statements that generated code cannot compile or that no GPU is
  available;
- old adaptive thresholds such as mean NNZ/row below 16;
- old fixed-format prompts, hidden ME-TCF fallbacks, and claims that arbitrary
  model-generated names represented distinct executable formats;
- line counts, operation counts, or pipeline-stage counts from a historical
  `.inc` file;
- claims that declaring/registering an atomic type proves a complete route.

Reproduce any of these before using it in a current report.

## Current durable records

Use these focused records before their older predecessors:

| Concern | Preferred record |
|---|---|
| Matrix features | `docs/codex/matrix_feature_extraction_v3.md` |
| General research format design | `docs/codex/format_design_agent_v6.md` |
| Historical fixed ME-TCF batch | `docs/codex/format_design_agent_v4.md` and `deepseek_format_design_26_run.md` |
| Atomic extension checklist | `docs/atomic_format_extension_playbook.md` |
| Compiler/format inventory | Current source, then `docs/atomic_format_analysis.md` |
| Historical codegen debugging | `docs/generated_kernel_validation.md`, then later backport notes |
| Performance experiments | Original fresh run logs/CSV, not prose summaries alone |

Documents named `PROJECT_ONBOARDING`, `latest_generated`, `ref_vs_generate`,
dated `sparsene_vs_cusparse*`, and optimization summaries are valuable for
reconstructing hypotheses and provenance, but are not present-state contracts.

## Security boundary

Never repeat a credential discovered while reading legacy material. Refer only
to an environment-variable name or to a redacted model configuration. If a
legacy credential may still be live, report that security issue separately and
recommend revocation/rotation; do not test the credential.

# MASK README rename and project positioning

Date: 2026-07-22

## Scope

The project name in the root README was changed from `LLM-SOG` to the tentative
name `MASK: Matrix-Aware Agentic SpMM Kernel Generation`.

This is a documentation-only change. No source code, generated CUDA artifacts,
benchmark outputs, or SOTA summary files were modified.

## Rationale

The new README presents the project around the three current research ideas:

1. matrix-aware sparse-format design from extracted matrix and hardware
   features;
2. agent-guided compiler and code-generation strategy selection;
3. compile-run-profile feedback that uses correctness, timing, and hardware
   evidence to tune later decisions.

The README still states that the compiler foundation builds on the original
Sparsene paper, but the project name itself no longer contains `Sparsene`.

## Source context used

- `.codex/skills/sparsene-development/SKILL.md`
- `.codex/skills/sparsene-development/references/project-map.md`
- `.codex/skills/sparsene-development/references/research-context.md`
- `.codex/skills/sparsene-development/references/agent-loop.md`
- `.codex/skills/sparsene-development/references/format-codegen.md`
- `docs/codex/format_design_agent_v6.md`
- `docs/codex/custom_format_operator_e2e_v6_all26_n512.md`
- current CLI help for `python/test/end2end_pipeline.py`
- current CLI help for `python/test/run_format_design_batch.py`

## Content changes

- Replaced the old `LLM-SOG` title and introduction with `MASK`.
- Reframed the architecture around the format-design agent, strategy agent, and
  hardware-feedback loop.
- Updated the quick-start command to current `end2end_pipeline.py` flags:
  `--n-cols`, `--iterations`, `--eval-warmup`, and `--eval-repeat`.
- Used the user-requested benchmark defaults `warmup=3` and `repeat=32` in the
  README examples.
- Added a concise description of the fixed performance contract
  `sparsene.kernel_only_post_conversion.v1`.
- Replaced deprecated design-document links with current `docs/codex` and
  `sota_results/summary` provenance pointers.
- Preserved the distinction between declared formats, integrated support, and
  separately validated generated-kernel results.

## Validation

Documentation validation only:

```text
git status --short
python/test/end2end_pipeline.py --help
python/test/run_format_design_batch.py --help
```

No GPU benchmark, paid model invocation, CUDA compilation, or Python unit test
was required for this documentation-only update.

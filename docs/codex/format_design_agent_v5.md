# Format Design Agent v5: General Research DSL Design

> 历史契约。v5 已证明通用研究设计方向可行，但其 JSON 示例、长度语法和
> 8192-token 运行配置导致 2026-07-18 批次全部失败。当前权威契约见
> `docs/codex/format_design_agent_v6.md`，不要把 v5/v6 结果混入同一 run。

## Goal

v5 changes the format agent from an ME-TCF tile selector into a research-format
designer. The agent now proposes a complete hierarchical Format DSL according
to matrix features even when the current repository cannot yet execute that
format.

The core contract is:

```text
matrix features + hardware
  -> unconstrained-by-backend research DSL proposal
  -> deterministic DSL validation (no structural rewrite)
  -> deterministic backend-capability classification
  -> later implementation and performance feedback
```

This directly addresses the 2026-07-17 batch in which all 26 raw model outputs
were ME-TCF/COO/16x8 because the prompt explicitly made the hierarchy immutable.
Those historical outputs remain useful as ME-TCF parameter analyses, but they
are not diverse sparse-format designs.

## Separation of design and execution

The paper's DSL is more expressive than the code paths implemented for the
paper evaluation. v5 therefore treats these as independent questions:

1. Is the proposed Format DSL structurally and semantically valid?
2. How much of this exact proposal can the current repository execute?

A design can be accepted by the format-design batch when question 1 is true
even if question 2 says that backend work is required. The implementation
status must not bias the model toward ME-TCF, but it must also never be hidden.

Capability levels are:

| Level | Meaning |
|---|---|
| `invalid` | DSL validation failed; no construction or execution is allowed |
| `frontend_research` | general DSL is valid and reaches Format/RTS, but no exact backend profile exists |
| `codegen_declared` | atomic/registry code exists, but the full host and numerical path is not validated |
| `standalone_validated` | a known format has a separate generated/testbed route, but not the unified e2e dispatcher |
| `integrated_e2e` | exact ME-TCF/COO profile and a currently accepted `16x8`/`16x16` tile |

The capability classifier matches hierarchy and atomic semantics, not the
model-generated descriptive format name. It separately records the backend
registry name such as `ME_TCF` or `BIT_TCF`.

## General DSL contract

The model may return one to six recursive levels. Each level contains two to
eight axes, covers ROW and COL directions, and uses:

- `df`: dense fixed;
- `dv`: dense variable;
- `sf`: sparse fixed;
- `sv`: sparse variable.

The atomic leaf may be `coo`, `mco`, `csr`, `ell`, `dia`, `dense`, or a fully
specified `custom` leaf. A custom leaf must describe value arrays, metadata
arrays, storage-to-logical coordinate mapping, and restore algorithm; it is
valid research DSL but necessarily classified as requiring backend extension.
Its `x_axis` and `y_axis` must reference ROW and COL axes in the deepest level.
Lengths may use `M`, `K`, `N`, integers, declared parameters, and simple
arithmetic. Reorder identifiers are allowed and preserved.

`levels + atomic` are the authoritative DSL. `transformation_plan` explains the
implied Split, Swap, Sparsify, Spawn, Atomicize, and optional reorder sequence.

The validator checks identifiers, level numbering/count, per-level axis names,
axis types/directions, length-expression symbols, parameter expressions,
reorder syntax, atomic type/layout, and deepest-level atomic references. It
then attempts Format construction and RTS derivation for capability assessment.

## Prompt design

The prompt no longer contains an immutable executable profile or the historical
global 16x8 performance prior. It teaches:

- the physical meaning of each axis type;
- metadata, occupancy, padding, and restore costs of every atomic family;
- ME-TCF, BIT-TCF, BIT-BSR, SR-BCRS, CSR/ELL/DIA, extra levels, and reordering as
  design references rather than a closed choice set;
- a required decision order: diagnose matrix, compare full formats, choose
  hierarchy, axes/reorder, atomic, then parameters;
- that novelty is not a goal and every added component needs evidence;
- that current backend support must not influence the research proposal;
- that candidate metrics are exact for ME-style condensation and only proxies
  when the proposal changes grouping/reordering/hierarchy.

The response must compare at least three materially different complete formats,
cite at least six exact feature values, explain physical semantics in Chinese,
and list implementation requirements and uncertainties.

## Feature digest

The compact digest includes:

- shape, density/sparsity, symmetry, and data quality;
- row/column distributions with tail, CV, Gini, and empty fractions;
- row-span, column-gap, bandwidth, and diagonal-band locality;
- adjacent-row/reordered overlap and unique-column reduction proxies;
- all nine `8/16/32 x 8/16/32` hierarchical candidates;
- tile counts, occupancy, padding, zero work, B-index padding, reuse, imbalance,
  and COO/mask/dense storage estimates;
- the minimum estimated storage tile for each encoding.

Evidence records use exact dotted paths and values. Ungrounded records are
removed, but this cleanup never changes the DSL hierarchy or atomic encoding.

## Failure and normalization policy

v4 replaced unsupported structures with canonical ME-TCF. v5 explicitly
forbids this behavior.

- Malformed explanatory/evidence entries may be removed and recorded in
  `normalization_actions`.
- Malformed structural containers become an invalid design so callers can
  safely serialize the result.
- Semantically invalid levels/axes/atomics are preserved for audit and reported
  in `dsl_validation.errors`.
- HTTP or JSON failure returns `FORMAT_DESIGN_FAILED`, not ME-TCF.
- Format-only batch quality gates reject invalid/fallback results while
  accepting valid `frontend_research` proposals.

Raw payload, deterministic response, token usage, exact prompts, and audit
artifacts remain separated in the common run directory.

## Batch and e2e behavior

`python/test/run_format_design_batch.py` now reports format name/family, atomic
type, reference tile, DSL status, support level, and matched profile. Summary
counts are grouped by family, atomic, support, and reference tile instead of
only `16x8/16x16`. Default paid attempts remain one.

`scripts/run_format_design_26_background.sh` keeps the required virtual
environment and makes the paid connectivity probe opt-in through
`RUN_CONNECTIVITY=1`.

`python/test/end2end_pipeline.py` stops with a precise message when a proposal
is valid but not supported by the integrated DTC path. It does not rewrite or
discard that research design. Once a new backend profile is implemented, the
capability registry and e2e dispatcher can be extended without changing the
research prompt back into a fixed-format selector.

## Validation performed

The focused local suite covers:

- open-search-space prompt content;
- valid novel MCO hierarchy preservation, Format construction, and RTS;
- invalid DSL reporting without ME-TCF rewrite;
- ME-TCF integrated classification;
- BIT-TCF standalone classification;
- fully specified custom-atomic validity and frontend-only classification;
- invalid external response rejection;
- evidence grounding without structural mutation.

No paid model call and no GPU execution are needed for these prompt/schema/
validator changes. The next paid 26-matrix run should be treated as a new v5
experiment rather than resumed into the v4 ME-TCF-only run directory.

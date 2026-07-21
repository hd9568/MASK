---
name: "format-design"
description: "Guides feature-grounded research design of general hierarchical sparse Format DSL proposals, independently of current backend support."
---

# Hierarchical Sparse Format Design

## Mission

Design the physical sparse representation that best matches the input matrix.
The result is a complete hierarchical Format DSL proposal, not a parameter
choice inside a fixed ME-TCF template.

Current code-generation support is an implementation concern evaluated after
design.  It must not collapse the research search space.  Never claim a novel
proposal is already executable; list the work needed to implement it.

## Benchmark Objective

The project's primary performance metric is the warmed GPU SpMM compute-kernel
time after format conversion, metadata construction, padding, and row/column
reordering have completed. These preprocessing transformations, any
correctness-only inverse permutation, allocation/copy, compilation, and
verification are outside the measured time.

Rank designs only by expected work inside the timed compute kernel: metadata
reads, indirect gathers, restore/scatter, padding-induced work, load balance,
locality, occupancy, MMA/accumulation, synchronization, and result writes.
Never reject or down-rank a design because offline conversion or reorder takes
longer. Still specify conversion/reorder correctness, metadata, permutation and
inverse-map semantics, storage, cache identity, and implementation work; label
these as untimed engineering requirements rather than kernel-performance cost.

## DSL Model

- A design has one to six recursive `Format` levels and exactly one atomic leaf.
- Each level has ordered axes and represents both ROW and COL directions.
- Axis types are:
  - `df`: dense, fixed length;
  - `dv`: dense, variable length;
  - `sf`: sparse, fixed length;
  - `sv`: sparse, variable length.
- Atomic types are `coo`, `mco`, `csr`, `ell`, `dia`, `dense`, and a fully
  specified `custom` research leaf.
- The atomic `x_axis` and `y_axis` reference ROW and COL axes in the deepest
  level.
- A `custom` leaf must define its value arrays, metadata arrays, storage-to-
  logical coordinate mapping, and restore algorithm. It is a legal research
  proposal but necessarily requires a new backend implementation.
- `level` is an unquoted, 1-based JSON integer (`1`, `2`, ...), never a string.
- Length is a logical domain bound or split factor. It uses only `M`, `K`, `N`,
  concrete positive integer parameters, parentheses, and `+ - * /`. Never use
  `ceil`, `floor`, `%`, `//`, dotted references, prose, or undeclared symbols.
- For `sv`/`dv`, the realized per-parent cardinality comes from runtime
  metadata while `length` remains a legal bound such as `K` or `BLK_K`.
  Source-style row blocking is `X_o(M)` plus `X_i(BLK_M)`, not
  `ceil(M/BLK_M)`.
- Optional reorder identifiers describe a stable row/column permutation.  A
  reorder must have feature evidence and its preprocessing/inverse-map contract
  must be acknowledged. Its latency is excluded from the kernel-only metric.

The hierarchy implies Split, Swap, Sparsify, Spawn, Atomicize, and optional
reorder transformations.  Explain this transformation plan, but treat
`levels + atomic` as the authoritative DSL.

## Source-Derived Building Blocks

The request prompt injects the complete machine-readable catalog derived from
`python/sparsene/formats/design_catalog.py`. Interpret it as composable legal
patterns, not a classifier menu:

- `ME_TCF` (`DTC_SpMM.py`): two-level BLK_M row-window column condensation,
  BLK_K blocking, then COO tile coordinates/values. Local condensation and
  sorting are not a global matrix reorder.
- `BIT_TCF_ACC` (`Acc_SpMM.py`): the same two-level hierarchy with an MCO
  bitmask+compact-values leaf. The source constant is BIT-TCF even though its
  current `Format.name` says `DTC-SpMM`.
- `CSR_TCF`, `ELL_TCF`, `DIA_TCF`: the same grouping hierarchy with distinct
  tile payload contracts: row_ptr streams, padded per-row slots, or diagonal
  offsets/padded diagonal values. Never describe these as renamed COO arrays.
- `SR_BCRS`: column-condensed active blocks with a dense leaf; worthwhile only
  when zero padding/work inside selected blocks is sufficiently small.
- `BIT_BSR_SPADEN`: one-level sparse active fixed blocks with an MCO leaf; it
  stores active block selection rather than traversing a full dense BSR grid.
- `HME_TCF`: an experimental five-level hierarchy that sparsifies and retiles
  inside an outer TCF tile. The source records unstable RTS behavior, so use it
  as a multi-scale idea rather than an execution claim.
- `ROW_REORDER_SR_BCRS`: sparse/variable row domain, row blocking, column
  condensation, and dense leaf. Despite its name, the current source sets no
  `Axis.reorder`; a sparse row axis is not a stable permutation. A true reorder
  proposal must specify permutation construction, metadata/payload movement,
  output inverse mapping, storage, and cache identity. These are untimed
  engineering requirements in the current benchmark.
- `dense`/`dense_T`: operand axis-order layouts with no atomic child. They teach
  layout ordering but are not complete sparse format-agent outputs.

The main design dimensions are grouping, column representation, hierarchy
depth, optional explicit reorder, and atomic encoding. Reuse, remove, extend,
or combine source patterns only when the feature/cost argument supports it.

These are references, not a closed menu.  Reuse one when it is best, combine
ideas when each component has a clear role, or propose a new hierarchy.  Do
not add complexity merely to look novel or different across datasets.

## Feature-to-Format Reasoning

- Use row/column percentiles, CV, Gini, empty fraction, and top-tail share to
  diagnose irregularity and load imbalance.
- Use adjacent-row overlap and unique-column reduction to judge row-window
  grouping and B-row reuse.
- Use normalized bandwidth, spans, gaps, and diagonal-band fraction to judge
  locality and DIA-like opportunities.
- Use reorder proxies only as estimated post-reorder kernel benefit; they do
  not predict exact kernel performance. Preprocessing latency is deliberately
  outside the primary metric.
- Use all supplied row-window/tile candidates to compare block count, density,
  padding, reuse, imbalance, and estimated COO/mask/dense storage.
- Candidate statistics are exact for the current ME-style row-window
  condensation analysis and only proxies for a different grouping hierarchy.
- Choose COO for very sparse/irregular tiles, MCO when mask metadata is cheaper,
  dense for sufficiently dense active blocks, CSR for row-skewed local tiles,
  ELL for uniform local row lengths, DIA for strong diagonal structure, and a
  custom leaf only when no existing encoding captures the observed structure.
- Global density alone is never sufficient to choose an atomic encoding.

## Required Decision Process

1. Diagnose the matrix structure and likely bottleneck.
2. Compare at least three materially different complete formats.
3. Choose hierarchy, axis order/types, optional reorder, atomic encoding, then
   parameters—in that order.
4. Cite at least six exact feature values using exact dotted paths.
5. Explain kernel-visible metadata, padding, restore, locality, and load-balance
   costs. Describe conversion/reorder implementation separately without using
   its latency to rank formats.
6. Explain why the selected complete format beats the alternatives.
7. List backend/host/atomic/correctness work required for implementation.
8. Lower confidence when features are missing or only proxy the proposal.
9. Before returning, audit integer level typing, expression grammar, declared
   parameters, deepest atomic axes, and exact feature path/value pairs.

## Output Contract

Return one raw JSON object matching the requested schema.  Include:

- descriptive format name/family/summary;
- complete levels, axes, atomic leaf, parameters, and transformation plan;
- optional reference tile from the supplied table;
- exact feature evidence;
- at least three complete-format alternatives;
- Chinese semantic explanation, decision basis, selection reason, and risks;
- implementation requirements and confidence.

Do not emit markdown.  Do not invent measurements.  Do not force ME-TCF, COO,
`16x8`, or `16x16` unless the matrix evidence independently justifies them.
Keep the JSON concise: six to eight evidence records and exactly three complete
alternatives are sufficient; do not repeat the same reasoning across fields.

# Format and Codegen

## Contents

- Format representation
- Implemented atomic families
- Lowering contracts
- Strategy-sensitive NVIR/codegen
- Extension checklist
- Current limitations and conflicts

## Format representation

`python/sparsene/format/format.py` defines four axis combinations:

- `df`: dense, fixed length;
- `dv`: dense, variable length;
- `sf`: sparse, fixed length;
- `sv`: sparse, variable length.

`Format` recursively contains axes and a child. `AtomicFormat` is the leaf storage contract. Symbolic lengths are SymPy expressions and must remain consistent with compile-time/runtime parameters.

## Implemented atomic families

The format layer currently declares factories for:

- dense;
- COO;
- MCO/mask-based;
- CSR;
- ELL;
- DIA.

Concrete format files include ME-TCF, BIT-TCF, SR-BCRS, CSR-TCF, ELL-TCF, DIA-TCF, HME-TCF, and row-reordered variants.

Do not equate declaration with full support. Verify each family through Computent metadata, OPIR generation, NVIR implementations/fusions, host preprocessing, launch interface, and numerical tests. `docs/atomic_format_analysis.md`, `docs/atomic_format_extension_playbook.md`, and format-specific development notes are useful inventories, but confirm every step against current source.

The current v6 format-agent capability registry classifies the exact known profiles as:

| Exact profile | Capability label |
|---|---|
| ME-TCF + COO | `integrated_e2e` only for reference tile `16x8`/`16x16` |
| Two-level column-condensed TCF + row-major MCO, SM80-compatible params | `integrated_e2e` through semantic backend `TCF_MCO_TENSORCORE_V1` |
| BIT-BSR + MCO | `standalone_validated` |
| SR-BCRS + dense | `standalone_validated` |
| CSR-TCF + CSR | `codegen_declared` |
| ELL-TCF + ELL | `codegen_declared` |
| DIA-TCF + DIA | `codegen_declared` |

These are deterministic repository capability labels, not fresh benchmark
claims. Recheck them when host bridges or tests change.

## Lowering contracts

### Format to RTS

`derive_rts()` turns nested format structure into transformations such as split, sparsify, swap, spawn/go-up, and atomicization.

### RTS to Computent

`computent_from_rts()` builds array definitions, schedule nodes, symbol mappings, varlen tables, and atomic metadata. Adding an atomic type normally requires a dedicated transformation application path and matching `AtomicFormatType` handling.

### Computent to OPIR

`generate_from_computent()` emits loops, array references, metadata loads, atomic restore operations, B loads, MMA, and C loads/stores. `VarlenLoweringPass` materializes variable-length semantics; `CValFlattenPass` rewrites output-tile handling.

### OPIR to NVIR

`generate_nvir()` builds global-memory interfaces, chooses memory spaces, lowers operations, inserts S2R/`ldmatrix` paths, and attaches strategy decisions. `compiler_driver.py` performs semantic fusion, stage partitioning, pipeline application, and final code emission.

### NVIR to CUDA and host

`codegen.py` and `templates.py` emit CUTE/CUDA classes and kernels. The host testbed supplies actual metadata arrays, template parameters, dispatch, matrix conversion, cuSPARSE comparison, and cache behavior.

Saved agent designs use `formats/dsl.py` and `formats/custom/loader.py` so
offline construction does not import or invoke an Agent. `backend_registry.py`
matches executable backends by physical semantics rather than dataset or
format-name strings. `generate_custom_format_operators.py` retains RTS and IR
artifacts and emits `<dataset>_<run_id>.inc`; `run_custom_format_e2e.py` builds
and validates the selected backend in the same e2e run directory.

### Preprocessing and reorder semantics

ME-TCF host preprocessing groups `BLK_M` rows, builds the sorted union of their
logical columns, stores the condensed-to-original column map in `val_sidx`,
partitions that condensed sequence by BLK_K, and writes tile-local swizzled COO
coordinates. The generated host bridge adapts this metadata to the lowered
kernel ABI.

This column condensation and local sorting are not a global row or column
reordering of the mathematical matrix. An `Axis.reorder` identifier or a
row-reordered format is a separate semantic promise: verify permutation
construction, metadata/payload permutation, output/inverse-map requirements,
extra storage, host dispatch, and cache identity before claiming reorder
support.

The generalized SM80 TCF-MCO host implements `row_rcm` as deterministic RCM on
the symmetrized structural row graph and restores original output rows outside
timing. A logical BLK_M=8 window remains an independent 8-row condensation
window but is physically zero-padded to the m16 MMA/store tile; padded GPU work
is timed. Extremely long per-window tile lists may be divided into bounded
partial-C execution segments followed by a timed GPU reduction. When no window
is split, the direct-C path avoids reduction overhead.

The current experimental objective is kernel-only post-conversion timing.
Format construction, metadata generation, row/column permutation and
correctness-only inverse permutation occur outside the primary timer and must
not be added to or used to penalize format ranking. They still need a correct,
reproducible implementation and separate engineering reporting. Conversely,
metadata loads, gathers, restore, padding work, synchronization, MMA and stores
that execute inside the timed SpMM kernel are part of the measured cost.

## Strategy-sensitive NVIR/codegen

The current strategy schema controls or records:

- `tile_b` in `{16, 32, 64}`;
- B gather materialization and async-copy mode;
- `val_sidx` materialization;
- COO idx/value placement in SMEM or RMEM;
- COO async or scalar load mode;
- SMEM alignment;
- async-pipeline enablement, synchronization, wait depth, and stage shift.

`StrategyDecision.from_dict()` canonicalizes some unsafe combinations. Read it before changing the schema. A prompt-visible knob is not necessarily independently safe, and a normalized decision may differ from the raw LLM response.

Canonical layout constraints are coupled to `ldmatrix` address generation. Also note that the end-to-end driver may override COO restore swizzle when `TC_BLOCK_K` changes. Inspect both strategy normalization and driver overrides before claiming a knob value reached codegen.

The strategy normalizer forces the default COO restore swizzle to
`swizzle_123`, but the current integrated driver changes the applied value to
`swizzle_124` when BLK_K is not 8. Log raw, normalized, driver-adjusted, and
generated metadata separately.

## Extension checklist

For a new format or atomic leaf:

1. Define exact logical coordinates, stored values, metadata, padding, and ordering.
2. Add format classes/factories and RTS atomicization.
3. Add schedule/array definitions and varlen mappings.
4. Add OPIR operation types and generation.
5. Add NVIR memory placement, shapes, implementations, and fusion rules.
6. Add global-memory interface hints and dtype/layout rules.
7. Add host preprocessing and device argument wiring.
8. Include every format/tile/preprocessing semantic in checkpoint and binary cache keys.
9. Test isolated structure, small numerical cases, real matrices, and performance.
10. Compare the generated route with cuSPARSE and, separately, any hand-tuned reference.

For CSR, ELL, DIA, or another leaf, require a dedicated payload contract rather
than reusing COO arrays by name. Examples of identity-bearing metadata include
CSR row pointers, ELL per-block row width/padding sentinels, DIA offsets, and
MCO masks. A shared loader skeleton is acceptable only when the host payload
and restore operands still implement the declared semantics.

## Current limitations and conflicts

- The v6 format agent designs against the general hierarchical DSL. It does not
  force ME-TCF or COO and does not rewrite unsupported proposals. Always inspect
  `dsl_validation` and `capability` separately.
- The legacy unified DTC route remains ME-TCF/COO-specific. A separate semantic
  custom-format route integrates two-level column-condensed TCF + row-major MCO
  on SM80, including optional row_rcm, logical BLK_M 8/16, BLK_K multiples of
  8, compilation, correctness, and cuSPARSE timing. Other hierarchies and
  atomic leaves remain research/declared/standalone until their own semantic
  backend and host contracts are validated.
- BIT-TCF lowering must preserve the BLK_MMA K dimension and loop A/B ldmatrix
  plus MMA for BLK_K>8. The historical flattened ACC register shapes silently
  compute only the first 8 K entries.
- SM80's grid.y limit requires multiple launches for large execution-window
  counts; all launches in one SpMM iteration are part of the primary timer.
- Format-agent `params` may contain concrete design values referenced symbolically
  by level length expressions. Capability checks must verify parameter/tile
  consistency before classifying an integrated profile.
- Historical docs disagree about adaptive thresholds, swizzles, supported atomics, and generated/reference gaps. Use current executed code.
- Current heuristic `SPARSENE_TC_BLOCK_K_AGGRESSIVE=1` selects 16 only for a
  positive mean NNZ/row below 6; this is an implementation heuristic, not a DSL
  restriction or a proven general threshold.
- Experimental grid/mixed format modules currently require full integration verification; do not present their design comments as measured behavior.

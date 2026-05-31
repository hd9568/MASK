---
name: "format-design"
description: "Guides sparse format selection and parameter choice. Invoke when an agent must choose BLK sizes, atomic format, or hierarchical sparse layout for a matrix."
---

# Format Design

## Mission

Use this skill when selecting or validating a sparse format from matrix statistics.

## Current Codegen Constraint

The current Tensor Core lowering only supports the standard 2-level ME-style structure:

- Outer level: `df(X_o, ROW, M) + sv(Y_s, COL, K) + df(X_i, ROW, BLK_M)`
- Inner level: `df(Y_so, COL, K/BLK_K) + df(Y_si, COL, BLK_K) + df(X_i, ROW, BLK_M)`
- Atomic leaf: `coo(X_i, Y_si)`, `mco(X_i, Y_si)`, or `dense(X_i, Y_si)`

Do not invent new axis names, directions, or level counts unless the backend support is also being extended.

## Decision Rules

- If row NNZ CV is above `0.3`, keep `sv_axis("Y_s", COL, "K")`.
- `BLK_M`:
  - `8` for very short or highly imbalanced rows.
  - `16` for medium row density and balanced padding/performance tradeoff.
  - `32` when rows are denser and load balance remains stable.
- `BLK_K`:
  - `16` when Tensor Core utilization dominates and padding is tolerable.
  - `8` when `16` causes excessive padding.
- Atomic type:
  - `mco` for very low per-block density.
  - `coo` for medium density and flexibility.
  - `dense` only when block fill rate is high enough to amortize padding.

## What Features Matter Most

- Global shape and NNZ
- Row/column NNZ distributions
- Row-length coefficient of variation
- Padding ratio for candidate `BLK_M x BLK_K`
- Per-block density percentiles
- TC-block load-balance indicators

## Output Style

Return structured JSON that includes:

- format name
- levels
- atomic descriptor
- `BLK_M`, `BLK_K`
- reorder decision
- rationale and confidence

## Validation Reminder

Even if a new format name is proposed, verify it still maps to supported runtime hints and gmem interface expectations before codegen.

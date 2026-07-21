# Format Design Agent v4

> Historical executable-profile design. Superseded for research format design
> by `docs/codex/format_design_agent_v5.md`. Keep this document only as the
> contract/provenance of the 2026-07-17 ME-TCF-only batch.

## Why the previous design was unsafe

The previous prompt mixed two incompatible contracts. Its DSL tutorial invited
one-, two-, or three-level formats, arbitrary axis types, COO/MCO/dense leaves,
and reordering, while a later rule required the exact two-level ME-TCF COO
layout. The response parser only forced `atomic.type="coo"`; it did not enforce
the level count, axis order, names, directions, types, expressions, or atomic
axis references. A plausible LLM response could therefore reach Format-to-RTS
despite being unsupported by the host and generated backend.

There was also no effective feature-driven format decision. `BLK_M` and
`BLK_K` were made symbolic, then the driver selected concrete values before
calling the agent and ignored the agent's recommendation. The format agent was
mostly producing a rationale around a fixed format.

## Executable profile

v4 separates the general DSL from the currently executable search space. The
profile `me_tcf_coo_sm80_v1` has one legal structure:

```text
row_window:
  df(X_o, ROW, M)
  sv(Y_s, COL, K)
  df(X_i, ROW, BLK_M)
tc_block:
  df(Y_so, COL, K/BLK_K)
  df(Y_si, COL, BLK_K)
  df(X_i, ROW, BLK_M)
atomic:
  coo(X_i, Y_si)
```

The executable candidate set is `16x8` and `16x16`. The DSL parameters remain
symbolic, while `selected_candidate` records the concrete choice that agent
mode applies to NVCC/host preprocessing. BLK_M=8, BLK_M=32, BLK_K=32, MCO,
dense, reorder, and extra hierarchy levels remain outside this profile.

`format_name` is fixed to `ME_TCF`, separately from the versioned profile name.
This is an execution ABI: the NVIR registry and semantic-fusion driver dispatch
on `ME_TCF`. A descriptive model-generated name would silently bypass the
ME-TCF-specific fusion path.

This is deliberately narrower than the abstract Sparsene DSL. A genuinely new
format must first gain matching RTS/Computent/OPIR/NVIR, global-memory hints,
host preprocessing, launch arguments, cache identity, and numerical tests; it
must not be enabled by prompt wording alone.

## Feature-to-decision reasoning

The prompt receives a compact v3 digest rather than the full JSON. It includes:

- shape, data quality, and row/column long-tail statistics;
- locality and reorder metrics, explicitly marked diagnostic-only;
- exact post-condensation `16x8` and `16x16` costs;
- padded COO work, TC-block count, unique columns, B-index padding, Tensor Core
  zero fraction, density percentiles, reuse, and window imbalance;
- explicit `16/8` ratios and deltas so the model compares candidates rather
  than interpreting unrelated absolute values.

Global matrix density is not accepted as a substitute for post-condensation
block density. The model must return at least three evidence records with exact
digest paths and copied values. The normalizer resolves every path and removes
evidence whose value does not match the supplied digest.

## Backend-calibrated prior

Static metadata reduction is not a complete performance model. In
`.tc_block_k_sweep_20260715_124737`, BLK_K=8 beat BLK_K=16 on 22 of 26 matrices,
and median `time16/time8` was 1.3137836463785613. BLK_K=16 usually halves the TC
block count, but it also adds K iterations and shared-memory/restore overhead.

Therefore:

- `16x8` is the deterministic fallback and measured safety prior;
- block-count reduction alone is not sufficient evidence for `16x16`;
- the agent may select `16x16`, but must justify it using matrix-specific
  candidate costs;
- future closed-loop format learning should replace this coarse prior with
  accumulated correct per-matrix measurements and uncertainty estimates.

## Deterministic correctness boundary

`normalize_format_payload()` treats model JSON as untrusted input. It validates
or repairs:

- the complete level/axis tree;
- the `ME_TCF` registry/semantic-fusion format name and executable profile;
- the COO atomic descriptor;
- symbolic DSL parameters;
- candidate membership and feature availability;
- invalid-entry safety fallback;
- evidence structure and exact value grounding;
- confidence vocabulary and confidence after repair.

Every repair is stored in `normalization_actions`. `raw_payload` remains
available for diagnosing model quality. `build_format_from_response()` repeats
the executable-profile checks and refuses a noncanonical response, so callers
cannot bypass the normalizer accidentally.

The recorded decision now distinguishes:

```text
raw LLM payload -> normalized legal DSL/candidate -> applied candidate
```

In agent mode, `end2end_pipeline.py` applies the normalized candidate to
`args.blk_m` and `args.tc_block_k`. Heuristic mode retains its separate policy.
The format-agent configuration now reads its default credential only from
`DEEPSEEK_API_KEY`; source code no longer contains a credential fallback.

## Validation evidence

- Six focused unit tests cover prompt scope, a valid response, hostile
  three-level/dense/numeric output, invalid input data, builder rejection, and
  canonical Format construction.
- Python compilation succeeds for the agent, end-to-end driver, and tests.
- A canonical response traverses Format -> RTS -> Computent -> lowered OPIR and
  produces the expected seven COO global-memory interfaces.
- All 26 persistent v3 files contain both executable candidates and can be
  summarized and normalized. The original static score experiment selected
  `16x16` for all 26, which contradicted measured sweep performance and was
  removed in favor of the measured-safe fallback.
- No live external-model call was required for structural validation. A future
  evaluation should replay several models with fixed prompts and measure raw
  validity, normalization rate, candidate accuracy, correctness, and runtime.

## Remaining limitation and next step

v4 performs meaningful parameterized design inside one executable format
family; it is not yet arbitrary sparse-format synthesis. The next principled
extension is a registry of independently validated executable profiles (for
example COO versus a fully integrated mask or dense leaf). The agent could then
classify among profiles, while each profile supplies its own strict schema,
normalizer, host contract, and correctness suite.

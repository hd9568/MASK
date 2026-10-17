# Project Map

## Contents

- System identity
- Authoritative paths
- Current execution routes
- Source-of-truth rules
- Known repository state distinctions

## System identity

**SparseKG: A Matrix- and Architecture-Aware Agentic Compiler for SpMM Kernel
Generation** is the current project identity. SparseKG builds on Sparsene's
format-driven SpMM compiler foundation and extends it with:

1. matrix feature extraction;
2. a format-design agent;
3. deterministic Format-to-CUDA lowering;
4. a strategy agent for tensor placement and pipeline decisions;
5. compile, correctness, timing, and NCU feedback;
6. iterative selection of the best correct generated kernel.

The core compiler route is:

`Format -> derive_rts() -> Computent -> OPIR -> VarlenLoweringPass -> CValFlattenPass -> NVIR -> semantic fusion/software pipeline -> generated .inc`

The paper names the fine-grained executable abstraction `pTask` and its dependency DAG `pGraph`. In the implementation, the closest concrete layers are OPIR/NVIR operations, `NvOp` dependency structures, pipeline plans, and simulator inputs; do not assume a one-to-one class named `pTask` exists.

## Authoritative paths

| Concern | Primary source |
|---|---|
| Project entry and user workflow | `README.md` |
| Paper | `paper/SC26-Sparsene-main/main.tex` and section `.tex` files |
| Format data model and atomic factories | `python/sparsene/format/format.py` |
| Concrete formats | `python/sparsene/formats/*.py` |
| Format to RTS | `python/sparsene/transform/rts.py`, `transformation.py` |
| RTS to schedule/arrays | `python/sparsene/op_gen/computent/` |
| OPIR generation and types | `python/sparsene/op_gen/opir/` |
| NVIR model and lowering | `python/sparsene/op_gen/nvir/nvop.py`, `generate.py` |
| Semantic fusion and pipeline application | `python/sparsene/op_gen/nvir/compiler_driver.py`, `plan.py`, `pipeline/` |
| CUDA emission | `python/sparsene/op_gen/nvir/codegen.py`, `templates.py` |
| Matrix feature extraction | `python/sparsene/analysis/matrix_features.py` and CUDA testbed feature extractor |
| Runtime LLM plumbing | `python/sparsene/agent/` |
| Format agent | `python/sparsene/op_gen/format_design_agent.py` |
| Strategy agent | `python/sparsene/op_gen/strategy_agent.py` |
| Fix agent | `python/sparsene/op_gen/fix_agent.py` |
| End-to-end driver | `python/test/end2end_pipeline.py` |
| Audited format-design batch | `python/test/run_format_design_batch.py` |
| Multi-dataset feedback search | `python/test/run_strategy_feedback_multidataset.py` |
| Basic generation route | `python/test/test_generate.py` |
| Declared multi-format generation | `python/test/generate_sparse_formats.py` |
| Runtime prompt skills | `.agent/skills/*/SKILL.md` |
| Host bridge/testbed | `examples/src_fp32/dtc/testbed/` |
| Current ME-TCF K sweep | `scripts/sweep_tc_block_k.sh` |
| Historical design and experiment records | `docs/` |

## Current execution routes

### End-to-end route

Use `python/test/end2end_pipeline.py` for the integrated feature-format-strategy loop. Inspect its `parse_args()` or run `--help` before composing commands; older README and docs examples use parameter names that have changed.

The current driver:

- extracts matrix features with the CUDA feature tool;
- chooses adaptive `TC_BLOCK_K` and pins `BLK_M` according to current policy;
- uses built-in ME-TCF in heuristic mode or calls the general research format
  agent in agent mode;
- continues into the current unified DTC strategy/codegen route only for an
  exact `integrated_e2e` ME-TCF proposal; other valid proposals remain format
  artifacts until their backend/host dispatcher is implemented;
- builds OPIR/NVIR artifacts once for the chosen format;
- evaluates iteration 0 with a heuristic strategy;
- iterates strategy decisions, generation, compilation, correctness, timing, and NCU;
- selects only from correct iterations;
- restores the default generated `.inc` in a `finally` block.

### Standalone generation route

`python/test/test_generate.py` currently exercises ME-TCF and writes a generated `.inc`. Much of the other-format code in that file is commented reference material, so imports alone do not establish validated support.

`python/test/generate_sparse_formats.py` lowers ME-TCF, BIT-TCF, BIT-BSR, and
SR-BCRS into `python/test/results/*.inc`. Successful source emission establishes
only that this declared generation route completed; it does not prove that the
matching host bridge compiled, ran, or passed numerical validation.

### Audited research-format route

`python/test/run_format_design_batch.py` consumes persistent
`*.features.v3.json` files and writes a phase-oriented run below
`logs/e2e_runs`. It accepts a valid general v6 DSL even when capability is only
`frontend_research`. It persists exact Agent messages/response/usage and keeps
per-dataset research formats separate from future strategy/codegen artifacts.

Do not resume a batch into a directory with an incompatible format-design
schema; the driver rejects schema mixing. The background wrapper activates the required venv and makes paid
connectivity probing opt-in.

### Multi-dataset strategy route

`python/test/run_strategy_feedback_multidataset.py` provides broader search, mutation/diversification, synthetic and dataset evaluation, profile aggregation, best-entry selection, and final all-dataset reporting.

## Source-of-truth rules

Use this precedence when claims conflict:

1. currently executed source and tests;
2. generated artifacts and fresh logs from the same revision;
3. current focused `docs/codex/` contracts that match source;
4. current root README for architecture orientation;
5. dated optimization notes and copied onboarding documents.

Historical documents are valuable for hypotheses, failed attempts, and benchmark provenance, but several contain obsolete paths, CLI flags, performance gaps, or capability claims.

## Known repository state distinctions

- `me_tcf_kernel3.inc`, `python/me_tcf_kernel3.inc`, generated per-run `.inc` files, and `examples/.../kernel.inc` can have different provenance and behavior.
- `docs/agent_runs/` and other run directories are observations, not source-of-truth implementation.
- `/mnt/lustre-client/zhangkaige/selected_mtx` contains raw matrices, while
  `/mnt/lustre-client/huoda/selected_mtx` contains persistent v3 feature
  artifacts. Do not treat the latter as a replacement raw-dataset root.
- `examples/cutlass/` is a large external dependency/build tree; do not treat it as first-party Sparsene logic unless the task targets CUTLASS integration.
- `.trae/skills` includes byte-for-byte copies of several docs plus older
  summaries. Use it to recover historical hypotheses, never as a current API or
  performance contract.

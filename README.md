# MASK: Matrix-Aware Agentic SpMM Kernel Generation

MASK is a matrix-specialized sparse GPU operator generation framework. It builds on
the format-centric compiler ideas from the Sparsene paper, but adds two LLM-based
agents around the deterministic compiler pipeline:

1. a format-design agent that proposes sparse formats from matrix and hardware
   features;
2. a strategy agent that selects code-generation, tensor placement, tiling, and
   pipeline choices;
3. a compile-run-profile feedback loop that measures generated kernels and feeds
   correctness, timing, and hardware evidence back into later decisions.

The current target is high-performance SpMM on NVIDIA SM80 GPUs, especially
A100/A800, with the main evaluation path using FP32 inputs/outputs and TF32
Tensor Core execution for `N=512`.

## System Overview

```text
Input: MatrixMarket sparse matrix (.mtx)
    |
    v
+----------------------------------------------------------+
| Phase 1: matrix and hardware feature extraction           |
| row/column NNZ, block density, locality, load balance     |
+-------------------------+--------------------------------+
                          |
                          v
+----------------------------------------------------------+
| Phase 2: format-design agent                              |
| matrix-aware hierarchy, axes, atomic format, reorder      |
| output: auditable Format DSL + capability classification  |
+-------------------------+--------------------------------+
                          |
                          v
+----------------------------------------------------------+
| Phase 3: deterministic compiler lowering                  |
| Format -> RTS -> Computent -> OPIR -> NVIR -> CUDA .inc   |
+-------------------------+--------------------------------+
                          |
                          v
+----------------------------------------------------------+
| Phase 4: strategy-agent feedback loop                     |
| heuristic baseline -> strategy -> codegen -> build        |
| -> correctness -> timing/profile -> next strategy         |
+-------------------------+--------------------------------+
                          |
                          v
                 Best correct generated SpMM kernel
```

The agents search and interpret design choices, but the compiler, validators,
host bridges, and benchmark harness are deterministic. A generated kernel is
ranked only after it compiles and passes numerical validation.

## Core Ideas

### Matrix-aware format design

The format-design agent consumes persistent matrix feature files and an SM80
hardware profile. It may propose one to six levels of hierarchical sparse format
DSL using dense/sparse, fixed/variable axes and atomic leaves such as COO, MCO,
CSR, ELL, DIA, dense, or a fully specified custom leaf.

The agent is not limited to choosing a known format name. Deterministic validation
preserves valid research designs and classifies implementation support
separately as `frontend_research`, `codegen_declared`, `standalone_validated`, or
`integrated_e2e`.

### Agent-guided compiler strategy

The strategy agent works after the format has been lowered into compiler
artifacts. Its current decisions include `tile_b`, B/COO memory placement,
async-copy behavior, shared-memory alignment, pipeline enablement, wait depth,
and stage shifts. The strategy response is normalized by deterministic
guardrails before CUDA code is generated.

### Hardware feedback loop

MASK keeps correctness and performance evidence in the run directory. Each
iteration records the requested strategy, normalized strategy, generated source,
compile result, correctness result, timing, and optional NCU profile. Later agent
decisions can use this structured feedback instead of relying only on static
rules.

## Compiler Pipeline

The main lowering path is:

```text
Format
  -> derive_rts()
  -> Computent
  -> OPIR
  -> VarlenLoweringPass
  -> CValFlattenPass
  -> NVIR
  -> semantic fusion / software pipeline
  -> generated CUDA include (.inc)
  -> host bridge / preprocessing / validation / timing
```

The paper-level concepts of fine-grained executable tasks and dependency graphs
map in this implementation to OPIR/NVIR operations, `NvOp` dependency
structures, pipeline plans, simulator inputs, and generated CUDA kernels.

## Performance Contract

All current format, strategy, and result artifacts use:

```text
sparsene.kernel_only_post_conversion.v1
```

The primary metric is warmed CUDA-event time for the SpMM compute kernel after
format conversion, metadata construction, padding, and row/column reordering
have completed.

Not included in the primary metric: format conversion, reorder and inverse
mapping, checkpoint I/O, allocation/copy, code generation, compilation,
correctness verification, and cuSPARSE preprocess.

Included in the primary metric: every GPU launch inside the timed compute path,
metadata reads, indirect B gathers, restore/scatter, atomics, padding work,
synchronization, MMA, accumulation, stores, and any required timed reduction.

## Repository Layout

```text
.
|-- python/sparsene/
|   |-- analysis/              # matrix feature extraction
|   |-- agent/                 # model runtime, JSON calls, audit logging
|   |-- format/                # Axis, Format, AtomicFormat data model
|   |-- formats/               # declared sparse format families and catalog
|   |-- op_gen/                # format agent, strategy agent, IR and codegen
|   `-- transform/             # Format -> RTS transformations
|-- python/test/
|   |-- end2end_pipeline.py    # single-matrix integrated agent/codegen loop
|   |-- run_format_design_batch.py
|   |-- run_custom_format_e2e.py
|   `-- run_builtin_backend_e2e.py
|-- examples/src_fp32/         # CUDA/CUTLASS host bridges and testbeds
|-- .agent/skills/             # runtime skills injected into project agents
|-- .codex/skills/             # Codex development skill
|-- docs/codex/                # current engineering records and contracts
|-- sota_results/summary/      # audited SOTA and generated-kernel summaries
|-- logs/e2e_runs/             # coordinated experiment artifacts
`-- paper/                     # original paper source used for context
```

## Environment

Use the project virtual environment before running Python entry points:

```bash
cd /home/huoda/sparsene
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
```

Common requirements:

- CUDA 12.x, NVCC, cuSPARSE, cuBLAS
- NVIDIA SM80 GPU such as A100 or A800
- Python 3.12 in `/mnt/lustre-client/huoda/venv/llm-sog`
- model credentials provided through environment variables or protected model
  configuration

Do not copy literal API keys into commands, docs, or logs.

## Quick Start

Run one matrix through the integrated feature-format-strategy loop:

```bash
PYTHONPATH=python python3 python/test/end2end_pipeline.py \
  --mtx /mnt/lustre-client/zhangkaige/selected_mtx/consph.mtx \
  --n-cols 512 \
  --mode agent \
  --format-model deepseek-v4-pro \
  --strategy-model deepseek-v4-pro \
  --iterations 1 \
  --eval-warmup 3 \
  --eval-repeat 32
```

Run the audited format-design phase over persistent 26-dataset features:

```bash
PYTHONPATH=python python3 python/test/run_format_design_batch.py \
  --feature-dir /mnt/lustre-client/huoda/selected_mtx \
  --run-dir logs/e2e_runs/<run_id> \
  --model deepseek-v4-pro \
  --max-tokens 32768 \
  --resume
```

Run generated custom-format operators from an existing e2e run directory:

```bash
PYTHONPATH=python python3 python/test/run_custom_format_e2e.py \
  logs/e2e_runs/20260718_165146_deepseek_v4_pro_format26_v6 \
  --n 512 \
  --warmup 3 \
  --repeat 32 \
  --jobs 1
```

GPU benchmark jobs should normally remain serial (`--jobs 1`) so independent
operator runs do not disturb one another.

## Important Entry Points

| Task | Entry point |
|---|---|
| Single-matrix integrated pipeline | `python/test/end2end_pipeline.py` |
| 26-dataset format-design batch | `python/test/run_format_design_batch.py` |
| Custom-format operator e2e run | `python/test/run_custom_format_e2e.py` |
| Built-in backend e2e run | `python/test/run_builtin_backend_e2e.py` |
| Multi-dataset strategy feedback | `python/test/run_strategy_feedback_multidataset.py` |
| Source-derived format vocabulary | `python/sparsene/formats/design_catalog.py` |
| Format capability matching | `python/sparsene/op_gen/backend_registry.py` |
| NVIR-to-CUDA emission | `python/sparsene/op_gen/nvir/codegen.py` |

Check `--help` on the relevant script before composing long runs; older notes
may contain historical flag names.

## Current Format Families

| Family | Leaf encoding | Current status |
|---|---|---|
| ME-TCF | COO | integrated e2e for exact supported tile profiles |
| TCF-MCO / BIT-TCF | MCO mask + compact values | integrated through the custom SM80 backend for supported profiles |
| BIT-BSR | MCO over active fixed blocks | standalone/custom validated for supported cases |
| SR-BCRS | dense active blocks | standalone validated with generated backend work in progress |
| CSR-TCF | CSR | codegen-declared; requires dedicated host/backend validation |
| ELL-TCF | ELL | codegen-declared; requires dedicated host/backend validation |
| DIA-TCF | DIA | codegen-declared; requires dedicated host/backend validation |
| HME-TCF and row-reorder variants | mixed/reordered | research or experimental unless separately validated |

Declaration alone is not executable support. New formats need matching
preprocessing, metadata ABI, lowering, host dispatch, numerical validation, and
benchmark provenance.

## Results and Provenance

Current project records separate historical paper/SOTA data from freshly
generated results:

- `sota_results/summary/`: audited A100/A800 CSV and Markdown summaries.
- `docs/codex/format_design_agent_v6.md`: current format-agent contract.
- `docs/codex/custom_format_operator_e2e_v6_all26_n512.md`: 26-dataset custom
  format e2e closure and known performance bottlenecks.
- `docs/codex/sparsene_a100_best_vs_current_a800_n512.md`: historical
  A100-versus-current A800 comparison methodology.
- `docs/codex/development_log.md`: chronological engineering record.

Historical logs and paper results are useful context, but new claims should be
backed by fresh artifacts from the same code revision, dataset, precision, `N`,
GPU, CUDA toolchain, warmup/repeat counts, and cache policy.

## Paper Context

The original paper source is kept under `paper/SC26-Sparsene-main/`. MASK uses
the paper's hierarchical sparse-format and automatic pipeline-construction ideas
as the compiler foundation, then extends the system with matrix-aware format
design, agent-guided compilation strategy, and hardware-feedback optimization.

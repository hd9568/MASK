---
name: "sparsene-framework"
description: "Explains the Sparsene compilation stack and data model. Invoke when an agent must reason about format -> RTS -> OPIR -> NVIR -> CUDA generation."
---

# Sparsene Framework

## Mission

Use this skill whenever a task touches sparse format semantics, code generation, or runtime integration inside Sparsene.

## Core Pipeline

The canonical pipeline is:

`Format -> derive_rts() -> Computent -> OPIR -> VarlenLoweringPass -> CValFlattenPass -> NVIR -> software pipeline/codegen -> .inc -> nvcc binary`

## What Each Layer Means

- `Format`: hierarchical sparse layout description with recursive `Format` nodes and an atomic leaf.
- `RTS`: flattened transformation sequence derived from the format tree.
- `Computent`: schedule plus array definitions inferred from RTS.
- `OPIR`: structured sparse operator IR with loops, loads, array refs, COO restore, and MMA.
- `NVIR`: CUDA-oriented lowering with explicit memory spaces and software-pipeline decisions.
- `.inc`: final generated CUDA/CUTLASS-style kernel implementation consumed by `host_program_generated.cu`.

## Key Format Concepts

- Axis factories: `df_axis`, `dv_axis`, `sf_axis`, `sv_axis`.
- `sv_axis` on the K/column side is the current mechanism for irregular row-length handling.
- Atomic formats currently relevant for ME-style Tensor Core paths are `coo`, `mco`, and `dense`.

## Files That Matter

- Format definitions: `python/sparsene/formats/*.py`
- Format data model: `python/sparsene/format/format.py`
- RTS derivation: `python/sparsene/transform/rts.py`
- Computent: `python/sparsene/op_gen/computent/computent.py`
- OPIR generation: `python/sparsene/op_gen/opir/generate.py`
- NVIR lowering: `python/sparsene/op_gen/nvir/generate.py`
- Software pipeline/codegen: `python/sparsene/op_gen/nvir/compiler_driver.py`, `codegen.py`, `templates.py`
- Runtime host path: `examples/src_fp32/dtc/testbed/host_program_generated.cu`, `main.cu`

## Agent Guardrails

- Do not free-form generate CUDA first; make structured decisions first.
- Prefer decisions that map onto existing format/codegen constraints.
- Preserve correctness on varlen sparse axes before chasing performance.

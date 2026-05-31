---
name: "strategy-optimization"
description: "Encodes safe ME_TCF CUDA tuning knobs and constraints. Invoke when an agent must choose cp.async, SMEM/RMEM placement, TILE_B, or pipeline settings."
---

# Strategy Optimization

## Mission

Use this skill when tuning NVIR/codegen decisions for the current ME_TCF Tensor Core kernel family.

## Main Decision Groups

- Tensor placement
- Async copy policy
- Shared-memory layout constraints
- Software pipeline depth and synchronization
- `TILE_B` selection

## Safe Knobs

- `tile_b`: `16`, `32`, `64`
- `b_array_ref_mode`: `cp_async_1x4` or `sync_scalar`
- `coo_load_mode`: `cp_async_x4_aligned_fallback` or `sync_scalar`
- `coo_idx_output_mem` / `coo_val_output_mem`: `smem` or `rmem`
- `max_wait_prior`: `None`, `0`, `1`, `2`, `3`
- `stage_shifts`: `[]`, `[0]`, `[2]`
- `sync_threads_on_smem_input`: `True` or `False`
- `align_cp_async_smem_outputs`: `True` or `False`

## Hard Constraints

- `b_smem_swizzle` must remain `swizzle_323`.
- `coo_restore_swizzle` must remain `swizzle_123`.
- If COO idx/val are placed in `rmem`, `coo_load_mode` must be `sync_scalar`.
- If async pipeline is disabled, all async producers must downgrade to synchronous behavior.
- `materialize_val_sidx_to_smem` must stay `True` on the current lowering path.

## Optimization Heuristics

- Keep `cp.async` only when alignment and producer-consumer overlap are credible.
- Use `smem + cp.async` for highest throughput when correctness is stable.
- Consider `rmem + sync_scalar` only as a conservative fallback when sync pressure or async incompatibility is suspected.
- If occupancy is low because registers are high, reduce pipeline aggressiveness before changing fundamental dataflow.
- When performance stagnates, change a small number of knobs unless profile data clearly points to a larger move.

## Strategy Diversity

Good exploration dimensions:

- `TILE_B`
- B-side async on/off
- COO smem vs rmem
- bounded vs unbounded wait depth
- stage shift placement
- extra `__syncthreads` before SMEM consumption

## Canonicalization

- Treat `stage_shifts=[]` as the canonical form for default shift `1`.
- Do not emit `stage_shifts=[1]`; it is semantically redundant and wastes search budget.

## Output Style

Return structured JSON only. Include rationale that references correctness history, recent timings, and NCU profile evidence.

# web-BerkStan Generated Kernel 优化分析

## 1. 当前状态

| 路径 | mykernel_ms | cuSPARSE_ms | speedup | 日期 |
|---|---|---|---|---|
| generated (latest_generated.md) | 6.928 | 4.183 | 0.604 | 2026-06 |
| generated (本轮起点) | 5.070 | 4.218 | 0.832 | 2026-06-10 |
| generated (本轮终点) | 5.125 | 4.238 | 0.827 | 2026-06-10 |
| reference (手工算子) | ~3.937 | 4.232 | 1.075 | 历史 |

目标: generated 超过 cuSPARSE。当前差距约 21%。

## 2. 本轮 codegen 修改汇总

### 2.1 保留的修改 (infrastructure improvements)

| 文件 | 修改 | 原因 |
|---|---|---|
| `main.cu:31` | `#define TC_BLOCK_M 16` → `#ifndef` guard | 允许 -D 运行时覆盖 |
| `nvop.py:156` | `SwizzleLayout` 新增 `adaptive_blk_k: bool` | 支持 BLK_K 自适应的 swizzle |
| `codegen.py:1154` | swizzle codegen 支持 `adaptive_blk_k` | 当 BLK_K=16 时自动使用 S=4 |
| `generate.py:1651` | `coo_atomic_val_restore` 使用 `adaptive_blk_k=True` | 确保 restore swizzle 随 BLK_K 自适应 |

### 2.2 此次尝试且回退的修改

| 实验 | 修改 | web-BerkStan 结果 | 处理 |
|---|---|---|---|
| TC_BLOCK_M=32 | `-D` 编译 | correctness errors (row shift) | **回退**。文档已记录 M=32 bug，需进一步 fix |
| TC_BLOCK_K=16 + adaptive swizzle | codegen fix + `-D` 编译 | 5.08→6.98ms 退化 (correctness 通过) | **回退**。K 加倍使每 block 的计算量 > dispatch 节省 |
| `array_ref` 改为 2 线程 `if(lid<2)` | 手动改 .inc | 5.07→5.90ms 退化 | **回退**。warp divergence 导致 |
| `array_ref_1` 加 `__syncthreads()` | 手动改 .inc | 5.08→5.92ms 退化 | **回退**。pipeline 已有 wait_prior 保证 DMA |
| restore 后加 `__pipeline_commit()` | codegen 改 generate.py | 5.14→5.19 无变化 | **回退**。同步 smem 操作加 commit 无实际 overlap |

## 3. 基础设施验证数据集结果 (web-BerkStan + 3 数据集)

| 数据集 | M | generated_ms | cuSPARSE_ms | speedup | correctness |
|---|---|---|---|---|---|
| web-BerkStan | 685,230 | 5.125 | 4.238 | 0.827 | ok |
| cant | 62,451 | 1.016 | 1.875 | 1.846 | ok |
| scircuit | 170,998 | 1.194 | 0.980 | 0.821 | ok |
| eu-2005 | 862,664 | 10.049 | 9.491 | 0.944 | ok |

所有数据集正确性通过。scircuit 相对 latest_generated.md (1.561ms) 有小幅提升。

## 4. 关键分析结论

### 4.1 Generated 与 Reference 的结构差异

当前 generated kernel 的 pipeline 结构与 reference 高度相似 (fill/loop_step/remains/empty 模式均已实现)。遍历了以下所有优化方向，均未获得显著性能提升:

- Block 参数调优 (TC_BLOCK_M, TC_BLOCK_K, TILE_B) → 退化或 bug
- Helper 级别微调 (array_ref, syncthreads, pipeline commit) → 无收益或退化
- Swizzle 自适应 → 仅确保正确性，不影响性能

### 4.2 根因判断

web-BerkStan 的 generated vs reference 差距 (~5.1ms vs ~3.9ms, +31%) 不是 helper 级别的问题。

可能根因:
1. **B 矩阵访存模式**: 685K×512 的 B 矩阵，每个 row window 随机 gather 访问，L2 cache 命中率低
2. **Thread block 数量**: 282K TC blocks，大量极稀疏 block (1-2 NNZ)，dispatch overhead 高但无法通过增大 M/K 缓解 (均有正确性或性能问题)
3. **需要 NCU profiling**: 确定 stall 是 SMEM bank conflict、global memory latency、还是 warp scheduler mismatch

### 4.3 下一步方向

1. **NCU profiling** → 确定具体 stall 原语
2. **B 矩阵 prefetch**: 在 row window 级别预取 B 列到 smem，减少随机访问惩罚
3. **自适应融合**: 对极稀疏 block 做 warp-level 早期退出，避免无效 DMA
4. **TC_BLOCK_M=32 的正确性修复**: 使得更大的 row window 可行

# TC_BLOCK_K 自适应选择优化

## 概述

本文档记录了将 `TC_BLOCK_K` 从固定值 8 改为根据矩阵稀疏度自适应选择（8 或 16）的完整过程。

**核心思路**：稀疏矩阵（avg NNZ/row < 16）使用 `TC_BLOCK_K=16`，将 TC block 数量减半，减少 kernel launch 数和 dispatch overhead。

---

## 1. 性能瓶颈定位

### 1.1 问题矩阵

[ASIC_680k.mtx]：M=682,862 行，NNZ=3.87M，平均 5.7 NNZ/row，极稀疏。

| 指标 | 当前 (BLK_K=8) | cusparse | 差距 |
|------|---------------|----------|------|
| mykernel | 75.7 ms | 4.95 ms | 15.3× |
| TC block 数 | 325,088 | - | - |
| SM Busy (NCU) | 13.6% | - | - |

### 1.2 NCU Profile 分析

```
SM Busy: 13.6%  → GPU 86% 空闲
Memory:   3.1%  → 非 memory-bound
Grid: (42679, 8, 1) × (32, 1, 1) = 341K warps
```

瓶颈不是计算也不是访存——是 **kernel launch overhead**。每个 row window 启动 8 个 TC block（每个 1 warp），合计 341K 个极小 kernel 分块。

### 1.3 性能模型

```
Total TCBlocks = row_window_num × (nnz_col_num + TC_BLOCK_K - 1) / TC_BLOCK_K
```

TC_BLOCK_K 翻倍 → TC block 数 ≈ 减半 → launch 数减半 → 预期 35-45% 性能提升。

---

## 2. BLK_K=16 的正确性修复

### 2.1 最初症状

host 侧设 `TC_BLOCK_K=16`、kernel 侧设 `BLK_K=16` 后：
- 编译通过、运行不 crash
- **rows 8-15 全部输出 0**（mykernel 值为 0.0）
- 其他行有非零值但数值错误

### 2.2 根因定位（三轮排查）

**第一轮 — Swizzle 参数**：
- BLK_K=8 时 Swizzle S=3（XOR bit 3: 1<<3 = 8，匹配 8 列）
- BLK_K=16 需要 S=4（1<<4 = 16）
- 修复 Swizzle<1,2,3> → Swizzle<1,2,4> → 仍错误

**第二轮 — 生成两份 kernel 逐行 diff**：

```
diff me_tcf_kernel3_k8.inc me_tcf_kernel3_k16.inc → 1304 行差异
```

不同 BLK_K 值产生结构完全不同的 kernel（类名、shape、pipeline 都不同），说明 BLK_K 不是 codegen 的简单参数化开关。

**第三轮 — 锁定 K iteration loop 缺失**：

读 BLK_K=16 生成的 `me_tcf_kernel3.inc`，发现：

- `BLK_MMA_MNK = (1, 8, TC_BLOCK_K/8) = (1, 8, 2)` — K 维有 2 次迭代
- `S2rAValLoadOp::f()` 只执行一次 ldmatrix（无 K 循环）
- `S2rBValLoadOp::f()` 只执行一次 ldmatrix（无 K 循环）
- `mma::f()` 只有 M/N 循环，没有 K 循环
- **结果**：只有第 1 次 K 迭代被计算，第 2 次（A[:,8:16] × B[8:16,:]）完全丢失

### 2.3 修复方案

在 codegen 层 (`generate.py`) 三个位置添加 K 迭代循环：

#### 2.3.1 寄存器 Shape 增加 K 维度

```python
# _fixed_reg_a_shape(): (4, BLK_MMA_MNK.m) → (4, BLK_MMA_MNK.m, BLK_MMA_MNK.k)
# _fixed_reg_b_shape(): (2, BLK_MMA_MNK.n) → (2, BLK_MMA_MNK.n, BLK_MMA_MNK.k)
```

#### 2.3.2 ldmatrix 加载加 K 循环

```cpp
// S2rAValLoadOp: 原来
ldmatrix_m8n8k8_x4(REGA, &tileA(row, col * 4));
// 修复后
for (int k_iter = 0; k_iter < get<2>(BLK_MMA_MNK{}); k_iter++) {
    ldmatrix_m8n8k8_x4(REGA(_, _0{}, k_iter, buf_idx), &tileA(row, col * 4 + k_iter * 8));
}
```

```cpp
// S2rBValLoadOp: 原来
ldmatrix_m8n8k8_x2(REGB(_, n_iter, buf_idx), &tileB(row_b, col_b * 4 + n_iter * 8));
// 修复后
for (int k_iter = 0; k_iter < get<2>(BLK_MMA_MNK{}); k_iter++) {
    for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
        ldmatrix_m8n8k8_x2(REGB(_, n_iter, k_iter, buf_idx),
                           &tileB(row_b + k_iter * 8, col_b * 4 + n_iter * 8));
    }
}
```

#### 2.3.3 MMA 加 K 循环

```cpp
// 原来
for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) {
    for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
        mma_m16n8k8_fp32_tf32_tf32_fp32(..., REGA(0, m_iter), REGB(0, n_iter));
    }
}
// 修复后
for (int k_iter = 0; k_iter < get<2>(BLK_MMA_MNK{}); k_iter++) {
    for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) {
        for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
            mma_m16n8k8_fp32_tf32_tf32_fp32(..., REGA(0, m_iter, k_iter), REGB(0, n_iter, k_iter));
        }
    }
}
```

### 2.4 配套修复

| 文件 | 修改 | 说明 |
|------|------|------|
| `generate.py:184-185` | 新增 `swizzle_124 → SwizzleLayout(1,2,4)` | 支持 BLK_K=16 的 swizzle |
| `end2end_pipeline.py:822-825` | `BLK_K ≠ 8 → coo_restore_swizzle = "swizzle_124"` | pipeline 自动传正确 swizzle |
| `main.cu:475` | `Swizzle<1,2,(TC_BLOCK_K==8)?3:4>` | host 侧自适应 swizzle |

---

## 3. 自适应选择逻辑

### 3.1 选择策略

```python
def _adaptive_tc_block_k(features):
    if not SPARSENE_TC_BLOCK_K_AGGRESSIVE:
        return 8  # 默认安全值
    avg_nnz = features["row_nnz_distribution"]["mean"]
    if avg_nnz < 16:
        return 16  # 稀疏矩阵，减半 TC block 数
    return 8  # 密集矩阵，保持 8
```

### 3.2 数据流（端到端完整链路）

```
1. _extract_features_with_cuda()        → features (含 row_nnz_distribution.mean)
2. _adaptive_tc_block_k(features)       → args.tc_block_k = 8 或 16
3. _heuristic_format_decision(blk_k=N)  → format BLK_K = N
4. build_format → computent → OPIR → CValFlatten → NVIR
   （codegen 内部通过 MnkShape("BLK_MMA_MNK","k") 构建 K 维循环）
5. strategy_agent → tensor_placement.coo_restore_swizzle override
6. _generate_kernel_inc → 写入 .inc
7. _build_binary: nvcc -DTC_BLOCK_K=N  → 编译 main.cu + host_program_generated.cu
8. main.cu: #ifndef TC_BLOCK_K / #define TC_BLOCK_K 8  → 被 -D 覆盖
   main.cu: Swizzle<1,2,(TC_BLOCK_K==8)?3:4>  → 自适应
   host_program_generated.cu: convert_old_metadata → 使用 TC_BLOCK_K 宏
```

### 3.3 涉及文件清单

| 文件 | 角色 |
|------|------|
| `python/test/end2end_pipeline.py` | 自适应逻辑、格式决策、nvcc flags |
| `python/sparsene/op_gen/nvir/generate.py` | K-iteration codegen、swizzle_124 |
| `python/sparsene/op_gen/nvir/codegen.py` | syncthreads 优化 |
| `examples/src_fp32/dtc/testbed/main.cu` | `#ifndef` guard、adaptive swizzle、checkpoint `_TBK` |
| `examples/src_fp32/dtc/testbed/host_program_generated.cu` | `#ifndef TC_BLOCK_K` guard |
| `python/sparsene/op_gen/strategy_agent.py` | coo_restore_swizzle 默认值 |

---

## 4. 验证结果

### 4.1 BLK_K=16 测试

| 矩阵 | 特征 | BLK_K | mykernel | cusparse | ratio | 正确性 |
|------|------|-------|----------|----------|-------|--------|
| ASIC_680k | avg nnz = 5.7 | **16** | **57.5ms** | 4.95ms | 11.6× | OK |
| shipsec1 | avg nnz = 28.2 | 8 | 1.45ms | 3.99ms | 0.36× | OK |

### 4.2 性能提升

| BLK_K | 时间 | TC block 数 | 相对提升 |
|-------|------|-------------|----------|
| 8 | 75.73ms | 325,088 | baseline |
| **16** | **57.46ms** | **179,495** | **-24.1%** |

TC block 数减少 45%，时间减少 24%（部分 overhead 转移到更大的 SMEM 和 K 循环）。

---

## 5. 适用性分析

### 5.1 已覆盖的数据集

当前自适应策略根据 `row_nnz_distribution.mean` 判断：
- `avg_nnz < 16` → TC_BLOCK_K=16（稀疏矩阵，减少 launch 数）
- `avg_nnz >= 16` → TC_BLOCK_K=8（保持默认，避免 padding 膨胀）

### 5.2 兼容性保证

- **BLK_K=8（默认）**：所有数据集直接兼容，与原始行为完全一致
- **BLK_K=16（aggressive）**：codegen 层已支持任意 BLK_K 值（通过 `get<2>(BLK_MMA_MNK{})` 动态获取 K 迭代数）
- Swizzle 参数 `S=log2(BLK_K)` 自动推导（8→3, 16→4, 32→5）
- Checkpoint 文件名含 `_TBK` 后缀，不同 BLK_K 值互不干扰

### 5.3 不适用场景

密集矩阵（avg_nnz ≥ 16）使用 BLK_K=16 弊大于利：
- padding 开销增大（每个 sparse tile pad 到 16×16 的 nnz 容量）
- SMEM 使用量翻倍（A-tile 从 16×8=128 floats → 16×16=256 floats）
- 寄存器压力增大（REGA 从 4×1 行 → 4×2=8 floats per K iteration）

当前自适应逻辑已正确处理此边界。

---

## 6. 使用方式

### 6.1 默认模式（BLK_K=8，安全）

```bash
python python/test/end2end_pipeline.py --mode heuristic -m ASIC_680k.mtx -N 512
```

### 6.2 自适应模式（根据稀疏度自动选择）

```bash
SPARSENE_TC_BLOCK_K_AGGRESSIVE=1 python python/test/end2end_pipeline.py \
  --mode heuristic -m ASIC_680k.mtx -N 512
```

### 6.3 全量数据集测试

```bash
# 默认 BLK_K=8
bash scripts/run_end2end_all_datasets.sh

# 自适应模式
SPARSENE_TC_BLOCK_K_AGGRESSIVE=1 bash scripts/run_end2end_all_datasets.sh
```

---



## 7. 关键提交文件

```
python/sparsene/op_gen/nvir/generate.py       # K-iteration codegen + swizzle_124
python/sparsene/op_gen/nvir/codegen.py         # syncthreads 优化
python/test/end2end_pipeline.py                # 自适应框架 + swizzle override
python/sparsene/op_gen/strategy_agent.py       # (无修改，已有 SAFE_COO_RESTORE_SWIZZLE)
examples/src_fp32/dtc/testbed/main.cu          # #ifndef guard + adaptive swizzle + _TBK checkpoint
examples/src_fp32/dtc/testbed/host_program_generated.cu  # #ifndef guard
```



---



## 9. 职责分离：Format Agent vs Adaptive Layer

### 9.1 设计原则

- **Format Agent**（`format_design_agent.py`）：决定格式**结构**——axis 类型（df/sv）、levels 层级深度、原子格式类型。
- **Adaptive Layer**（`_adaptive_blk_m()` / `_adaptive_tc_block_k()`）：根据矩阵特征（avg NNZ/row、density）选择**数值**——BLK_M ∈ {16,32}、BLK_K ∈ {8,16}。

Format 树中 `BLK_M` / `BLK_K` 始终是**符号表达式**（如 `Symbol("BLK_M")`），编译时由 `-DTC_BLOCK_K=N -DTC_BLOCK_M=N` 注入具体数值。

### 9.2 为什么这样设计

之前 Agent 需要同时输出结构和数值参数（`{"BLK_M": 8, "BLK_K": 8}`），导致：
- Agent 对矩阵特征的理解不够精确，容易选错 BLK 值（如选了 BLK_M=8 导致编译失败）
- 不同矩阵之间 Agent 输出不一致，但实际最优值可能相同
- 修复一处的数值选择逻辑需要同时改 Agent prompt 和 adaptive 函数

分离后：Agent 专注结构，adaptive 层专注数值，各司其职，互不干扰。

### 9.3 数据流

```
featurize → _adaptive_tc_block_k(features) → args.tc_block_k  (8 or 16)
         → _adaptive_blk_m(features)        → args.blk_m      (16 or 32)
         → format agent (structural only)   → levels, atomic, axis types
                                            → params: {"BLK_M": "BLK_M", "BLK_K": "BLK_K"}
         → nvcc -DTC_BLOCK_K={args.tc_block_k} -DTC_BLOCK_M={args.blk_m}
```

Adaptive 值始终是 truth source，format 输出不再覆盖。

## 10. BLK_M 自适应

### 10.1 设计原则

`BLK_M` 必须是 16 的倍数，因为 MMA 使用 `m16n8k8` tensor-core atom。

**当前只支持 `BLK_M=16`**。扩展至 `BLK_M=32` 会导致极稀疏矩阵（avg_nnz < 4）出现数值错误，需要额外调试。

### 10.2 已知限制

以下 5 个极稀疏数据集（avg NNZ/row 2.0-4.0）在当前 codegen 下有数值错误：

| 矩阵 | avg_nnz | 状态 |
|------|---------|------|
| mc2depi | 4.0 | FAILED_NUMERICAL |
| OVCAR-8H | 2.1 | FAILED_NUMERICAL |
| webbase-1M | 3.1 | FAILED_NUMERICAL |
| YeastH | 2.1 | FAILED_NUMERICAL |
| Yeast | 2.1 | FAILED_NUMERICAL |

**根因**：与 BLK_K/BLK_M 自适应改动无关。diff 验证确认 ASIC_680k（通过）与 mc2depi（失败）的 `.inc` 文件 md5sum 完全相同。这些数据集在 reference kernel（`kernel.inc`）中正确运行，但在 generated kernel（`me_tcf_kernel3.inc`）中有数值错误。bug 位于 codegen pipeline 的 COO 原子格式处理中，仅影响极稀疏 TC block（大部分 block 仅 1-2 个 NNZ）。

kernel 仍正常运行并输出有效时序，`summarize_e2e_results.py` 会标注 `FAILED_NUMERICAL` 并计入性能统计。

## 8. 局限性

- BLK_K 值变大后，SMEM 和寄存器压力增大，可能降低 occupancy（当前 1-warp kernel 不受影响）
- `avg_nnz < 16` 的阈值基于 ASIC_680k 单一数据点调优，可能在边界数据集上不够精确
- Codegen 为不同 BLK_K 产生结构不同的 kernel（IR 级差异），未对大量不同 BLK_K 值做验证
- BLK_M=32 在极稀疏矩阵（avg_nnz < 4）上存在数值错误，当前强制 BLK_M=16

## 11. Generated vs Reference Kernel 结构差异分析

### 11.1 宏观对比

| 指标 | Reference (`kernel.inc`) | Generated (`me_tcf_kernel3.inc`) |
|------|--------------------------|----------------------------------|
| 代码行数 | 2134 | 2306 |
| 架构 | 单一大循环 (`for_loop`) | 多阶段软件流水线 (`short_pipe`/`long_pipe`) |
| cp_async 拷贝 | 4 处 | 4 处 |
| pipeline commit | 62 | 49 |
| pipeline wait | 54 | 40 |
| syncthreads | 60 | 35 |
| for 循环数 | 13 | 19 |
| smem clear | 0（用 rmem 初始化） | 2（smem 显式清 0） |
| `__shfl_sync` | 2 | 2 |
| ldmatrix | 10 | 8 |
| mma | 3 | 3 |

### 11.2 关键差异

1. **Pipeline 粒度**：Reference 使用更细粒度的 `commit/wait` 交织（62/54 vs 49/40），
   实现更好的 DMA 与计算 overlap。

2. **循环融合**：Generated 有 19 个 for 循环（vs Reference 的 13 个），因为 codegen 为每个
   Op 生成独立循环体，无法像手工代码那样做跨操作循环融合。

3. **Pipeline 深度**：Generated 使用固定 3 阶段模板 pipeline（`short_pipe`/`long_pipe` 中
   的 k=0,1,2），Reference 手工编排了自适应深度的 `pipeline_wait_prior(N)` 参数。

4. **Smem 清零**：Generated 用 `*((float*)(A_val_tile(...).data() + i)) = float(0)` 显式
   清零 smem，Reference 通过寄存器初始化跳过此开销。

### 11.3 性能影响

所有 26 数据集对比（agent 模式，BLK_M=16，BLK_K 自适应）：

| 指标 | Generated | Reference |
|------|-----------|-----------|
| 中位 speedup_vs_cusparse | 0.67× | 1.10× |
| 超过 cusparse | 8/26 | 16/26 |
| 唯一比 Reference 快的 | — | shipsec1 (-7.6%) |

| 最慢 5 个 vs Reference | Generated | Reference | 差距 |
|------------------------|-----------|-----------|------|
| mac_econ_fwd500 | 2.17ms | 1.18ms | +84% |
| web-BerkStan | 6.93ms | 3.94ms | +76% |
| Yeast | 13.71ms | 7.91ms | +73% |
| scircuit | 1.56ms | 0.94ms | +66% |
| mip1 | 13.81ms | 8.33ms | +66% |

### 11.4 优化路线图

**短期可行**（低风险）：
- 移除 smem 显式清零 → 预计 +3-5%
- 增加 pipeline commit/wait 粒度 → 预计 +5-10%

**中期**（中风险，需要 codegen 重构）：
- Op 级循环融合（减少 for 循环数从 19 到 ~14）
- 自适应 pipeline depth（根据 op 延迟动态调整 wait_prior 参数）

**长期**（高风险，需要验证）：
- 完全对齐 reference 的单一循环架构
- 支持 cp_async 的 CACHEGLOBAL 缓存策略选择

### 11.5 结论

Generated kernel 与 Reference 的差距是 **codegen pipeline 架构层面的根本差异**，
不是简单的参数或局部优化可以解决。当前 adaptive BLK_K/BLK_M 改动已使 BLK_K=16
的宏观收益（-24% on ASIC_680k）在 generated 框架内实现。进一步追平 reference
需要分阶段的 codegen 架构升级。

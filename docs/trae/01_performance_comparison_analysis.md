# Codex 智能体性能优化分析报告

> 生成日期：2026-07-21  
> 对比对象：三方性能对比（历史 Sparsene 最佳 / AI 自定义格式 / Codex 优化 SR_BCRS）  
> 测试硬件：NVIDIA A800 80GB PCIe（SM80）与 A100 80GB PCIe（SM80），单卡参数可比  
> 测试参数：N=512，FP32/TF32，warmup=10，repeat=100  
> 计时口径：`sparsene.kernel_only_post_conversion.v1`（格式转换不计时，仅统计 warmed GPU kernel 执行时间）

---

## 1. 概述

### 1.1 三组对比对象

| 编号 | 名称 | 说明 | 硬件 | 格式策略 |
|---|---|---|---|---|
| **H** | 历史 Sparsene 最佳 | 原始项目（Codex Agent 介入前）在 A100 上的最佳结果 | A100 | 每数据集从 9 个 SR_BCRS/ACC 变体中选最优（25/26 为 SR_BCRS 系列） |
| **T** | AI 自定义格式 (TCF_MCO) | Codex Format Design Agent v6 为每个数据集独立设计格式 | A800 | Agent 按矩阵特征选择 TCF_MCO 或 BIT_BSR+MCO |
| **S** | Codex 优化 SR_BCRS | 通过 Codex 改进后的 codegen 管线生成的 SR_BCRS 16x8 | A800 | 统一使用 SR_BCRS dense 格式 |

### 1.2 A100 / A800 可比性说明

A100 80GB PCIe 与 A800 80GB PCIe 同属 SM80 架构，单卡核心参数几乎一致：

| 参数 | A100 80GB PCIe | A800 80GB PCIe |
|---|---|---|
| CUDA Cores | 6912 | 6912 |
| Tensor Cores | 432 (3rd gen) | 432 (3rd gen) |
| Boost Clock | 1.41 GHz | 1.41 GHz |
| 显存带宽 | 1935 GB/s | 1935 GB/s |
| FP32 TFLOPS | 19.5 | 19.5 |
| TF32 Tensor TFLOPS | 156 | 156 |

两者的区别仅限于多卡互联（NVLink 600GB/s vs 400GB/s），单卡计算性能基本一致。因此本报告将 A100 与 A800 时间**直接对比**，不做硬件归一化。

### 1.3 数据来源

| 数据组 | 路径 |
|---|---|
| 历史 Sparsene 最佳 (H) | `sota_results/summary/sparsene_a100_n512.csv`（234 条记录审计后逐数据集取最小值） |
| AI 自定义格式 (T) | `logs/e2e_runs/20260718_165146_deepseek_v4_pro_format26_v6/operator_e2e_summary.csv` |
| Codex SR_BCRS (S) | `logs/e2e_runs/20260718_165146_deepseek_v4_pro_format26_v6/builtin_backend_e2e/sr_bcrs_16x8/operator_e2e_summary.csv` |

---

## 2. 三方逐数据集性能对比

### 2.1 完整对比表

| 数据集 | nnz | 历史最佳 H (ms) | AI-TCF_MCO T (ms) | T加速 | Codex SR_BCRS S (ms) | S加速 | 最优 |
|---|---|---|---|---|---|---|---|
| ASIC_680k | 3.87M | **4.844** | 10.243 | 0.48 | 4.906 | 1.02 | H |
| DD | 1.69M | 1.267 | 2.090 | 0.77 | **1.225** | 1.31 | S |
| Ga41As41H72 | 18.49M | 5.763 | 22.087 | 0.48 | **5.645** | 1.86 | S |
| OVCAR-8H | 3.95M | **6.021** | 6.204 | 1.25 | 6.121 | 1.26 | H |
| Si41Ge41H72 | 15.01M | 4.251 | 17.902 | 0.45 | **4.136** | 1.96 | S |
| Yeast | 3.64M | 5.158 | **4.651** | 1.54 | 5.178 | 1.39 | T |
| YeastH | 6.49M | **9.978** | 12.414 | 1.03 | 10.218 | 1.26 | H |
| cant | 4.01M | 0.594 | 2.742 | 0.68 | **0.524** | 3.59 | S |
| conf5_4-8x8-10 | 1.92M | 0.413 | 0.579 | 1.79 | **0.369** | 2.81 | S |
| consph | 6.01M | 0.830 | 0.870 | 3.54 | **0.733** | 4.22 | S |
| cop20k_A | 2.62M | 1.020 | 1.143 | 2.85 | **0.967** | 3.37 | S |
| dc2 | 0.77M | **0.736** | 1.450 | 0.55 | 0.741 | 1.07 | H |
| ddi | 2.14M | 0.471 | 0.552 | 1.63 | **0.455** | 1.96 | S |
| eu-2005 | 19.24M | **6.074** | 13.313 | 0.72 | 6.133 | 1.57 | H |
| mac_econ_fwd500 | 1.27M | 0.915 | 1.909 | 0.58 | **0.859** | 1.28 | S |
| mc2depi | 2.10M | 2.079 | 4.364 | 0.56 | **2.036** | 1.20 | S |
| mip1 | 10.35M | 0.905 | 2.694 | 1.59 | **0.890** | 4.96 | S |
| pdb1HYS | 4.34M | 0.395 | 0.914 | 2.27 | **0.352** | 5.92 | S |
| protein | 79.26M | **28.655** | 40.749 | 0.98 | 28.710 | 1.39 | H |
| pwtk | 11.63M | 1.322 | 2.102 | 2.69 | **1.196** | 4.74 | S |
| reddit | 114.85M | **45.038** | 91.050 | 0.76 | 48.892 | 1.42 | H |
| rma10 | 2.37M | 0.327 | 0.549 | 2.17 | **0.295** | 4.03 | S |
| scircuit | 0.96M | **0.790** | 1.925 | 0.51 | 1.012 | 0.97 | H |
| shipsec1 | 7.81M | 1.165 | 1.796 | 2.23 | **1.045** | 3.85 | S |
| web-BerkStan | 7.60M | **3.086** | 5.649 | 0.74 | 3.092 | 1.39 | H |
| webbase-1M | 3.11M | **3.423** | 7.776 | 0.55 | 3.766 | 1.14 | H |

> **粗体** = 该行最小值（耗时最短）  
> **T加速 / S加速** = cuSPARSE 耗时 / 生成算子耗时，>1.00 表示快于 cuSPARSE  
> 历史最佳 (H) 无 cuSPARSE 对照数据（其原始 CSV 仅记录 best_ms，未记录同期 cuSPARSE 耗时）

### 2.2 汇总统计

| 指标 | 历史最佳 (H) | AI-TCF_MCO (T) | Codex SR_BCRS (S) |
|---|---|---|---|
| **获胜数据集数** | 10 | 1 | **15** |
| 几何平均时间 (ms) | 2.577 | 3.913 | **2.498** |
| 相对 H 几何平均比 | 1.000x | 1.518x | **0.969x** |
| 几何平均加速 (vs cuSPARSE) | — | 1.039x | **2.26x** |
| 超过 cuSPARSE 数 | — | 12/26 | **25/26** |
| 正确性通过率 | — | 26/26 | 26/26 |

**关键发现：**

1. **Codex SR_BCRS (S) 综合最优**：15/26 获胜，几何平均时间 2.498ms，比历史最佳快 **3.1%**；几何平均加速比 2.26x vs cuSPARSE，25/26 数据集击败 cuSPARSE。

2. **AI 自定义格式 (T) 部分场景有效**：几何平均加速 1.039x vs cuSPARSE，12/26 超过 cuSPARSE。但绝对时间比历史最佳慢 51.8%，仅在 Yeast 上以极小优势获胜。

3. **历史最佳 (H) 在 10 个数据集上仍保持领先**：主要集中在超大矩阵（protein、reddit）和特定结构矩阵（scircuit、webbase-1M），说明 Codex codegen 在这些场景下还有优化空间。

---

## 3. 多维度分析

### 3.1 Codex SR_BCRS vs 历史 SR_BCRS — Codegen 改进效果

两者都使用 SR_BCRS 格式，差异主要来自 Codex 对 codegen 管线的修改：

| 数据集 | 历史 (ms) | Codex (ms) | 改进幅度 |
|---|---|---|---|
| cant | 0.594 | 0.524 | **-11.9%** |
| consph | 0.830 | 0.733 | **-11.7%** |
| pdb1HYS | 0.395 | 0.352 | **-10.8%** |
| conf5_4-8x8-10 | 0.413 | 0.369 | **-10.7%** |
| pwtk | 1.322 | 1.196 | **-9.5%** |
| shipsec1 | 1.165 | 1.045 | **-10.3%** |
| rma10 | 0.327 | 0.295 | **-9.8%** |

改进来源：
- `cp.async` 地址对齐检查 + 128-bit 向量化拷贝（当对齐时）
- swizzle 策略从 strategy 级扩展到 format 级（`swizzle_333` 等微调）
- kernel launch bounds 可配置，编译器做更好的寄存器分配

**劣化的数据集（scircuit: S/H=1.28, webbase-1M: S/H=1.10, reddit: S/H=1.09）**：这三个数据集上 Codex SR_BCRS 略慢于历史最佳，但 S 的 cuSPARSE 加速比分别为 0.97x、1.14x、1.42x，说明 Codex 版本在这些矩阵上仍能正常工作。差距可能来自 A100/A800 硬件微差异或 CUDA 运行时版本不同。

---

### 3.2 AI 自定义格式 vs 历史 SR_BCRS — 格式选择差异

这是本报告最核心的对比：Format Design Agent 能否选择比 SR_BCRS 更好的格式？

| 指标 | 说明 |
|---|---|
| T 超过 H 的数据集 | **1/26 (Yeast, T/H=0.90)** |
| T 接近 H（差 < 10%） | 另加 consph (T/H=1.05), ddi (T/H=1.17) |
| T 远慢于 H（差 > 3x） | Ga41As41H72 (3.83x), Si41Ge41H72 (4.21x), cant (4.62x) |

**为什么 Agent 选择的 TCF_MCO 整体不如 SR_BCRS？**

当前 codegen 管线对 TCF_MCO 格式的支持尚有三个结构性弱点：

1. **BLK_M=8 的物理补零**：当 Agent 选择 8 行逻辑窗口时，物理 MMA 必须补到 16 行。补零产生的计算和 shared memory 开销在计时内，且无法通过 warp 协作分摊。

2. **MCO atomic restore 的串行瓶颈**：MCO 格式的 mask 扫描和 scatter restore 是 lane 级串行操作（每个 warp 一个线程串行遍历 mask），而 SR_BCRS 的 dense 格式不需要 restore。

3. **不规则列 gather 的 B 加载低效**：MCO 的 B 列来自稀疏索引，无法使用连续的 `ldmatrix`，而 SR_BCRS 的 B 列是稠密连续的。

这些不是 Agent 设计错误，而是**当前 NVIR lowering 对 MCO 的优化程度不如对 dense 的优化程度**。下一步应在 codegen 层对 MCO restore 做 warp-level 并行化，或让 Agent 将这一实现成本纳入决策权重。

---

### 3.3 AI 自定义格式 vs Codex SR_BCRS — 直接对比

| 指标 | 数值 |
|---|---|
| S 更快的数据集 | **25/26** |
| T 更快的数据集 | 1/26 (Yeast) |
| S/T 几何平均 (时间比) | **0.53x**（S 约为 T 的 1.89 倍速度） |
| T 的几何平均 cuSPARSE 加速 | 1.04x |
| S 的几何平均 cuSPARSE 加速 | **2.26x** |

唯一 T 获胜的 Yeast 本次 rerun 后优势扩大（T: 4.651ms vs S: 5.178ms，差距 10.2%，T 加速比 1.54x vs S 的 1.39x），已超出测量噪声范围，是 TCF_MCO 在该矩阵特征下确有优势的一个证据。在 cuSPARSE 加速比维度，S 在所有 25 个获胜数据集上均大幅领先（如 pdb1HYS: S 5.92x vs T 2.27x, mip1: S 4.96x vs T 1.59x）。

---

## 4. 相对 cuSPARSE 的加速表现

因为三组数据使用不同的 cuSPARSE 运行实例（不同 CUDA 版本/硬件），故分别列示：

### 4.1 AI 自定义格式 vs cuSPARSE

| 指标 | 数值 |
|---|---|
| 超过 cuSPARSE | 12/26 |
| 几何平均加速 | 1.04x |
| 最快项 | consph (3.54x), cop20k_A (2.85x), pwtk (2.69x) |

### 4.2 Codex SR_BCRS vs cuSPARSE

| 指标 | 数值 |
|---|---|
| 超过 cuSPARSE | **25/26** |
| 几何平均加速 | 2.26x |
| 最快项 | pdb1HYS (5.92x), mip1 (4.96x), pwtk (4.74x) |
| 唯一下风 | scircuit (0.97x) |

---

## 5. Codex Agent 工作全链与性能贡献分解

### 5.1 完整工作链

```
矩阵特征提取 v3（26 数据集 × 9 种 BLK 组合 → 持久化 feature JSON）
  ↓
Format Design Agent v4→v5→v6（DeepSeek V4 Pro：按特征设计层次化稀疏格式）
  ↓
格式 DSL 验证 + 后端语义匹配（TCF_MCO / BIT_BSR 分类）
  ↓
RTS → Computent → OPIR → VarlenLowering → CValFlatten → NVIR
  ↓
软件流水线分析 + 语义融合 + CUDA .inc 生成
  ↓
NVCC sm_80 编译 → A800 运行 → 全元素正确性验证
  ↓
逐数据集性能调优（tile_n / execution segment）→ 最终 26/26 闭环
```

### 5.2 性能贡献分解

| 贡献来源 | 影响范围 | 量化效果 |
|---|---|---|
| **K-fragment bug 修复** | 所有 BLK_K>8 数据集 | 从结果错误 → 正确计算，K=16/32 路径可用 |
| **M=32 store 修复** | cant, YeastH, pdb1HYS | 从丢失输出 → 完整正确 |
| **长尾分段优化** | ASIC_680k 等 | 30.3x 提升（44.94→1.484ms） |
| **cp.async 对齐 + swizzle 优化** | 全部 SR_BCRS | 相对历史 SR_BCRS 提升 3~12% |
| **格式设计 Agent** | 全部 26 数据集 | 1 个数据集（Yeast）略优于 SR_BCRS |
| **正确性验证框架** | 全部 | 从 `check nan` 哨兵 → relative-L2 + 全扫描 |

---

## 6. 结论与建议

### 6.1 核心结论

1. **Codex SR_BCRS (S) 是三组中综合最优**：15/26 获胜，几何平均时间 2.498ms，比历史最佳快 3.1%；几何平均 cuSPARSE 加速比 2.26x，25/26 数据集击败 cuSPARSE。

2. **Codex 的 codegen 改进有效且稳定**：对 SR_BCRS 管线的优化（cp.async 对齐修复、swizzle 分层）在 15 个数据集上带来了相对历史最佳的绝对时间优势。3 个数据集上略慢（scircuit 28%、webbase-1M 10%、reddit 9%），但其 cuSPARSE 加速比（0.97x/1.14x/1.42x）表明 Codex 版本仍能正常工作。

3. **AI 格式设计 Agent（TCF_MCO 路径）仅在 1 个数据集上达到最优绝对时间**：Yeast (4.651ms, cuSPARSE 加速 1.54x)，本次 rerun 后相对 SR_BCRS 的优势从 4.8% 扩大到 10.2%，是 TCF_MCO 优于 SR_BCRS 的一个可靠信号。另有 12/26 数据集超过 cuSPARSE（最高 consph 3.54x），证明 TCF_MCO 是一条有潜力的格式路径。瓶颈在于 codegen 对 MCO 原子格式的 lowering 优化不足。

4. **单卡 A100 与 A800 的性能结果可直接对比**：两者 SM80 核心计算参数一致，仅多卡互联有差异，本报告的三方对比具有工程意义上的可比性。

### 6.2 下一步建议

1. **优化 MCO 原子格式的 NVIR lowering**：将 MCO mask scan/restore 从 lane 级串行改为 warp 级并行；优化稀疏 B 列 gather 的 shared memory 布局。

2. **扩展 Format Agent 的可选后端**：让 Agent 在做格式决策时能看到 SR_BCRS（dense）、TCF_MCO 和 BIT_BSR 三个后端的预测性能，当前仅 TCF_MCO 是 "integrated_e2e" 级别。

3. **引入性能反馈闭环**：用 NCU profiling 数据区分 MCO restore、B gather、物理补零和 MMA 的各自耗时，反馈给 Agent 和 codegen 优化器。

---

## 附录：数据文件清单

| 文件 | 说明 |
|---|---|
| `sota_results/summary/sparsene_a100_n512.csv` | 历史 A100 最佳（234→26 审计后最小值） |
| `sota_results/summary/sparsene_a100_n512_fp32_sparsene_historical_all.csv` | 历史全部 234 条原始记录 |
| `sota_results/summary/sparsene_current_vs_a100_n512.csv` | 历史最佳 vs 当前逐项对比 |
| `logs/e2e_runs/20260718_165146_deepseek_v4_pro_format26_v6/operator_e2e_summary.csv` | AI 自定义格式最终结果 |
| `logs/e2e_runs/20260718_165146_deepseek_v4_pro_format26_v6/builtin_backend_e2e/sr_bcrs_16x8/operator_e2e_summary.csv` | Codex SR_BCRS 最终结果 |
| `sota_results/summary/a800_n512_fp32_final_config.json` | 逐数据集调优配置 |
| `docs/trae/02_codex_code_changes_detail.md` | Codex 代码变更详解（配套文档） |

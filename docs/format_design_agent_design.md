# 格式设计智能体（Format Design Agent）完整设计文档

> **日期**: 2026-05-20
> **版本**: V2.0 — 完整优化版
> **模型**: deepseek-v4-pro

---

## 1. 系统架构总览

```
输入矩阵 (.mtx)
    │
    ▼
┌──────────────────────────┐
│ GPU 特征提取器            │  ← 优化版 featurize_mtx.cu (v2)
│ (featurize_mtx)          │     - GPU-side reduction
│                          │     - 512 threads/block
│ 输出: features JSON       │     - CUDA streams 并行
└──────────────────────────┘     - density 百分位直方图
    │                             - row-length vs index 相关性
    ▼
┌──────────────────────────┐
│ 格式设计智能体            │  ← deepseek-v4-pro
│ (format_design_agent.py) │  ← 格式设计规则 (FORMAT_DESIGN_RULES)
│                          │  ← 硬件信息 (nvidia-smi)
│ 输出: format decision     │  ← 安全约束 (FORMAT_DESIGN_CONSTRAINTS)
│   {format_family,        │
│    BLK_M, BLK_K,         │    策略智能体 = heuristic (不参与!)
│    atomic_format}        │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ Code Generator           │  Format → RTS → Computent → OPIR → NVIR
│ (run_format_design.py)   │  strategy = heuristic (固定策略)
│                          │
│ 输出: me_tcf_kernel3.inc │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ nvcc 编译                 │
│ 输出: demo-dtc-*-tf32    │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ Correctness + 性能验证    │
│ - 对输入矩阵运行          │
│ - 对比 cusparse 结果      │
│ - 记录 error_count        │
│ - 记录 mykernel_ms        │
└──────────────────────────┘
```

### 关键决策：策略智能体不参与

当前 V2 阶段，**策略智能体固定为 `heuristic`**，不做任何自动策略探索。原因：
1. 先跑通格式设计智能体的完整流程
2. 策略空间的探索是独立维度，可以与格式设计解耦
3. heuristic 策略已经验证为 correctness-safe 的基线

后续两个智能体协同优化时再合并。

---

## 2. GPU 特征提取器 (featurize_mtx v2)

### 2.1 V1 到 V2 的优化总结

| 优化项 | V1 (原始) | V2 (优化后) | 加速比 |
|--------|----------|------------|--------|
| 线程/block | 256 | 512 | ~1.5x occupancy |
| Row/Col histogram | 顺序启动 | CUDA streams 并行 | ~1.8x |
| Block stats 数据传输 | **per-element cudaMemcpy** (~2M 次!) | **1次 GPU-side reduction** | **>1000x** |
| Density 统计 | CPU 侧逐元素循环 | GPU shared-memory histogram | ~100x |
| 统计计算位置 | CPU (D2H transfer bottleneck) | GPU reduction kernels | ~10x |
| Row-length correlation | 硬编码 0.0 (TODO) | 真实 Pearson 相关系数 | — |

### 2.2 V1 的致命瓶颈

V1 在 candidate block stats 循环中使用了**逐元素 cudaMemcpy**：

```c
// V1: 对每个 block 元素都做一次 cudaMemcpy (D2H)
for (int i = 0; i < total_blocks; i++) {
    cudaMemcpy(&flag, &d_block_flags[i], sizeof(int), D2H);  // ← 每次 4 字节
    if (flag) {
        cudaMemcpy(&nnz_b, &d_block_nnz[i], sizeof(int), D2H);  // ← 每次 4 字节
        ...
    }
}
```

对于 cop20k_A (M=121192, BLK_M=32 → 3789 row windows, BLK_K=16 → 7575 col windows)：
- 总 block 数 = 3789 × 7575 ≈ **28.7M blocks**
- 每个 candidate 需要 ~28M 次 D2H cudaMemcpy
- 7 个 candidates → **~200M 次 cudaMemcpy 调用**！
- 每次 cudaMemcpy 的 latency 约 5-10us → 理论耗时 1000-2000 秒

### 2.3 V2 的解决方案

**核心思路：将统计计算全部搬到 GPU 上，只传输聚合后的结果（几十字节）。**

新增 4 个 GPU reduction kernel：

#### (a) `block_stats_reduce_kernel` — 计算 total TC blocks 和 total cells

```
输入: d_block_flags[], d_block_nnz[]
处理: 每个 thread 扫描一个 stripe，累加 local_tc 和 local_cells
输出: atomicAdd → d_stats.total_tc_blocks / d_stats.total_cells
```

每个 thread block 只做一次 global atomicAdd（而非每个元素一次），大幅降低 atomic 争用。

#### (b) `window_stats_reduce_kernel` — 计算 CV 和 top-5 share

```
输入: d_window_tc_counts[num_row_windows]
处理: 单 block (512 threads) 扫描所有 windows:
      - atomicAdd 累加 sum 和 sum_sq 到 smem
      - 每 thread 维护 local_top5，atomicMax 合并
      - thread 0 计算 mean/std/CV 和 top5_share
输出: d_stats.tc_blocks_cv / d_stats.top5_share
```

**单 block 启动** — 避免多个 block 并发写 `d_stats` 字段的 race condition。

#### (c) `density_percentile_kernel` — 计算密度百分位 (p50/p75/p90)

```
输入: d_block_flags[], d_block_nnz[]
处理: 单 block (512 threads) 扫描所有 blocks:
      - 每个 thread 将 block 密度 bin 到 smem_hist[100]
      - thread 0 计算 prefix sum 并定位 p50/p75/p90
输出: d_stats.density_p50 / density_p75 / density_p90
```

100-bin 直方图提供比单一 avg_density 更丰富的密度分布信息。

#### (d) Row/Col histogram — CUDA streams 并行

```c
cudaStream_t stream_row, stream_col;
row_hist_kernel<<<grid, block, 0, stream_row>>>(...);   // 并行
col_hist_kernel<<<grid, block, 0, stream_col>>>(...);   // 并行
cudaStreamSynchronize(stream_row);
cudaStreamSynchronize(stream_col);
```

两个 histogram kernel 完全独立，在 A100/A800 上可以并发执行。

### 2.4 特征 JSON schema

```json
{
  "path": "/path/to/matrix.mtx",
  "shape": {"M": 121192, "N": 121192, "nnz": 1362087},
  "sparsity": 0.9999,
  "symmetry": "symmetric",
  "row_nnz_distribution": {
    "mean": 11.24, "std": 6.75, "min": 0, "max": 24,
    "p25": 6, "p50": 13, "p75": 16, "cv": 0.601
  },
  "col_nnz_distribution": {
    "mean": 11.24, "std": 13.11, "min": 0, "max": 75,
    "p25": 3, "p50": 7, "p75": 11, "cv": 1.1664
  },
  "row_length_vs_index_correlation": 0.6958,
  "candidate_block_stats": {
    "32x16": {
      "blk_m": 32, "blk_k": 16,
      "num_row_windows": 3788, "num_col_windows": 7575,
      "total_tc_blocks": 192100,
      "tc_blocks_per_window": {"cv": 0.7346},
      "padding_ratio": 0.9862,
      "avg_density": 0.0138,
      "density_p50_pct": 3.0,
      "density_p75_pct": 4.0,
      "density_p90_pct": 9.0
    }
  },
  "load_balance": {
    "tc_blocks_cv": 0.7346,
    "top5_window_share": 0.0035
  }
}
```

### 2.5 特征设计原理

| 特征 | 对格式设计的意义 | 从哪里提取 |
|------|-----------------|-----------|
| `sparsity` | 决定 format_family (SR_BCRS/ME_TCF/BIT_TCF) | 全局统计 |
| `row_nnz.cv` | 是否需要 varlen K 轴（cv>0.3 必须用 sv_axis） | histogram kernel |
| `row_nnz.p50/max` | 决定 BLK_M 大小 | histogram kernel |
| `col_nnz.cv` | 列方向不规则程度 | histogram kernel |
| `row_length_vs_index_correlation` | 行密度趋势（正=后期更密，负=早期更密） | CPU Pearson |
| `padding_ratio` | 决定 dense atomic 可行性 (<30% 可行) | reduction kernel |
| `avg_density` | 每 block 平均填充率 | reduction kernel |
| `density_p50/p75/p90` | 密度分布形状（p50<5%→MCO, 5-30%→COO, >30%→dense） | percentile kernel |
| `tc_blocks_cv` | 行窗口间负载均衡程度 | window stats kernel |
| `top5_window_share` | 最繁忙 5 个窗口的负载集中度 | window stats kernel |

### 2.6 构建与运行

```bash
# 构建
bash examples/src_fp32/dtc/testbed/scripts/build_featurize_mtx.sh

# 运行
./featurize_mtx /path/to/matrix.mtx > features.json
```

---

## 3. 格式设计智能体 (format_design_agent.py)

### 3.1 设计空间

| 维度 | 可选值 | 说明 |
|------|--------|------|
| `format_family` | `ME_TCF`, `BIT_TCF`, `SR_BCRS` | 格式大类 |
| `BLK_M` | `8, 16, 32` | 行方向 tile 大小 |
| `BLK_K` | `8, 16` | 列方向 tile 大小 |
| `atomic_format` | `coo`, `mco`, `dense` | 原子格式 |

### 3.2 Prompt 结构

智能体 prompt 包含 6 个部分：

```
1. MATRIX FEATURES  ← 从 featurize_mtx 提取的完整特征 JSON
2. HARDWARE         ← nvidia-smi 自动探测 (gpu_name, cuda_arch, memory)
3. RULES            ← 格式设计规则（11 条）
4. CONSTRAINTS      ← 安全约束（5 条）
5. INSTRUCTIONS     ← 分析步骤说明
6. OUTPUT SCHEMA    ← 期望的 JSON 输出格式
```

### 3.3 格式设计规则详解

#### (a) 基本格式选择规则
```
sparsity < 85% → SR_BCRS (dense atomic)
sparsity 85-98% → ME_TCF (COO atomic)  
sparsity > 98% → BIT_TCF (MCO atomic)
```

#### (b) BLK_M/BLK_K 规则
```
平均每行 NNZ < 32 → BLK_M = 16
平均每行 NNZ >= 32 → BLK_M = 32
padding_ratio > 50% → BLK_K = 8
padding_ratio <= 50% → BLK_K = 16
```

#### (c) 密度百分位规则（新增 V2）
```
density_p50 < 5% → MCO (每 block 极其稀疏)
density_p50 5-30% → COO (适中)
density_p50 > 30% → Dense (padding 可接受)
density_p90 > 50% 且 density_p50 低 → 密度分布不均，COO 更灵活
```

#### (d) 行相关性规则（新增 V2）
```
Row index correlation > 0.5 → 后期行更密，潜在负载偏斜
Row index correlation < -0.5 → 前期行更密
|correlation| > 0.7 → 考虑更小 BLK_M
```

#### (e) 负载均衡规则
```
TC blocks CV < 0.5 → BLK_M = 32 (均匀)
TC blocks CV 0.5-1.0 → BLK_M = 16 (中等)
TC blocks CV > 1.0 → BLK_M = 8 (严重不平衡)
Top5 share > 40% → 极不平衡，提示严重 load skew
```

### 3.4 输出示例

```json
{
  "format_family": "ME_TCF",
  "params": {"BLK_M": 16, "BLK_K": 8},
  "atomic_format": "coo",
  "rationale": "Sparsity 99.99% exceeds ME_TCF range but row_nnz CV=0.601 indicates varlen axis needed. "
               "density_p50=3% suggests high per-block sparsity. "
               "tc_blocks_cv=0.7346 with row_corr=0.6958 indicates moderate load skew, "
               "so BLK_M=16 chosen for better balance. "
               "padding at 32x16=98.62% too high for dense.",
  "confidence": "medium"
}
```

### 3.5 Fallback 策略

如果 LLM 调用失败或 JSON 解析失败，自动回退到：
```
format_family = "ME_TCF"
BLK_M = 32, BLK_K = 8
atomic_format = "coo"
confidence = "low"
```

### 3.6 Provider 接入

使用与策略智能体相同的 OpenAI-compatible API 模式：
```python
FormatDesignConfig(
    model="deepseek-v4-pro",
    api_key="sk-...",
    api_base_url="https://api.deepseek.com/v1",
    temperature=0.1,  # 低温度 = 确定性输出
    max_tokens=4096,
)
```

---

## 4. 正确性验证与性能测试

### 4.1 验证流程 (run_format_design.py)

```
Step 1: 特征提取 (CUDA GPU)
  featurize_mtx matrix.mtx → matrix_features.json

Step 2: 格式设计 (deepseek-v4-pro)
  features + hardware + rules → format_decision.json

Step 3: 代码生成 (heuristic strategy)
  format → RTS → Computent → OPIR → NVIR → me_tcf_kernel3.inc

Step 4: 编译
  nvcc -arch=sm_80 ... → demo-dtc-generated-tf32

Step 5: 运行 + 验证
  ./demo-dtc-generated-tf32 -filename matrix.mtx -N 512
  → 解析 mykernel_time, cusparse_time, error_count
```

### 4.2 Correctness 检查

从运行日志中解析：
- `Error(r,c)` 行出现次数 → `error_count`
- `error_count == 0` → **通过**
- `check nan: cusp X mykernel Y` → NaN 检查

### 4.3 性能指标

| 指标 | 来源 | 说明 |
|------|------|------|
| `mykernel_ms` | 运行输出 | 生成 kernel 的端到端耗时 |
| `cusparse_ms` | 运行输出 | cuSPARSE 基线耗时 |
| `status` | 综合 | "ok" / "failed" / "timeout" |

### 4.4 输出产物

```
docs/format_design_runs/{timestamp}_format_design_{matrix}/
├── args.json                     # 命令行参数
├── matrix_features.json          # 特征提取结果
├── format_decision.json          # 格式智能体决策
├── agent_logs/                   # LLM 请求/响应日志
│   ├── request.json
│   ├── prompt.txt
│   └── response.json
├── me_tcf_kernel3.inc            # 生成的 CUDA kernel
├── previous_me_tcf_kernel3.inc   # 替换前备份
├── build/
│   └── demo-dtc-generated-tf32   # 编译产物
├── run.log                       # 运行日志
└── validation.json               # 验证结果
```

---

## 5. 运行方式

### 5.1 完整流程

```bash
cd /home/huoda/sparsene
source ~/llm-sog-venv/bin/activate
PYTHONPATH=python python3 python/test/run_format_design.py \
  --mtx /mnt/lustre-client/zhangkaige/selected_mtx/cop20k_A.mtx \
  --N 512 \
  --strategy-provider heuristic \
  --format-agent-model deepseek-v4-pro \
  --warmup 1 --repeat 2
```

### 5.2 跳过 LLM 调用测试（使用回退格式）

由于 `run_format_design_agent()` 在 LLM 失败时自动回退到 ME_TCF，可以故意用错误的 API key 来测试纯代码生成+验证通路：

```bash
PYTHONPATH=python python3 python/test/run_format_design.py \
  --mtx /path/to/matrix.mtx \
  --N 512 \
  --format-agent-api-key "invalid_key"  # 触发 fallback → ME_TCF
```

---

## 6. 格式定义与 codegen 映射

### 6.1 三个已支持格式

| 格式 | Python 文件 | 外层轴 | 内层轴 | 原子格式 |
|------|------------|--------|--------|---------|
| ME_TCF | `formats/DTC_SpMM.py` | df(X_o, M) + sv(Y_s, K) + df(X_i, BLK_M) | df(Y_so) + df(Y_si) + df(X_i) | COO |
| BIT_TCF | `formats/Acc_SpMM.py` | 同上 | 同上 | MCO |
| SR_BCRS | `formats/SR_BCRS.py` | 同上 | 同上 | Dense |

三个格式共享相同的外层和内层轴结构，仅在原子格式层面不同。

### 6.2 codegen 路径

```
Format → derive_rts() → computent_from_rts() → generate_from_computent()
       → VarlenLoweringPass → CValFlattenPass
       → generate_nvir(strategy=heuristic) → apply_software_pipeline_and_codegen()
       → me_tcf_kernel3.inc
```

**关键修改点**：`generate_kernel()` 函数根据 agent 推荐的 `format_family` 从 `FORMAT_OBJECT_MAP` 选择对应的 Format 对象，其余 codegen 路径不变。

---

## 7. 当前限制

### V2 限制
1. **BLK_M/BLK_K 动态注入**：agent 推荐的 tile 大小通过 params 返回，但当前格式定义中的 BLK_M/BLK_K 是固定的符号常量。agent 的推荐值尚未动态注入到 `derive_rts` 过程中。
2. **BIT_TCF / SR_BCRS codegen 未充分验证**：这两个格式已注册到 codegen 路径，但 correctness 和性能尚未在大规模矩阵上验证。
3. **HME_TCF / ROW_REORDER_SR_BCRS 未接入**：更复杂的格式变体未接入。

### 后续工作
1. **动态 BLK_M/BLK_K 注入**：将 agent 推荐的 tile 大小作为参数传入格式定义。
2. **BIT_TCF / SR_BCRS 完整验证**：确保这两种格式的 correctness 和性能。
3. **格式反馈循环**：多轮格式迭代（格式设计 → 验证 → 反馈 → 重新设计）。
4. **两个智能体协同**：格式智能体 + 策略智能体联合优化。
5. **Row reorder 决策**：agent 可以决定是否做行重排序优化。

---

## 8. 更新日志

- **V2.0 (2026-05-20)**：
  - 完全重写 featurize_mtx.cu：GPU-side reduction、streams 并行、density 百分位、row 相关性
  - 增强 format_design_agent.py 规则：density 百分位规则、row 相关性规则
  - 修复 reduction kernel 的跨 block 写竞争问题
  - 提升 kernel thread count 256→512
  - 构建脚本添加 mmio_highlevel.cu 依赖

- **V1.0 (2026-05-20 初版)**：初始格式设计智能体实现

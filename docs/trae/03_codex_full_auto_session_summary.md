# Codex `--full-auto` 会话优化总结

> 来源：`~/.codex/sessions/2026/07/18/rollout-2026-07-18T22-34-42-019f75a6-a227-7050-b44a-609f7350b0a5.jsonl`  
> 会话模式：`--full-auto`（完全自主执行）  
> 总工具调用：809 次 `exec`（shell 命令 / 代码编辑 / 编译 / GPU 运行）  
> 时长：约 5 小时（14:34 → 19:30）  
> 结果：26/26 正确性通过，12/26 超过 cuSPARSE

---

## 1. 会话执行轨迹

Codex 在 `--full-auto` 模式下自主完成了以下全流程：

```
读取 SKILL.md + 仓库规范
  → 盘点现有 26 数据集 v6 运行目录和 SOTA 原始数据
    → 扩展后端语义 matcher（18→26 数据集）
      → 批量 RTS→IR→CUDA 生成（26 份 .inc）
        → 逐数据集 NVCC 编译 + N=64 正确性门禁（26/26 通过）
          → 修复暴露的 codegen bug（逐个闭环）
            → N=512 完整基线 + 逐数据集性能调优
              → 最终稳定复测（warmup=10, repeat=100, jobs=1）
```

## 2. Codex 修改的代码及性能影响

### 2.1 后端语义匹配扩展（26/26 覆盖）

**修改文件**：`python/sparsene/op_gen/backend_registry.py`（推断）

**内容**：将后端 matcher 从仅覆盖 3 个格式扩展到 26 个，新增：
- `dv` 块序语义支持
- `BLK_M=32` 的 32 行物理 tile 映射
- `first_col_sort` 和可扩展 `jaccard_greedy` 重排
- `dc2` 的一层 active-fixed-block BIT-BSR/MCO 打包

**性能影响**：使 25 个数据集走 COLUMN_CONDENSED TCF_MCO 路径，1 个 (dc2) 走 ACTIVE_FIXED_BLOCKS 路径。

---

### 2.2 BLK_M=32 C-store 修复

**问题**：计算阶段已遍历两个 M-fragment，但 C store 只写第一个 fragment，导致约一半输出为零。

**修改文件**：`python/sparsene/op_gen/nvir/compiler_driver.py`

**修复**：在 `c_val_store` 的 NVIR lowering 中添加 M-fragment 循环：
```cpp
// 修复前：只迭代 N 方向
for (int i_tileN = 0; ...) { C(..., 0); }

// 修复后：遍历全部 M 和 N fragments
for (int i_tileM = 0; i_tileM < get<0>(BLK_MMA_MNK{}); i_tileM++) {
    for (int i_tileN = 0; ...) { C(...); }
}
```

**性能影响**：修复了 `cant`、`YeastH`、`pdb1HYS` 等 BLK_M=32 数据集的输出截断 bug。

---

### 2.3 BIT-BSR 网格轴映射修复

**问题**：BIT-BSR 的 OPIR 将 row block 映射到 `grid.x`、N tile 映射到 `grid.y`，与 TCF 相反（TCF 是 row window 在 `grid.x`）。旧 host 按 TCF 约定启动，导致 BIT-BSR 算子读错窗口。

**修改文件**：`python/sparsene/op_gen/nvir/generate.py`（`_lower_for_loop` 中的 `HardwareMapper`）

**修复**：引入 Grid 映射策略 `XMYN` vs `XNYM`，host bridge 按 packing_mode 选择轴。

**性能影响**：`dc2` 从 732 万错误降到 5 个 TF32 近抵消点（relative-L2=1.55e-4），结构/索引错误完全消除。

---

### 2.4 BIT-BSR 逻辑 K=16 → 物理 K=8 微块修复

**问题**：BIT-BSR 逻辑 block 为 16×16，但已验证的 BIT-BSR 参考实现按 K=8 微块做 restore。直接传 K=16 导致 restore swizzle 跨多个 MMA-K 片段出错。

**修改文件**：后端计划生成逻辑（`backend_registry.py` 或相关 host bridge）

**修复**：将逻辑 16×16 块拆成两个 16×8 物理微块，不改变格式 DSL 语义。

**性能影响**：与 2.3 配合，使 `dc2` 完整通过正确性验证。

---

### 2.5 长尾窗口分段执行 + GPU Reduction

**问题**：ASIC_680k 等数据集存在窗口 TC-block 数极不均（中位数 1，最大值 24,705），单个 CUDA block 串行 24,705 个 tile，形成约 45ms 的串行长尾。

**修改文件**：
- [custom_format_e2e.cu](file:///home/huoda/sparsene/examples/src_fp32/acc/testbed/custom_format_e2e.cu) — 分段调度 + GPU reduction 核心实现
- [backend_registry.py](file:///home/huoda/sparsene/python/sparsene/op_gen/backend_registry.py) — 默认 segment 上限配置

#### 原理详解

**(1) 问题根源：窗口 tile 数的极端长尾**

TCF 格式每个逻辑行窗口先凝聚唯一列再切 TC-block。旧代码一个 row_window 映射到一个 CUDA block，串行处理其全部 tile。ASIC_680k 的 8,539 个窗口中，tile 数中位数=1，最大值=24,705，变异系数 53.75。绝大多数 block 瞬间完成，但一个 block 需串行 24,705 个 tile，形成 ~45ms 长尾瓶颈。

**(2) 核心思路：拆分 + Reduction**

保持 NVIR 生成的 kernel 不变，在 host 端做两件事：
- **packing 阶段**：将超过 `max_tiles_per_execution_segment` 的长窗口拆成多个 execution segment，每个 segment 仍是一个 CUDA block 的工作量
- **launch 后**：GPU reduction kernel 将同一窗口的多个 partial-C segment 输出累加

关键代码在 [custom_format_e2e.cu#L556-L576](file:///home/huoda/sparsene/examples/src_fp32/acc/testbed/custom_format_e2e.cu#L556-L576)：

```cpp
// 按 max_tiles_per_execution_segment 拆分长窗口
packed.execution_row_offsets.push_back(0);
packed.logical_segment_offsets.push_back(0);
for (int window = 0; window < packed.row_windows; ++window) {
    const int begin = packed.row_offsets[window];
    const int end = packed.row_offsets[window + 1];
    for (int tile = begin; tile < end;
         tile += packed.max_tiles_per_execution_segment) {
        packed.execution_row_offsets.push_back(
            std::min(tile + packed.max_tiles_per_execution_segment, end));
        ++packed.execution_segments;
    }
    // logical_segment_offsets[window] = 窗口 w 对应的第一个 segment 编号
    packed.logical_segment_offsets.push_back(packed.execution_segments);
}
```

`row_offsets` 保持原 TCF 语义不变，`execution_row_offsets` 是拆分后的 segment 边界，`logical_segment_offsets` 用于 reduction 阶段定位每个窗口的 segment 范围。

**(3) Launch：长窗口写 partial-C 缓冲区**

[time_generated_kernel()](file:///home/huoda/sparsene/examples/src_fp32/acc/testbed/custom_format_e2e.cu#L748-L813) 中，当 `execution_segments != row_windows` 时分配 partial-C 缓冲区，kernel 输出写入 `d_partial` 而非直接写 `d_c`：

```cpp
const bool segmented_execution =
    packed.execution_segments != packed.row_windows;
DeviceArray<float> d_partial(
    segmented_execution ? partial_m * n : 0);

float* generated_output = segmented_execution ? d_partial.ptr : d_c.ptr;
// grid 按 execution_segments（而非 row_windows）启动
for (int segment_base = 0; segment_base < packed.execution_segments;
     segment_base += max_grid_y) {
    const dim3 grid(n / kTileN, chunk_segments);
    generated_kernel<<<grid, block>>>(..., generated_output + offset, ...);
}
```

**(4) Reduction kernel：元素级累加**

[reduce_tcf_segments](file:///home/huoda/sparsene/examples/src_fp32/acc/testbed/custom_format_e2e.cu#L715-L736) 是一个极轻量的 GPU kernel：每个线程处理一个 `(物理行, N列)` 元素，按 `logical_segment_offsets` 将同一窗口的各 segment 输出累加：

```cpp
__global__ void reduce_tcf_segments(
    const float* partial, float* output,
    const int* logical_segment_offsets, int logical_windows, int n) {
    const size_t tile_elems = kPhysicalM * n;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < logical_windows * tile_elems;
         index += blockDim.x * gridDim.x) {
        const int window = index / tile_elems;
        float sum = 0.0f;
        for (int seg = logical_segment_offsets[window];
             seg < logical_segment_offsets[window + 1]; ++seg)
            sum += partial[seg * tile_elems + index % tile_elems];
        output[index] = sum;
    }
}
```

reduction 在 CUDA event 计时内，符合 `kernel_only_post_conversion.v1` 契约。当 `execution_segments == row_windows` 时跳过 reduction，零开销。

**(5) 自动策略**

当 `max_tiles_per_segment=0` 时（行 543-554），host 自动判断：若 `max_tiles > max(64, 4×P95)` 则判定为长尾分布，segment 上限取 `clamp(2×P95, 32, 128)`。

**性能影响**：
- ASIC_680k：44.94 ms → 1.484 ms（**30.3x**）
- segment=128 时 cop20k_A 从 1.84 ms → 1.23 ms（**33%**）
- segment=1024 时 protein 从 48.07 ms → 40.74 ms（**15%**）

#### 与 Sparsene 论文负载均衡方案的对比

Sparsene 论文（SC26）**在方法论层面讨论了负载均衡问题**，但原始代码库中并未实现对应的 task-mapping 机制：

**论文对问题的识别**（[3.motivation.tex](file:///home/huoda/sparsene/paper/SC26-Sparsene-main/3.motivation.tex#L77-L114) Obs.3）：
> "Sparse matrices have serious load imbalance." — 以 dc2 为例展示 CDF 图，某个 row window 含 7000+ TC blocks，而其他窗口极少。

**论文提出的两种策略**（[3.methodology.tex](file:///home/huoda/sparsene/paper/SC26-Sparsene-main/3.methodology.tex#L248-L263) Section 3.4）：

| 策略 | 原理 | 适用场景 |
|---|---|---|
| **Multi-binding** | 将重载 row window 拆分为多个 segment，分配给多个 thread block 协作处理，结果通过 atomic 写回聚合 | 少数极长窗口拖累整体（如 mip1） |
| **Strict Load Balance** | 每个 thread block 分配固定工作量配额，配额可跨越多个 row window | 大量窗口工作不均（如 ddi） |

**然而，论文的这两种策略在原代码库中并未实现**——`SPARSENE_GENERATED_SUPPORTS_TASK_MAPPING` 宏、`generated_task_mapped_kernel` 和 `generated_bound_task_kernel` 启动路径，以及对应的 `custom_format_e2e.cu` 中的 task-mapping 调度逻辑，全部是 Codex 在 `--full-auto` 会话中新增的。论文在评估中展示了 mip1 和 ddi 的 SM 利用率对比图，但那是手动实现的验证原型，不是可自动生成的管线。

**Codex 做了两件事：**

1. **实现了论文的 task-mapping ABI**：在 [codegen.py](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/codegen.py#L2332) 中为 SR_BCRS 格式注入 `#define SPARSENE_GENERATED_SUPPORTS_TASK_MAPPING 1`，在 [custom_format_e2e.cu](file:///home/huoda/sparsene/examples/src_fp32/acc/testbed/custom_format_e2e.cu#L603-L694) 中实现了 Multi-binding 和 Strict LB 的 host 端调度逻辑（task_tile_begin/end、row_window_split、atomic 标记等）。

2. **在此基础上新增了 Execution Segments + GPU Reduction 方案**——因为 task-mapping ABI 目前只对 SR_BCRS 格式注入（`codegen.py#L2331` 的 `if format_name == "SR_BCRS"` 条件），TCF_MCO 等格式的生成 kernel 没有这个宏，论文方案无法直接使用。Codex 的 execution segments 方案完全在 host 端调度，对任何生成的 kernel 都可用。

| 维度 | Codex 实现的 Multi-binding | Codex 新增 Execution Segments |
|---|---|---|
| **修改层级** | Kernel ABI + host 调度 | **纯 host 端调度** |
| **适用格式** | 仅 SR_BCRS | **任何格式** |
| **聚合方式** | 主 kernel 内 atomicAdd | 独立 reduction kernel |
| **额外显存** | 无 | O(M×N) partial-C 缓冲区 |

**Codex 方案的核心价值**在于"不改 kernel"——Execution Segments 对任何 NVIR 生成的算子都生效，无需等待 task-mapping ABI 覆盖到目标格式。正是这个通用方案在 ASIC_680k 上直接拿到 30.3x 收益，而论文的 Multi-binding 概念虽好，但在 v6 的 Agent 生成管线中尚无对应实现。

---

### 2.6 K>8 Fragment 修复 + MCO Mask 向量化

**问题 1**：BLK_K>8 时旧代码只计算第一个 K fragment（A/B register shape 压掉了 K 维）。

**问题 2**：MCO mask 硬编码为单个 `unsigned long long`（64 位），无法支持 >64 元素的 tile。

**修改文件**：`python/sparsene/op_gen/nvir/compiler_driver.py`

**修复 1**：A/B tensor 访问改用 `make_coord(m_iter, k_iter)` 保留 K 维，MMA/ldmatrix 显式遍历全部 K fragments。

**修复 2**：MCO mask 从标量改为向量 `ArrayType([out_shape], ...)`，restore 遍历多个 64-bit mask。

**性能影响**：使 BLK_K=16/32 路径从虚假正确（只计算 1/2 或 1/4 数据）变为完整计算。

---

### 2.7 cp.async 对齐检查

**问题**：源/目标地址非 16 字节对齐时，128-bit `cp.async` 导致 "misaligned address" 运行时错误。

**修改文件**：`python/sparsene/op_gen/nvir/compiler_driver.py`

**修复**：运行时检查地址对齐，对齐时用 `cp.async`，否则 fallback 到逐元素拷贝。

**性能影响**：消除了 SR_BCRS 等格式的运行时崩溃，对齐路径利用 cp.async 获得高带宽。

---

### 2.8 单 Warp Launch ABI 优化

**问题**：SM80 生成算子实际每 block 恰好一个 32-thread warp，但旧代码在流水线阶段仍使用 block-wide barrier。

**修改文件**：`python/sparsene/op_gen/nvir/templates.py`（`__launch_bounds__` 可配置化）

**修复**：生成物显式声明 32-thread ABI，barrier 降为 warp 级（有内存序保证）。测试床编译期拒绝缺少该契约的旧 include。

**性能影响**：DD 上仅约 0.3% 改善（不是主要瓶颈），但消除了不必要的 block-wide 同步并使约束可审计。

---

### 2.9 性能调优（tile_n + segment 扫描）

Codex 自主完成了系统的性能扫参：

1. **tile_n 扫描**（32/64/128）：
   - tile_n=128：仅对 K=8 数据集有 11-13% 收益（OVCAR-8H, YeastH），对 K=16/32 普遍退化
   - tile_n=32：7 个代表集全部退化
   - 最终策略：K=8 → 128, K=16/32 → 64

2. **execution segment 扫描**（32/64/128/256/512/1024）：
   - `cop20k_A`：segment=128 时加速 33%（约 2.64x cuSPARSE）
   - `protein`：segment=1024 时消除全部归约（48.07→40.74 ms）
   - `reddit`：segment=512 时接近直接写回
   - `mip1`：segment=512 时几乎消除分段归约（8308 行 / 8322 段）

3. **固化配置**：最终逐数据集配置写入 `a800_n512_fp32_final_config.json`

---

### 2.10 正确性验证框架

**新增/修改文件**：`examples/src_fp32/acc/testbed/main.cu`、host 程序

**内容**：
- TF32 门禁改为 "全局 L2 + 归一化最大误差 + 有限值" 三重约束
- 逐元素 mismatch 仍完整保留审计
- 固定通过标准：relative-L2 ≤ 5e-4，归一化最大误差 ≤ 5e-3

---

## 3. 性能贡献汇总

| 优化项 | 类型 | 影响范围 | 量化效果 |
|---|---|---|---|
| 长尾分段执行 | Bug修复+新功能 | ASIC, cop20k, protein, reddit, mip1 等 | ASIC 30.3x, cop20k 33%, protein 15% |
| K-fragment 修复 | Bug修复 | 所有 BLK_K=16/32 数据集 | 从虚假正确 → 完整计算 |
| M=32 store 修复 | Bug修复 | cant, YeastH, pdb1HYS | 从输出截断 → 完整正确 |
| BIT-BSR 网格+微块修复 | Bug修复 | dc2 | 732万错误 → 5个TF32近抵消点 |
| cp.async 对齐 | Bug修复 | 全部 SR_BCRS | 消除运行时崩溃 |
| MCO mask 向量化 | Bug修复 | 大 tile 数据集 | 支持 >64 元素 tile |
| Swizzle/ABI 优化 | 微优化 | 全部格式 | 3-12% SR_BCRS 改善, 编译期约束可审计 |
| tile_n 调优 | 调优 | K=8 数据集 | 11-13% |
| segment 调优 | 调优 | 长尾数据集 | 5-33% |

**核心结论**：Codex `--full-auto` 会话最大的性能贡献来自 **codegen bug 修复**（尤其是长尾分段执行和 K-fragment/Store 修复），而非格式选择智能体的 TCF_MCO 设计。SR_BCRS 通过修复后的管线获得了稳定提升，TCF_MCO 路径则在部分中小矩阵上展现了潜力。

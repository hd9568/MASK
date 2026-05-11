# INC Performance Benchmark vs cuSPARSE

本文记录 `examples/src_fp32/dtc/testbed` 下 `.inc` 算子的性能测试入口、当前 generated/reference 的实测结果，以及如何与 cuSPARSE 做对比。

## 现有 testbed 结构

- `main.cu` 负责：
  - 构造或读取稀疏矩阵
  - 转换到 ME-TCF metadata
  - 调用 `dtc_spmm_test_val_sidx_bind_reorder(...)`
  - 调用 `cusparse_spmm_all(...)`
  - 做最终数值校验
- `host_program.cu` 对应 reference `kernel.inc`
- `host_program_generated.cu` 对应 generated `python/me_tcf_kernel3.inc` 或 patched `python/me_tcf_kernel3_patched.inc`
- `scripts/validate_reference_kernel.sh` 和 `scripts/validate_generated_kernel.sh` 原本更偏 correctness 验证

本轮已补齐两点：

- `host_program_generated.cu` 现在会真正使用 `-warmup/-repeat`，并打印 `mykernel_time`
- `main.cu` 里的 `cusparse_spmm_all(...)` 现在也会按同样的 `-warmup/-repeat` 计时，并打印 `cusparse_time`

因此现在可以在同一组输入参数下直接比较：

- 自定义 `.inc` kernel 平均耗时
- cuSPARSE SpMM 平均耗时
- 两者 speedup

## 新增脚本

新增脚本：

- `examples/src_fp32/dtc/testbed/scripts/benchmark_inc_against_cusparse.sh`
- `examples/src_fp32/dtc/testbed/scripts/benchmark_inc_multi_dataset.sh`

支持三种模式：

- `MODE=generated`：测试当前 `python/me_tcf_kernel3.inc`
- `MODE=patched`：测试 `python/me_tcf_kernel3_patched.inc`
- `MODE=reference`：测试 `examples/src_fp32/dtc/testbed/kernel.inc`

输出会自动解析：

- `mykernel_time: ... ms`
- `cusparse_time: ... ms`
- `speedup_vs_cusparse=...x`

并把完整原始输出保存到：

- `examples/src_fp32/dtc/testbed/build_perf_logs/*.log`

其中 `benchmark_inc_multi_dataset.sh` 额外会输出：

- `summary.csv`：便于后处理或导入表格工具
- `summary.md`：直接可读的 Markdown 性能对比表
- 每个数据集对应的独立 `*.log`

## 推荐用法

### 1. 测试当前 generated `.inc`

```bash
cd /home/huoda/sparsene/examples/src_fp32/dtc/testbed
MODE=generated FORCE_RUN=1 CUDA_ARCH=sm_80 \
  ./scripts/benchmark_inc_against_cusparse.sh \
  -mtx_flag 0 -M 128 -K 128 -N 64 -sparsity 0.9 -warmup 50 -repeat 500
```

### 2. 测试 reference `kernel.inc`

```bash
cd /home/huoda/sparsene/examples/src_fp32/dtc/testbed
MODE=reference FORCE_RUN=1 CUDA_ARCH=sm_80 \
  ./scripts/benchmark_inc_against_cusparse.sh \
  -mtx_flag 0 -M 128 -K 128 -N 64 -sparsity 0.9 -warmup 50 -repeat 500
```

### 3. 测试 patched `.inc`

```bash
cd /home/huoda/sparsene/examples/src_fp32/dtc/testbed
MODE=patched FORCE_RUN=1 CUDA_ARCH=sm_80 \
  ./scripts/benchmark_inc_against_cusparse.sh \
  -mtx_flag 0 -M 128 -K 128 -N 64 -sparsity 0.9 -warmup 50 -repeat 500
```

### 4. 批量测试多个 `.mtx` 数据集

可以直接使用用户提供的数据集目录：

`/mnt/lustre-client/zhangkaige/selected_mtx`

例如测试 reference `kernel.inc` 在两个数据集上的性能表：

```bash
cd /home/huoda/sparsene
FORCE_RUN=1 MODE=reference \
DATASETS=dc2.mtx,ddi.mtx \
N_COLS=128 WARMUP=5 REPEAT=20 \
./examples/src_fp32/dtc/testbed/scripts/benchmark_inc_multi_dataset.sh
```

例如测试 generated `.inc`：

```bash
cd /home/huoda/sparsene
FORCE_RUN=1 MODE=generated \
DATASETS=dc2.mtx \
N_COLS=128 WARMUP=5 REPEAT=20 \
./examples/src_fp32/dtc/testbed/scripts/benchmark_inc_multi_dataset.sh
```

也可以不显式给 `DATASETS`，直接扫目录下全部 `.mtx`：

```bash
cd /home/huoda/sparsene
FORCE_RUN=1 MODE=reference \
MAX_DATASETS=10 \
N_COLS=128 WARMUP=5 REPEAT=20 \
./examples/src_fp32/dtc/testbed/scripts/benchmark_inc_multi_dataset.sh
```

脚本支持的关键环境变量：

- `MODE=generated|patched|reference`
- `DATASET_DIR=/mnt/lustre-client/zhangkaige/selected_mtx`
- `DATASETS=a.mtx,b.mtx`：只跑指定子集
- `MAX_DATASETS=10`：从目录里按文件名排序取前若干个
- `N_COLS=128`：dense B 的列数
- `WARMUP=5`
- `REPEAT=20`
- `TIMEOUT_SEC=600`

输出目录形如：

- `examples/src_fp32/dtc/testbed/build_perf_logs/multi_dataset_reference_*/summary.md`
- `examples/src_fp32/dtc/testbed/build_perf_logs/multi_dataset_reference_*/summary.csv`
- `examples/src_fp32/dtc/testbed/build_perf_logs/multi_dataset_generated_*/summary.md`

## 无 checkpoint 模式

为了避免根分区空间不足，并满足“直接读 `.mtx` 解析后计算、不依赖 checkpoint”的需求，本轮新增了：

- `SPARSENE_DISABLE_MTX_CACHE=1`

在这个模式下：

- 不读取 `mtx_cache` 中已有 checkpoint
- 不向 `mtx_cache` 写入新的 checkpoint
- 直接从 `.mtx` 读取 CSR，再现场构建 ME-TCF metadata
- `mykernel_time` 只统计 metadata 构建完成后的纯 GPU kernel 时间

当前 benchmark 脚本 `benchmark_inc_multi_dataset.sh` 已默认导出：

```bash
SPARSENE_DISABLE_MTX_CACHE=1
```

同时，`mykernel_time` 和 `cusparse_time` 现在统一使用 `cudaEventElapsedTime(...)`，因此计时口径是：

- 不含 `.mtx` 读取
- 不含 CSR -> ME-TCF 格式转换
- 不含 H2D metadata / B 矩阵拷贝
- 不含最终 correctness copy-back
- 只含 warmup 之后 repeated GPU kernel / cuSPARSE SpMM 的 device 时间

如果根分区空间紧张，也可以把日志目录切到 Lustre：

```bash
RESULT_ROOT=/mnt/lustre-client/zhangkaige/sparsene_perf_logs
```

## 当前实测结果

在当前环境中，`128x128x64, sparsity=0.9, warmup=50, repeat=500` 的可靠结果如下。

### generated

命令：

```bash
MODE=generated FORCE_RUN=1 CUDA_ARCH=sm_80 \
  ./scripts/benchmark_inc_against_cusparse.sh \
  -mtx_flag 0 -M 128 -K 128 -N 64 -sparsity 0.9 -warmup 50 -repeat 500
```

结果：

```text
mykernel_time: 0.064136 ms
cusparse_time: 0.014494 ms
generated: mykernel_time=0.064136 ms, cusparse_time=0.014494 ms, speedup_vs_cusparse=0.2260x
```

说明：

- 当前 generated kernel 约为 cuSPARSE 的 `0.226x`
- 换句话说，当前 generated 约比 cuSPARSE 慢 `4.42x`
- 但该结果对应的 correctness 已通过同一用例验证

日志：

- `examples/src_fp32/dtc/testbed/build_perf_logs/generated_20260511_215332.log`

### reference

命令：

```bash
MODE=reference FORCE_RUN=1 CUDA_ARCH=sm_80 \
  ./scripts/benchmark_inc_against_cusparse.sh \
  -mtx_flag 0 -M 128 -K 128 -N 64 -sparsity 0.9 -warmup 50 -repeat 500
```

结果：

```text
mykernel_time: 0.020986 ms
cusparse_time: 0.019034 ms
reference: mykernel_time=0.020986 ms, cusparse_time=0.019034 ms, speedup_vs_cusparse=0.9070x
```

说明：

- reference kernel 已经明显更接近 cuSPARSE
- 当前 reference 大约是 cuSPARSE 的 `0.907x`

日志：

- `examples/src_fp32/dtc/testbed/build_perf_logs/reference_20260511_215430.log`

### 多数据集脚本试跑

reference 模式，`DATASETS=dc2.mtx,ddi.mtx, N_COLS=128, WARMUP=5, REPEAT=20`：

```text
| dataset | M | K | N | mykernel_ms | cusparse_ms | speedup_vs_cusparse | status | reason |
|---|---|---|---|---|---|---|---|---|
| dc2.mtx | 116835 | 116835 | 128 | 13.041750 | 0.223900 | 0.0172 | ok | - |
| ddi.mtx | 4267 | 4267 | 128 | 0.426650 | 0.228350 | 0.5352 | ok | - |
```

generated 模式，`DATASETS=dc2.mtx, N_COLS=128, WARMUP=5, REPEAT=20`：

```text
| dataset | M | K | N | mykernel_ms | cusparse_ms | speedup_vs_cusparse | status | reason |
|---|---|---|---|---|---|---|---|---|
| dc2.mtx | 116835 | 116835 | 128 | 124.042850 | 0.223450 | 0.0018 | ok | - |
```

对应输出目录：

- `examples/src_fp32/dtc/testbed/build_perf_logs/multi_dataset_reference_20260511_221532/`
- `examples/src_fp32/dtc/testbed/build_perf_logs/multi_dataset_generated_20260511_221630/`

## 全量 26 个数据集结果

本轮在以下配置下完成了 `selected_mtx` 目录的全量 generated benchmark：

```bash
cd /home/huoda/sparsene
RESULT_ROOT=/mnt/lustre-client/zhangkaige/sparsene_perf_logs \
FORCE_RUN=1 MODE=generated N_COLS=128 WARMUP=5 REPEAT=20 \
./examples/src_fp32/dtc/testbed/scripts/benchmark_inc_multi_dataset.sh
```

结果文件：

- `/mnt/lustre-client/zhangkaige/sparsene_perf_logs/multi_dataset_generated_20260511_232325/summary.md`
- `/mnt/lustre-client/zhangkaige/sparsene_perf_logs/multi_dataset_generated_20260511_232325/summary.csv`

摘要：

- 总数据集数：`26`
- 成功数：`26`
- 失败数：`0`
- generated `mykernel_ms` 中位数：`8.2889 ms`
- cuSPARSE `cusparse_ms` 中位数：`1.1897 ms`
- `speedup_vs_cusparse` 中位数：`0.1062x`
- 最好数据集：`cop20k_A.mtx`，约 `0.5237x`
- 最差数据集：`dc2.mtx`，约 `0.0018x`

几个最慢的 generated case：

```text
ASIC_680k.mtx  mykernel=423.874707 ms  cusparse=1.244314 ms  speedup=0.0029x
reddit.mtx     mykernel=395.197119 ms  cusparse=26.548328 ms speedup=0.0672x
protein.mtx    mykernel=142.115332 ms  cusparse=9.754419 ms  speedup=0.0686x
dc2.mtx        mykernel=123.844299 ms  cusparse=0.223846 ms  speedup=0.0018x
mip1.mtx       mykernel=59.686603 ms   cusparse=1.096806 ms  speedup=0.0184x
```

## reference `kernel.inc` 全量结果

在与 generated 完全相同的口径下，本轮也测试了 `examples/src_fp32/dtc/testbed/kernel.inc`：

```bash
cd /home/huoda/sparsene
RESULT_ROOT=/mnt/lustre-client/zhangkaige/sparsene_perf_logs \
FORCE_RUN=1 MODE=reference N_COLS=128 WARMUP=5 REPEAT=20 \
./examples/src_fp32/dtc/testbed/scripts/benchmark_inc_multi_dataset.sh
```

结果文件：

- `/mnt/lustre-client/zhangkaige/sparsene_perf_logs/multi_dataset_reference_20260511_233110/summary.md`
- `/mnt/lustre-client/zhangkaige/sparsene_perf_logs/multi_dataset_reference_20260511_233110/summary.csv`

reference 摘要：

- 总数据集数：`26`
- 成功数：`26`
- 失败数：`0`
- reference `speedup_vs_cusparse` 中位数：`0.9775x`

这说明：

- reference `kernel.inc` 整体已经非常接近 cuSPARSE
- generated `.inc` 的慢，并不是因为 testbed benchmark 口径有问题
- generated `.inc` 的慢，主要来自 codegen 产物本身没有对齐 reference 的高效实现

几个代表性 reference case：

```text
web-BerkStan.mtx ref=1.058355 ms  cusparse=1.061222 ms  speedup=1.0027x
protein.mtx      ref=9.968384 ms  cusparse=10.043546 ms speedup=1.0075x
eu-2005.mtx      ref=2.267648 ms  cusparse=2.362470 ms  speedup=1.0418x
DD.mtx           ref=0.454502 ms  cusparse=0.415693 ms  speedup=0.9146x
reddit.mtx       ref=20.285338 ms cusparse=17.104897 ms speedup=0.8432x
```

## generated vs reference

把 generated 与 reference 在同一批 26 个数据集上直接对比后，可得到：

- generated `speedup_vs_cusparse` 中位数：`0.1062x`
- reference `speedup_vs_cusparse` 中位数：`0.9775x`
- generated / reference 的 kernel 时间中位数比值：`10.88x`

也就是说，当前 generated kernel 在中位数上大约比 reference 慢 `10.9x`。

一些差距特别大的 case：

```text
reddit.mtx      generated/reference = 19.48x
shipsec1.mtx    generated/reference = 19.09x
Ga41As41H72.mtx generated/reference = 14.88x
protein.mtx     generated/reference = 14.26x
Si41Ge41H72.mtx generated/reference = 14.24x
```

## generated vs reference 性能归因

结合 `generated .inc` 与 `reference kernel.inc` 的逐段对比，当前最可能影响最大的几个位置是：

1. `python/me_tcf_kernel3.inc` 中的 `array_ref_1::f(...)`
   - 当前实现是逐元素 gather：
   ```cpp
   for (int linear = lid; linear < rows * cols; linear += 32) {
       int row = linear / cols;
       int col = linear % cols;
       int src_row = operand_1(row);
       result_0(row, col, buf_idx) = operand_0(src_row, col);
   }
   ```
   - 对应 reference `kernel.inc` 中的 `G2sMatrixBLoadOp::f(...)`，后者使用 `_4` 粒度和 `SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>` 把 B tile 搬到 shared。
   - 这是当前 B-side 最显著的实现差异之一。

2. `python/me_tcf_kernel3.inc` 中的 `coo_atomic_format_load_val::f(...)`
   - 当前实现也是逐元素 global -> shared：
   ```cpp
   for (int i_load = lid; i_load < nnz_tile; i_load += 32) {
       int src_idx = operand_1 + i_load;
       coo_val(i_load, buf_idx) = src_idx < get<0>(shape_i0) ? operand_0(src_idx) : float(0);
   }
   ```
   - 对应 reference `kernel.inc` 中的 `G2sSparseCooValLoadOp::f(...)`，后者同样使用 `_4` 粒度 `cp.async`。
   - 这意味着 generated 的 A-value 装载形态也弱于 reference。

3. `python/me_tcf_kernel3.inc` 中的 `coo_atomic_format_load_idx::f(...)`
   - 当前实现逐元素装到寄存器，再由 `coo_atomic_val_restore` 做恢复。
   - reference 的 `G2sSparseCooIdxLoadOp::f(...)` 则是 `_4` 粒度 `cp.async` 到 shared，并同时产出 range 信息。
   - generated 在 COO index/value 的组织方式上更碎，容易拉低流水线效率。

4. `python/me_tcf_kernel3.inc` 中的 `for_loop` / `loop_step` 调度
   - 例如：
   ```cpp
   array_ref_v.template f<0>(...);
   __pipeline_commit();
   __pipeline_wait_prior(0);
   __syncthreads();
   array_ref_1_v.template f<0>(...);
   ```
   - reference 在同类路径上会先尽量把 sparse-index / COO idx / COO val 的异步装载全部发出，再在更晚的位置 wait。
   - generated 的 wait/commit/sync 更保守、更碎，容易提前打断 pipeline overlap。

## 最新源码回灌回归

本轮没有继续保留那些会破坏 correctness 的 helper 形态改写，而是只保留了 `generate.py` 里的安全语义修正：

- `array_ref` 默认 `cp_async=False`
- `coo_atomic_format_load_idx/val`
- `coo_atomic_val_restore`
- `mco_atomic_format_load_mask/val`
- `mco_atomic_val_restore`

对应效果是：

- 新生成的 `python/me_tcf_kernel3.inc` 中，`__pipeline_wait_prior(...)` / `__pipeline_commit()` 已经不再出现
- 仍然保留必要的 `__syncthreads()`，所以 current generated kernel 仍是同步 shared-memory helper 路线
- correctness 恢复到通过状态

本轮实际闭环命令：

```bash
cd /home/huoda/sparsene/python
source ~/llm-sog-venv/bin/activate
python test/test_generate.py

cd /home/huoda/sparsene/examples/src_fp32/dtc/testbed
SPARSENE_DISABLE_MTX_CACHE=1 \
  ./build_generated/demo-dtc-generated-tf32 \
  -M 128 -K 128 -N 64 -sparsity 0.9 -mtx_flag 0 -warmup 20 -repeat 100

SPARSENE_DISABLE_MTX_CACHE=1 \
  ./build_generated/demo-dtc-generated-tf32 \
  -filename /mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx \
  -mtx_flag 1 -N 128 -warmup 20 -repeat 100

SPARSENE_DISABLE_MTX_CACHE=1 \
  ./build_reference/demo-dtc-reference-tf32 \
  -filename /mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx \
  -mtx_flag 1 -N 128 -warmup 20 -repeat 100
```

实际结果：

```text
synthetic 128x128x64:
generated mykernel_time: 0.063826 ms
generated cusparse_time: 0.014531 ms
check nan: cusp 122.0 mykernel 122.0

shipsec1.mtx, N=128:
generated mykernel_time: 4.923709 ms
generated cusparse_time: 1.015286 ms
check nan: cusp 357.0 mykernel 357.0

reference mykernel_time: 0.389253 ms
reference cusparse_time: 1.015265 ms
check nan: cusp 357.0 mykernel 357.0
```

与此前全量 sweep 中的 generated `shipsec1.mtx, N=128` 基线相比：

- 旧 generated：`7.402010 ms`
- 新 generated：`4.923709 ms`
- 下降到原来的约 `66.5%`
- 等价于这一轮仅靠语义位修正拿到了约 `1.50x` 的加速

但与 reference 对比仍有明显差距：

- 新 generated / reference = `4.923709 / 0.389253 ≈ 12.65x`

这说明本轮回灌已经证明：

- `cp_async` 语义错标确实会引入多余的 pipeline 原语并伤害性能
- 修正这些语义位不会破坏 correctness，且能带来可观收益
- 但它只能回收“调度层误伤”的部分成本，还不足以让 generated 接近 reference

因此下一阶段最该优先处理的仍是 helper 实现形态，而不是继续围绕 `wait/commit` 做微调：

- `array_ref_1`
- `coo_atomic_format_load_idx`
- `coo_atomic_format_load_val`

它们目前虽然已经在语义上被正确标成“非 cp.async helper”，但真正的访存粒度、lane 分工和 shared-memory 装载形态，仍然和 reference 相差很远。

## 进一步回灌：保留 COO `(ll, rr)` 边界

在上一轮基础上，继续向上游追查后发现，generated 与 reference 在 COO 路线上的更大差异，不只是 helper 形态，而是 COO load 接口本身：

- reference 更接近 `load_offset -> (ll, rr) -> idx/value load -> range-aware restore`
- generated 当时已经在 `varlenLoweringPass` 中把这组显式边界重新压成了单一 flat offset

本轮最终保留下来的有效回灌不是直接改 `.inc`，而是：

- 在 `opir/generate.py` 中恢复 `LoadOffsetOp -> (ll, rr)`
- 在 `opir/ops.py` 中让 `CooAtomicFormatLoadIdxOp` 重新产出 `(coo_idx, coo_range)`
- 在 `varlenLoweringPass.py` 中对 `CooAtomicFormatLoadIdxOp / CooAtomicFormatLoadValOp` 加专门豁免：
  - 如果它们已经拿到显式 `[ll, rr]`
  - 就不要再被 generic array-index flatten 逻辑压回单一 flat index
- 在 `nvir/generate.py` 中为这组三输入 / 双输出 COO helper 补齐 lowering

这次回灌后，新生成 `.inc` 的关键变化是：

```cpp
coo_atomic_format_load_idx_v.template f<0>(
    val_coo_idx_new_1Op_v.template output<0>(),
    load_offset_1_v.template output<0, 0>()(0),
    load_offset_1_v.template output<1, 0>()(0)
);
coo_atomic_format_load_val_v.template f<0>(
    val_coo_val_new_1Op_v.template output<0>(),
    load_offset_1_v.template output<0, 0>()(0),
    load_offset_1_v.template output<1, 0>()(0)
);
```

对应 helper 也重新变成了 range-aware 版本：

```cpp
CUTE_DEVICE void f(Tensor_i0 operand_0, TI1 operand_1, TI2 operand_2) {
    coo_range(0, buf_idx) = operand_2 - operand_1;
    int nnz_tile = get<0>(BLK_MNK{}) * get<2>(BLK_MNK{});
    for (int i_load = lid; i_load < nnz_tile; i_load += 32) {
        int src_idx = operand_1 + i_load;
        coo_idx(i_load, buf_idx) = src_idx < operand_2 ? operand_0(src_idx) : 0;
    }
}
```

闭环命令：

```bash
cd /home/huoda/sparsene/python
source ~/llm-sog-venv/bin/activate
python test/test_generate.py

cd /home/huoda/sparsene/examples/src_fp32/dtc/testbed
nvcc -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo \
  -ftemplate-backtrace-limit=0 -arch=sm_80 \
  -ccbin /opt/rh/gcc-toolset-13/root/usr/bin/g++ \
  -Xcompiler -fopenmp -Df32 \
  -I./cutlass/include -I./cutlass/examples/common -I./cutlass/tools/util/include \
  -I./src -I. \
  host_program_generated.cu main.cu src/mmio.cu src/mmio_highlevel.cu src/utils.cu \
  -lcusparse -lcublas \
  -o build_generated/demo-dtc-generated-tf32

SPARSENE_DISABLE_MTX_CACHE=1 \
  ./build_generated/demo-dtc-generated-tf32 \
  -M 128 -K 128 -N 64 -sparsity 0.9 -mtx_flag 0 -warmup 20 -repeat 100

SPARSENE_DISABLE_MTX_CACHE=1 \
  ./build_generated/demo-dtc-generated-tf32 \
  -filename /mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx \
  -mtx_flag 1 -N 128 -warmup 20 -repeat 100
```

实际结果：

```text
synthetic 128x128x64:
generated mykernel_time: 0.083724 ms
generated cusparse_time: 0.019132 ms
check nan: cusp 122.0 mykernel 122.0

shipsec1.mtx, N=128:
generated mykernel_time: 3.051046 ms
generated cusparse_time: 1.760729 ms
check nan: cusp 357.0 mykernel 357.0

reference:
mykernel_time: 0.388547 ms
cusparse_time: 1.015716 ms
check nan: cusp 357.0 mykernel 357.0
```

与上一轮安全基线相比：

- 上一轮 generated：`4.923709 ms`
- 本轮 generated：`3.051046 ms`
- 新一轮继续下降到原来的约 `61.97%`
- 相对上一轮再拿到约 `1.61x` 加速

若与最早全量 sweep 的 `shipsec1.mtx, N=128` 基线对比：

- 旧 generated：`7.402010 ms`
- 当前 generated：`3.051046 ms`
- 总体约 `2.43x` 加速

与 reference 的距离也继续缩小：

- 上一轮 generated / reference ≈ `12.65x`
- 当前 generated / reference ≈ `3.051046 / 0.388547 ≈ 7.85x`

这说明这次真正有效的性能回灌，不是局部 helper 循环改写，而是：

- 在 OPIR 层保住 COO range 信息
- 避免在 `varlenLoweringPass` 中丢掉 `(ll, rr)` 这组边界语义

换句话说，generated 之前有相当一部分慢因来自：

- codegen 过早把 COO tile 的边界信息压平成了单一偏移
- 进而导致后续 restore/load helper 只能走更弱的“长度/偏移近似”路径

## 继续回灌：恢复 COO/B-side 的 cp.async 访存

在保住 `(ll, rr)` 边界之后，再继续对齐 reference 的 helper 访存形态，终于拿到了第二轮显著收益。

### 1. COO idx/value load 恢复 cp.async

在 `nvir/generate.py` 中，把三输入版本的：

- `coo_atomic_format_load_idx`
- `coo_atomic_format_load_val`

改成和 reference 一致的 `_4` 粒度 `local_tile + cp.async copy` 形式，同时重新把它们标记成真正的 `cp_async` helper。

但为了避免 synthetic case 的 `misaligned address (716)`，这里额外保留了一个保护：

- 若 `ll % 4 == 0`，走 `_4` 粒度 `cp.async`
- 否则自动回退到原来的标量 loop

### 2. 只对 cp.async 的 shared 输出做 16B 对齐

`codegen.py` 中的 shared output buffer 声明改成：

- 普通 SMEM 输出仍保持原状
- 只有 `out.attrs["cp_async"] == True` 的 shared buffer 才加 `__align__(16)`

这一步是必须的，否则 synthetic case 会因为 shared 目标地址不满足 16B 对齐而报错。

### 3. B-side `array_ref_1` 恢复 reference 风格 `cp.async`

在 `array_ref` lowering 中，仅对 `B_val -> tileB` 这一路恢复 reference 风格的 `_1x4` 搬运：

- `array_ref_role == "b_val"` 时
- 使用 `local_tile(B_val, make_shape(_1{}, _4{}), ...)`
- `copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>{}, ...)`
- 并把该 `array_ref` 重新标成 `cp_async=True`

而 `val_sidx` 的 1D gather 仍然保持同步版本，没有一起恢复 `cp.async`，以降低 correctness 风险。

### 新结果

```text
synthetic 128x128x64:
generated mykernel_time: 0.024668 ms
generated cusparse_time: 0.014356 ms
check nan: cusp 122.0 mykernel 122.0

shipsec1.mtx, N=128:
generated mykernel_time: 0.523653 ms
generated cusparse_time: 1.014999 ms
check nan: cusp 357.0 mykernel 357.0
```

这里对比前一轮：

- 前一轮 generated：`3.051046 ms`
- 加入 COO/B-side `cp.async` 后：`0.523653 ms`
- 相对前一轮再次拿到约 `5.83x` 加速

若与最早 `shipsec1.mtx, N=128` 基线相比：

- 最早 generated：`7.402010 ms`
- 当前 generated：`0.523653 ms`
- 总体约 `14.13x` 加速

若与之前测得的 reference 基线相比：

- reference：`0.388547 ms`
- 当前 generated：`0.523653 ms`
- generated / reference ≈ `1.35x`

这说明 generated 现在已经不再是“量级上明显慢于 reference”，而是进入了同一数量级的区间。

## manual `.inc` 试探与回退

本轮对 generated `.inc` 做了两次直接性能试探，但都因为 correctness 失败而回退：

1. **试探 A：把 `array_ref_1::f(...)` 改成 `_4` 粒度、`cp.async` 风格搬运**
   - 小 synthetic case 上 `mykernel_time` 从约 `0.064 ms` 降到了约 `0.025 ms`
   - 但 `128x128x64` correctness 立刻失配
   - `shipsec1.mtx` 也出现 `199` 个 correctness errors
   - 结论：shared/swizzle + gather 语义比表面更敏感，不能直接替换成 reference 的拷贝模板

2. **试探 B：把 `array_ref_1 / coo_atomic_format_load_idx / coo_atomic_format_load_val` 改成 4 元素展开**
   - 目标是不改异步语义，只减少循环开销
   - 仍然破坏 correctness
   - 结论：generated 当前这些 load/restore helper 的执行顺序、lane 映射和缓冲区语义都高度脆弱，不能仅做局部循环改写

最终保留策略：

- 回退 `me_tcf_kernel3.inc` 到 correctness 基线
- 仅保留 benchmark 脚本与 validator 的流程性改进：
  - `SKIP_GENERATE=1`
  - `SPARSENE_DISABLE_MTX_CACHE=1`
  - `cudaEvent` 纯 GPU 计时

这说明下一阶段如果要真正提升 generated 性能，最合理的路径不是继续手工 patch `.inc` 的局部循环，而是把 `array_ref_1 / coo_atomic_format_load_* / default_schedule()` 等关键模式系统性回灌到 codegen，让生成物在“访存形态 + pipeline 语义”上同时接近 reference。

## 如何理解这些结果

从这组数据可以得到两个直接结论：

- correctness 已经打通，但 generated 的性能仍明显落后于 reference 和 cuSPARSE
- 这意味着下一阶段如果要做性能优化，优先级应放在 generated 与 reference 的实现差异上，而不是先去追 correctness

更具体地说：

- reference `kernel.inc` 已接近 cuSPARSE
- generated `me_tcf_kernel3.inc` 当前仍慢约 `3x` 于 reference
- 因此 codegen 后续的重点应该是继续对齐 reference 的访存、流水与寄存器/共享内存使用模式

## 当前环境注意事项

当前环境在更大规模 case 上偶发 CUDA `304`：

```text
CUDA API failed ... OS call failed or operation not supported on this OS (304)
```

因此：

- 小规模且已验证 correctness 的 case，可以作为当前阶段的可靠性能基线
- 较大规模 case 如果出现 `304`，不要直接把该次 `mykernel_time` 当成真实性能数据
- 更稳妥的做法是优先复用已经稳定的参数组合，或在宿主机原生 shell 里重跑

# Codegen Backport GPU Validation

本文记录本轮将 `patched inc` 必要修复回灌到 Python codegen 之后，对 generated 路线做的真实 GPU 验证结论。

## 最新结果

截至当前轮次，generated 路线已经在真实 GPU 上通过 `128x128x64` 验证。

在本轮继续做“从 `generate.py` / `codegen.py` 回灌性能差异点”的过程中，最新稳定结论进一步收敛为：

- 会破坏 correctness 的 helper 形态改写已经全部回退
- 当前只保留安全的 `cp_async=False` 语义标注修正
- 基于这组修正重新 `python test/test_generate.py` 生成后的 `python/me_tcf_kernel3.inc`，synthetic correctness 继续通过
- `shipsec1.mtx, N=128` 下 generated 从历史基线 `7.402010 ms` 降到 `4.923709 ms`
- 同 case 的 reference 仍是 `0.389253 ms`，因此 generated 还有约 `12.65x` 差距

也就是说，这一轮源码回灌已经完成了一个明确闭环：

- correctness 没有因为性能回灌而再次退化
- `cp_async` 语义错标的确是 generated 慢因之一
- 但它不是主导慢因，主导差距仍在 helper 访存形态

## 2026-05-11 最新验证

重新生成：

```bash
cd /home/huoda/sparsene/python
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
python test/test_generate.py
```

synthetic correctness：

```bash
cd /home/huoda/sparsene/examples/src_fp32/dtc/testbed
SPARSENE_DISABLE_MTX_CACHE=1 \
  ./build_generated/demo-dtc-generated-tf32 \
  -M 128 -K 128 -N 64 -sparsity 0.9 -mtx_flag 0 -warmup 20 -repeat 100
```

输出：

```text
mykernel_time: 0.063826 ms
cusparse_time: 0.014531 ms
check nan: cusp 122.0 mykernel 122.0
```

`shipsec1.mtx` 单点回归：

```bash
SPARSENE_DISABLE_MTX_CACHE=1 \
  ./build_generated/demo-dtc-generated-tf32 \
  -filename /mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx \
  -mtx_flag 1 -N 128 -warmup 20 -repeat 100

SPARSENE_DISABLE_MTX_CACHE=1 \
  ./build_reference/demo-dtc-reference-tf32 \
  -filename /mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx \
  -mtx_flag 1 -N 128 -warmup 20 -repeat 100
```

输出：

```text
generated:
mykernel_time: 4.923709 ms
cusparse_time: 1.015286 ms
check nan: cusp 357.0 mykernel 357.0

reference:
mykernel_time: 0.389253 ms
cusparse_time: 1.015265 ms
check nan: cusp 357.0 mykernel 357.0
```

## 当前保留结论

当前真正应该保留到 codegen 的只有两类东西：

1. 正确性修复
   - induction var 绑定 `l + {c}`
   - shape/materialize/swizzle 等此前已经验证过的必要修复

2. 安全的性能修复
   - 对非 `cp.async` helper 显式打 `cp_async=False`
   - 让 `default_schedule()` 不再为这些同步 helper 发出无意义的 `__pipeline_commit()` / `__pipeline_wait_prior(...)`

当前明确不应保留的东西是：

- 把 `array_ref_1`
- `coo_atomic_format_load_idx`
- `coo_atomic_format_load_val`
- 1D `val_sidx` helper

直接改写成 `_4` 粒度或 `cp.async` 风格的局部实现

因为这些改动虽然可能进一步降低 `mykernel_time`，但已经被实际验证会破坏 correctness。

## 2026-05-12 进一步回灌

继续向上游排查后，发现 generated 与 reference 在 COO 路线上的一个更关键差异，不是 helper 内部 loop，而是 COO 边界信息是否在 lowering 过程中被保住：

- 旧 generated：`coo_len -> flat offset`
- reference-like 路线：`load_offset -> (ll, rr) -> idx/value load -> range-aware restore`

本轮最终保留下来的有效修复是：

1. 在 `opir/generate.py` 中恢复 `LoadOffsetOp` 产生 `(ll, rr)`
2. 在 `opir/ops.py` 中让 `CooAtomicFormatLoadIdxOp` 重新返回两个结果：
   - `coo_idx`
   - `coo_range`
3. 在 `varlenLoweringPass.py` 中对 `CooAtomicFormatLoadIdxOp / CooAtomicFormatLoadValOp` 增加专门豁免：
   - 如果 operands 已经是显式 `[array, ll, rr]`
   - 就不要再被 generic flatten 改写成单一 flat index
4. 在 `nvir/generate.py` 中补齐这组三输入 / 双输出 helper 的 lowering

这次回灌的关键效果不是“删掉更多 loop”，而是：

- Stage0 从错误试探阶段的 `14 ops`
- 回落到保留 `load_offset_1 + coo_atomic_*` 的 `9 ops`

说明 `(ll, rr)` 已经成功穿透到 generated pipeline，而没有再次退化成一串 `load/div/add` 中间标量。

### 实测结果

重新生成并重编 generated binary 后：

```text
synthetic 128x128x64:
mykernel_time: 0.083724 ms
cusparse_time: 0.019132 ms
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

与上一轮稳定 generated 基线相比：

- 上一轮：`4.923709 ms`
- 本轮：`3.051046 ms`
- 继续取得约 `1.61x` 加速

与最初 `shipsec1.mtx, N=128` generated 基线相比：

- 最初：`7.402010 ms`
- 当前：`3.051046 ms`
- 总体约 `2.43x` 加速

与 reference 的差距从：

- `12.65x`

收敛到：

- `3.051046 / 0.388547 ≈ 7.85x`

### 当前结论更新

现在可以更明确地说，generated 的主导慢因至少包括两层：

1. `cp_async` 语义错标
   - 已修复
   - 贡献了第一段约 `1.50x` 的安全加速

2. COO range 信息在 lowering 中丢失
   - 本轮修复
   - 又贡献了约 `1.61x` 的额外加速

因此后续继续提速时，优先级应该是：

- 继续守住 OPIR/NVIR 接口层的结构信息
- 再考虑 helper 级别的访存形态微调

而不是反过来先去改局部循环展开或单独恢复 `cp.async`。

## 2026-05-12 再进一步：COO/B-side cp.async 回灌

在 `(ll, rr)` 结构已经保住之后，再继续向 reference 靠拢，最终真正起作用的是两条 helper 访存回灌：

1. `coo_atomic_format_load_idx / coo_atomic_format_load_val`
   - 恢复 `_4` 粒度 `local_tile + cp.async copy`
   - 重新标成 `cp_async=True`
2. `array_ref_1` 对应的 `B_val -> tileB`
   - 恢复 reference 风格 `_1x4` 的 `cp.async`
   - 只对 `array_ref_role == "b_val"` 生效

同时，为了避免 synthetic case 出现：

```text
CUDA API failed ... misaligned address (716)
```

额外加了两条保守约束：

- `coo_atomic_*` 只有在 `ll % 4 == 0` 时才走 `_4` 粒度 `cp.async`
- `codegen.py` 只对 `cp_async=True` 的 shared 输出加 `__align__(16)`

这两条约束加完之后，generated 再次通过了：

- synthetic correctness
- `shipsec1.mtx, N=128` correctness

并拿到了目前为止最大的单轮收益：

```text
synthetic 128x128x64:
mykernel_time: 0.024668 ms
cusparse_time: 0.014356 ms
check nan: cusp 122.0 mykernel 122.0

shipsec1.mtx, N=128:
generated mykernel_time: 0.523653 ms
generated cusparse_time: 1.014999 ms
check nan: cusp 357.0 mykernel 357.0
```

对比上一轮：

- 上一轮 generated：`3.051046 ms`
- 本轮 generated：`0.523653 ms`
- 再取得约 `5.83x` 加速

对比最早基线：

- 最早 generated：`7.402010 ms`
- 当前 generated：`0.523653 ms`
- 总体约 `14.13x` 加速

对比此前稳定测得的 reference：

- reference：`0.388547 ms`
- 当前 generated：`0.523653 ms`
- generated / reference ≈ `1.35x`

这说明 generated 现在已经进入“与 reference 同一数量级”的区间，而不再是早期那种 `10x+` 的明显落后状态。

### 结论更新

到这一轮为止，已经能把 generated 的性能差距明确拆成三类：

1. `cp_async` 语义错标
2. COO range 信息在 lowering 中丢失
3. reference 风格的 `cp.async` helper 访存形态没有被生成出来

其中前两类解决后，generated 从 `7.40 ms` 降到 `3.05 ms`；
第三类再补齐之后，又进一步降到 `0.52 ms`。

因此，后续如果还要继续逼近 reference，优先级已经非常清晰：

- 先继续守住“结构信息 + cp.async helper 形态”这两类源码回灌
- 再考虑剩余的小差距是否来自 schedule 细节、寄存器压力或个别 helper 的 lane 分工

闭环命令：

```bash
cd /home/huoda/sparsene/python
python test/test_generate.py

cd /home/huoda/sparsene/examples/src_fp32/dtc/testbed
FORCE_RUN=1 PATCHED_ONLY=0 CUDA_ARCH=sm_80 \
  ./scripts/validate_generated_kernel.sh \
  -mtx_flag 0 -M 128 -K 128 -N 64 -sparsity 0.9 -warmup 1 -repeat 1
```

当前真实输出已经从此前的大量 `Error(i, j): ...` 收敛为：

```text
Checkpoint loaded successfully from: /home/huoda/sparsene/examples/src_fp32/dtc/testbed/mtx_cache/_M128_K128_METCF.checkpoint
[generated-meta] val_len_offset[0..2]=(0,13,27) first_row_blocks=13 coo_offsets[0..2]=(0,11,33) sidx[0..3]=(0,1,2,6)
check nan: cusp 122.0 mykernel 122.0
```

这说明本轮 generated 路线已经满足：

- `test_generate.py` 可重新生成 `python/me_tcf_kernel3.inc`
- generated validator 可成功编译
- kernel 可在真实 GPU 上运行
- 当前验证用例下不再出现结果比对报错

## 本轮最终起作用的修复

本轮最后真正保留下来的修复，不是继续增加新的 schedule 逻辑，而是把 `python/sparsene/op_gen/nvir/codegen.py` 里那段“跨调度批次 flush B-side `array_ref_1`”的实验逻辑回退掉，恢复到更朴素的 `default_schedule()`。

原因是：

- 去掉 `deferred_b_ldmatrix_across_steps` 的试探是负收益，GPU 误差扩散到了更多 row-window
- 改成“下一调度批次开头 flush 上一拍 `array_ref_1`”后，虽然代码轨迹形式上更像 `patched inc`，但 GPU 数值反而更差
- 这说明当前 generated 路线剩余的正确性，并不取决于再额外人为调整 B-side 发射顺序
- 在已有关键修复已经回灌的前提下，额外的跨批次 flush 反而破坏了原本正确的 A/B pairing 与 wait 语义

最终保留的结论是：

- `generate.py` 中 induction var 从 `"_idx"` 改为 `"l + {c}"` 是把 generated 从“全 0”推进到“非零并接近正确”的决定性修复
- `codegen.py` 中 `is_mma` 识别、`div(..., BLK_K)` 的 very narrow loop-bound 特判、`BLK_M/N/K` 参数 materialize 等修复是必要的
- 最近两轮试探性 schedule 修改都不应保留，正确做法是回退

## 目标

验证以下闭环是否成立：

- `python/test/test_generate.py` 重新生成 `python/me_tcf_kernel3.inc`
- `examples/src_fp32/dtc/testbed/scripts/validate_generated_kernel.sh` 使用 generated 路线编译 validator
- 在真实 GPU 上运行 generated kernel
- 与 cuSPARSE 输出做数值对比

## 本轮关键回灌点

本轮已回灌的关键修复集中在：

- `python/sparsene/op_gen/nvir/generate.py`
- `python/sparsene/op_gen/nvir/codegen.py`
- `examples/src_fp32/dtc/testbed/host_program_generated.cu`

主要包括：

- `val_sidx_new` 作为 `array_ref` 输入时，shape 从错误的 `[nnz_dim_K, BLK_K]` 规范化为 `[BLK_K, nnz_dim_K]`
- `BLK_M/N/K` 参数表达式统一 materialize 成 `get<...>(BLK_MNK{})`
- `array_ref_1` 和 `coo_atomic_val_restore` 的 shared-memory layout / swizzle 回灌
- `host_program_generated.cu` 的 kernel launch 参数与新生成的 `generated_kernel(...)` 对齐

## 实际 GPU 验证命令

在真实 GPU 环境里执行：

```bash
cd /home/huoda/sparsene/examples/src_fp32/dtc/testbed
FORCE_RUN=1 PATCHED_ONLY=0 CUDA_ARCH=sm_80 \
  ./scripts/validate_generated_kernel.sh \
  -mtx_flag 0 -M 128 -K 128 -N 64 -sparsity 0.9 -warmup 1 -repeat 1
```

对应日志：

- `docs/log/codegen_backport_validate_gpu2.log`

## 历史结论（已过时）

generated 路线已经达到：

- 可以重新生成 `python/me_tcf_kernel3.inc`
- 可以编译 generated validator
- 可以在真实 GPU 上启动运行

但是当前还没有数值正确。

日志显示：

- checkpoint 成功加载
- kernel 已实际运行到结果比对阶段
- 结果与 cuSPARSE 大量不一致
- `mykernel` 输出在大量位置为 `0.0`

典型日志片段：

```text
[3/3] Running validator
Checkpoint loaded successfully from: /home/huoda/sparsene/examples/src_fp32/dtc/testbed/mtx_cache/_M128_K128_METCF.checkpoint
Error(0, 0): cusp(122.0) mykernel(0.0)
Error(0, 1): cusp(114.0) mykernel(0.0)
Error(0, 2): cusp(89.0) mykernel(0.0)
...
check nan: cusp 122.0 mykernel 0.0
```

因此，本轮 GPU 验证的准确结论是：

- generated 路线的“生成 -> 编译 -> GPU 运行”已经打通
- 但“数值正确性”仍未通过

## 当前最重要的判断

从现象看，当前问题已经不再是：

- C++ 代码生成不合法
- kernel launch 参数不匹配
- shape assert
- shared-memory misaligned address

当前更像是 kernel 主计算路径虽然运行了，但结果没有正确写回到 `C`，或者上游 tile 在进入 `mma` 前已经被清空为 0。

结合之前 patched 路线已经对齐的情况，下一步最值得优先核查的是：

- `array_ref_1 -> coo_atomic_val_restore -> S2rBValLoadOp` 这一段 B-side 数据链路是否与 patched 版完全一致
- `mma` 输入 fragment 是否在 generated 路线里仍被错误置零
- `c_val_store` 之后的最终写回是否真的落到 `dC`

## 最新调试收敛

本轮又做了两类更靠近根因的验证：

- 在 host bridge 中打印 generated kernel 的入口元数据摘要
- 在 `c_val_store::f()` 前打印 `for_loop` 最终传入写回阶段的 4 个 fragment 值

对应日志：

- `docs/log/codegen_backport_validate_gpu6.log`
- `docs/log/manual_kernel_debug_frag.log`

观察结果：

```text
[generated-meta] val_len_offset[0..2]=(0,13,27) first_row_blocks=13
coo_offsets[0..2]=(0,11,33) sidx[0..3]=(0,1,2,6)
DBG c_val_store frag: 0.000000 0.000000 0.000000 0.000000
```

这说明：

- generated 路线进入 kernel 之前的 host metadata 不是空的，至少 block 0 的 row-window trip count 和 compact COO offset 都正常
- 问题不在最终 `c_val_store` 写回映射
- `for_loop` 在进入 `c_val_store` 之前，累加寄存器 fragment 本身就已经是全 0

因此根因已经继续收缩到 kernel 主循环内部，更具体地说是：

- `coo_atomic_val_restore`
- `S2rAValLoadOp`
- `S2rBValLoadOp`
- `mma`

这条链路中的某一环仍未与 patched 路线完全对齐

## 最新源码判断

结合当前 generated 与 patched 的 diff，最可疑的剩余系统性差异是：

- generated 目前仍把 `coo_atomic_format_load_idx/val` 的输出 shape 固定化
- generated 目前仍把 `coo_atomic_val_restore` 的输入 shape 固定化
- patched 路线在这些位置仍保留动态 `VARLEN` 模式长度，并在构造时显式传入 `BLK_K_BLK_M`

这组差异可能导致 A-side 恢复链路虽然合法运行，但恢复出的 tile 实际上始终为空，从而让 `mma` 前的累加片段保持全 0

## 建议的下一步调试顺序

建议继续沿着首次“全 0”出现的 stage 做最小化排查：

1. 对照 `python/me_tcf_kernel3.inc` 与 `python/me_tcf_kernel3_patched.inc`
2. 重点检查 `array_ref_1`、`coo_atomic_val_restore`、`S2rBValLoadOp`、`mma`、`c_val_store`
3. 做 stage-level dump 或最小 instrumentation，确认第一个变成 0 的张量
4. 只把真正必要的差异继续回灌到 Python codegen

## 本轮结论摘要

- GPU 可用，验证已在真实设备上执行
- generated 路线已不再停留在 compile/runtime blocker
- 当前 remaining bug 是数值错误，不是环境错误
- 典型现象是 `mykernel` 输出大面积为 `0.0`

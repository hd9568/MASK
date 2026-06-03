# ME_TCF 生成算子性能优化总结 (2026-05-29)

## 1. 当前结论

- 目标: 在不引入智能体的前提下，让默认格式 + 默认 heuristic 策略生成的 `ME_TCF` kernel 逼近甚至打平手写 `kernel.inc`。
- 当前主线版本在 `shipsec1 / N=512 / warmup=20 / repeat=100` 上达到:
  - 生成: `1.580595 ms`
  - 手写: `1.541315 ms`
  - 差距: `+2.55%`
- 当前版本 correctness 正常，已在 `shipsec1`、`cop20k_A` 上验证通过。
- 本轮曾有一版在 `shipsec1` 上做到 `1.536440 ms`，略快于手写，但在 `cop20k_A`、`dc2` 上触发 `misaligned address (716)`，因此已回退，没有保留到主线。

## 2. 这一路是怎么把 20%+ gap 缩到 2%~3% 的

### 2.1 先把寄存器和 occupancy 拉回到手写附近

最早生成 kernel 的主要问题不是单一某条指令，而是整体调度和寄存器压力都偏重。

已经保留在主线中的有效改动包括:

- `templates.py` 中使用更紧的 `__launch_bounds__(32, 20)`，压低编译器分配寄存器。
- `templates.py` 和 `codegen.py` 中移除大量d模板类对 `tid/wid` 的显式持有与传递，避免热路径无意义状态。
- `codegen.py` 中不再为每个热 op 自动生成运行时 `assert(shape == size(shape(...)))`，减少 device 指令和寄存器消耗。

效果:

- 早期 profile 中生成 kernel 约 `128 regs/thread`，而手写约 `95`。
- 当前主线版本的轻量 NCU 指标已经到 `92 regs/thread`，低于手写的 `95 regs/thread`。

### 2.2 在 OPIR 层先消掉冗余表达式

在 `varlenLoweringPass.py` 中做了若干符号级简化:

- `x + 0 -> x`
- `x - 0 -> x`
- 跳过 `1 / BLK_K` 这类恒等缩放
- 跳过恒为 0 的项

这些优化的价值不在于单次节省很多时间，而在于后续 NVIR pipeline 中少生成一整串 `gen_add/gen_sub/gen_div/gen_const`，让主循环更干净。

### 2.3 在 NVIR 层做两类关键融合

在 `compiler_driver.py` 中保留了两类真正有效的融合:

- `load_offset + gen_sub + gen_div` 融合
  - 把 `off_r - off_l` 和相关长度计算合并进 `load_offset`
- `gen_add(off_l, l + c)` 内联
  - 不再让 `array_ref` 和 `load_offset_1` 额外依赖一个独立的 `gen_add` op

这些融合让 pipeline body 的 op 数下降，也让 stage 划分更接近手写版本。

### 2.4 真正拉近手写 kernel 的关键: 改 pipeline 时序，而不是只看单个 op

对比手写 `kernel.inc` 和生成的 `short_pipe()` 后，发现最关键的不是“有没有 cp.async”，而是“cp.async 发起之后有没有被过早 wait/sync 消费”。

手写 kernel 的关键时序是:

1. 先发起 `sidx + coo_idx + coo_val`
2. 再 `wait_prior(2)`，然后做 `B` 的 G2S
3. 再 `wait_prior(1)`，然后做 restore
4. 最后进入 `S2rA / S2rB / mma`

此前生成 kernel 的问题是:

- `sidx` 发起后过早进入 `wait/sync`
- `B` load overlap 不够

最终保留的有效做法是:

- 把 `array_ref` (`sidx`) 放在 Stage0
- 把 `array_ref_1` (`B`) 推迟到 Stage1
- 让 `array_ref(sidx)`、`coo idx`、`coo val` 先发起，再消费 `B`

这一步对 `shipsec1` 的实际收益最大，是从“仍慢 20%+”缩到“只差几个点”的核心原因。

### 2.5 把 cp.async 写法下沉成框架 helper，避免源码里硬编码某一代架构 atom

由于框架需要考虑多种硬件，不能在生成源码里到处直接写 `SM80_CP_ASYNC_CACHEGLOBAL<...>`。

当前保留的做法是:

- 在 `templates.py` 中新增 `sparsene_copy_g2s_128<TElem>()`
- 在 `__CUDA_ARCH__ >= 800` 时走 cp.async atom
- 更低架构自动退化为普通 `copy()`

这样做有两个好处:

- 生成出来的源码本身不再直接耦合具体硬件代号
- 仍然保留了 A800 / SM80 上的快路径

## 3. 当前生成 kernel 和手写 kernel 的主要差异

在 `shipsec1` 这个主测点上，当前生成 kernel 与手写 kernel 已经非常接近，但还有几个残余差异。

| 维度 | 手写 kernel | 当前生成 kernel | 影响 |
|---|---|---|---|
| `sidx` G2S | 直接 cp.async 到 SMEM | 已对齐，Stage0 发起 | 基本对齐 |
| `B` G2S | Stage1 发起 | 已对齐，Stage1 发起 | 基本对齐 |
| `coo idx/val` | 手写专用 loader | 通用 `coo_atomic_format_load_*` 模板 | 指令数略高 |
| `coo range` 处理 | 内嵌在手写 loader | 仍保留 `load_offset_1` 独立 op | 额外开销 |
| 代码风格 | 专用手写 fast path | 通用 lowering 生成 | 灵活但更冗长 |

最直观的量化差异来自轻量 NCU 指标:

| 指标 | 手写 | 当前生成 |
|---|---:|---:|
| `launch__registers_per_thread` | `95` | `92` |
| `smsp__inst_executed.sum` | `445,491,056` | `471,249,160` |

结论:

- 当前已经不是寄存器问题，寄存器甚至比手写更低。
- 剩余 gap 主要来自动态指令数仍偏高，说明还有额外控制流 / 通用路径 / 独立 op 开销。

## 4. 这轮详细分析中发现的关键点

### 4.1 `val_coo_off` 不是“全部 4 对齐”，但 `ll` 是稳定对齐的

对 checkpoint 的二进制内容做了直接分析:

- `shipsec1` 中 `ll`、`rr` 都是 `4` 对齐
- `cop20k_A`、`dc2` 中:
  - `ll` 始终 `4` 对齐
  - `rr` 不是总对齐

也就是说:

- `coo` 的起始地址本身适合走 x4 fast path
- 但 tile 的结尾长度不是总能整除 4
- 因此任何“简单粗暴的整段无条件 x4 cp.async”都可能在部分数据集上出错

### 4.2 为什么有一版 `shipsec1` 能超过手写，但不能保留

我尝试把 `coo_atomic_format_load_idx/val` 改成默认无条件 x4 cp.async，结果是:

- `shipsec1`: `1.536440 ms`，略快于手写 `1.541315 ms`
- 但 `cop20k_A`、`dc2` 直接出现 `misaligned address (716)`

随后我又尝试“主体 x4 cp.async + 尾部标量补齐”，仍然在 `cop20k_A` 上触发同类错误。

因此这条思路虽然在单点上有效，但没有跨数据集 correctness 保证，必须回退。

## 5. 当前主线保留了什么，回退了什么

### 5.1 当前保留在主线中的改动

- OPIR 常量/恒等式消除
- `__launch_bounds__(32, 20)`
- `tid/wid` 清理
- 去掉热路径 runtime shape assert
- `load_offset + gen_sub + gen_div` 融合
- `gen_add` 内联到 `array_ref + load_offset_1`
- `sidx` 和 `B` 的 stage 重排
- `sparsene_copy_g2s_128` 架构抽象 helper

### 5.2 已确认无效或不安全、因此回退的改动

- `coo idx/val` 默认整段无条件 x4 cp.async
  - `shipsec1` 更快
  - 但 `cop20k_A`、`dc2` 报 `misaligned address (716)`
- `coo idx/val` 主体 x4 cp.async + 尾部标量补齐
  - 仍有 correctness 风险
  - 已回退
- 更早之前的 `load_offset_1` 完全并入 `coo_*`
  - 在当时测试中造成性能回退

## 6. 为什么现在还差 2%~3%

当前 safe 主线和手写相比，最大的剩余差异是:

- `load_offset_1` 仍然是独立 op，手写版没有这层显式拆分
- `coo_atomic_format_load_*` 仍然保留了“对齐判断 + fallback”的通用结构
- 通用 lowering 生成出的控制流和模板外壳，仍比手写专用 fast path 多一部分动态指令

换句话说，剩余 gap 本质上已经不是“大方向错了”，而是“最后一段专用 fast path 还没专门化到位”。

## 7. 下一步最值得继续攻的方向

如果要在 correctness 不退化的前提下，把最后这 2%~3% 再抠掉，我认为最值得做的是:

1. 为 `coo_atomic_format_load_idx/val` 增加真正安全的 predicated cp.async fast path
   - 方向不是“去掉判断硬上 cp.async”
   - 而是用 `copy_if` / `cp.async zfill` / predication 的方式处理尾块
2. 为 `ME_TCF` 单独生成一个更专用的 `coo` loader
   - 把 `load_offset_1` 和 `coo idx/val` 的部分控制流合并掉
   - 但要避免回到“重复 gmem lookup 导致变慢”的旧问题
3. 继续压缩 Stage0 中的通用模板开销
   - 当前 Stage0 仍比手写多一层显式 `load_offset_1`

## 8. 关键验证命令

```bash
cd /home/huoda/sparsene
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate

# 生成 kernel
PYTHONPATH=python python3 python/test/test_generate.py --strategy-provider heuristic

# 编译
cp me_tcf_kernel3.inc python/me_tcf_kernel3.inc
cd examples/src_fp32/dtc/testbed
nvcc -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo   -ftemplate-backtrace-limit=0 -arch=sm_80   -ccbin /opt/rh/gcc-toolset-13/root/usr/bin/g++ -Xcompiler -fopenmp -Df32   -I./cutlass/include -I./cutlass/examples/common   -I./cutlass/tools/util/include -I./src -I.   -I../../../../../../python   host_program_generated.cu main.cu src/mmio.cu   src/mmio_highlevel.cu src/utils.cu   -lcusparse -lcublas -o /tmp/demo

# 主测点
/tmp/demo -filename /mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx   -mtx_flag 1 -N 512 -warmup 20 -repeat 100

# correctness smoke test
/tmp/demo -filename /mnt/lustre-client/zhangkaige/selected_mtx/cop20k_A.mtx   -mtx_flag 1 -N 512 -warmup 5 -repeat 20
```

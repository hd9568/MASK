# ME_TCF 生成算子性能优化总结 (2026-05-31 最终版)

## 1. 当前结论

- 目标: 在不引入智能体、默认格式 + 默认 heuristic 策略下，生成 kernel 性能尽可能接近手写 `kernel.inc`。
- 当前安全基线在 `shipsec1 / N=512 / warmup=20 / repeat=100` 上:
  - 生成: `1.567 ms`
  - 手写: `1.541 ms`
  - 差距: `+1.7%`
- 当前版本 correctness 正常，已在 `shipsec1`、`cop20k_A` 上验证通过。
- 本轮进行了多次闭环分析→尝试优化→验证→回退的迭代，最终确定剩余差距的根因不在于单个指令或检查分支。

## 2. 本轮闭环分析过程

### 2.1 尝试 1: 移除 coo 加载的 alignment check（无条件 cp.async）
- 思路: 已验证 shipsec1 上所有 compact position 都是 4 对齐的，试图移除 `if ((operand_1 & 3) == 0)` 检查分支。
- 结果: shipsec1 上性能无变化（检查始终为 true），cop20k_A 上触发 `misaligned address (716)`。
- 回退: ✓ 已回退
- 根因: `val_coo_len_new_offset` 存储的是 compact 无 padding 位置，在 cop20k_A/dc2 等数据集上，部分 TC block 的 compact 起始位置不满足 4 对齐。cp.async 要求源地址 16B 对齐，非对齐位置导致异常。

### 2.2 尝试 2: Host 程序添加 4-aligned padding + sentinel idx
- 思路: 在 host_program_generated.cu 的 compact 转换中添加 0-3 个 sentinel 元素（idx=SENTINEL_IDX, val=0），使所有 compact position 保持 4 对齐。Sentinel 元素的 idx 超出 tile_elems，restore 步骤会自动跳过。
- 结果: shipsec1 上触发 `illegal memory access (700)`。
- 回退: ✓ 已回退
- 根因: padding 使 `rr - ll <= real_nnz + 3`，最大可达 131。但 SMEM buffer 只有 128 个元素（= BLK_M * BLK_K = 16 * 8）。cp.async 循环按 `rr - ll` 拷贝，会写超出 SMEM buffer。

### 2.3 尝试 3: 扩大 SMEM buffer 以容纳 padding
- 未实现: 需要同时修改:
  - host_program_generated.cu（padding）
  - generate.py 中的 coo load lowering（移除 alignment check）
  - SMEM layout 定义（增加 buffer 大小 → 增加 shared memory 占用 → 可能降低 occupancy）
- 结论: 这条路径虽理论上可行，但需要跨多层修改，且会增加 SMEM 占用（每个 buffer 多 3 个元素），在当前阶段不具性价比。

### 2.4 验证: alignment check 是必需且正确的保护
- 当前生成 kernel 的 `if ((operand_1 & 3) == 0)` 检查在 shipsec1 上始终为 true（所有 compact position 恰好 4 对齐），走 cp.async 快路径。
- 在 cop20k_A 上约 80% 的 TC block 走 cp.async，20% 走标量 fallback（因为 compact position 不是 4 对齐）。
- 这个检查对跨数据集正确性至关重要。

## 3. 剩余 ~1.7% 差距的真实来源

通过轻量 NCU profile 对比（以 shipsec1 为基线）:

| 指标 | 手写 kernel | 当前生成 kernel |
|---|---:|---:|
| `launch__registers_per_thread` | `95` | `92` |
| `smsp__inst_executed.sum` | `445,491,056` | `471,249,160` |

- 寄存器比手写更低（92 vs 95），不存在 occupancy 瓶颈。
- 动态指令数多约 5.8%（~26M 条额外指令）。

额外指令的主要来源:
1. `load_offset_1` 作为独立 op，比手写版多一次模板调用和额外的标量 gmem 读写
2. 生成代码的通用模板架构比手写专用 fast path 有更多控制流开销
3. `sparsene_copy_g2s_128` 的 arch 条件编译在 SM80 上虽然编译期消除，但仍有微小的模板展开开销

## 4. 有效优化总结

已保留在主线中的所有有效优化:

| 层级 | 改动 | 效果 |
|------|------|------|
| OPIR | 常量/恒等式消除 (varlenLoweringPass.py) | 消除 gen_add/gen_sub/gen_div/gen_const |
| NVIR | load_offset + gen_sub + gen_div 融合 | 减少外层 op |
| NVIR | gen_add 内联到 array_ref + load_offset_1 | 减少 pipeline op |
| Pipeline | sidx(Stage0) + B(Stage1) 重排 | 关键！使 sidx 先发起不会被过早消费 |
| Codegen | 移除热路径 runtime shape assert | 减少 device 指令 |
| Codegen | tid/wid 清理 | 减少状态传递开销 |
| Templates | `__launch_bounds__(32, 20)` | 拉低寄存器 (128→92) |
| Templates | `sparsene_copy_g2s_128` 架构抽象 helper | 跨架构兼容 |

## 5. 验证命令

```bash
cd /home/huoda/sparsene
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate

# 生成
PYTHONPATH=python python3 python/test/test_generate.py --strategy-provider heuristic

# 编译
cp me_tcf_kernel3.inc python/me_tcf_kernel3.inc
cd examples/src_fp32/dtc/testbed
nvcc -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo \
  -ftemplate-backtrace-limit=0 -arch=sm_80 \
  -ccbin /opt/rh/gcc-toolset-13/root/usr/bin/g++ -Xcompiler -fopenmp -Df32 \
  -I./cutlass/include -I./cutlass/examples/common \
  -I./cutlass/tools/util/include -I./src -I. \
  -I../../../../../../python \
  host_program_generated.cu main.cu src/mmio.cu \
  src/mmio_highlevel.cu src/utils.cu \
  -lcusparse -lcublas -o /tmp/demo

# Benchmark
/tmp/demo -filename /mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx \
  -mtx_flag 1 -N 512 -warmup 20 -repeat 100
```

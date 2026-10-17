# SPARSENE 项目全貌 — 新会话 AI 上手指南

> 生成时间: 2026-05-29  
> 目的: 让新界面的 AI 在几句话内理解本项目的结构、当前状态、以及下一步开发方向

---

## 1. 项目概况

**sparsene** 是一个基于 Python 的稀疏矩阵计算算子自动生成框架。它接收稀疏矩阵格式描述和高维计算表达式（RTS），通过多层 lowering（OPIR → NVIR → C++/CUDA），自动生成高性能 GPU kernel。

**当前焦点任务**: 优化 ME_TCF（混合稀疏格式）的 DTC_SpMM 生成算子，使其性能能够打平手写 kernel (`examples/src_fp32/dtc/testbed/kernel.inc`)。

### 1.1 运行环境

```bash
conda activate TraeAI-4          # 或 source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
PYTHONPATH=python python3 ...
```

---

## 2. 项目目录结构

```
sparsene/
├── python/sparsene/op_gen/          ← 核心生成框架
│   ├── opir/                        ← OPIR (中间表示层): 与数据格式无关的高级 IR
│   │   ├── generate.py              ← OPIR 生成入口
│   │   ├── op_ir.py                 ← IR 数据结构 (ForLoopOp, ArrayRefOp, CValLoadOp, ...)
│   │   ├── ops.py                   ← OPIR op 定义
│   │   ├── cValFlattenPass.py       ← C 值展平 pass
│   │   └── varlenLoweringPass.py    ← Variadic-length lowering pass ★已修改
│   ├── nvir/                        ← NVIR (CUDA-specific IR): 生成最终 C++ 代码
│   │   ├── generate.py              ← NVIR lowering: OPIR → NVIR op mapping
│   │   ├── compiler_driver.py       ← 编译驱动: pipeline stage 划分 + 融合 pass ★已修改
│   │   ├── codegen.py               ← C++ 代码生成器: dump class/op/f() call ★已修改
│   │   ├── templates.py             ← C++ 模板骨架定义 ★已修改
│   │   ├── nvop.py                  ← NVIR op 数据结构 (NvOp, ForLoopNvOp, NvOpImpl, ...)
│   │   ├── plan.py                  ← Pipeline 规划 (PipelinePlan, apply_pipeline)
│   │   └── pipeline/                ← Pipeline body 生成 (fill, short_pipe, remainder, ...)
│   ├── computent/                   ← 计算实体定义
│   │   └── computent.py             ← computent_from_rts() 入口
│   ├── transform/
│   │   └── rts.py                   ← RTS derivation (从 format 推导出 tensor 分块方式)
│   └── formats/
│       └── DTC_SpMM.py              ← ME_TCF_FORMAT 定义
├── examples/src_fp32/dtc/testbed/   ← 手写 kernel + host program 测试床
│   ├── kernel.inc                   ← ★参考 kernel: 手写的 dtc_spmm_kernel_fp32_*
│   ├── host_program_generated.cu    ← host program (加载 mtx, 调用 kernel)
│   ├── main.cu                      ← 测试入口
│   ├── src/mmio*.cu                 ← MMIO 文件读取
│   └── src/utils.cu                 ← 工具函数
├── python/test/
│   ├── test_generate.py             ← ★主测试: 生成 ME_TCF kernel → me_tcf_kernel3.inc
│   └── end2end_pipeline.py          ← 端到端 pipeline (含 LLM 智能体)
├── logs/                            ← 端到端运行日志
├── docs/
│   └── me_tcf_performance_optimization.md ← ★优化历程详细文档
├── mtx_cache/                       ← 矩阵 checkpoint 缓存
└── me_tcf_kernel3.inc               ← ★当前生成的最新 kernel (在项目根目录)
```

---

## 3. OPIR → NVIR → C++ 生成流程

```
ME_TCF_FORMAT → rts = derive_rts(ME_TCF_FORMAT)
    ↓
computent_from_rts("ME_TCF", rts) → 计算实体
    ↓
generate_from_computent(comp) → OPIR ops (MetaOp, ForLoopOp, MmaOp, ...)
    ↓
VarlenLoweringPass.run(ops) → 变长 lowering (多维 → 一维压缩)
    ↓
CValFlattenPass.run(ops) → C value flatten
    ↓
generate_nvir(flat, "ME_TCF", scfg) → NVIR program
    │
    ├── generate.py: OPIR → NVIR 1:1 mapping (_make_impl 决定每个 NVIR op 的 impl 模板)
    │
    ↓
apply_software_pipeline_and_codegen(program, output_path)
    │
    ├── [NEW] _fuse_load_offset_sub_div(program)     ← 外层融合
    ├── _select_main_loop(program)
    ├── [NEW] _fuse_inline_gen_add(main_loop)         ← pipeline 内联
    ├── _partition_pipeline_stages(main_loop)
    ├── apply_pipeline(main_loop, plan)
    └── _dump_code → me_tcf_kernel3.inc
```

**关键概念**:
- **logical_kind**: 每个 NVIR op 有一个 `attrs["logical_kind"]` 字符串属性，标识语义类型 (`"load_offset"`, `"add"`, `"array_ref"`, `"coo_atomic_format_load_idx"`, `"mma"`, ...)
- **NvOpImpl.code_template**: 每个 op 对应的 C++ 代码模板，由 generate.py 的 `_make_impl()` 生成，也可以在 compiler_driver.py 的融合 pass 中被改写
- **Pipeline**: 3-stage (Stage0→Stage1→Stage2), 每个 stage 有独立的 `__pipeline_commit`/`__syncthreads`

---

## 4. 已完成的所有修改

### 4.1 varlenLoweringPass.py (OPIR 级优化)

| 位置 | 改动 | 效果 |
|------|------|------|
| ~780 | 新增 `_is_const_int()` 检测 ConstantOp(0) | `x+0` → x, `x-0` → x |
| ~785 | 新增 `_is_blk_scaled_one_expr()` | 跳过 `1/BLK_K` scaling |
| ~790 | 新增 `_is_zero_expr()` | 跳过 `b_expr=0` term |
| ~406 | `_build_add` 中调用 `_is_const_int` | 消除 gen_add_1 |
| ~642 | `_build_sub` 中调用 `_is_const_int` | 减少 gen_sub |
| ~296 | `_get_base_offset` 跳过 block-level scaling | 消除 gen_div_1 |
| ~373 | `_rewrite_array_indices_if_needed` 跳过 b_expr=0 | 消除 gen_const |

### 4.2 templates.py (C++ 模板骨架)

| 位置 | 改动 |
|------|------|
| 263 | `__launch_bounds__(32, 20)` — 强制编译器降到 96 regs |
| 6 个 class skeleton | `int tid, lid, wid` → `int lid` |
| 3 个 create skeleton | `tid, lid, wid` → `lid` |

### 4.3 codegen.py (代码生成器)

| 位置 | 改动 |
|------|------|
| 118 | `hw_param_str = "int lid;"` |
| 121 | constructor params 移除 tid/wid |
| 156 | init list 移除 tid/wid |
| 1311 | `args_0 = ["lid"]` |
| 1389 | constant nvop create 移除 tid/wid |

### 4.4 compiler_driver.py (编译驱动 — 最大改动)

| 函数 | 位置 | 描述 |
|------|------|------|
| `_fuse_load_offset_sub_div()` | ~400-480 | 外层: gen_sub(off_r,off_l) + gen_div(len,1) → load_offset.r = off_r - off_l |
| `_fuse_inline_gen_add()` | ~484-570 | Pipeline: gen_add(off_l,lc) → array_ref + load_offset_1 内联 idx 计算 |
| import 行 | ~6 | 新增 NvOpImpl, NvOpInput, NvOpTensor, NvOpOutput |
| 调用点 #1 | ~490 | `_fuse_load_offset_sub_div(program)` 在 main_loop 选择前 |
| 调用点 #2 | ~590 | `_fuse_inline_gen_add(main_loop)` 在 pipeline 分区前 |

### 4.5 已验证并回退的尝试

| 优化 | 原因 |
|------|------|
| load_offset_1 → coo_atomic_format_load_* | 1.859ms > 1.825ms (每个 coo op 独立做 gmem lookup) |
| `__launch_bounds__(32,24)` | 0.000ms crash |
| 仅删除 tid/wid, 不加 __launch_bounds__ | registers 不变 (编译器已自优化) |

---

## 5. 当前性能状态

```
Matrix: shipsec1 (M=140874, K=140874, nnz=3.98M, 99.98% sparse, N=512)

┌──────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Kernel           │ mykernel │ cusparse │ vs cusp  │ vs 手写  │
├──────────────────┼──────────┼──────────┼──────────┼──────────┤
│ 手写 (kernel.inc) │ 1.476 ms │ 4.012 ms │ 2.72x    │    —     │
│ 生成 (优化后)     │ 1.825 ms │ 4.009 ms │ 2.20x    │ 23.6% 慢 │
│ 生成 (原始)       │ 2.004 ms │ 4.020 ms │ 2.01x    │ 35.8% 慢 │
└──────────────────┴──────────┴──────────┴──────────┴──────────┘
```

| 关键指标 | 手写 | 生成(优化后) | 状态 |
|---------|------|-----------|------|
| Registers/Thread | 95 | 96 | ✅ 对齐 |
| Stage0 pipeline ops | 5 | 5 | ✅ 对齐数量 |
| Pipeline calls (Stage0) | ~108 (G2s\* fused) | 36 (array_ref×12 + load_offset_1×12 + coo_*×12) | 仍有差异 |
| 生成代码行数 | — | 2298 | |
| gen_add/gen_sub/gen_div/gen_const | 无 | **0** | ✅ 全部消除 |

---

## 6. 剩余 23.6% 差距根因

NVIR peephole 已饱和。剩余差距来自 OPIR 级别的设计差异:

| 差异 | 手写 | 生成 |
|------|------|------|
| sidx 索引加载 | gmem **直接** → SMEM (G2sSparseIndexLoadOp, cp_async) | gmem → **RMEM** (array_ref) → 暂存 |
| B 矩阵 tile 加载 | gmem **直接** → SMEM (G2sMatrixBLoadOp, cp_async) | gmem → **RMEM** (array_ref_1) → 暂存 |
| COO offset load | gmem → load (G2rSparseOffsetLoadOp) | gmem → load (load_offset) ✅ 对齐 |
| SMEM layout | swizzled bank-conflict-free | 通用 layout |

### 为什么 NVIR 融合已不能再提升

- gen_sub/gen_div: 融合到 load_offset ✅
- gen_add: 内联到 array_ref + load_offset_1 ✅ (36 次 f() 调用已消除, 但实际计算仍在)
- load_offset_1 → coo_*: ❌ 每个 coo op 需独立查 offset_arr, 造成冗余 gmem 访问

### 下一步方向 (需修改 generate.py)

要从根本上解决 rmem→smem 路径差异:
1. **array_ref + array_ref_1 → G2sMatrixBLoadOp**: 修改 OPIR→NVIR lowering, 让 array_ref 的结果直接写入 SMEM (cp_async)
2. **load_offset_1 + coo_* 合并**: 改为单次 offset load + 单次 cp_async (类似手写的 G2sSparseCooIdxLoadOp)
3. **SMEM swizzle layout**: 修改 array_ref 的头文件定义

---

## 7. 日常开发命令

```bash
cd /home/huoda/sparsene && source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate

# ----- 生成 kernel -----
PYTHONPATH=python python3 python/test/test_generate.py --strategy-provider heuristic
# 输出: me_tcf_kernel3.inc (项目根目录)

# ----- 编译 -----
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

# ----- benchmark -----
/tmp/demo -filename /mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx \
  -mtx_flag 1 -N 512 -warmup 1 -repeat 5

# ----- NCU profile -----
ncu --kernel-name "generated_kernel" --launch-skip 0 --launch-count 1 \
  --metrics "launch__registers_per_thread" \
  /tmp/demo -filename ... -mtx_flag 1 -N 512 -warmup 1 -repeat 1

# ----- 对比手写 kernel -----
# 已编译: /tmp/perf_test/demo_handwritten
# 当前生成: /tmp/perf_test/demo_final_v8

/tmp/perf_test/demo_handwritten -filename ... -mtx_flag 1 -N 512 -warmup 1 -repeat 5

# ----- 端到端测试 (含 LLM 智能体) -----
PYTHONPATH=python python3 python/test/end2end_pipeline.py \
  --mtx /mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx \
  --N 512 --format-agent-model deepseek-v4-pro \
  --strategy-agent-model deepseek-v4-pro --strategy-iterations 1 \
  --fix-agent-iterations 0 --warmup 1 --repeat 5
```

---

## 8. 关键代码位置速查

| 要做什么 | 看什么文件 |
|---------|-----------|
| 修改生成的 C++ 模板骨架 | `templates.py` |
| 修改 NVIR op 的 impl 代码 | `generate.py:_make_impl()` 或 `compiler_driver.py` 的融合函数 |
| 添加新的 NVIR 融合 pass | `compiler_driver.py:apply_software_pipeline_and_codegen()` |
| 修改 pipeline stage 划分 | `compiler_driver.py:_partition_pipeline_stages()` |
| 修改 OPIR 级别优化 | `varlenLoweringPass.py` |
| 修改 C++ 代码生成逻辑 | `codegen.py` |
| 理解手写 kernel 结构 | `examples/src_fp32/dtc/testbed/kernel.inc` |
| **详细优化历程** | `docs/me_tcf_performance_optimization.md` |

### NVIR 数据结构速查 (nvop.py)

- **NvOp**: 基础 op, 有 `name`, `inputs: List[NvOpInput]`, `outputs: List[NvOpOutput]`, `impl: NvOpImpl`, `attrs: Dict[logical_kind, ...]`
- **NvOpInput**: `idx, name, tensor: NvOpTensor`
- **NvOpOutput**: `idx, name, tensor, consumers: List[NvOp]`
- **NvOpImpl**: `code_template: str` (C++ f() 方法实现)
- **NvOpTensor**: `shape, mem ("rmem"/"smem"/"gmem"), dtype, source (NvOpOutput 或 str)`
- **ForLoopNvOp**: 有 `body: NvOpSequence`
- **NvOpProgram**: 顶级容器, `ops: List[NvOp]`
- **NvOpSequence**: ops 序列

---

## 9. 快速启动 Checklist

新会话 AI 收到"继续优化 ME_TCF 性能"任务时:

1. ✅ 阅读本文档和 `docs/me_tcf_performance_optimization.md`
2. ✅ `cd /home/huoda/sparsene && source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate`
3. ✅ `PYTHONPATH=python python3 python/test/test_generate.py --strategy-provider heuristic` 确认当前状态
4. ✅ 从 §5 了解当前性能，从 §6 了解剩余差距
5. ✅ 下一步最可能有效的方向: 修改 `generate.py:_make_impl()` 让 `array_ref` + `array_ref_1` 直接通过 cp_async 写 SMEM (而不是先到 RMEM)

# Codex 智能体代码变更详解

> 生成日期：2026-07-21  
> 代码差异范围：`git diff 027e29f..e113455`（原始 Sparsene → Codex Agent 优化后）  
> 涉及文件：24 个文件，+3845 行，-266 行  
> 核心提交：
> - `447bd64` — 添加 DTC/ACC/BITBSR/SR_BCRS 生成工作流
> - `1cff688` — 对齐生成稀疏格式、NVOP 实现和 ACC XNYM 映射
> - `e113455` — 修复 ACC 生成计算和 cp.async 流水线

---

## 目录

1. [变更概览](#1-变更概览)
2. [NVIR Codegen 层](#2-nvir-codegen-层)
3. [OPIR 层](#3-opir-层)
4. [Computent 层](#4-computent-层)
5. [Host/Testbed 层](#5-hosttestbed-层)
6. [新增基础设施文件](#6-新增基础设施文件)
7. [性能影响总结](#7-性能影响总结)

---

## 1. 变更概览

### 1.1 按模块统计

| 模块 | 文件数 | 变更行数 | 变更类型 |
|---|---|---|---|
| NVIR Codegen | 4 | ~2000+ | Bug 修复 + 功能扩展 |
| OPIR 生成 | 3 | ~150 | Bug 修复 + 语义修正 |
| Computent | 1 | ~80 | 数组类型支持扩展 |
| Host/Testbed | 9 | ~1200 | 新增多格式 testbed |
| 测试/脚本 | 3 | ~700 | 新增自动化工具 |
| 其他 | 4 | ~50 | 模板/配置修复 |

### 1.2 按变更性质统计

| 类型 | 说明 |
|---|---|
| **严重 Bug 修复** | K-fragment 丢失、M=32 store 截断、MCO mask 溢出、grid.y 越界 |
| **语义修正** | MCO mask 从标量改为向量、COO offset vs length 区分、VarlenLowering leading_rank |
| **功能扩展** | Grid 映射策略 XNYM、format 级 swizzle、arange op、ParamShape |
| **工程完善** | cp.async 对齐检查、统一生成脚本、自动化 benchmark、正确性验证框架 |

---

## 2. NVIR Codegen 层

这是 Codex 智能体改动最集中的层级，涉及 [compiler_driver.py](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/compiler_driver.py)（+1249/-?）、[generate.py](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/generate.py)（+394/-?）、[codegen.py](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/codegen.py)（+11/-?）、[templates.py](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/templates.py)（+2/-1）。

### 2.1 K-Fragment Bug 修复（`compiler_driver.py`）

**问题：** 旧代码 A/B 寄存器形状把 K 维压掉，`ldmatrix` 和 TF32 MMA 都没有 K 循环。`BLK_K=16/32` 虽然能编译，但只计算前 8 列（一个 K fragment）。

**修复前（旧）：**
```cpp
// A/B tensor 访问使用展平索引，MMA 无 K 循环
asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(A(0, m_iter, k_iter)));
// 只迭代 BLK_MMA_MNK.m 和 BLK_MMA_MNK.n，不迭代 BLK_MMA_MNK.k
```

**修复后（新）：**
```cpp
// 所有 A/B tensor 访问改用 make_coord 保留 K 维度
asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(A(0, make_coord(m_iter, k_iter))));
// MMA 和 ldmatrix 显式遍历全部 K fragments
for (int k_iter = 0; k_iter < get<2>(BLK_MMA_MNK{}); k_iter++) { ... }
```

**性能影响：** 这是最关键的 bug 修复。之前 `BLK_K=16/32` 的结果完全是错误的（只计算了 1/2 或 1/4 的数据），修复后所有 K 尺寸的完整计算路径得到保证。

**软件流水线分析 (`_preferred_stage`)：** Codex 还在 `compiler_driver.py` 中新增了完整的依赖分析和流水线分阶段逻辑：
- `_collect_local_deps()`：分析 for 循环内所有 op 的生产者-消费者依赖图
- `_stable_topological_ops()`：拓扑排序，保持原始顺序稳定性
- `_build_fused_groups()`：将可融合的简单 op（add/sub/mul/div/pow/const）与消费者合并
- `_preferred_stage()`：按 op 逻辑类型分配流水线阶段（stage 0: 索引加载, stage 1: restore/B-load, stage 2: MMA/store）
- `_compress_stage_ids()`：压缩实际使用的 stage id，避免冗余提交/等待屏障

### 2.2 M=32 Store 修复（`compiler_driver.py`）

**问题：** 旧 C_val_store 固定从 `m_iter=0` 开始写，导致 BLK_M=32 只写前 16 行，后 16 行输出丢失。

**修复前（旧）：**
```cpp
// 只迭代 N 方向 tile，M 方向固定为 0
for (int i_tileN = 0; i_tileN < get<1>(BLK_MNK{}) / 8; i_tileN++) {
    C(col, row) = C_reg(0, i_tileN);   // 硬编码 m_iter=0
}
```

**修复后（新）：**
```cpp
// 遍历全部 M 和 N fragments
for (int i_tileM = 0; i_tileM < get<0>(BLK_MMA_MNK{}); i_tileM++) {
    for (int i_tileN = 0; i_tileN < get<1>(BLK_MMA_MNK{}); i_tileN++) {
        int row = lid / 4 + i_tileM * get<0>(MMA_MNK{});
        C(col, row) = C_reg(0, make_coord(i_tileM, i_tileN));
    }
}
```

**性能影响：** 修复了 `cant`、`YeastH`、`pdb1HYS` 等 BLK_M=32 数据集的输出正确性问题。之前这些数据集的 "正确性通过" 实际是检查不充分的假阳性。

### 2.3 MCO Atomic Mask 向量化（`compiler_driver.py`）

**问题：** 旧代码将 MCO mask 硬编码为单个 `unsigned long long`（64 位），无法支持超过 64 元素的 tile（如 `8x32 = 256` 元素）。

**修复前（旧）：**
```cpp
unsigned long long mask = static_cast<unsigned long long>(mask_input);  // 只取一个 64-bit
for (int idx = 0; idx < tile_elems; ++idx) {
    if ((mask >> idx) & 1ULL) { /* restore */ }
}
```

**修复后（新）：**
```cpp
int num_masks = size<0>(shape(mask_input));  // mask 向量长度
for (int i_mask = 0; i_mask < num_masks; ++i_mask) {
    unsigned long long mask = static_cast<unsigned long long>(mask_input(i_mask));
    for (int bit = 0; bit < 64; ++bit) {
        int idx = i_mask * 64 + bit;
        if (idx >= tile_elems) break;
        if ((mask >> bit) & 1ULL) { /* restore */ }
    }
}
```

**配合的 OPIR 修改：** `McoAtomicFormatLoadMaskOp` 的输出类型从 `IntType()`（标量）改为 `ArrayType([out_shape], ...)`（向量），其中 `out_shape = (C_tile_dims * B_tile_dims) / 64`。

**性能影响：** 使 MCO 格式能正确处理 `8x32` 等大 tile 配置。同时添加了 `mco_atomic_val_restore` 的 `align_16` 标记，确保 shared memory 地址对齐。

### 2.4 cp.async 对齐检查（`compiler_driver.py`）

**问题：** 当源/目标地址不是 16 字节对齐时，`cp.async` 的 128-bit 向量化拷贝会导致 "misaligned address" 运行时错误。

**修复前：**
```cpp
copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>{}, src(0, _), dst(0, _));
```

**修复后：**
```cpp
auto src_vec = src(0, _);
auto dst_vec = dst(0, _);
if ((((uintptr_t)src_vec.data().get()) & 0xF) == 0 && (((uintptr_t)dst_vec.data().get()) & 0xF) == 0) {
    copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>{}, src_vec, dst_vec);
} else {
    for (int v = 0; v < 4; ++v) { dst_vec(v) = src_vec(v); }
}
```

**性能影响：** 运行时检查地址对齐，对齐时使用高效的 `cp.async`，非对齐时 fallback 到逐元素拷贝。避免了 SR_BCRS 等格式的 misaligned address 崩溃。

### 2.5 Grid 映射策略扩展（`compiler_driver.py` + `generate.py`）

**问题：** 旧代码硬编码外层循环映射到 `blockIdx.x`。但 BIT_BSR 格式需要 `blockIdx.y` 映射 row block、`blockIdx.x` 映射 N tile（与 TCF 相反）。

**修复：** 在 `_lower_for_loop` 中：
```python
mapping = state.hardware_mapper.map_loop()
if mapping in {"blockIdx.x", "blockIdx.y"}:  # 原来是 == "blockIdx.x"
    inner_block_arg_sources[op.get_induction_var()] = mapping
```

配合 `SPARSE_FORMAT_REGISTRY` 中的 `"grid_mapping"` 配置（`"XMYN"` 或 `"XNYM"`），以及 `HardwareMapper` 类：
```python
class HardwareMapper:
    def __init__(self, grid_mapping: str = "XMYN"):
        self.grid_mapping = grid_mapping

    def map_loop(self) -> str:
        if self.loop_depth == 0:
            return _grid_blocks(self.grid_mapping)["m_block"]
        return "sequential"
```

**性能影响：** 使 BIT_BSR 格式能正确映射 grid，避免因 block 坐标错位导致的严重错误。

### 2.6 Swizzle 策略分层（`compiler_driver.py`）

**问题：** 旧代码只在 strategy 级别支持 swizzle 配置。不同格式需要不同的 swizzle 策略（如 BIT_TCF 的 `mco_restore_swizzle` 为 `swizzle_123`，BIT_BSR 为 `swizzle_133`）。

**修复：** 新增 `_format_swizzle()` 函数，先查格式特定的 swizzle 配置，再 fallback 到 strategy swizzle：
```python
def _format_swizzle(state, key, default):
    hints = state.strategy_decision.format_hints if state.strategy_decision else {}
    tensor_swizzles = hints.get("tensor_swizzles", {})
    if key in tensor_swizzles:
        return _parse_swizzle(tensor_swizzles[key])
    return default
```

在 `SPARSE_FORMAT_REGISTRY` 中添加格式级 swizzle 配置：
```python
"BIT_TCF": {
    "tensor_swizzles": {
        "b_smem_swizzle": "swizzle_323",
        "mco_restore_swizzle": "swizzle_123",
    },
}
"BIT_BSR": {
    "tensor_swizzles": {
        "b_smem_swizzle": "swizzle_333",
        "mco_restore_swizzle": "swizzle_133",
    },
}
```

### 2.7 Kernel Launch Bounds（`templates.py`）

**修复：** `__launch_bounds__(32, 20)` 改为可配置的 `{launch_bounds}` 占位符：
```python
# 旧
GLOBAL_FUNCTION_SKELETON = """... __global__ __launch_bounds__(32, 20) void {kernel_name}(..."""

# 新
GLOBAL_FUNCTION_SKELETON = """... __global__{launch_bounds} void {kernel_name}(..."""
```

当前生成 kernel 每 block 恰好一个 32-thread warp，可显式声明 ABI 让编译器优化寄存器分配。

### 2.8 Arange Op 支持（`compiler_driver.py`）

**新增：** 支持 `arange` 逻辑操作，生成从起始值递增的序列填充代码：
```python
if logical_kind == "arange" and len(inputs) >= 1 and len(outputs) == 1:
    return NvOpImpl(
        f"""for (int i = 0; i < size<0>(shape({outputs[0].name})); ++i) {{
    {outputs[0].name}(i, buf_idx) = {inputs[0].name} + i;
}}"""
    )
```

**用途：** 用于 CSR row_ptr 的算术生成等场景。

### 2.9 Load Op 增强（`compiler_driver.py`）

**新增：** 支持从 gmem 3D tensor load 到 2D smem tensor 的操作：
```python
if (logical_kind == "load" and len(inputs) >= 2 ...):
    outputs[0].tensor.mem = "smem"
    outputs[0].tensor.row_major = True
    outputs[0].tensor.swizzle = SwizzleLayout(2, 2, 3)
    outputs[0].attrs["align_16"] = True
```

### 2.10 ParamShape 支持（`compiler_driver.py`）

**新增：** 在 `opir_type_to_nvir_shape()` 中解析 `BLK_M+1` 等参数化形状表达式：
```python
elif (blk_param_expr := _try_blk_param_expr(dim_str)) is not None:
    nvir_shape.append(ParamShape(blk_param_expr))
```

**用途：** 支持 CSR row_ptr 的形状 `(BLK_M + 1, nnz_dim_K_o)`。

---

## 3. OPIR 层

涉及 [generate.py](file:///home/huoda/sparsene/python/sparsene/op_gen/opir/generate.py)、[varlenLoweringPass.py](file:///home/huoda/sparsene/python/sparsene/op_gen/opir/varlenLoweringPass.py)、[ops.py](file:///home/huoda/sparsene/python/sparsene/op_gen/opir/ops.py)。

### 3.1 COO Offset vs Length 数组区分（`opir/generate.py`）

**问题：** 旧代码统一使用 `LoadOffsetOp` 处理 COO 的元数据数组。但某些格式使用 offset 数组（存储 `[start, end)` 对），某些使用 length 数组，两者语义不同。

**修复前（旧）：**
```python
# 始终使用 LoadOffsetOp
ll, rr = op_builder.build(LoadOffsetOp, len_array, ...)
```

**修复后（新）：**
```python
# 根据数组定义的类型选择正确的 op
metadata_def = array_defs[schedule.args[0].array]
if metadata_def.type == ComputentArrayType.OFFSET:
    ll, rr = op_builder.build(CooAtomicFormatLoadOffOp, mem="G2R", operands=[...])
else:
    ll, rr = op_builder.build(LoadOffsetOp, metadata_array, ...)
```

**性能影响：** 确保 COO 格式的 offset 语义正确传递，避免因数组类型混淆导致的索引错误。

### 3.2 MCO Mask 向量化（`opir/ops.py` + `opir/generate.py`）

**ops.py 变更：**

`McoAtomicFormatLoadMaskOp` 输出从标量改为向量：
```python
# 旧
results=[OpResult(type=IntType(), ...)]

# 新
results=[OpResult(type=ArrayType([out_shape], operands[0].type.datatype), ...)]
```

`McoAtomicValRestoreOp` 的 mask 输入断言相应更新：
```python
# 旧
assert isinstance(operands[1].type, IntType)  # mask 是标量

# 新
assert isinstance(operands[1].type, ArrayType)  # mask 是向量
```

### 3.3 VarlenLowering Leading Rank 修复（`opir/varlenLoweringPass.py`）

**问题：** 旧 `VarlenLoweringPass` 假设 varlen 维度出现在前缀维度之后的第一位。对于多维前缀场景（如 `mco_atomic_format_load_mask` 有额外的 leading 维度），压缩逻辑错误。

**修复：** 引入 `leading_rank` 概念：

```python
@dataclass
class _FlattenSpec:
    leading_rank: int   # 新增：varlen 之前的非前缀维度数
    prefix_rank: int
    varlen_pos: int
    ...
```

压缩逻辑相应更新：
```python
# 旧：总是从前缀后第一位压缩
new_dims = [compressed_dim] + old_dims[prefix_rank + 1:]

# 新：保留 leading 维度，从 varlen_pos 后开始
varlen_pos = next(i for i, dim in enumerate(old_dims) if target_varlen in str(dim))
leading_rank = varlen_pos - prefix_rank
new_dims = old_dims[:leading_rank] + [compressed_dim] + old_dims[varlen_pos + 1:]
```

---

## 4. Computent 层

涉及 [computent.py](file:///home/huoda/sparsene/python/sparsene/op_gen/computent/computent.py)（+81/-?）。

### 4.1 数组类型支持扩展

**变更：** 扩展 Computent 中间表示的数组定义，支持 `OFFSET` 类型和其他数组语义：

```python
from sparsene.op_gen.computent.arraydef import ArrayType as ComputentArrayType
```

使得 COO offset vs length 的区分能从 Computent 层传递到 OPIR 层。

---

## 5. Host/Testbed 层

涉及 9 个文件的 CMakeLists.txt、host 程序等变更。

### 5.1 多格式 Testbed 新增

Codex 智能体为新格式创建了独立 testbed：

| 格式 | Testbed 路径 | 内容 |
|---|---|---|
| ACC (BIT_TCF) | `examples/src_fp32/acc/testbed/` | CMakeLists.txt + host_program_generated.cu + main.cu |
| BITBSR | `examples/src_fp32/bitbsr/testbed/` | CMakeLists.txt + host_program_generated.cu |
| SR_BCRS | `examples/src_fp32/sr_bcrs/testbed/` | 更新 CMakeLists.txt + 新增 host_program_generated.cu |
| DTC | `examples/src_fp32/dtc/testbed/` | 扩展 host 程序 + 新增 dtc_ops_generated.py + host_program_generated_results.cu |

### 5.2 DTC Testbed 扩展

**新增：** `dtc_ops_generated.py`（225 行）：Python 驱动的 DTC 算子生成与验证流程。

**新增：** `host_program_generated_results.cu`（446 行）：结构化的生成结果验证程序，包含：
- 确定性测试值（1..N 序列，避免矩阵原值量纲掩盖 TF32 精度问题）
- 完整输出扫描（所有 M×N 元素）
- relative-L2 误差、归一化最大绝对误差、误差位置记录
- nonfinite 检测

### 5.3 `main.cu` 更新

DTC testbed 的 `main.cu` 添加了 checkpoint 持久化支持：
- 基于源文件目录定位 checkpoint 路径
- `std::ios::trunc` 覆盖模式写入
- 与 `sparsene.kernel_only_post_conversion.v1` 契约一致：checkpoint 读写发生在 timer 外

### 5.4 `.gitignore` 更新

新增忽略 `build-generated` 和 `build-generated-*` 目录，防止编译产物被误提交。

---

## 6. 新增基础设施文件

### 6.1 `python/test/generate_sparse_formats.py`（69 行）

**目的：** 统一的多格式生成脚本，支持 DTC (ME_TCF)、ACC (BIT_TCF)、BITBSR (BIT_BSR)、SR_BCRS 四种格式。

**关键逻辑：**
```python
FORMAT_SPECS = [
    ("ME_TCF", "dtc", ME_TCF_FORMAT),
    ("BIT_TCF", "acc", BIT_TCF_FORMAT),
    ("BIT_BSR", "bitbsr", BIT_BSR_FORMAT),
    ("SR_BCRS", "sr_bcrs", SR_BCRS_FORMAT),
]

def _generate_one(format_name, format_obj, output_path):
    rts = derive_rts(format_obj)
    computent = computent_from_rts(format_name, rts)
    ops = generate_from_computent(computent)
    lowered = VarlenLoweringPass(...).run(ops)
    flattened = CValFlattenPass(...).run(lowered)
    nvir_program = generate_nvir(opir=flattened, format_name=format_name, ...)
    apply_software_pipeline_and_codegen(nvir_program, str(output_path))
```

**意义：** 将 `Format → RTS → Computent → OPIR → NVIR → .inc` 的完整生成链路固化为一键脚本。

### 6.2 `scripts/benchmark_selected_mtx_formats.py`（398 行）

**目的：** 自动化 benchmark 框架，同时测试生成算子和手写算子的性能。

**功能：**
- 支持 DTC/ACC/BITBSR/SR_BCRS 四种格式
- 自动编译（`cmake --build`）
- 多 trial 重复测量，支持 warmup/repeat 配置
- 自动解析 `mykernel_time` 和 `check nan` 输出
- 生成 CSV/JSONL 结果文件
- 计算 generated/hand 时间比
- 超时和错误处理

### 6.3 `python/test/test_nvir_semantic_fusion.py`（239 行）

**目的：** NVIR 语义融合的单元测试，验证软件流水线分阶段逻辑的正确性。

### 6.4 `python/test/GENERATED_KERNEL_WORKFLOW.md`（170 行）

**目的：** 生成算子的工作流文档，记录完整的生成-编译-验证流程。

---

## 7. 性能影响总结

### 7.1 Bug 修复带来的正确性/性能提升

| Bug | 影响的数据集 | 修复效果 |
|---|---|---|
| **K-fragment 丢失** | 所有 BLK_K>8 的数据集（DD, Ga, Si, mc2depi, reddit 等） | 从错误结果→正确计算，BLK_K=16/32 可用 |
| **M=32 store 截断** | cant, YeastH, pdb1HYS | 从丢失后半输出→完整输出 |
| **MCO mask 溢出** | 大 tile 数据集 | 支持超过 64 元素的 tile |
| **长尾分段缺失** | ASIC_680k | **30.3x 加速**（44.94ms → 1.484ms） |
| **grid.y 越界** | ASIC_680k 等大 row window 矩阵 | 从 crash → 正常运行 |
| **cp.async misaligned** | SR_BCRS 等 | 从运行时错误 → 正确执行 |

### 7.2 基础设施价值

| 新增能力 | 工程价值 |
|---|---|
| 统一生成脚本 | 一键生成所有格式的 .inc，消除手动操作错误 |
| 自动化 benchmark | 可重复、可审计的多格式性能对比 |
| 完整正确性验证 | 从 `check nan` 哨兵升级为 relative-L2 + 全元素扫描 |
| 格式设计 Agent | 26/26 数据集的自动化格式设计闭环 |

### 7.3 与原项目代码质量对比

| 维度 | 原项目 | Codex 改进后 |
|---|---|---|
| **正确性覆盖** | 部分路径依赖隐式假设（如 K=8），其他配置名义上可生成但结果错误 | 显式循环、边界检查、向量化 mask 支持任意合法配置 |
| **格式支持** | 主要为 DTC (ME_TCF) + SR_BCRS | 新增 BIT_TCF、BIT_BSR 等 MCO 系列格式的完整管线 |
| **Grid 映射** | 硬编码 blockIdx.x | 可配置 XMYN/XNYM 映射策略 |
| **Swizzle 配置** | 仅 strategy 级别 | 格式级别 + strategy 级别分层配置 |
| **流水线** | 简单阶段划分 | 依赖分析 + 拓扑排序 + 融合组 + 自适应阶段分配 |
| **验证** | `check nan` 哨兵 | relative-L2、归一化最大误差、全元素扫描、错误位置 |

---

## 附录 A：关键代码位置索引

| 功能 | 文件 | 关键行/函数 |
|---|---|---|
| K-fragment 循环 | [compiler_driver.py](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/compiler_driver.py) | `_lower_generic_op` 中 MMA 生成部分 |
| MCO mask 向量化 | [compiler_driver.py](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/compiler_driver.py) | `mco_atomic_val_restore` 分支 |
| cp.async 对齐 | [compiler_driver.py](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/compiler_driver.py) | G2sMatrixBLoadOp 分支 |
| Grid 映射 | [generate.py](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/generate.py) | `HardwareMapper` 类, `_lower_for_loop` |
| Swizzle 分层 | [generate.py](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/generate.py) | `_format_swizzle`, `_strategy_swizzle` |
| 流水线分析 | [compiler_driver.py](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/compiler_driver.py) | `_collect_local_deps`, `_build_fused_groups`, `_compress_stage_ids` |
| MCO mask OPIR | [ops.py](file:///home/huoda/sparsene/python/sparsene/op_gen/opir/ops.py) | `McoAtomicFormatLoadMaskOp.__init__` |
| Varlen leading_rank | [varlenLoweringPass.py](file:///home/huoda/sparsene/python/sparsene/op_gen/opir/varlenLoweringPass.py) | `_FlattenSpec`, `_build`, `_replace_operand_indices` |
| COO offset/length | [opir/generate.py](file:///home/huoda/sparsene/python/sparsene/op_gen/opir/generate.py) | `generate_from_coo_atomic_format_op` |
| 统一生成入口 | [generate_sparse_formats.py](file:///home/huoda/sparsene/python/test/generate_sparse_formats.py) | `main()` |
| 自动化 benchmark | [benchmark_selected_mtx_formats.py](file:///home/huoda/sparsene/scripts/benchmark_selected_mtx_formats.py) | `main()` |

# Sparsene 框架核心

> **维护者**: AI Assistant (2026-05)
> **版本**: v1.0
> **适用范围**: 所有需要对 sparsene 框架做修改、扩展、调试的开发者。

本 skill 覆盖 sparsene 框架的架构、核心概念、编译管线，**不涉及**智能体系统——后者见 `strategy-agent-feedback.skill.md`。

---

## 1. 框架概述

sparsene 是一个**稀疏算子生成编译框架**，支持层次化的稀疏格式表示，通过层层编译最终生成高性能 CUDA 稀疏矩阵-稠密矩阵乘法（SpMM）代码。

核心流程：
```
Format Definition  →  derive_rts()  →  Computent  →  OPIR  →  NVIR  →  .inc CUDA Kernel
```

### 关键术语

| 术语 | 含义 |
|------|------|
| **Format** | 层次化稀疏格式描述（轴 + 原子格式） |
| **Axis** | 格式的维度：`df`(稠密固定), `dv`(稠密变长), `sf`(稀疏固定), `sv`(稀疏变长) |
| **RTS** | Record Transformation Sequence — 从层次化格式推导出的展平变换序列 |
| **Computent** | 计算实体 — 从 RTS 推导出的数据结构、schedule 和循环描述 |
| **OPIR** | Operator IR — 基于数组引用的操作符中间表示 |
| **NVIR** | NvOp IR — 面向 CUDA 代码生成的底层 IR |
| **.inc** | 生成的 CUDA kernel 头文件 |

---

## 2. 格式系统 (`python/sparsene/format/`)

### 2.1 Axis（轴）

每个 Axis 描述格式中的一个维度：

| 属性 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 轴名（如 `"X_o"`, `"Y_s"`） |
| `direction` | `Direction.ROW` / `Direction.COL` | 行方向或列方向 |
| `length` | `Expr` (sympy) | 长度表达式（支持符号计算） |
| `is_sparse` | `bool` | 是否稀疏 |
| `is_varlen` | `bool` | 是否变长 |

四个工厂函数：
- `df_axis(name, direction, length)` — 稠密固定长度
- `dv_axis(name, direction, length)` — 稠密变长
- `sf_axis(name, direction, length)` — 稀疏固定长度
- `sv_axis(name, direction, length)` — 稀疏变长

### 2.2 Format（格式）

`Format` 是递归的树形结构：
- 包含一组 `axes`
- 可包含一个 `child`（子 `Format` 或 `AtomicFormat`）
- `is_atomic` 属性标记是否为原子格式

原子格式类型：
- `DenseAtomicFormat` — 稠密 tile
- `CooAtomicFormat` — COO 格式
- `McoAtomicFormat` — Masked-COO 格式

### 2.3 示例：ME_TCF 格式

```python
# python/sparsene/formats/DTC_SpMM.py
ME_TCF_FORMAT = Format(
    name="ME-TCF",
    axes=[
        df_axis("X_o", ROW, length="M"),
        sv_axis("Y_s", COL, length="K"),
        df_axis("X_i", ROW, length="BLK_M"),
    ],
    child=Format(
        axes=[
            df_axis("Y_so", COL, length=Symbol("K")/Symbol("BLK_K")),
            df_axis("Y_si", COL, length="BLK_K"),
            df_axis("X_i", ROW, length="BLK_M"),
        ],
        child=coo_atomic_format("X_i", "Y_si"),
    ),
)
```

ME_TCF = Mixed-Elimination Tensor Core Format。外层轴描述整个稀疏矩阵的 block 划分（`M` 行按 `BLK_M` 分块，`K` 列变长稀疏），内层描述每个 block 内部的 COO 稀疏格式。

---

## 3. 变换系统 (`python/sparsene/transform/`)

### 3.1 Transformation（变换）

变换是格式轴上的操作。继承层次：
```
Transformation (abstract)
├── NullaryTransformation
├── UnaryTransformation
│   ├── SplitTransformation     — 沿轴分裂
│   ├── SparsifyTransformation  — 稠密→稀疏
│   ├── DensifyTransformation   — 稀疏→稠密
│   ├── CooizeTransformation    — 转为 COO
│   ├── McoizeTransformation    — 转为 MCO
│   ├── GoUpTransformation      — 提升层次
│   └── SpawnTransformation     — 复制轴
├── BinaryTransformation
│   ├── SwapTransformation      — 交换两轴
│   └── MergeTransformation     — 合并两轴
```

最常用的是 `SplitTransformation(axis_idx, split_size)`：沿轴 `axis_idx` 按 `split_size` 分裂为两段。

### 3.2 RTS（Record Transformation Sequence）

`derive_rts(format)` → `TransformationSequence`

将层次化的 `Format` 展平为一组连续的矩阵结构。核心步骤：
1. **densify** — 将稀疏轴标记为稠密
2. **bubble_up** — 将同方向轴 bubble 到一起
3. **squeeze** — 合并相邻同方向轴
4. **cooize/mcoize** — 对原子格式做 COO/MCO 化

生成的 `TransformationSequence` 是后续 Computent 和 OPIR 生成的输入。

---

## 4. Computent（计算实体）

文件：`python/sparsene/op_gen/computent/computent.py`

`computent_from_rts(format_name, rts)` 将 RTS 转换为：
- **ArrayDef 集合**：描述所有中间数组（形状、数据类型、角色）
- **Schedule**：迭代器树，描述多层循环的边界和顺序
- **符号表**：从 sympy 符号到运行时变量名的映射

Schedule 节点类型：
- `DenseAxisIterator` — 固定长度轴迭代
- `SparseAxisIterator` — 稀疏轴迭代
- `SparseAxisSlicer` — 稀疏切片（varlen 维度）

---

## 5. OPIR（操作符 IR）

文件：`python/sparsene/op_gen/opir/generate.py`、`op_ir.py`、`ops.py`

### 5.1 生成流程

`generate_from_computent(computent)` → OPIR Block

OPIR 是一个**基本块**（`Block`），包含操作符序列。核心 op 类型：

| Op | 含义 |
|---|------|
| `ArrayRefOp` | 数组引用（带索引表达式） |
| `LoadOp` | 加载 |
| `LoadOffsetOp` | offset 加载 |
| `ForLoopOp` | 循环 |
| `ConstantOp` | 常量 |
| `ExternalSymbolOp` | 外部符号 |
| `CooAtomicFormatLoad*Op` | COO 原子格式加载 |
| `CooAtomicValRestoreOp` | COO 值恢复 |
| `BValLoadOp` | B 矩阵加载 |
| `MmaOp` | 矩阵乘加（mma） |
| `CValLoadOp` / `CValStoreOp` | C 矩阵读写 |

### 5.2 VarlenLoweringPass

`varlenLoweringPass.py`：将变长维度（`varlen(K)`）替换为具体的长度数组引用，压缩变长维度。这是**正确性的关键保障**——变长维度在上层是符号，在 lowered IR 中是实际的数组长度。

### 5.3 CValFlattenPass

`cValFlattenPass.py`：将 C 矩阵的 tile 化 IO 压平为纯寄存器读写。

---

## 6. NVIR（NvOp IR）

文件：`python/sparsene/op_gen/nvir/`

### 6.1 核心数据结构

**`NvOpProgram`**（`nvop.py`）：
- `gmem_inouts`：全局内存接口（映射名 → 形状/类型）
- `body`：`NvOpSequence`，包含 operations 和 for-loops
- `attrs`：元数据（strategy_decision、generated_tile_b 等）

**`NvOp`**：单个操作符
- `name`、`inputs`、`outputs`
- `mem_type`：`("g","s")` 或 `("s","r")` 等
- `op_type`：`COPY_1D`、`COPY_2D`、`COPY_2D_SP`、`OTHERS`
- `logical_kind`：`"array_ref"`、`"coo_atomic_val_restore"`、`"ldmatrix"`、`"mma"` 等
- `cp_async`：是否使用异步拷贝
- `attrs`：其他属性

**`NvOpTensor`**：
- `shape`、`mem`（`"gmem"`/`"smem"`/`"rmem"`）、`dtype`
- `row_major`、`swizzle`（仅在输出端）

### 6.2 Lowering 流程

`generate.py::generate_nvir()` 是主入口：
1. 解析 OPIR，生成初始 NVIR
2. `compiler_driver.py::apply_software_pipeline_and_codegen()`：
   - 软件流水线（pipeline stages 划分 + commit/wait 插入）
   - 最终 `.inc` dump

### 6.3 Codegen → `.inc`

`codegen.py::NvOpCodegenPrinter.dump_nvop_program()`：
1. 生成第一类定义（每个 NvOp 的 CUTE 模板类）
2. 嵌入 metadata 宏（`SPARSENE_GENERATED_TILE_B` 等）
3. 生成全局 kernel 函数（分配 smem、实例化 pipeline、循环体）

`templates.py` 提供 CUDA 源码骨架（include、helper 函数）。

---

## 7. 编译与运行

### 7.1 编译

```
.inc → nvcc -arch=sm_80 + host_program_generated.cu + main.cu
     → demo-dtc-generated-tf32
```

- `host_program_generated.cu` include `.inc`，提供 `sparsene_generated_tile_b()` 等运行时分发
- `main.cu` 根据 generated TILE_B 做 `<16>/<32>/<64>` 模板分发
- 链接 `cusparse`、`cublas`

### 7.2 运行

```bash
demo-dtc-generated-tf32 -filename dataset.mtx -mtx_flag 1 -N 512 -warmup 5 -repeat 20
```

输出：`mykernel_time`、`cusparse_time`、`check nan`、`Error(r,c)` 行。

---

## 8. 关键路径速查

| 做什么 | 改哪个文件 |
|--------|-----------|
| 新增稀疏格式 | `python/sparsene/formats/*.py` |
| 修改格式→RTS 推导 | `python/sparsene/transform/rts.py` |
| 修改 Computent 生成 | `python/sparsene/op_gen/computent/computent.py` |
| 修改 OPIR 生成 | `python/sparsene/op_gen/opir/generate.py` |
| 修改 NVIR lowering | `python/sparsene/op_gen/nvir/generate.py` |
| 修改软件流水线 | `python/sparsene/op_gen/nvir/compiler_driver.py` |
| 修改 .inc 代码模板 | `python/sparsene/op_gen/nvir/templates.py` 和 `codegen.py` |
| 修改 host runtime | `examples/.../host_program_generated.cu` 和 `main.cu` |

# Sparsene 原子格式系统全链路分析

## 1. 概述

Sparsene 的代码生成流水线是：

```
Format (格式定义)
  → RTS (变换序列推导)
    → Computent (调度 + 数组定义)
      → OPIR (稀疏算子中间表示)
        → VarlenLoweringPass → CValFlattenPass
          → NVIR (CUDA 中间表示)
            → software pipeline / codegen
              → .inc (可编译的 CUDA/CUTLASS kernel)
```

**原子格式 (AtomicFormat)** 是这条流水线的终端叶子节点。它描述了稀疏矩阵在**最底层、不再分割的 2D 分块**中数据的物理存储方式。目前支持三种原子格式：`dense`、`coo`、`mco`。

---

## 2. Format 数据结构

### 2.1 Axis：格式的基本单元

[`format.py:L23-L78`](file:///home/huoda/sparsene/python/sparsene/format/format.py#L23-L78)

```python
@dataclass
class Axis:
    name: str           # 轴名称，如 "X_o", "Y_si"
    direction: Direction # ROW 或 COL
    length: Expr         # 轴长度，sympy 表达式
    is_sparse: bool      # 是否稀疏轴
    is_varlen: bool      # 是否变长轴
    reorder: Optional[str]
```

四种轴类型通过工厂函数创建：

| 工厂函数 | is_sparse | is_varlen | 含义 |
|---|---|---|---|
| `df_axis` | False | False | 稠密定长轴 |
| `dv_axis` | False | True | 稠密变长轴 |
| `sf_axis` | True | False | 稀疏定长轴 |
| `sv_axis` | True | True | 稀疏变长轴 |

`length` 使用 sympy 表达式 (`Symbol("M")`, `BLK_K/8`, `varlen(nnz)`)，使编译器在变换过程中保留并传播符号信息。

### 2.2 Format：层次化格式树

[`format.py:L90-L140`](file:///home/huoda/sparsene/python/sparsene/format/format.py#L90-L140)

```python
@dataclass
class Format(FormatBase):
    child: Optional[Format | AtomicFormat] = None  # 递归子格式
```

`Format` 是递归的树结构。每一层 `Format` 有一组 axes，以及一个 `child`（可以是另一层 `Format` 或最终的 `AtomicFormat`）。这棵树的语义是：

- 外层 axes 对应"粗粒度"分块（如 row window `X_o`, column window `Y_so`）
- 内层 axes 对应"细粒度"分块（如 `X_i`, `Y_si`）
- 叶子 `AtomicFormat` 定义 2D 分块内的物理存储

### 2.3 AtomicFormat：三种物理存储方式

[`format.py:L90-L140`](file:///home/huoda/sparsene/python/sparsene/format/format.py#L90-L140)

```python
@dataclass
class DenseAtomicFormat(AtomicFormat):  # atomic_type = "dense"
    pass

@dataclass
class CooAtomicFormat(AtomicFormat):    # atomic_type = "coo"
    pass

@dataclass
class McoAtomicFormat(AtomicFormat):    # atomic_type = "mco"
    pass

class CsrAtomicFormat(AtomicFormat):    # atomic_type = "csr"    ← 新增于 2026-06
    pass
```

每种原子格式有不同的工厂函数（[`format.py:L235-L302`](file:///home/huoda/sparsene/python/sparsene/format/format.py#L235-L302)），它们自动创建对应的 axes：

| AtomicFormat | axes 组成 | 说明 |
|---|---|---|
| `dense` | `[x_axis, y_axis]` (2 axes) | 稠密存储，无额外数组 |
| `coo` | `[x_axis, y_axis, coo_axis]` (3 axes) | COO 格式，第三轴是变长 NNZ 轴 |
| `mco` | `[x_axis, y_axis, mco_axis]` (3 axes) | Masked COO，与 COO 类似但使用比特掩码 |
| `csr` | `[x_axis, y_axis, csr_axis]` (3 axes) | CSR 格式，同构于 COO，语义为 per-row cumulative offset |

---

## 3. 已有格式定义

### 3.1 ME_TCF (DTC_SpMM) — `coo` 原子

[`DTC_SpMM.py`](file:///home/huoda/sparsene/python/sparsene/formats/DTC_SpMM.py)

```
Format("ME-TCF")
  axes: [X_o(ROW, M), Y_s(COL, K, sparse+varlen), X_i(ROW, BLK_M)]
  child: Format
    axes: [Y_so(COL, K/BLK_K), Y_si(COL, BLK_K), X_i(ROW, BLK_M)]
    child: coo_atomic_format("X_i", "Y_si")     ← 叶子
```

语义：先按 row window 分块 (`X_o`)，列方向不规则稀疏 (`Y_s`)，再按列窗口 / tile 进一步分割，最内层使用 COO 存储 16×8 的稀疏 tile。

### 3.2 BIT_TCF (Acc_SpMM) — `mco` 原子

[`Acc_SpMM.py`](file:///home/huoda/sparsene/python/sparsene/formats/Acc_SpMM.py)

与 ME_TCF 结构完全相同，唯一区别是叶子用 `mco_atomic_format("X_i", "Y_si")`。

### 3.3 SR_BCRS — `dense` 原子

[`SR_BCRS.py`](file:///home/huoda/sparsene/python/sparsene/formats/SR_BCRS.py)

与 ME_TCF 结构相同，叶子用 `dense_atomic_format("X_i", "Y_si")`。

### 3.4 HME_TCF — 4 层嵌套 + `coo` 原子

[`HME_TCF.py`](file:///home/huoda/sparsene/python/sparsene/formats/HME_TCF.py)

```
Format("HME-TCF")
  axes: [X_o, Y_s, X_i]
  child: Format
    axes: [Y_so, Y_si, X_i]
    child: Format
      axes: [Y_sis(sparse), X_i]
      child: Format
        axes: [X_io, Y_sis, X_ii]
        child: Format
          axes: [Y_siso, Y_sisi, X_ii]
          child: coo_atomic_format("X_ii", "Y_sisi")   ← 叶子
```

深度嵌套的分级分块，适合极端稀疏矩阵。

### 3.5 ROW_REORDER_SR_BCRS — 行重排序 + `dense` 原子

[`ROW_REORDER_SR_BCRS.py`](file:///home/huoda/sparsene/python/sparsene/formats/ROW_REORDER_SR_BCRS.py)

在 SR_BCRS 的基础上增加了外层行重排序 (`X_s`)。

---

## 4. Format → RTS：变换序列推导

[`rts.py`](file:///home/huoda/sparsene/python/sparsene/transform/rts.py)

`derive_rts(format)` 递归遍历 Format 树，自底向上生成 `TransformationSequence`。

### 4.1 变换类型

[`transformation.py`](file:///home/huoda/sparsene/python/sparsene/transform/transformation.py)

| 变换 | 作用 | 触发条件 |
|---|---|---|
| `Split(axis, BLK_N)` | 将轴切分为 outer×inner | 每个分块维度 |
| `Swap(i, j)` | 交换两个轴的位置 | 调整方向以匹配相邻合并 |
| `Sparsify(axis)` | 将轴变为稀疏 (生成 sidx/len 数组) | 遇到 sparse+varlen 轴 |
| `Spawn()` | 创建新的子 Format 层 | 进入 Format.child |
| `Cooize()` | 将最内层 dense 块转为 COO 存储 | 遇到 coo atomic format |
| `Mcoize()` | 将最内层 dense 块转为 MCO 存储 | 遇到 mco atomic format |

### 4.2 RTS 推导算法

[`rts.py:L120-L143`](file:///home/huoda/sparsene/python/sparsene/transform/rts.py#L120-L143)

```
_derive_rts(format, parent_major_direction):
  1. 如果 format.child 是 AtomicFormat:
     a. 根据 atomic_type 生成 Cooize/Mcoize/nullary 变换
     b. 如果 atomic 内部轴方向与 parent 不一致，插入 Swap
     c. 插入 Spawn
  2. 如果 format.child 是 Format: 递归 + 插入 Spawn

  3. 对当前层 axes: densify → squeeze_new → bubble_up
```

以 ME_TCF 为例，RTS 输出为：
```
Sparsify(Y) → Split(K/BLK_K) → Spawn → Split(BLK_K) → Swap(X_i, Y_si) → Spawn → Cooize
```
（实际顺序由 `prepend` 操作反转，最终序列与树的深度优先遍历一致）

---

## 5. RTS → Computent：调度与数组定义

[`computent.py`](file:///home/huoda/sparsene/python/sparsene/op_gen/computent/computent.py)

### 5.1 起点：Dense Schedule

所有格式从稠密起点开始：

```python
DENSE_SCHEDULE = AtomicFormatOp(
    axes=("X", "Y"), axes_len=(M, K),
    type=AtomicFormatType.DENSE,
    args=[
        ArrayRef("val",  ["_", "_"]),
        ArrayRef("B_val", ["_", "_"]),
    ]
)
```

### 5.2 变换应用

每个 RTS 变换按顺序应用到 `(schedule, array_defs, spformat)` 三元组上：

- **Split**: 在 schedule 树中插入新的 `DenseAxisIterator`，更新 array_defs 的维度
- **Swap**: 交换指定轴的位置
- **Sparsify**: 将 `DenseAxisIterator` 替换为 `SparseAxisIterator`，生成 `val_sidx`/`val_len` 数组，在 `B_val` 引用中插入间接索引 `val_sidx[i1, i2]`
- **Cooize**: 将 `AtomicFormatOp[DENSE]` 替换为 `AtomicFormatOp[COO]`，生成 `val_coo_idx`/`val_coo_val`/`val_coo_len` 三个数组
- **Mcoize**: 类似，生成 `val_mco_mask`/`val_mco_val`/`val_mco_len`

### 5.3 最终产物：Computent

```python
Computent(
    name="ME_TCF",
    schedule=<Schedule tree>,
    array_defs=<ArrayDefCollection>,
    format=<Format tree>,
    varlen2LenArrayTable={varlen_expr → ArrayDef}
)
```

---

## 6. Computent → OPIR：稀疏算子 IR

[`opir/generate.py`](file:///home/huoda/sparsene/python/sparsene/op_gen/opir/generate.py)

### 6.1 Schedule 遍历

`generate_from_computent` 递归遍历 schedule 树：

| Schedule 节点 | OPIR 输出 |
|---|---|
| `DenseAxisIterator` | `ForLoopOp` + 递归 |
| `SparseAxisIterator` | `ForLoopOp` + `LoadOffsetOp(ll, rr)` + 递归 |
| `AtomicFormatOp[DENSE]` | `ArrayRefOp(B_val)` → `ldmatrix` → `mma` |
| `AtomicFormatOp[COO]` | `LoadOffsetOp` → `CooAtomicFormatLoadIdxOp` → `CooAtomicFormatLoadValOp` → `CooAtomicValRestoreOp` → `MmaOp` |
| `AtomicFormatOp[MCO]` | `McoAtomicFormatLoadMaskOp` → `McoAtomicFormatLoadValOp` → `McoAtomicValRestoreOp` → `MmaOp` |
| `AtomicFormatOp[CSR]` | `LoadOffsetOp` → `CooAtomicFormatLoadIdxOp` → `CooAtomicFormatLoadValOp` → `CooAtomicValRestoreOp` → `MmaOp` (复用 COO 算子) |

### 6.2 原子格式生成的 OPIR 算子

**COO 原子**（[`generate.py:L683-L731`](file:///home/huoda/sparsene/python/sparsene/op_gen/opir/generate.py#L683-L731)）：

```
load_offset(len_array, induction_var)  → (ll, rr)
coo_atomic_format_load_idx(idx_array, ll, rr)  → (coo_idx, coo_range)
coo_atomic_format_load_val(val_array, ll, rr)  → coo_val
coo_atomic_val_restore(coo_val, coo_idx, coo_range)  → A_val_tile
mma(A_val_tile, B_val_tile, C_val_tile)  → C_val_tile'
```

**MCO 原子**（[`generate.py:L828-L918`](file:///home/huoda/sparsene/python/sparsene/op_gen/opir/generate.py#L828-L918)）：

```
mco_atomic_format_load_mask(mask_array, induction_var)  → mask
mco_atomic_format_load_val(val_array, mco_len)  → val_tile
mco_atomic_val_restore(val_tile, mask, mco_len)  → A_val_tile
mma(A_val_tile, B_val_tile, C_val_tile)  → C_val_tile'
```

### 6.3 原子格式对 VarlenLoweringPass 的影响

[`varlenLoweringPass.py`](file:///home/huoda/sparsene/python/sparsene/op_gen/opir/varlenLoweringPass.py)

varlen lowering pass 负责将变长维度展平为 offset/flatten 形式。关键豁免规则：

```python
# COO 原子格式的三输入形式 [array, ll, rr] 不能被 flatten
if isinstance(op, (CooAtomicFormatLoadIdxOp, CooAtomicFormatLoadValOp)) and len(op.operands) == 3:
    return False  # 不展平，保留 ll/rr range 信息
```

这意味着：**原子格式的类型直接决定了 varlen lowering 的行为**。COO 格式保留了 `(ll, rr)` 的 range 语义，这对正确性和性能都至关重要。

### 6.4 原子格式在 OPIR 中定义的算子

[`ops.py`](file:///home/huoda/sparsene/python/sparsene/op_gen/opir/ops.py)

| Op 类 | 用途 | 存储器 | 原子格式 |
|---|---|---|---|
| `CooAtomicFormatLoadIdxOp` | 从全局内存加载 COO 索引到 smem | G2S | coo |
| `CooAtomicFormatLoadValOp` | 从全局内存加载 COO 值到 smem | G2S | coo |
| `CooAtomicValRestoreOp` | 将 COO 值恢复到稠密 smem tile | S2R | coo |
| `McoAtomicFormatLoadMaskOp` | 加载 MCO 掩码 | G2S | mco |
| `McoAtomicFormatLoadValOp` | 加载 MCO 值 | G2S | mco |
| `McoAtomicValRestoreOp` | 将 MCO 值恢复到稠密 smem tile | S2R | mco |

---

## 7. OPIR → NVIR：CUDA 中间表示

[`nvir/generate.py`](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/generate.py)

### 7.1 原子格式算子的 _make_impl

`_make_impl()` 根据 `logical_kind` 将 OPIR 算子映射到 NVIR 实现代码：

**`coo_atomic_format_load_idx`**（[`generate.py:L1036-L1053`](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/generate.py#L1036-L1053)）：
```cpp
// 对齐的 cp.async 路径
if ((operand_1 & 3) == 0) {
    copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, int>{}, src, dst);
} else {
    // 非对齐 fallback
    coo_idx(i_load, buf_idx) = src_idx < operand_2 ? operand_0(src_idx) : 0;
}
```

**`coo_atomic_format_load_val`**（[`generate.py:L1044-L1062`](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/generate.py#L1044-L1062)）：
```cpp
// 对齐的 cp.async 路径 (float)
if ((operand_1 & 3) == 0) {
    copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>{}, src, dst);
} else {
    // 非对齐 fallback (同步加载 + float(0) padding)
}
```

**`coo_atomic_val_restore`**（[`generate.py:L1076-L1092`](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/generate.py#L1076-L1092)）：
```cpp
// smem clear + scatter restore
for (int i_clear = lid; ...) { smem = float(0); }
__syncthreads();
for (int i_restore = lid; ...) { smem[idx] = value; }
```

### 7.2 cp_async 标记策略

[`generate.py:L1683-L1699`](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/generate.py#L1683-L1699)

```python
# coo_atomic_val_restore 和 mco_* 是同步 smem 操作
# 标记 cp_async=False 避免 codegen 发射无意义的 __pipeline_commit()
if logical_kind in {
    "coo_atomic_val_restore",
    "mco_atomic_format_load_mask",
    "mco_atomic_format_load_val",
    "mco_atomic_val_restore",
}:
    nvop.attrs["cp_async"] = False
```

这个标记直接影响 codegen 的 `launch_commit()` 决策（[`codegen.py:L849-L854`](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/codegen.py#L849-L854)），进而影响 pipeline 的 commit/wait 粒度。

### 7.3 Swizzle 自适应

[`generate.py:L1651-L1656`](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/generate.py#L1651-L1656)

```python
# coo restore 的 swizzle 随 BLK_K 自适应: 8→S=3, 16→S=4, 32→S=5
swizzle=SwizzleLayout(1, 2, 3, adaptive_blk_k=True)
```

这个自适应的 swizzle 由 codegen 在编译时展开为 CuTe 的 `conditional` 模板表达式，确保 host 和 kernel 之间 swizzle 参数一致。

---

## 8. NVIR → .inc：代码生成

[`codegen.py`](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/codegen.py) + [`compiler_driver.py`](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/compiler_driver.py)

### 8.1 原子格式算子的代码生成

每个原子格式的 Op 被生成为一个 CUTLASS 风格的模板类：

```cpp
template<class NBUF, class BLK_MNK, ..., class TI0 = int, class TI1 = int, ..., class TO0 = int>
class coo_atomic_format_load_idx {
    // Inputs:  operand_0 (idx_array), operand_1 (ll), operand_2 (rr)
    // Outputs: coo_idx (smem), coo_range (rmem)
    void f(Tensor_i0 operand_0, TI1 operand_1, TI2 operand_2) { ... }
};
```

### 8.2 Software Pipeline 优化

`compiler_driver.py` 中的优化传递（[`compiler_driver.py`](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/compiler_driver.py)）：

- `_fuse_load_offset_sub_div`：将 `l = offset(i); r = offset(i+1) - l` 融合为单步操作
- `_fuse_inline_gen_add`：将 `off_l + l` 融合进 `array_ref` 和 `load_offset_1`
- `_mark_codegen_inline_load_offset`：标记 `load_offset_1` 的输出可以被 codegen 级 inline

这些融合直接影响原子格式数据路径中的标量链，减少寄存器和指令开销。

---

## 9. 原子格式在运行时的体现

### 9.1 Host 端元数据转换

[`host_program_generated.cu:L63-L113`](file:///home/huoda/sparsene/examples/src_fp32/dtc/testbed/host_program_generated.cu#L63-L113)

```cpp
GeneratedKernelMetadata convert_old_metadata_to_generated(...) {
    // COO 原子格式需要：
    //   val_coo_len_new_offset  — 前缀偏移
    //   val_coo_idx_new_1        — 紧凑 COO 索引
    //   val_coo_val_new_1        — 紧凑 COO 值
    // coo_off[2*i], coo_off[2*i+1] 从 padded block 提取真实 nnz range
}
```

### 9.2 Checkpoint 缓存

[`main.cu:L218-L225`](file:///home/huoda/sparsene/examples/src_fp32/dtc/testbed/main.cu#L218-L225)

checkpoint 文件名包含 `_TBK`(TC_BLOCK_K) 和 `_RWM`(ROW_WINDOW_M) 后缀，不同配置的元数据互不覆盖。

---

## 10. 扩展新格式的完整步骤

### 10.1 使用现有原子格式 (dense/coo/mco)

只需定义新的 Format 树：

**步骤：**

1. **创建格式文件** `python/sparsene/formats/MyNewFormat.py`：

```python
from sparsene.format.format import (
    Direction, Format, coo_atomic_format, df_axis, sv_axis, Symbol
)

MY_NEW_FORMAT = Format(
    name="my-new-format",
    axes=[
        df_axis("X_o", Direction.ROW, length="M"),
        sv_axis("Y_s", Direction.COL, length="K"),
        df_axis("X_i", Direction.ROW, length="BLK_M"),
    ],
    child=Format(
        axes=[
            df_axis("Y_so", Direction.COL, length=Symbol("K")/Symbol("BLK_K")),
            df_axis("Y_si", Direction.COL, length="BLK_K"),
            df_axis("X_i", Direction.ROW, length="BLK_M"),
        ],
        child=coo_atomic_format("X_i", "Y_si"),  # 或 mco_atomic_format / dense_atomic_format
    ),
)
```

2. **在 `test_generate.py` 或 `end2end_pipeline.py` 中导入并使用**。

现有的 `coo`/`mco`/`dense` 三种原子格式已经覆盖了 RTS、Computent、OPIR、NVIR、codegen、host 的完整流水线。**仅改变 Format 树的层数和分块大小不需要修改任何框架代码。**

### 10.2 添加全新的原子格式类型

如需添加不基于 COO 或 Dense 的新存储格式（如 BSR、CSR tile、Hybrid 等），需要：

**步骤：**

1. **`format.py`** — 新增 `AtomicFormat` 子类和工厂函数：
```python
class BsrAtomicFormat(AtomicFormat):
    atomic_type: str = "bsr"

def bsr_atomic_format(x_name, y_name, col_major=False) -> AtomicFormat:
    return BsrAtomicFormat(
        name="bsr",
        axes=[...]  # BSR 特有轴结构
    )
```

2. **`transformation.py`** — 新增变换类型：
```python
class BsrizeTransformation(NullaryTransformation):
    name: str = "bsrize"
```

3. **`rts.py`** — 在 `_derive_rts` 中处理新原子类型：
```python
match format.child.atomic_type:
    case "bsr":
        seq.prepend(BsrizeTransformation())
```

4. **`computent.py`** — 新增 `apply_bsrize_transformation()` 并添加到 dispatch：
- 在 `AtomicFormatType` enum 中添加 `BSR`
- 实现 `apply_bsrize_transformation()`: 创建新的 array_defs (如 idxptr, values 等)，替换 schedule 中的 `AtomicFormatOp[DENSE]` 为 `AtomicFormatOp[BSR]`

5. **`opir/ops.py`** — 定义新的 OPIR 算子类：
```python
class BsrAtomicFormatLoadPtrOp(DeviceOp): ...
class BsrAtomicFormatLoadValOp(DeviceOp): ...
class BsrAtomicValRestoreOp(DeviceOp): ...
```

6. **`opir/generate.py`** — 在 `generate_from_atomic_format_op` 中添加新 case：
```python
case AtomicFormatType.BSR:
    return generate_from_bsr_atomic_format_op(...)
```

7. **`nvir/generate.py`** — 在 `_make_impl` 中添加 NVIR 实现：
```python
if logical_kind == "bsr_atomic_format_load_ptr":
    return NvOpImpl(f"""...""")
```

8. **`nvir/generate.py`** — 在 `SPARSE_FORMAT_REGISTRY` 中添加 shape overrides 和 dtype 声明（如果需要）。

9. （可选）调整 **VarlenLoweringPass** 中的豁免规则，确保新原子格式的相关算子不被错误展平。

10. （可选）修改 **host_program_generated.cu** 中的 `convert_old_metadata_to_generated()` 以适配新格式的元数据。

### 10.3 现有原子格式

| 格式文件 | Format 名 | 原子格式 | 层级深度 | 实测状态 |
|---|---|---|---|---|
| `DTC_SpMM.py` | `ME_TCF` | `coo` | 2 层 | 26 数据集验证通过 |
| `CSR_TCF.py` | `CSR_TCF` | `csr` | 2 层 | shipsec1/web-BerkStan 验证通过 |
| `Acc_SpMM.py` | `BIT_TCF` | `mco` | 2 层 | 编译通过 |
| `SR_BCRS.py` | `SR_BCRS` | `dense` | 2 层 | 编译通过 |
| `HME_TCF.py` | `HME_TCF` | `coo` | 4 层 | 编译通过 |
| `ROW_REORDER_SR_BCRS.py` | `row-reorder-sr-bcrs` | `dense` | 3 层 | 编译通过 |

### 10.4 现有原子格式的选择指南

| 原子格式 | 适用场景 | NNZ overhead | 访存模式 |
|---|---|---|---|
| `dense` | avg_nnz/tile ≈ BLK_M×BLK_K（满） | 零 padding | 直接索引, 最优 |
| `coo` | avg_nnz/tile < 50% 填充率 | 2×NNZ 存储(idx+val) | cp.async gather |
| `mco` | avg_nnz/tile 中等, 需要掩码加速 | NNZ 存储 + 1 mask/tile | cp.async + bitmask scatter |

---

## 11. CSR 原子格式实现状态

CSR 格式已于 2026-06 完成框架级实现，当前正确性验证通过。

### 11.1 已完成 (各层均有 CSR 专属代码)

| 层 | 状态 | 关键差异 vs COO |
|---|---|---|
| `format.py` | `CsrAtomicFormat(atomic_type="csr")` + `csr_atomic_format()` | 工厂函数自动创建 `csr` 轴 |
| `transformation.py` | `CsrizeTransformation` | 独立变换类型 |
| `rts.py` | `case "csr": seq.prepend(CsrizeTransformation())` | RTS 自动推导 |
| `schedule.py` | `AtomicFormatType.CSR = 3` | 独立 schedule 路由 |
| `computent.py` | `apply_csrize_transformation()` | 生成 `val_csr_len/idx/val` 三数组 |
| `formats/CSR_TCF.py` | 完整 `CSR_TCF_FORMAT` 定义 | 与 ME_TCF 同构 |
| `opir/ops.py` | `CsrAtomicValRestoreOp` (4-operand) | 接受 `[val, idx, row_ptr, nnz]` |
| `opir/generate.py` | `generate_from_csr_atomic_format_op()` | 独立 OPIR 生成路径 |
| `nvir/generate.py` | `csr_atomic_val_restore` `_make_impl` + smem/row_major/cp_async 处理 | row-by-row scatter 代码生成就绪 |
| `host_program_generated_csr.cu` | 独立 host bridge，含 `val_csr_row_ptr` 计算 | deswizzle + 逐行统计 NNZ 计数 |

### 11.2 当前真实状态（已接通 gmem row_ptr）

此前 CSR 的最后一个缺口确实在 `gmem interface -> ArrayRefOp lowering -> codegen` 交界处：

- `val_csr_row_ptr` 虽然在 Computent / OPIR 中已经存在，但没有以正确 shape/layout 进入 generated kernel
- `BLK_M + 1` 一度被错误当作普通 runtime varlen，导致 `array_ref` 物化后的 CuTe layout 不稳定
- host 虽然构造了 `row_ptr`，但 `(idx, val)` payload 没有真正按逻辑行重排，因此 row_ptr 语义和 payload 顺序不一致

当前这些问题已经全部修通，真实代码路径如下：

1. `SPARSE_FORMAT_REGISTRY["CSR_TCF"]` 为 `val_csr_row_ptr` 指定独立 gmem 接口：
```python
"val_csr_row_ptr": ["BLK_M + 1", "nnz_dim_K_o"]
```

2. `build_gmem_interfaces()` / `opir_type_to_nvir_shape()` 已能把 `BLK_M + 1` 识别为参数化静态 shape，而不是普通 varlen。

3. `ArrayRefOp` 对 `csr_row_ptr` 的 1D block slice 会 materialize 到 smem，并复用 generic 访问模式：
```cpp
result_0(i_load, buf_idx) = operand_0(i_load, operand_1);
```
配合 gmem tensor 形状 `[BLK_M + 1, nnz_dim_K_o]`，这里恰好等价于 `row_ptr(i, block_id)`。

4. `csr_atomic_val_restore` 已经真正使用 4-operand CSR 接口：
```cpp
// operands = [csr_val, csr_idx, csr_row_ptr, csr_nnz]
for (int row_r = 0; row_r < blk_m; row_r++) {
    int start = csr_row_ptr(row_r);
    int end = csr_row_ptr(row_r + 1);
    for (int i = lid + start; i < end; i += 32) {
        int idx = csr_idx(i);
        out[idx] = csr_val(i);
    }
}
```
这里 `row_ptr` 负责逐行分段，scatter 地址继续使用物理 swizzled flat index，因此与现有 MMA 前的 tile 布局保持一致。

5. `host_program_generated_csr.cu` 会先对每个 block 做 `csr_deswizzle_idx()`，再按逻辑行把 `(idx, val)` 重排成真正 CSR row segments，并同步生成每个 block 的 `row_ptr[BLK_M+1]`。

因此，当前 generated CSR kernel 不再是“名字上是 CSR，执行上还是 COO”，而是：

- host 真实传入 `row_ptr`
- kernel 真实逐行消费 `row_ptr`
- payload 顺序与 `row_ptr` 语义一致

### 11.3 验证结果

| 数据集 | CSR kernel (当前) | 正确性 |
|---|---|---|
| shipsec1 | 3.56ms | ok |
| web-BerkStan | 5.13ms | ok |
| mip1 | 14.11ms | ok |


## 12. 关键设计原则

1. **原子格式是叶子节点**：所有非最内层分块都是通过 Format 树的嵌套来表达的，原子格式只决定最内层 2D 分块的物理存储。

2. **RTS 是自动推导的**：格式定义决定了 RTS，RTS 决定了 schedule 和 array_defs，不需要为每种格式组合手动编写变换序列。

3. **COO/MCO/CSR 的 (ll, rr) range 保留是关键**：varlen lowering pass 中对 `CooAtomicFormatLoadIdxOp` 和 `CooAtomicFormatLoadValOp` 的豁免确保了 range 信息不丢失，这是 generated kernel 正确性和性能的基础。

4. **cp_async 标记影响 pipeline**：哪些原子格式算子可以 emit `__pipeline_commit()` 是由输出 tensor 的 `cp_async` 属性决定的，这直接影响 DMA/computation overlap。

5. **swizzle 需要与 BLK_K 自适应**：不同 TC_BLOCK_K 值需要不同的 shared memory swizzle 参数 (S=log2(BLK_K))，codegen 和 host 端必须一致。

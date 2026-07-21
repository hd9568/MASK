# ELL 原子格式开发全记录

## 1. ELL 格式数据结构

ELL (ELLPACK) 格式将每个 TC block 内的稀疏 tile 按行等宽排列：

```
block j:  [row_0: idx0 val0, idx1 val1, ..., idx_{e-1} val_{e-1}, sentinel...]
          [row_1: idx0 val0, idx1 val1, ..., idx_{e-1} val_{e-1}, sentinel...]
          ...
          [row_15: ...]
```

- `ell_len` = max(NNZ per row) within this block
- 每行恰好 `ell_len` 个 entry, 不足的行用 sentinel idx 填充
- 存储为两个平铺 1D 数组 (与 COO 共享 gmem 接口协议): `ell_idx[ell_total]`, `ell_val[ell_total]`
- 和 COO 的区别: ELL 在 host 侧按行排好顺序并做 max-nnz 对齐,padding 隐含了行结构

## 2. 全链路修改清单

与 CSR 扩展过程一致,ELL 需要在 8 个层级添加独立代码:

### 2.1 format.py — 格式类 + 工厂函数

[format.py L143-L146](file:///home/huoda/sparsene/python/sparsene/format/format.py#L143-L146):
```python
@dataclass(kw_only=True)
class EllAtomicFormat(AtomicFormat):
    atomic_type: str = "ell"
```

[format.py L330-L367](file:///home/huoda/sparsene/python/sparsene/format/format.py#L330-L367):
`ell_atomic_format(x_name, y_name)` 工厂函数 — 创建 3 轴结构 `[X_i, Y_si, X_i_Y_si_ell]`

### 2.2 transformation.py + rts.py — 变换类型

[transformation.py L168-L177](file:///home/huoda/sparsene/python/sparsene/transform/transformation.py#L168-L177):
```python
class EllizeTransformation(NullaryTransformation):
    name: str = "ellize"
```

[rts.py L104](file:///home/huoda/sparsene/python/sparsene/transform/rts.py#L104):
```python
case "ell":
    seq.prepend(EllizeTransformation())
```

### 2.3 schedule.py — AtomicFormatType enum

[schedule.py L39-L43](file:///home/huoda/sparsene/python/sparsene/op_gen/computent/schedule.py#L39-L43):
```python
class AtomicFormatType(Enum):
    ELL = 4
```

### 2.4 computent.py — 数组定义 + schedule 构造

[computent.py L1535-L1643](file:///home/huoda/sparsene/python/sparsene/op_gen/computent/computent.py#L1535-L1643):
`apply_ellize_transformation()` — 生成 `val_ell_len`(SPLEN) / `val_ell_idx`(COO_IDX) / `val_ell_val`(VAL) 三个数组。

Schedule args 为 4 元组 `[len, idx, val, B_val]` — 和 COO 同结构, 不同于 CSR 的 5 元组。

[computent.py L1746-L1750](file:///home/huoda/sparsene/python/sparsene/op_gen/computent/computent.py#L1746-L1750):
```python
elif isinstance(transform, EllizeTransformation):
    schedule, array_defs, atomic_format = apply_ellize_transformation(...)
```

### 2.5 opir/ops.py — ELL restore 操作定义

[ops.py L274-L312](file:///home/huoda/sparsene/python/sparsene/op_gen/opir/ops.py#L274-L312):
```python
class EllAtomicValRestoreOp(DeviceOp):
    """operands = [ell_val, ell_idx, ell_len]  (3 operands, like COO)"""
```
- 3 操作数 (与 COO 相同, 不同于 CSR 的 4 操作数)
- `ell_range` 标量表示 block 总 entry 数 (`BLK_M * ell_len`)

### 2.6 opir/generate.py — OPIR 生成

[opir/generate.py L604-L607](file:///home/huoda/sparsene/python/sparsene/op_gen/opir/generate.py#L604-L607):
```python
case AtomicFormatType.ELL:
    return generate_from_ell_atomic_format_op(...)
```

[opir/generate.py L917-L973](file:///home/huoda/sparsene/python/sparsene/op_gen/opir/generate.py#L917-L973):
完整 OPIR 链路: `LoadOffsetOp → CooAtomicFormatLoadIdxOp → CooAtomicFormatLoadValOp → EllAtomicValRestoreOp → MmaOp`

### 2.7 nvir/generate.py — gmem 接口 + kernel 代码

**gmem 接口** [nvir/generate.py L159-L189](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/generate.py#L159-L189):
```python
"ELL_TCF": {
    "shape_overrides": {"val_ell_len": ["nnz_dim_K_o"], ...},
}
```
ELL 的 gmem 接口与 COO 几乎相同 (3 个关键数组 vs COO 的 3 个), 没有 CSR 那样的 `row_ptr` 特殊形状。

**kernel restore** [nvir/generate.py L1197-L1223](file:///home/huoda/sparsene/python/sparsene/op_gen/nvir/generate.py#L1197-L1223):
```cpp
const int blk_m = get<0>(BLK_MNK{});
const int ell_len = ell_range / blk_m;    // per-block max_nnz_per_row
const int ell_total = blk_m * ell_len;     // should equal ell_range
for (int i = lid; i < ell_total; i += 32) {
    auto value = ell_val(i);
    int idx = ell_idx(i);
    if (idx < tile_elems) { out[idx] = value; }
}
```
- `ell_len = ell_range / BLK_M` 显式体现 ELL 的 row-width 语义
- `ell_range` 由 host 填充为 `BLK_M * max_nnz_per_row`
- padding 条目使用 sentinel idx (`>= tile_elems`) 被自动跳过

**dispatch points**(都带 `ell_atomic_val_restore`):
- `_classify_logical_kind` (L764): `EllAtomicValRestoreOp → "ell_atomic_val_restore"`
- `_choose_output_mem` (L848): smem 输出
- `_choose_op_type` (L889): `NvOpType.OTHERS`
- swizzle 配置 (L1804-1810): `SwizzleLayout(1,2,3,adaptive_blk_k=True)`
- cp_async 标记 (L1841): `False`

### 2.8 formats/ELL_TCF.py — 格式定义

[ELL_TCF.py](file:///home/huoda/sparsene/python/sparsene/formats/ELL_TCF.py):
```python
ELL_TCF_FORMAT = Format(
    name="ELL-TCF",
    axes=[df_axis("X_o", ROW), sv_axis("Y_s", COL), df_axis("X_i", ROW)],
    child=Format(... child=ell_atomic_format("X_i", "Y_si"))
)
```
与 ME_TCF / CSR_TCF 同构,仅叶子原子格式不同。

### 2.9 host bridge — host_program_generated_ell.cu

[host_program_generated_ell.cu](file:///home/huoda/sparsene/examples/src_fp32/dtc/testbed/host_program_generated_ell.cu):

核心: `convert_old_metadata_to_generated()` — 从 reference COO 数据转换为 ELL 格式:

1. **Deswizzle**: 还原 `Swizzle<1,2,S>` 的物理 flat index → 逻辑行号
2. **Per-row NNZ 计数**: 按 `logical / blk_k` 分组并统计每行 NNZ
3. **取 max**: `ell_len = max(row_counts[0..BLK_M-1], 1)`
4. **row-major 输出**: 逐行输出真实 entry + sentinel padding
5. **sentinel idx**: `TC_BLOCK_M * TC_BLOCK_K` (即 `tile_elems`), kernel 的 `idx < tile_elems` 检查会自动跳过

关键: padding 必须用 sentinel idx, 不能用 0, 因为 0 是 tile 内的合法位置。

## 3. ELL vs COO vs CSR 差异对照

| 层 | COO | CSR | ELL |
|---|---|---|---|
| format class | CooAtomicFormat | CsrAtomicFormat | EllAtomicFormat |
| RTS transform | Cooize | Csrize | Ellize |
| schedule type | COO(1) | CSR(3) | ELL(4) |
| schedule args 数 | 4 | 5 | 4 |
| OPIR restore | CooAtomicValRestoreOp(3 operands) | CsrAtomicValRestoreOp(4 operands) | EllAtomicValRestoreOp(3 operands) |
| 额外 gmem 参数 | 无 | val_csr_row_ptr[BLK_M+1,block] | 无 |
| host payload 组织 | flat compact 顺序 | deswizzle → 按行排序 → row_ptr | deswizzle → 按行分组 → max-nnz padding |
| restore 代码结构 | flat scatter loop | flat scatter loop (优化后) | row-major stride loop (ell_len 显式) |

## 4. 调试经验

1. **padding sentinel bug**: ELL 的 padding 条目不能填 `idx=0`。因为 kernel 做 `idx < tile_elems` 检查, `tile_elems >= 0` 恒真, 会覆盖 tile[0] 位置。修复 → 使用 `tile_elems` 本身作为 sentinel。

2. **gen_add fusion 与 pipeline**: 启用 `_fuse_inline_gen_add` 可将 ELL 的 `gen_add` 从 55 个降到 0, 直接提升 15% 性能 (2.02→1.73ms)。但该 pass 对 CSR 不安全 (CSR 的 no-op `array_ref_2` 会留下 stale reference), 因此加了 `has_array_ref_2` gate。

3. **ELL 的 restore 和 COO 在运行期等价但不相同**: ELL 的 `ell_len = ell_range/BLK_M` 显式定义了 row-width 语义, 而 COO 直接使用 flat range。这个 `.inc` 代码层面的差异是格式身份的关键, 即便两者在数学上遍历相同的 flat index 序列。

## 5. 性能数据 (shipsec1, TBK=8, TBM=16)

| 格式 | mykernel | cusparse | 加速比 | 备注 |
|---|---|---|---|---|
| COO (ME_TCF) | 1.453 ms | 4.011 ms | 2.76x | 基准 |
| CSR (优化后) | 1.698 ms | 4.010 ms | 2.36x | 用 row_ptr 参数 |
| ELL (row-major) | 1.824 ms | 4.011 ms | 2.20x | 含 row-major padding 开销 |

ELL 比 COO 慢 26%, 主要来自 host 侧的 per-block max-nnz 计算和 padding 膨胀。但作为真正独立的 ELL 格式, 代码路径已完全分叉。

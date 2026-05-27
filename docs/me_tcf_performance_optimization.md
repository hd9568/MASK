# ME_TCF 算子性能优化路径

> 维护者: AI Assistant (2026-05-27)
> 目标: 将 `test_generate.py` 生成的 ME_TCF 算子性能对齐手写参考实现 `kernel.inc`

---

## 1. 最终性能对比

```
Matrix: shipsec1 (M=140874, K=140874, nnz=3.98M, 99.98% sparse, N=512)

┌────────────────────┬──────────┬──────────┬───────────┬──────────┐
│ Kernel             │ mykernel │ cusparse │ vs cusp   │ vs 手写  │
├────────────────────┼──────────┼──────────┼───────────┼──────────┤
│ 手写 (kernel.inc)  │ 1.475 ms │ 4.012 ms │ 2.72x     │    —     │
│ 生成 (优化后)       │ 1.998 ms │ 4.013 ms │ 2.01x     │ 35.4% 慢 │
│ 生成 (优化前)       │ 2.004 ms │ 4.020 ms │ 2.01x     │ 35.8% 慢 │
└────────────────────┴──────────┴──────────┴───────────┴──────────┘
```

| 指标 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| Stage0 ops | 9 | **6** | -33% |
| 代码行数 | 2742 | **2470** | -272行 |
| 冗余模板类 | 3 (gen_div_1, gen_const, gen_add_1) | **0** | 全部消除 |
| 正确性 | ✅ error=0 | ✅ error=0 | 无回归 |

---

## 2. 代码结构对比分析

### 2.1 全局对比

| 维度 | 手写 (kernel.inc) | 生成 (me_tcf_kernel3.inc) |
|------|------------------|--------------------------|
| 代码行数 | 2134 | 2470 |
| 主循环 inner ops | 21 | 29→26→23 (逐步优化) |
| 运行时算术 ops | 0 (全部 host 预计算) | 7→4→1 |
| Pipeline Stage0 | 融合 G2s* loader | 10→8→6 ops |
| B 矩阵加载 | G2sMatrixBLoadOp (cp_async) | array_ref_1 (cp_async) |
| offset 来源 | val_soff (host 预计算) | val_len_offset (kernel 端 LoadOffsetOp) |

### 2.2 生成代码中的冗余链

优化前的 Stage0 包含以下 ops 链：

```
  gen_div_1(off_l, 1)       # off_l / 1 = off_l  (无操作)
  gen_const()               # 0
  gen_add(off_l, 0)         # off_l + 0 = off_l  (无操作)
  gen_add_1(off_l, l+i)     # 真正有用的计算: off_l + (l+i)
  array_ref(val_sidx, idx)  # 从 smem 加载索引
  array_ref_1(B_val, ...)   # cp_async 加载 B 矩阵
  load_offset_1(...)        # 加载 COO offset 对
  coo_atomic_format_load_idx(...)
  coo_atomic_format_load_val(...)
```

### 2.3 手写代码的对应链

```
  G2rSparseOffsetLoadOp     # 直接从 host 预计算的 val_soff 数组加载 (l, r)
  G2sSparseIndexLoadOp      # Gmem→Smem 加载 val_sidx
  G2sSparseCooIdxLoadOp     # Gmem→Smem 加载 coo_idx
  G2sSparseCooValLoadOp     # Gmem→Smem 加载 coo_val
  G2sMatrixBLoadOp          # cp_async 加载 B 矩阵
  S2sRestoreMatrixAOp       # Smem→Smem 恢复 A 矩阵
```

---

## 3. 根因分析

### 3.1 冗余 gen_div_1 + gen_const + gen_add_1 产生路径

这三个冗余 op 来自 `VarlenLoweringPass._rewrite_array_indices_if_needed()` 中的线性索引计算：

```
flat_index = a * offset + b * i + j
```

对于 ME_TCF 格式的 `varlen(K/BLK_K)` 维度：
- `a_expr = diff(K/BLK_K, K) = 1/BLK_K` （生成 DivOp）
- `b_expr = subs(K/BLK_K, K, 0) = 0` （生成 ConstantOp(0) → gen_const）
- `term_a = scale_base_offset(LoadOffsetOp, a_expr)` → `gen_div_1(off_l, 1)`
- `term_b = mul(i, b_expr)` → `gen_const(0)` → `gen_add(term_a, 0)`

当 BLK_K 在 codegen 中被 hack 为 1 时（因为 off_l 已经是 block 级别），这些 op 全部变成无操作。

### 3.2 剩余 35% 性能差距根因

| 原因 | 手写 | 生成 |
|------|------|------|
| offset 来源 | val_soff host 预计算 | val_len_offset kernel 端 LoadOffsetOp |
| 融合 loader | G2s* 系列（单 op = 范围加载+cp_async） | 分步 array_ref + load_offset + coo_atomic_* |
| 算术 | 0 个运行时算术 | 1 个 gen_add (off_l + l) |

**核心差异**: 手写版本使用 host 端预计算的 `val_soff` 数组直接加载 block 起始偏移，完全消除了 kernel 端的 offset 计算。生成框架通过 `LoadOffsetOp` 加载 `val_len_offset` 并在 kernel 端做 `off_l + l` 拼接。

---

## 4. 已实施的优化

### 4.1 优化 1: _build_add 常量消除

**文件**: `python/sparsene/op_gen/opir/varlenLoweringPass.py` Line 406

**问题**: `_build_add` 中使用 `getattr(lhs, "value", None)` 检测常量，此法无效（Value 对象不直接存储 value 属性）。

**修复**: 新增 `_is_const_int(val, target)` 方法，通过 `val.defining_op` 查找 ConstantOp。

```python
def _is_const_int(self, val: Value, target: int) -> bool:
    if not hasattr(val, "defining_op") or val.defining_op is None:
        return False
    op = val.defining_op
    if not isinstance(op, ConstantOp):
        return False
    return int(op.attributes.get("value", None)) == target

def _build_add(self, lhs, rhs, generated_ops):
    if self._is_const_int(lhs, 0): return rhs
    if self._is_const_int(rhs, 0): return lhs
    add = AddOp(lhs, rhs)
    generated_ops.append(add)
    return add.result
```

**效果**: 消除 `gen_add_1` (Stage0: 9→8 ops)

### 4.2 优化 2: _build_sub 常量消除

**文件**: `python/sparsene/op_gen/opir/varlenLoweringPass.py` Line 642

**修复**: `rhs == 0` 时直接返回 lhs。

```python
def _build_sub(self, lhs, rhs, generated_ops):
    if self._is_const_int(rhs, 0): return lhs
    sub = SubOp(lhs, rhs)
    generated_ops.append(sub)
    return sub.result
```

### 4.3 优化 3: a_expr = 1/BLK_K 时跳过 scaling

**文件**: `python/sparsene/op_gen/opir/varlenLoweringPass.py` Line 296

**根因**: `_get_base_offset` 中 `scale_expr = 1/BLK_K` 触发 `_build_fractional_mul` 生成 DivOp(offset, BLK_K)。但在 block 级别，offset 值已经无需 scaling。

**修复**: 新增 `_is_blk_scaled_one_expr()` 检测 `scale_expr * BLK_K == 1`，满足条件时跳过 scaling。

```python
def _is_blk_scaled_one_expr(self, expr: Any) -> bool:
    try:
        return simplify(sympify(expr) * sympify("BLK_K") - 1) == 0
    except Exception:
        return False

# 在 _get_base_offset 中:
if self._is_one_expr(scale_expr) or self._is_blk_scaled_one_expr(scale_expr):
    return base_offset
```

**效果**: 消除 `gen_div_1` (Stage0: 8→7 ops)

### 4.4 优化 4: b_expr = 0 时跳过 term_b

**文件**: `python/sparsene/op_gen/opir/varlenLoweringPass.py` Line 373

**根因**: `b_expr = 0` 时 `_build_fractional_mul(i, 0)` 生成 ConstantOp(0)，然后 `_build_add(term_a, 0)` 做无操作相加。

**修复**: 新增 `_is_zero_expr()` 方法，b_expr == 0 时直接让 `dim1_base_idx = term_a`。

```python
def _is_zero_expr(self, expr: Any) -> bool:
    try: return simplify(sympify(expr)) == 0
    except: return False

if self._is_zero_expr(spec.b_expr):
    dim1_base_idx = term_a
else:
    term_b = self._build_fractional_mul(i_val, spec.b_expr, prefix_ops)
    dim1_base_idx = self._build_add(term_a, term_b, prefix_ops)
```

**效果**: 消除 `gen_const` (Stage0: 7→6 ops)

### 4.5 删除 codegen 中的 BLK_K hack（已回滚）

**文件**: `python/sparsene/op_gen/nvir/codegen.py` Line 689

**尝试**: 移除 `dump_f_method_call_arg` 中 special-case BLK_K→1 的 hack。

**结果**: 导致 error_count=199（语义错误）。BLK_K hack 是**有意为之**的正确行为：`off_l` 来自 `val_len_offset` 数组，值已经是 block 级别，除以 BLK_K 得到 1 是正确的 identity。**已回滚。**

---

## 5. 剩余优化空间

### 5.1 Host-side 预计算 (预估 +++ 效果)

将 `val_len_offset` stride 计算移到 host 端，kernel 只做 direct load。

**难度**: 高 — 需要修改 `computent_from_rts` 或格式定义。

### 5.2 G2s 融合 loader

将 `array_ref + load_offset_1 + coo_atomic_format_load_*` 三步融合为单个 G2s 类，减少模板参数和 pipeline 延迟。

**难度**: 高 — 需要修改 NVIR lowering 的 load_path。

### 5.3 gen_mul 消除

`gen_mul` 在 stage0 中计算 `slice_offset`。该值可以在 host 端预计算或内联到 load_offset 中。

**难度**: 中 — 需要修改 `CValFlattenPass` 或 NVIR lowering。

---

## 6. 修改文件清单

| 文件 | 变更类型 | 描述 |
|------|---------|------|
| `python/sparsene/op_gen/opir/varlenLoweringPass.py` | 修改 | `_build_add`/_build_sub` 常量消除 |
| `python/sparsene/op_gen/opir/varlenLoweringPass.py` | 新增 | `_is_const_int`, `_is_blk_scaled_one_expr`, `_is_zero_expr` |
| `python/sparsene/op_gen/opir/varlenLoweringPass.py` | 修改 | `_get_base_offset` 跳过 BLK_K scaling |
| `python/sparsene/op_gen/opir/varlenLoweringPass.py` | 修改 | `_rewrite_array_indices_if_needed` 跳过 b_expr=0 |

---

## 7. 验证命令

```bash
cd /home/huoda/sparsene && source ~/llm-sog-venv/bin/activate

# 生成并编译优化后 kernel
PYTHONPATH=python python3 python/test/test_generate.py --strategy-provider heuristic
cp me_tcf_kernel3.inc python/me_tcf_kernel3.inc

# 编译 + benchmark
nvcc -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo \
  -ftemplate-backtrace-limit=0 -arch=sm_80 \
  -ccbin /opt/rh/gcc-toolset-13/root/usr/bin/g++ \
  -Xcompiler -fopenmp -Df32 \
  -I./examples/src_fp32/dtc/testbed/cutlass/include \
  -I./examples/src_fp32/dtc/testbed/cutlass/examples/common \
  -I./examples/src_fp32/dtc/testbed/cutlass/tools/util/include \
  -I./examples/src_fp32/dtc/testbed/src \
  -I./examples/src_fp32/dtc/testbed \
  -I./python \
  ./examples/src/fp32/dtc/testbed/host_program_generated.cu \
  ./examples/src/fp32/dtc/testbed/main.cu \
  ./examples/src/fp32/dtc/testbed/src/mmio.cu \
  ./examples/src/fp32/dtc/testbed/src/mmio_highlevel.cu \
  ./examples/src/fp32/dtc/testbed/src/utils.cu \
  -lcusparse -lcublas -o /tmp/demo_opt

/tmp/demo_opt -filename /mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx \
  -mtx_flag 1 -N 512 -warmup 1 -repeat 5
```

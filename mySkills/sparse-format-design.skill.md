# 稀疏格式设计

> **维护者**: AI Assistant (2026-05)
> **版本**: v1.0
> **适用范围**: 需要理解或修改 sparsene 稀疏格式定义的开发者、格式智能体设计者。

本 skill 专注于 sparsene 的稀疏格式定义系统和已实现的格式，帮助理解如何为不同的稀疏矩阵特征选择或设计合适的数据布局。

---

## 1. 格式系统的设计哲学

sparsene 的格式系统是**层次化**和**声明式**的。一个格式定义包含：
- 外层：粗粒度的 block 划分轴
- 内层：细粒度的 tile 内部布局
- 底层：原子格式（dense / COO / MCO）

格式 ≠ 物理存储。格式是**逻辑布局描述**，通过 RTS 变换系统映射到实际的内存访问模式。

---

## 2. 轴类型速查

| 缩写 | 含义 | 密度 | 长度 | 示例 |
|------|------|------|------|------|
| `df` | Dense Fixed | 稠密 | 固定/符号 | `df_axis("X_o", ROW, "M/BLK_M")` |
| `dv` | Dense Varlen | 稠密 | 变长 | `dv_axis("nnz", ROW)` |
| `sf` | Sparse Fixed | 稀疏 | 固定 | `sf_axis("bits", COL, "NNZ/BLK_M")` |
| `sv` | Sparse Varlen | 稀疏 | 变长 | `sv_axis("Y_s", COL, "K")` |

关键区别：
- **sv** 轴是稀疏 + 变长：每一行（block）的列长度不同。这对于不规则稀疏矩阵至关重要。
- **df** 轴是稠密 + 固定：所有 block 大小一致。

---

## 3. 已实现的格式

### 3.1 ME_TCF（Mixed-Elimination TCF）

**文件**：`python/sparsene/formats/DTC_SpMM.py`

**结构**：
```
Format("ME-TCF")
├── df_axis("X_o", ROW, "M")           # M 行，block 划分
├── sv_axis("Y_s", COL, "K")           # K 列，变长稀疏
├── df_axis("X_i", ROW, "BLK_M")       # 每 block 的 M 维度
└── child: Format
    ├── df_axis("Y_so", COL, "K/BLK_K") # K 维度的 block 划分
    ├── df_axis("Y_si", COL, "BLK_K")   # 每 block 的 K 维度
    ├── df_axis("X_i", ROW, "BLK_M")    # 每 block 的 M 维度
    └── child: coo_atomic_format("X_i", "Y_si")  # COO 稀疏
```

**特点**：
- 双层：外层粗 block 索引（`X_o`, `Y_so`），内层每 block COO
- `Y_s` 是 sv 轴，支持 row-level elimination（不同 row 有不同 nnz）
- 适合：中等稀疏度、block-wise 可计算的矩阵

**使用场景**：当前项目唯一使用过的格式。RTS 变换由此格式推导，OPIR 据此生成。

### 3.2 BIT_TCF（Bitmask TCF）

**文件**：`python/sparsene/formats/Acc_SpMM.py`

**结构**：
```
Format("DTC-SpMM")
├── df_axis("X_o", ROW, "M")
├── sv_axis("Y_s", COL, "K")
├── df_axis("X_i", ROW, "BLK_M")
└── child: Format
    ├── df_axis("Y_so", COL, "K/BLK_K")
    ├── df_axis("Y_si", COL, "BLK_K")
    ├── df_axis("X_i", ROW, "BLK_M")
    └── child: mco_atomic_format("X_i", "Y_si")  # Masked-COO
```

**与 ME_TCF 的区别**：使用 **MCO**（Masked-COO）而非 COO。MCO 使用 bitmask 而非索引数组来标记非零位置。

**适合场景**：高稀疏度矩阵，bitmask 比完整索引更紧凑。

### 3.3 HME_TCF（Hybrid ME TCF）

**文件**：`python/sparsene/formats/HME_TCF.py`

ME_TCF 的混合变体，在细粒度 block 层级提供不同的消除策略。

### 3.4 ROW_REORDER_SR_BCRS

**文件**：`python/sparsene/formats/ROW_REORDER_SR_BCRS.py`

行重排序 + SR-BCRS 格式。在 ME_TCF 基础上增加行重排序（row reorder）步骤以改善访问局部性。

### 3.5 SR_BCRS

**文件**：`python/sparsene/formats/SR_BCRS.py`

标准的 Split-Row BCRS 格式，无 Tensor Core 适配。

---

## 4. 格式设计决策树

当为新矩阵选择或设计格式时，考虑以下因素：

```
矩阵特征
├── 能否 fit Tensor Core tile (BLK_M=16/32, BLK_K=16)?
│   ├── 能 → 考虑 TCF 家族（ME_TCF / BIT_TCF）
│   └── 不能 → 考虑 SR_BCRS / 通用格式
│
├── 稀疏度水平
│   ├── > 95% → BIT_TCF (bitmask 开销小)
│   ├── 80-95% → ME_TCF (COO 索引适中)
│   └── < 80% → 考虑 row reorder 优化
│
├── 稀疏结构规则性
│   ├── 规则（每 row 的 nnz 类似）→ df 轴多，sv 少
│   └── 不规则 → sv 轴必须保留
│
└── 矩阵规模
    ├── 大矩阵 → 多层 block 划分（更多 df 层级）
    └── 小矩阵 → 简化层次
```

---

## 5. 格式 → 生成的映射关系

格式定义直接影响生成的 CUDA kernel：

| 格式特征 | 生成的代码影响 |
|----------|---------------|
| `sv_axis` | 生成 varlen 循环 + length 数组查找 |
| `coo_atomic_format` | 生成 COO idx/val 加载 + 恢复循环 |
| `mco_atomic_format` | 生成 bitmask 加载 + 条件恢复 |
| 多层 Format | 生成多级 block 循环 |
| `df_axis` 的 length 为符号 | 生成模板参数 |

---

## 6. 新增格式的开发步骤

1. 在 `python/sparsene/formats/` 创建新文件
2. 用 `df_axis` / `sv_axis` / `sf_axis` / `dv_axis` 定义格式轴
3. 如果涉及新原子格式 → 在 `format.py` 和 `computent.py` 中增加处理
4. 在 `test_generate.py` 中注册新格式
5. 运行 `python test/test_generate.py` 验证 OPIR 生成
6. 运行 `validate_generated_kernel.sh` 验证 correctness

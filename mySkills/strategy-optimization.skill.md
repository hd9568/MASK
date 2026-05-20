# CUDA 算子优化策略

> **维护者**: AI Assistant (2026-05)
> **版本**: v1.0
> **适用范围**: 需要理解 ME_TCF 的 CUDA 优化策略空间、设计新 knob、或修改现有约束的开发者。

本 skill 基于对 sparsene ME_TCF 格式的 extensive 实验（~200+ 次 kernel 编译和验证），汇总了所有已验证安全和不安全的优化策略组合。

---

## 1. 策略空间总览

在 ME_TCF 的 SpMM kernel 中，策略决策分为**张量放置**（`TensorPlacementStrategy`）和**流水线**（`PipelineStrategy`）两个维度。

### 1.1 TensorPlacementStrategy（张量放置策略）

| Knob | 安全值 | 默认 | 含义 |
|------|--------|------|------|
| `tile_b` | `16, 32, 64` | `64` | N 方向 tile 大小。控制 B 矩阵的列方向分块大小 |
| `materialize_b_array_ref_to_smem` | `True, False` | `True` | B-side gather 是否物化到 smem |
| `materialize_val_sidx_to_smem` | **`True` only** | `True` | val_sidx 是否物化到 smem。**不可改** |
| `enable_b_array_ref_cp_async` | `True, False` | `True` | B-side 加载是否用 cp.async |
| `b_array_ref_mode` | `cp_async_1x4, sync_scalar` | `cp_async_1x4` | B-side 加载模式 |
| `b_smem_swizzle` | **`swizzle_323` only** | `swizzle_323` | B staging smem 的 swizzle 布局。**不可改** |
| `coo_idx_output_mem` | `smem, rmem` | `smem` | COO idx 的存放位置 |
| `coo_val_output_mem` | `smem, rmem` | `smem` | COO val 的存放位置 |
| `coo_load_mode` | `cp_async_x4_aligned_fallback, sync_scalar` | `cp_async_x4_aligned_fallback` | COO 加载模式 |
| `coo_restore_swizzle` | **`swizzle_123` only** | `swizzle_123` | COO restore smem 的 swizzle 布局。**不可改** |
| `align_cp_async_smem_outputs` | `True, False` | `True` | cp.async 输出是否 16 字节对齐 |

### 1.2 PipelineStrategy（流水线策略）

| Knob | 安全值 | 默认 | 含义 |
|------|--------|------|------|
| `enable_async_pipeline` | `True, False` | `True` | 是否启用异步软件流水线 |
| `sync_threads_on_smem_input` | `True, False` | `True` | 消费 smem 前是否加 `__syncthreads` |
| `max_wait_prior` | `None, 0, 1, 2, 3` | `None` | 流水线最大提前量（`__pipeline_wait_prior`） |
| `stage_shifts` | `[], [0], [1], [2]` | `[]` | 流水线 stage 偏移 |

---

## 2. 约束规则（CRITICAL — 违反则崩溃或 correctness 错误）

### 2.1 swizzle 锁定

**规则**：`b_smem_swizzle` 必须为 `swizzle_323`，`coo_restore_swizzle` 必须为 `swizzle_123`。

**根因**：`_insert_ldmatrix_if_needed()` 中的 `S2rAValLoadOp`/`S2rBValLoadOp` 地址映射是硬编码的 lane→smem 地址公式，不随 swizzle 变化。改 swizzle 后 `ldmatrix` 指令读到错误的 smem 地址。

**证据**：修改 `coo_restore_swizzle=none` → `Error(row=4, col=1-199)` 整段偏差；修改 `b_smem_swizzle=swizzle_123` + `coo_restore_swizzle=swizzle_323` → `Error(row=0, col=0-199)` 从首元素开始整段偏差。

### 2.2 COO rmem + cp.async = 崩溃

**规则**：`coo_idx_output_mem=rmem` 或 `coo_val_output_mem=rmem` 时，`coo_load_mode` **必须**是 `sync_scalar`。

**根因**：COO rmem staging 与 async-copy lowering 不兼容，触发了对非 GPU 地址的 cp.async，产生 `illegal memory access`。

**编码位置**：`StrategyDecision.from_dict()` 中有自动归一化逻辑。

### 2.3 pipeline=false 必须降级所有 async

**规则**：`enable_async_pipeline=False` 时，必须同时：
- `enable_b_array_ref_cp_async=False`
- `b_array_ref_mode=sync_scalar`
- 如果 `coo_load_mode=cp_async_x4_aligned_fallback`，则降为 `sync_scalar`

**根因**：关 pipeline 后 producer/consumer 同步前提被破坏。保留任一 `cp.async` 时 `__pipeline_commit`/`__pipeline_wait_prior` 语义不成立。

**编码位置**：`StrategyDecision.from_dict()`。

### 2.4 val_sidx 必须留在 smem

**规则**：`materialize_val_sidx_to_smem` **必须**为 `True`。

**根因**：val_sidx 是 B 矩阵 gather 的索引表，当前 lowering 依赖它在 smem 中的存在。改 `False` 导致从 `(0,0)` 开始整段偏差。

### 2.5 checkpoint cache 污染

**问题**：synthetic case 的 checkpoint key 原来不包含 `sparsity`，不同稀疏度的 case 复用同一缓存。

**修复**：`main.cu` 中 synthetic checkpoint key 加入 `formatSparsityToken(sparsity)`；feedback loop 中 synthetic 执行设置 `SPARSENE_DISABLE_MTX_CACHE=1`。

---

## 3. 有效搜索空间

```
pipeline=True:
  tile_b(3) × b_cp_async(2) × coo_placement(3) × max_wait_prior(5) × stage_shifts(4) × sync_smem(2) × align_16(2)
  = 3 × 2 × 3 × 5 × 4 × 2 × 2 = 1,440

pipeline=False (b_cp_async 强制关, COO 强制 sync_scalar 且 smem):
  tile_b(3) × coo_placement(1) × max_wait_prior(5) × stage_shifts(4) × sync_smem(2) × align_16(2)
  = 3 × 1 × 5 × 4 × 2 × 2 = 240
  (coo_rmem + sync_scalar 在 pipeline=false 下也可行，所以实际是 240 × 2 = 480)
  但 b_cp_async 强制关时 b_array_ref_mode 固定 sync_scalar

总计: ~1,920 种有效组合
```

**COO placement 的三态**（在 `pipeline=True` 下）：
| 组合 | coo_idx_output_mem | coo_val_output_mem | coo_load_mode |
|------|-------------------|-------------------|---------------|
| A (baseline) | smem | smem | cp_async_x4_aligned_fallback |
| B | smem | smem | sync_scalar |
| C | rmem | rmem | sync_scalar (强制联动) |

---

## 4. 各 knob 的优化语义

### 4.1 tile_b（N 方向 tile 大小）
- `64`：baseline。大的 N tile 提高 Tensor Core 按列复用，但增加 block 总量
- `32`：减少每个 block 的工作量，增加并行度。验证通过
- `16`：单个 warp 的 N 方向更小

**选择依据**：
- N 较小（<256）→ `16` 或 `32`
- N 较大（≥512）→ `64`
- 需要配合 N 是否整除 tile_b（否则触发 fallback dispatch）

### 4.2 B-side cp.async
`enable_b_array_ref_cp_async=True` + `b_array_ref_mode=cp_async_1x4`：
- B 矩阵 gather 使用 `cp.async` 从 global→smem
- 与 async pipeline 配合时效果最好
- 寄存器压力稍高（~128 regs vs ~126）

`enable_b_array_ref_cp_async=False` + `b_array_ref_mode=sync_scalar`：
- B 矩阵使用标量 load，降低寄存器压力
- 吞吐量下降（~3.5x slower on shipsec1）

### 4.3 COO 加载模式

**cp_async_x4_aligned_fallback**（默认）：
- 当 COO 起始地址 4 字节对齐时使用 `cp.async` 128-bit
- 否则回退到标量 load
- 最高吞吐量，但需要 smem staging

**sync_scalar**：
- 全部 COO 数据走标量 load
- 降低寄存器压力和同步复杂度
- 吞吐量下降但可靠性高

### 4.4 COO 放置（smem vs rmem）

**smem**（默认）：
- COO idx/val 先经由 smem staging，再 restore 到 A_val_tile
- 支持 cp.async 加载
- 额外 smem 占用和同步点

**rmem**（联动 sync_scalar）：
- COO idx/val 直接装载到寄存器
- 减少 smem traffic 和同步点
- 不兼容 cp.async，必须 sync_scalar
- 对 shipsec1 测试通过但性能下降

### 4.5 max_wait_prior
- `None`：流水线无深度限制，最大重叠
- `0`：无流水线重叠，每个 stage 等待上一个完成
- `1-3`：有界深度，减少 register pressure 但限制 overlap

已测试通过：`0`, `1`, `2`, `3`, `None`。

### 4.6 stage_shifts
- `[]`（默认）：实际 shift=1，相邻 stage 间延迟 1
- `[0]`：无延迟
- `[1]`：显式 shift=1
- `[2]`：双倍延迟

不同 shift 会影响 `__pipeline_wait_prior` 的参数。

### 4.7 sync_threads_on_smem_input
- `True`（默认）：消费 smem 前加 `__syncthreads`，确保 producer 完成
- `False`：移除同步，依赖 pipeline 顺序担保

### 4.8 align_cp_async_smem_outputs
- `True`（默认）：cp.async 输出对齐到 16 字节，多占 smem
- `False`：不强制对齐，可能引发 misalignment 但省 smem

---

## 5. 新增 Knob 的安全测试流程

```bash
# 1. 生成 manual json（改 baseline 的一个字段）
python -c "
import json
base = json.load(open('/tmp/sparsene_manual_tests/baseline.json'))
base['tensor_placement']['new_knob'] = 'new_value'
json.dump(base, open('/tmp/test.json','w'), indent=2)
"

# 2. 生成 + 编译 + 运行
python test/test_generate.py --strategy-provider manual_json \
  --strategy-manual-json /tmp/test.json
PATCHED_ONLY=0 SKIP_GENERATE=1 \
  bash examples/src_fp32/dtc/testbed/scripts/validate_generated_kernel.sh \
  -N 512 -warmup 1 -repeat 2

# 3. 检查 correctness
grep -c 'Error(' /path/to/output.log
grep -c 'illegal memory' /path/to/output.log
```

---

## 6. 未来优化方向

1. **S2R/ldmatrix layout-aware 改造**：让 `b_smem_swizzle` 和 `coo_restore_swizzle` 可安全变动（→ 搜索空间翻 9 倍）
2. **COO rmem + async 支持**：补齐 COO rmem 路径与 async-copy 的兼容性
3. **materialize_val_sidx_to_smem** 的 correctness 问题排查
4. **ldmatrix load mode 选择**（x2 vs x4、trans vs 非 trans）
5. **stage partition 策略**：当前是编译器自动划分，可暴露为策略 knob

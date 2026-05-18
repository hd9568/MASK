# sparsene 项目开发技能文档

> 本文档涵盖项目全链路：代码生成、策略智能体、反馈循环、已知陷阱、脚本启动方式。
> 当前（2026-05-14）项目处于策略反馈优化阶段，重点是对 ME_TCF 格式的 SpMM 进行自动调优。

---

## 1. 工程概览

### 1.1 项目目标
为稀疏矩阵-稠密矩阵乘（SpMM）自动生成 CUDA kernel。核心格式是 **ME_TCF**（Mixed-Elimination Tensor Core Format）。
- 输入：稀疏格式描述（OPIR）
- 中间表示：NVIR（NvOp IR）
- 输出：`.inc` CUDA kernel + host wrapper
- 策略智能体决定张量放置、流水线、swizzle 等底层选择。

### 1.2 生成链路
```
format definition (DTC_SpMM.py)
  → derive_rts() → computent
  → opir.generate_from_computent() → OPIR ops
  → VarlenLoweringPass / CValFlattenPass
  → nvir.generate_nvir() → NVIR
  → compiler_driver.apply_software_pipeline_and_codegen() → .inc
```
关键入口文件：
- `python/test/test_generate.py` — 单次生成入口
- `python/sparsene/op_gen/nvir/generate.py` — NVIR lowering（最核心，含所有策略接入点）
- `python/sparsene/op_gen/nvir/codegen.py` — `.inc` dump（含 metadata 宏生成）
- `python/sparsene/op_gen/nvir/templates.py` — CUDA 源码骨架
- `python/sparsene/op_gen/nvir/compiler_driver.py` — 软件流水线 + codegen 驱动

### 1.3 编译与运行
编译链：
```
.inc 文件 → nvcc + host_program_generated.cu + main.cu → demo-dtc-generated-tf32
```
- `host_program_generated.cu` include `.inc`，提供 `sparsene_generated_tile_b()` 等运行时分发函数
- `main.cu` 根据 generated TILE_B 做 `<16>/<32>/<64>` dispatch
- 动态链接 `cusparse`、`cublas`
- 编译 arch：`sm_80`

---

## 2. 环境变量与路径

### 2.1 虚拟环境
```bash
VENV_PATH="${HOME}/llm-sog-venv"
source "${VENV_PATH}/bin/activate"
```

### 2.2 关键环境变量
| 变量 | 默认值 | 作用 |
|------|--------|------|
| `PYTHONPATH` | `${ROOT}/python` | Python 模块搜索路径 |
| `SPARSENE_MTX_CACHE_DIR` | `/mnt/lustre-client/huoda/mtx_cache` | ME_TCF checkpoint 缓存目录 |
| `SPARSENE_DISABLE_MTX_CACHE` | `1` (validate 脚本) | 禁用 cache（feedback loop 中必须设置） |
| `CUDA_ARCH` | `sm_80` | 编译目标架构 |
| `NVCC_CCBIN` | `/opt/rh/gcc-toolset-13/root/usr/bin/g++` | nvcc 宿主编译器 |
| `PATCHED_ONLY` | `1` | 只编译/运行 patched inc，不重新生成 |
| `SKIP_GENERATE` | `0` | 跳过生成步骤 |
| `FORCE_RUN` | `0` | 无 GPU 检测时仍运行 |

### 2.3 数据集路径
- **真实数据集**：`/mnt/lustre-client/zhangkaige/selected_mtx/`（`.mtx` 文件，不要递归搜索）
- **当前默认反馈数据集**：`shipsec1.mtx`（路径：`/mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx`）
- **MTX cache 目录**：`/mnt/lustre-client/huoda/mtx_cache`（已从本地迁移，避免磁盘满）
- 原本地路径 `/home/huoda/sparsene/examples/src_fp32/dtc/testbed/mtx_cache` 已废弃

### 2.4 日志输出目录
- 反馈循环输出：`docs/agent_feedback_runs/`
- 智能体日志：`docs/agent_runs/` 或各 iter 下的 `agent/` 子目录
- 目录格式：`{timestamp}_me_tcf_{provider}_{label}_multidataset`

---

## 3. 策略智能体系统

### 3.1 核心文件
- [strategy_agent.py](file:///home/huoda/sparsene/python/sparsene/op_gen/strategy_agent.py) — 所有策略定义、schema、prompt、providers

### 3.2 策略 knob 完整列表

#### TensorPlacementStrategy
| Knob | 有效值 | 默认 | 说明 |
|------|--------|------|------|
| `tile_b` | `16, 32, 64` | `64` | N 方向 tile 大小 |
| `materialize_b_array_ref_to_smem` | `True, False` | `True` | B-side gather 是否放到 smem |
| `materialize_val_sidx_to_smem` | `True` | `True` | **不可改**（已验证不安全） |
| `enable_b_array_ref_cp_async` | `True, False` | `True` | B-side 是否用 cp.async |
| `b_array_ref_mode` | `cp_async_1x4, sync_scalar` | `cp_async_1x4` | B-side 加载模式 |
| `b_smem_swizzle` | `swizzle_323` | `swizzle_323` | **锁定**（S2R 路径未 layout-aware） |
| `coo_idx_output_mem` | `smem, rmem` | `smem` | COO idx 存放位置 |
| `coo_val_output_mem` | `smem, rmem` | `smem` | COO val 存放位置 |
| `coo_load_mode` | `cp_async_x4_aligned_fallback, sync_scalar` | `cp_async_x4_aligned_fallback` | COO 加载模式 |
| `coo_restore_swizzle` | `swizzle_123` | `swizzle_123` | **锁定**（S2R 路径未 layout-aware） |
| `align_cp_async_smem_outputs` | `True, False` | `True` | 16 字节对齐 cp.async 输出 |

#### PipelineStrategy
| Knob | 有效值 | 默认 | 说明 |
|------|--------|------|------|
| `enable_async_pipeline` | `True, False` | `True` | 异步流水线 |
| `sync_threads_on_smem_input` | `True, False` | `True` | 消费 smem 前加 `__syncthreads` |
| `max_wait_prior` | `None, 0, 1, 2, 3` | `None` | 流水线最大提前量 |
| `stage_shifts` | `[], [0], [1], [2]` | `[]` | 流水线 stage 偏移 |

### 3.3 当前搜索空间估算
- `pipeline=True`：`3×2×3×5×4×2×2 = 1,440`
- `pipeline=False`（b_cp_async 强制关，COO 只有 2 种模式）：`3×1×2×5×4×2×2 = 480`
- **总计：~1,920 种有效组合**

### 3.4 Providers
- `heuristic` — 返回固定已知良好策略（不需要 API key）
- `manual_json` — 从 JSON 文件读取（用于单因子测试）
- `openai_http` — OpenAI 兼容 API（用于 mimo / deepseek）
- `anthropic_http` — Anthropic 兼容 API（用于 mimo）
- `subprocess_json` — 调用外部命令

---

## 4. 模型 API 接入

### 4.1 可用模型
| 模型 | API Key | Base URL |
|------|---------|----------|
| `mimo-v2.5-pro` | `tp-ckk8svdzeha2h3dly4ec5984bq914cpty4shelmntgg8c83p` | `https://token-plan-cn.xiaomimimo.com/v1` |
| `deepseek-v4-flash` | `sk-eb7f7960366d41a5a4310277168e93da` | `https://api.deepseek.com/v1` |
| `deepseek-v4-pro` | `sk-eb7f7960366d41a5a4310277168e93da` | `https://api.deepseek.com/v1` |

### 4.2 DeepSeek 特殊注意事项
- DeepSeek 会先消耗 reasoning token，再输出 JSON 内容
- `strategy_max_tokens` 需要设大（建议 `4096`）否则容易空内容或截断
- 在 `strategy_agent.py` 的 `_try_parse_json()` 中若遇到空 `content`，检查是否有 `reasoning_content` 字段

### 4.3 模型连通性测试
```bash
python test/check_strategy_model_connectivity.py --deepseek-api-key 'sk-...'
```

---

## 5. 已知陷阱（CRITICAL — 避免浪费时间重复踩坑）

### 5.1 绝对不能改的 Knob（会引起 correctness 错误或崩溃）
1. **`b_smem_swizzle` ≠ `swizzle_323`** → 能编译但语义错，从 `(0,0)` 开始整段偏差
   - 根因：`ldmatrix` 的 S2R 地址映射在 `_insert_ldmatrix_if_needed()` 中是硬编码 lane 映射，不随 swizzle 变化
2. **`coo_restore_swizzle` ≠ `swizzle_123`** → 同上
3. **`coo_idx_output_mem=rmem` 或 `coo_val_output_mem=rmem` + `coo_load_mode=cp_async`** → **illegal memory access**
   - 根因：COO rmem staging 与 async-copy lowering 不兼容
   - **安全用法**：必须同时 `coo_load_mode=sync_scalar`
   - 这个约束已在 `StrategyDecision.from_dict()` 中编码
4. **`enable_async_pipeline=False` + 任一路径保留 `cp.async`** → correctness error（row 17574 整段偏差）
   - 根因：关 pipeline 后 producer/consumer 同步前提被破坏
   - 约束已编码：关 pipeline 自动关 B cp.async + 降级 COO 为 sync_scalar
5. **`materialize_val_sidx_to_smem=False`** → correctness error，从 `(0,0)` 整段偏差
   - 不能改，当前 lowering 依赖它

### 5.2 Synthetic Checkpoint Cache 污染
- **问题**：`main.cu` 中 synthetic case 的 checkpoint key 原来不包含 sparsity
- **现象**：sparsity 0.9 和 0.95 复用同一缓存文件，导致 correctness 伪失败
- **修复**：
  - `main.cu` 中 synthetic 路径的 checkpoint key 已加入 `formatSparsityToken(sparsity)`
  - feedback loop 中 synthetic 执行和 ncu profile 都设置 `SPARSENE_DISABLE_MTX_CACHE=1`

### 5.3 `.inc` metadata 与 host fallback 重复定义
- **问题**：曾出现 `.inc` 中用 `static constexpr` 定义 swizzle metadata，`host_program_generated.cu` 用 `#ifndef` 包裹同名变量 fallback，但 `#ifndef` 只检查宏不检查变量
- **修复**：统一使用 `#define` 宏定义（`SPARSENE_GENERATED_TILE_B` / `B_SMEM_SWIZZLE` / `COO_RESTORE_SWIZZLE`）

### 5.4 ncu CSV 混入程序 stdout
- `ncu.csv` 中会出现 `mykernel_time: 765... ms` 这种行，这不是 ncu metric，是程序自己打印的时间
- 需要过滤 `Duration` 和 `mykernel_time` 行

### 5.5 DeepSeek 的 `max_tokens` 不足导致空响应
- DeepSeek 先消耗 reasoning token 再输出 JSON
- 如果 `max_tokens=1200`（原默认），可能 reasoning 就用完了，JSON 内容为空
- 对 deepseek 建议 `max_tokens=4096`

### 5.6 最终 Benchmark 挂死/慢
- **问题**：最终 benchmark 串行跑所有数据集（26 个），每个 `warmup=20, repeat=100`，大矩阵可能 malloc 崩溃导致进程挂死
- **现象**：`ASIC_680k` 等大数据集会触发 `Fatal glibc error: malloc.c: sysmalloc assertion failed`
- **修复**（2026-05-14）：
  - `_run_command_killable()` — 使用 `subprocess.Popen` + `os.killpg(SIGKILL)` 确保超时后真正杀掉进程
  - `_run_final_best_benchmark()` — 添加 `--final-benchmark-parallel N`（默认 4）使用 `ThreadPoolExecutor` 并行执行
  - 添加 `--final-benchmark-timeout N`（默认 120 秒）每个数据集有独立超时
  - 每个数据集有 try/except 容错，崩溃/超时只标记 fail，不阻塞整体流程
  - 添加 `progress.log` 实时输出进度

---

## 6. 脚本入口

### 6.1 单次生成 + 编译 + 运行验证
```bash
# 重新生成 .inc → 编译 → 运行（默认是 1024×1024 synthetic）
PATCHED_ONLY=0 bash examples/src_fp32/dtc/testbed/scripts/validate_generated_kernel.sh

# 指定 .mtx 数据集
PATCHED_ONLY=0 bash examples/src_fp32/dtc/testbed/scripts/validate_generated_kernel.sh \
  -filename /mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx \
  -mtx_flag 1 -N 512 -warmup 1 -repeat 3

# 跳过生成，直接用已有 .inc
PATCHED_ONLY=0 SKIP_GENERATE=1 bash examples/src_fp32/dtc/testbed/scripts/validate_generated_kernel.sh

# 手动指定策略生成
cd python
python test/test_generate.py --strategy-provider manual_json \
  --strategy-manual-json /tmp/my_strategy.json
```

### 6.2 多数据集反馈循环
```bash
# 使用 heuristic provider，3 轮，固定 shipsec1 做反馈
bash examples/src_fp32/dtc/testbed/scripts/run_agent_feedback_multidataset.sh \
  --strategy-provider heuristic \
  --iterations 3 \
  --max-datasets 1 \
  --feedback-dataset /mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx \
  --eval-warmup 1 --eval-repeat 2 \
  --final-warmup 1 --final-repeat 2 \
  --profile-warmup 1 --profile-repeat 1
```

### 6.3 DeepSeek 正式实验（一键顺序跑 flash + pro）
```bash
# 默认各 5 轮，feedback dataset = shipsec1
bash examples/src_fp32/dtc/testbed/scripts/run_deepseek_feedback_shipsec1.sh \
  'sk-eb7f7960366d41a5a4310277168e93da'

# 自定义迭代次数
ITERATIONS=10 bash examples/src_fp32/dtc/testbed/scripts/run_deepseek_feedback_shipsec1.sh \
  'sk-eb7f7960366d41a5a4310277168e93da'
```

### 6.4 其他在线模型实验
```bash
# 示例：mimo 模型
bash examples/src_fp32/dtc/testbed/scripts/run_agent_feedback_multidataset.sh \
  --strategy-provider openai_http \
  --strategy-api-key 'tp-ckk8svdzeha2h3dly4ec5984bq914cpty4shelmntgg8c83p' \
  --strategy-api-base-url 'https://token-plan-cn.xiaomimimo.com/v1' \
  --strategy-model mimo-v2.5-pro \
  --iterations 5 \
  --feedback-dataset /mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx \
  --run-label mimo-v2.5-pro
```

---

## 7. 代码关键位置速查

| 用途 | 文件 |
|------|------|
| 策略 schema/feeder | `python/sparsene/op_gen/strategy_agent.py` |
| NVIR lowering（swizzle/tile 接入点） | `python/sparsene/op_gen/nvir/generate.py` |
| `.inc` metadata 宏生成 | `python/sparsene/op_gen/nvir/codegen.py`（`dump_nvop_program()`） |
| CUDA 源码模板 | `python/sparsene/op_gen/nvir/templates.py` |
| 软件流水线 + codegen 主驱动 | `python/sparsene/op_gen/nvir/compiler_driver.py` |
| S2R/ldmatrix 实现 | `generate.py` 中 `_insert_ldmatrix_if_needed()` |
| COO restore 实现 | `generate.py` 中 `_make_impl()` 的 `coo_atomic_val_restore` 分支 |
| host fallback 和 TILE_B dispatch | `examples/src_fp32/dtc/testbed/host_program_generated.cu` |
| main binary（checkpoint / synthetic / 数据集） | `examples/src_fp32/dtc/testbed/main.cu` |
| feedback loop 主脚本 | `python/test/run_strategy_feedback_multidataset.py` |
| 单因子手动测试 JSON 目录 | `/tmp/sparsene_manual_tests/` |
| 模型连通性测试 | `python/test/check_strategy_model_connectivity.py` |

---

## 8. 反馈循环产物说明

每轮 run 输出目录结构：
```
run_dir/
├── baseline_heuristic/          # heuristic baseline
│   ├── agent/                   # 智能体日志
│   ├── me_tcf_kernel3.inc
│   ├── strategy.json
│   ├── iteration_summary.json
│   ├── dataset_results.csv
│   └── dataset_logs/
├── iter_01/                     # 第 N 轮迭代（同样结构）
│   ...
├── iter_NN/
├── summary.json                 # 全部历史 + best_iteration
├── performance_curve.csv        # 每轮性能曲线数据
├── performance_curve.png        # matplotlib 曲线图
├── best_iteration_kernel.inc    # 最优 kernel 副本
├── best_iteration_all_datasets_eval.csv/md   # 最优迭代在当前评测集上的结果
└── best_iteration_all_datasets_final.csv/md  # 最优迭代在全部真实数据集上的结果
```

### 关键指标
- `mykernel_ms` vs `cusparse_ms`
- `gmean_ratio_vs_cusparse` — 几何平均比值（越低越好）
- `score` — 综合评分（包含惩罚项）

---

## 9. 新增 knob 的安全测试流程

当需要测试某个新 knob 是否安全时，使用 manual_json 单因子测试：

```bash
cd python
# 1. 生成 manual json（改 baseline 的一个字段）
python -c "
import json
base = json.load(open('/tmp/sparsene_manual_tests/baseline.json'))
base['tensor_placement']['new_knob'] = 'new_value'
json.dump(base, open('/tmp/sparsene_manual_tests/test_new.json','w'), indent=2)
"

# 2. 生成 + 编译 + 运行
python test/test_generate.py --strategy-provider manual_json \
  --strategy-manual-json /tmp/sparsene_manual_tests/test_new.json
cd ..
PATCHED_ONLY=0 SKIP_GENERATE=1 \
  bash examples/src_fp32/dtc/testbed/scripts/validate_generated_kernel.sh \
  -N 512 -warmup 1 -repeat 2

# 3. 检查输出中是否有 Error( 行或 illegal memory
grep -c 'Error(' /path/to/output.log
```

---

## 10. 修改策略约束时必改文件清单

如果要增减/修改 knob 的有效值，以下文件**必须同步更新**：

| 文件 | 需要改的位置 |
|------|-------------|
| `strategy_agent.py` | `TensorPlacementStrategy` 默认值、`from_dict()` 归一化、`_default_output_schema()`、`build_strategy_request()` 的 constraints、`build_sparse_format_profile()` 的 decision_targets、`render_strategy_prompt()` 的 optimization_guidance |
| `run_strategy_feedback_multidataset.py` | `_strategy_signature()`、`_make_mutation()`、`mutation_pool`、`_build_exploration_notes()` 的候选列表 |
| `generate.py` | `_strategy_swizzle()`、`_choose_output_mem()`、`generate_nvir()` 中的策略读取点 |

---

## 11. GPU 环境

- GPU 类型：SM80（A100/A800 级别）
- ncu 路径：`~/local/cuda/bin/ncu`
- IDE 终端可能检测不到 GPU（`nvidia-smi` 不可用），需要用 `FORCE_RUN=1` 或者在有 GPU 的 shell 中跑

---

## 12. 下一步开发方向（已排期/建议）

1. **S2R/ldmatrix layout-aware 改造**：让 `b_smem_swizzle` 和 `coo_restore_swizzle` 可安全变动（当前搜索空间 1920，放开这个可再翻倍）
2. **COO rmem + async 支持**：补齐 COO rmem 路径与 async-copy 的兼容性
3. **多数据集正式实验**：deepseek-v4-flash / deepseek-v4-pro 各 50 轮在全部 selected_mtx 数据集上
4. **`materialize_val_sidx_to_smem`** 的 correctness 问题排查（当前不可改）
5. **patched.inc 路径**的独立编译问题

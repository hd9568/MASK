# 策略智能体与反馈循环

> **维护者**: AI Assistant (2026-05)
> **版本**: v1.0
> **适用范围**: 需要理解第二个智能体（策略智能体）和反馈循环的开发者。

本 skill 覆盖策略智能体的架构、prompt 设计、多轮反馈闭环、以及实际运行方式。第一个智能体（格式选择智能体）尚未实现。

---

## 1. 系统架构

```
┌─────────────┐    strategy JSON    ┌──────────────┐
│ 策略智能体   │ ─────────────────→ │ NVIR Lowering │ ──→ kernel.inc
│ (agent)     │                    │ (generate.py) │
└─────────────┘                    └──────────────┘
       ↑                                  │
       │ feedback (perf + ncu +           │
       │   correctness)                   ↓
       │                          ┌──────────────┐
       └──────────────────────────│ Feedback Loop │
                                  │ (run script)  │
                                  └──────────────┘
```

- **策略智能体** (`strategy_agent.py`)：接收硬件信息、格式描述、历史反馈，返回结构化策略 JSON
- **NVIR Lowering** (`generate.py`)：接收策略，控制张量放置、swizzle、cp.async、流水线参数
- **Feedback Loop** (`run_strategy_feedback_multidataset.py`)：每轮迭代：生成 → 编译 → 运行 → ncu profile → 反馈给 agent

---

## 2. 策略智能体 (`strategy_agent.py`)

### 2.1 输入结构（StrategyRequest）

智能体接收的 context 包括：

```
StrategyRequest
├── hardware: HardwareProfile (GPU 型号、SM 数量、显存等)
├── sparse_format: SparseFormatProfile
│   ├── tensors: List[FormatTensorSummary] (所有 gmem tensor 的名称/dtype/shape)
│   ├── decision_targets: List[str] (需要决策的维度清单)
│   └── summary: str (格式描述)
├── constraints: List[str] (安全约束)
├── observations: Dict (known_good_baseline 等)
├── output_schema: Dict (期望的输出 JSON schema)
└── feedback_context: StrategyFeedbackContext (多轮反馈时)
    ├── iteration / max_iterations
    ├── history (前几轮的策略 + 性能 + correctness)
    ├── best_iteration
    ├── latest_profile (ncu summary)
    └── correctness_feedback (结构化错误信息)
```

### 2.2 输出结构（StrategyDecision）

```json
{
  "provider": "string",
  "rationale": "string",
  "tensor_placement": {
    "tile_b": 64,
    "materialize_b_array_ref_to_smem": true,
    "materialize_val_sidx_to_smem": true,
    "enable_b_array_ref_cp_async": true,
    "b_array_ref_mode": "cp_async_1x4",
    "b_smem_swizzle": "swizzle_323",
    "coo_idx_output_mem": "smem",
    "coo_val_output_mem": "smem",
    "coo_load_mode": "cp_async_x4_aligned_fallback",
    "coo_restore_swizzle": "swizzle_123",
    "align_cp_async_smem_outputs": true
  },
  "pipeline": {
    "enable_async_pipeline": true,
    "sync_threads_on_smem_input": true,
    "max_wait_prior": null,
    "stage_shifts": [],
    "rationale": "string"
  }
}
```

### 2.3 归一化与约束

`StrategyDecision.from_dict()` 在解析 AI 返回的 JSON 后执行多道安全检查：

1. `tile_b` 不在 `{16,32,64}` → 钳回 `64`
2. `coo_idx_mem/coo_val_mem ∈ {smem, rmem}`；若 `rmem` → 强制 `coo_load_mode=sync_scalar`
3. `b_smem_swizzle` → 强制 `swizzle_323`
4. `coo_restore_swizzle` → 强制 `swizzle_123`
5. `enable_async_pipeline=False` → 强制降级所有 `cp.async` 为 `sync_scalar`

这样即使 AI 返回了不安全组合，代码层也会拦截。

### 2.4 Providers

| Provider | 用途 |
|----------|------|
| `heuristic` | 返回固定已知良好策略（不需要 API） |
| `manual_json` | 从 JSON 文件读取（单因子测试用） |
| `openai_http` | OpenAI 兼容 API（mimo / deepseek） |
| `anthropic_http` | Anthropic 兼容 API（mimo） |
| `subprocess_json` | 调用外部命令 |

### 2.5 可用模型

| 模型 | API Key | Base URL |
|------|---------|----------|
| `mimo-v2.5-pro` | `tp-ckk8svdzeha2h3dly4ec5984bq914cpty4shelmntgg8c83p` | `https://token-plan-cn.xiaomimimo.com/v1` |
| `deepseek-v4-flash` | `sk-eb7f7960366d41a5a4310277168e93da` | `https://api.deepseek.com/v1` |
| `deepseek-v4-pro` | `sk-eb7f7960366d41a5a4310277168e93da` | `https://api.deepseek.com/v1` |

---

## 3. 反馈循环 (`run_strategy_feedback_multidataset.py`)

### 3.1 运行模式

两种反馈模式：
- **Synthetic mode**（默认）：每轮在随机生成的矩阵上采集性能
- **Dataset mode**（`--feedback-dataset`）：每轮在指定的 .mtx 数据集上采集

### 3.2 每轮迭代流程

```
Iteration N:
  1. Agent 输出策略 JSON
  2. 策略归一化 (from_dict)
  3. manual_json → test_generate.py → kernel.inc
  4. nvcc 编译 binary
  5. 运行 binary on feedback dataset → mykernel_time, correctness
  6. ncu profile → throughput, occupancy metrics
  7. 汇总结果 → feedback_context → 下一轮 agent
```

### 3.3 最终 Benchmark

迭代结束后：
- 选出 best iteration（按 `score` 排序，`score = gmean_ratio + 失败惩罚`）
- 用 best kernel 在**全部 discovered 数据集**上串行评测
- 输出 `best_iteration_all_datasets_final.csv/md`

### 3.4 输出产物

```
run_dir/
├── baseline_heuristic/
├── iter_01/ ... iter_NN/
├── summary.json                 # 全部历史
├── performance_curve.csv/png    # 性能曲线
├── best_iteration_kernel.inc    # 最优 kernel
├── best_iteration_all_datasets_final.csv/md  # 最终全量评测
└── best_iteration_final_eval/   # 最终评测详情
```

### 3.5 Correctness Feedback 结构化

每个 dataset 的运行日志被解析为：

```json
{
  "error_count": 199,
  "first_error": {"row": 0, "col": 0, "cusparse": 312.0, "mykernel": 200.0, "delta": -112.0},
  "bias_direction": "mykernel_lt_cusparse",
  "mean_delta": -135.38,
  "max_abs_delta": 185.0,
  "row_range": [0, 0],
  "col_range": [0, 198]
}
```

该信息通过 `StrategyFeedbackContext.correctness_feedback` 传入 agent prompt，帮助 agent 避免重复错误策略。

---

## 4. 脚本入口

### 4.1 通用入口

```bash
bash examples/src_fp32/dtc/testbed/scripts/run_agent_feedback_multidataset.sh \
  --strategy-provider openai_http \
  --strategy-model mimo-v2.5-pro \
  --iterations 5 \
  --feedback-dataset /path/to/dataset.mtx \
  --run-label my_model_name
```

### 4.2 DeepSeek 一键实验

```bash
bash examples/src_fp32/dtc/testbed/scripts/run_deepseek_feedback_shipsec1.sh \
  'sk-eb7f7960366d41a5a4310277168e93da'
```

默认：flash 5 轮 → pro 5 轮，feedback dataset = `shipsec1.mtx`。

### 4.3 模型连通性测试

```bash
python test/check_strategy_model_connectivity.py \
  --deepseek-api-key 'sk-...'
```

---

## 5. DeepSeek 特殊注意事项

- DeepSeek 先消耗 **reasoning token** 再输出 JSON
- `--strategy-max-tokens` 必须设大（建议 4096），否则 reasoning 就耗尽了，JSON 为空
- 响应中可能出现 `reasoning_content` 字段，解析时需兼容

---

## 6. 已知运行陷阱

1. **Lustre 挂载不可用**：`_discover_datasets()` 现在 tolerate 空根目录，依赖 `--feedback-dataset` 的绝对路径
2. **最终 benchmark 挂死**：大矩阵（如 ASIC_680k）可能 malloc 崩溃。单个数据集有独立超时和 try/except 容错
3. **单 GPU 并行污染**：`--final-benchmark-parallel` 默认 `1`，多 GPU 才调大
4. **matplotlib 缺失**：IDE 环境下可能没有，需在 venv 中装

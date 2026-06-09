# LLM-SOG — LLM-Driven Sparse Operator Generation Based on Sparsene

基于大语言模型（LLM）驱动的层次化稀疏格式设计与高性能 CUDA 算子自动生成框架。

## 架构总览

```
输入: .mtx 稀疏矩阵
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 1: GPU 特征提取 (featurize_mtx)                    │
│  → row/col NNZ 分布, block density 百分位, load balance  │
└────────────────────────┬─────────────────────────────────┘
                         │ features.json
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 2: 格式设计智能体 (deepseek-v4-pro)                │
│  → 基于层次化格式 DSL 自主设计稀疏格式                      │
│  → 输出: 轴类型/层次结构/原子格式/BLK_M/BLK_K             │
└────────────────────────┬─────────────────────────────────┘
                         │ format JSON
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 3: 格式构建 → RTS → Computent → OPIR → NVIR       │
│  → 层次化稀疏格式 → 展平变换 → 计算实体 → 操作符IR → .inc │
└────────────────────────┬─────────────────────────────────┘
                         │ kernel.inc
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 4-6: 策略智能体反馈循环 (N iterations)              │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Iter 0: heuristic baseline                        │  │
│  │ Iter 1..N:                                        │  │
│  │   Agent 输出策略 → Codegen → 编译 → 运行 + 验证     │  │
│  │   → NCU Profile → 性能分析 → 反馈给 Agent           │  │
│  └────────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
              🏆 最优 .inc CUDA Kernel
```

## 目录结构

```
sparsene/
├── python/
│   ├── sparsene/              # 核心库
│   │   ├── format/            # 格式 DSL (Axis, Format, AtomicFormat)
│   │   ├── formats/           # 已实现格式 (ME_TCF, BIT_TCF, SR_BCRS…)
│   │   ├── transform/         # RTS 变换系统
│   │   ├── op_gen/            # 代码生成
│   │   │   ├── computent/     # 计算实体
│   │   │   ├── opir/          # 操作符 IR
│   │   │   ├── nvir/          # NVIR → CUDA .inc
│   │   │   ├── format_design_agent.py  # 格式设计智能体
│   │   │   └── strategy_agent.py       # 策略优化智能体
│   │   └── analysis/          # 特征提取 (Python)
│   └── test/
│       ├── end2end_pipeline.py          # ★ 端到端入口
│       ├── run_strategy_feedback_multidataset.py  # 多数据集策略搜索
│       ├── test_generate.py             # 代码生成测试
│       └── check_strategy_model_connectivity.py   # API 连通性测试
├── examples/                  # CUDA host program + kernel
│   └── src_fp32/dtc/testbed/
│       ├── featurize_mtx.cu   # GPU 特征提取器
│       ├── host_program_generated.cu
│       ├── main.cu
│       └── scripts/           # 编译 & 验证脚本
├── mySkills/                  # 技能定义文件
├── paper/                     # Sparsene 论文 (SC26)
├── docs/                      # 设计文档
├── logs/                      # 运行时日志 (gitignore)
└── README.md
```

## 快速开始

### 模型配置

所有可用的 LLM 模型定义在 `llm_models.yaml` 中。支持增删模型：

```yaml
# llm_models.yaml
models:
  deepseek-v4-pro:
    provider: openai_compatible
    model: deepseek-v4-pro
    api_base_url: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY        # 优先从环境变量读取
    api_key: sk-...                       # fallback 硬编码 key
    defaults:
      temperature: 0.1
      max_tokens: 8192
      timeout_sec: 300
```

测试所有模型连通性：
```bash
PYTHONPATH=python python3 python/test/test_run_llm.py
```

### 环境依赖

- CUDA 12.x / sm_80 (A100/A800)
- NVCC + gcc 13 toolset
- Python 3.12 + venv (`/mnt/lustre-client/huoda/venv/llm-sog`)
- DeepSeek API key
- cuSPARSE / cuBLAS

### 端到端运行

```bash
cd /home/huoda/sparsene
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate

# 最小示例：1 个矩阵，1 轮策略迭代
PYTHONPATH=python python3 python/test/end2end_pipeline.py \
  --mtx /mnt/lustre-client/zhangkaige/selected_mtx/consph.mtx \
  --N 512 \
  --format-agent-model deepseek-v4-pro \
  --strategy-agent-model deepseek-v4-pro \
  --strategy-iterations 1 \
  --warmup 1 --repeat 2
```

### 完整运行（推荐）

```bash
PYTHONPATH=python python3 python/test/end2end_pipeline.py \
  --mtx /mnt/lustre-client/zhangkaige/selected_mtx/consph.mtx \
  --N 512 \
  --format-agent-model deepseek-v4-pro \
  --strategy-agent-model deepseek-v4-pro \
  --strategy-iterations 5 \
  --warmup 1 --repeat 2
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mtx` | (必需) | .mtx 文件路径或数据集目录下的文件名 |
| `--N` | 512 | 随机 B 矩阵列数 |
| `--format-agent-model` | deepseek-v4-pro | 格式设计智能体模型 (llm_models.yaml key) |
| `--strategy-agent-model` | deepseek-v4-pro | 策略智能体模型 (llm_models.yaml key) |
| `--model-config` | llm_models.yaml | 模型配置文件路径 |
| `--strategy-iterations` | 5 | 策略智能体反馈迭代次数 |
| `--warmup` / `--repeat` | 1 / 2 | Benchmark warmup/repeat 次数 |
| `--strategy-temperature` | 0.35 | Agent 温度参数 |
| `--strategy-max-tokens` | 4096 | Agent max_tokens |
| `--skip-feature-extract` | - | 跳过特征提取（用已有结果） |
| `--skip-format-design` | - | 跳过格式设计（用已有结果） |
| `--run-label` | - | 自定义运行标签 |

### 测试 DeepSeek API 连通性

```bash
PYTHONPATH=python python3 python/test/check_strategy_model_connectivity.py \
  --deepseek-api-key 'sk-...'
```

## 日志结构

每次运行会在 `logs/<timestamp>_<matrix_name>/` 下生成完整链路日志：

```
logs/20260524_120000_consph/
├── args.json                     # 运行参数
├── matrix_features.json          # Phase 1: GPU 特征提取结果
├── format_decision.json          # Phase 2: 格式设计智能体决策
├── format_agent/                 # 格式设计 agent 对话
│   └── 20260524_*/               #   prompt.txt + response.json + request.json
├── iter_00_heuristic/            # 启发式基线
│   ├── strategy.json             #   策略
│   ├── kernel_heuristic.inc      #   生成的 CUDA kernel
│   ├── build/                    #   编译产物
│   ├── run.log                   #   运行日志
│   └── ncu.csv                   #   NCU profile
├── iter_01/                      # 策略迭代 1
│   ├── strategy.json
│   ├── kernel_iter01.inc
│   ├── build/
│   ├── run.log
│   └── ncu.csv
├── ... (iter_02 .. iter_N)
├── strategy_agent/               # 策略 agent 对话
│   └── .../                      #   prompt.txt + response.json + request.json
├── strategy_history.json         # 所有迭代历史
└── summary.json                  # 最终汇总
```

## 核心概念

### 层次化稀疏格式 DSL

格式由 `Axis`（轴）、`Format`（层次）、`AtomicFormat`（原子格式）递归组合：

```
轴类型: df(稠密固定) / dv(稠密变长) / sf(稀疏固定) / sv(稀疏变长)
原子格式: coo(坐标) / mco(mask-COO) / dense(稠密)
```

示例（ME_TCF 格式）：
```
Format("ME-TCF")
├── df(X_o, ROW, M) + sv(Y_s, COL, K) + df(X_i, ROW, BLK_M)
└── Format("tc_block")
    ├── df(Y_so, COL, K/BLK_K) + df(Y_si, COL, BLK_K) + df(X_i, ROW, BLK_M)
    └── coo_atomic_format(X_i, Y_si)
```

### 格式设计智能体

- 模型：deepseek-v4-pro
- 输入：GPU 特征提取的矩阵特征 + 硬件信息
- 输出：完整格式定义（轴类型、层次、BLK_M/BLK_K、原子格式）
- 智能体已学习完整的 DSL 语法，可自主设计新格式

### 策略智能体

- 模型：deepseek-v4-pro（可配置）
- 控制决策：张量放置（tile_b, cp.async, COO 位置…）+ 流水线参数
- 反馈循环：N 轮迭代，每轮根据性能 + NCU profile 调整策略

## 已实现格式

| 格式 | 原子格式 | 文件 |
|------|---------|------|
| ME_TCF | COO | `python/sparsene/formats/DTC_SpMM.py` |
| BIT_TCF | MCO | `python/sparsene/formats/Acc_SpMM.py` |
| SR_BCRS | Dense | `python/sparsene/formats/SR_BCRS.py` |
| HME_TCF | 混合 | `python/sparsene/formats/HME_TCF.py` |
| ROW_REORDER_SR_BCRS | 重排 | `python/sparsene/formats/ROW_REORDER_SR_BCRS.py` |

## 设计文档

| 文档 | 内容 |
|------|------|
| [docs/format_design_agent_design.md](docs/format_design_agent_design.md) | 格式设计智能体完整设计 |
| [docs/strategy_agent_design.md](docs/strategy_agent_design.md) | 策略智能体设计 |
| [docs/project_code_explanation.md](docs/project_code_explanation.md) | 项目代码结构说明 |
| [docs/llm_agent_design.md](docs/llm_agent_design.md) | LLM Agent 通用设计 |
| [docs/ref_vs_generate.md](docs/ref_vs_generate.md) | Reference vs Generate 对比 |
| [docs/inc_perf_benchmark_vs_cusparse.md](docs/inc_perf_benchmark_vs_cusparse.md) | vs cuSPARSE 性能对比 |

## 论文

[paper/SC26-Sparsene-main/](paper/SC26-Sparsene-main/) — Sparsene: A Format-Centric Sparse Operator Generation Framework

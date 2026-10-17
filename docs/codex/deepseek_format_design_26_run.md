# DeepSeek V4 Pro — 26 数据集格式设计运行说明

> 这是固定 ME-TCF executable profile 的历史 v4 运行说明。v6 通用 DSL
> 设计不要 resume 到该目录，否则会混合两种 schema 和实验含义；应新建
> run。当前契约见 `docs/codex/format_design_agent_v6.md`。

## 目标与统一目录

格式设计、后续策略搜索、算子生成、编译、正确性、性能和 NCU 应属于同
一次可复现执行。固定根目录为：

```text
/home/huoda/sparsene/logs/e2e_runs/<run_id>/
```

当前已经建立、可继续使用的 run 是：

```text
/home/huoda/sparsene/logs/e2e_runs/20260717_233737_deepseek_v4_pro_format26
```

目录契约：

```text
<run>/
├── connectivity/                         # 模型连通性请求与 usage
├── run_manifest.json                     # 环境、revision、模型、数据集和阶段布局
├── events.jsonl                          # 跨阶段事件流
├── formats/<dataset>.format.json         # 通过质量门的最终格式
├── format_design_summary.{json,csv}      # 26 数据集汇总
├── token_usage_total.json                # 包括历史 attempt 的 token 总计
└── datasets/<dataset>/
    ├── matrix_features.json              # 本次输入特征快照
    ├── input_provenance.json             # 原始路径、schema、SHA256
    ├── format_agent/<attempt>/            # 一次真实提问的完整审计
    │   ├── messages.json                 # system + user 完整消息
    │   ├── system_prompt.txt
    │   ├── user_prompt.txt               # 包括注入的 runtime skills
    │   ├── request.json
    │   ├── response_text.txt             # 模型完整文本
    │   ├── raw_response.json             # API 原始响应和 usage
    │   ├── parsed.json
    │   ├── response.json
    │   ├── usage.json                    # 输入/输出/reasoning/总 token
    │   └── audit.json                    # 内容摘要、SHA256、token 来源、状态
    ├── raw_model_decision.json
    ├── format_decision.json              # 原始、规范化和 applied candidate
    ├── format_design_basis_zh.md          # 人可读中文依据
    ├── token_usage.json
    ├── strategy/                          # 后续阶段预留
    ├── codegen/                           # 后续阶段预留
    └── iterations/                        # build/correctness/perf/NCU 预留
```

## 必须使用的环境

所有项目 Python 和 Agent 命令都先在同一个 shell 中执行：

```bash
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
```

系统 Python 缺少 `yaml`，不能替代该环境。

## 连通性结果

2026-07-17 在指定 venv 中执行了带审计的单模型测试：

```text
deepseek-v4-pro: ok
prompt_tokens: 5
completion_tokens: 32
```

审计目录位于当前 run 的 `connectivity/20260717_233827_model_connectivity/`。
全配置测试还发现 `deepseek-v4-flash` 可用，而 `mimo-v2.5-pro` 因其自身配置
凭证无效返回 401；这不影响本次 DeepSeek V4 Pro 格式设计。

## 推荐启动方式

继续当前 run，复用已经完成且合格的 `ASIC_680k`，不要重复付费：

```bash
cd /home/huoda/sparsene
bash scripts/run_format_design_26_background.sh \
  /home/huoda/sparsene/logs/e2e_runs/20260717_233737_deepseek_v4_pro_format26
```

脚本会自行 source 指定 venv；检测到 `run_manifest.json` 后自动传入
`--resume`。它不会设置 `--max-tokens`，因此沿用 `llm_models.yaml` 中模型
配置的输出上限。默认 `MAX_ATTEMPTS=1`，避免因非关键质量告警自动增加付费
调用。

若希望完全新建一次 run：

```bash
cd /home/huoda/sparsene
bash scripts/run_format_design_26_background.sh
```

若已确认模型可用且不希望新 run 再发送一次最小连通性请求：

```bash
SKIP_CONNECTIVITY=1 bash scripts/run_format_design_26_background.sh
```

脚本输出 PID、run 路径和 batch log。监控命令：

```bash
tail -f <run>/format_design_batch.log
find <run>/formats -maxdepth 1 -name '*.format.json' | wc -l
cat <run>/format_design_batch.pid
```

不要用并发 26 请求来缩短时间：顺序执行更容易控制费用、限流、审计顺序，
也便于后续把每个数据集的策略与生成产物接到同一目录。

## Prompt 与输出设计

提示词不把“格式设计”描述成不受约束的 DSL 文本生成，而是要求模型区分：

1. 逻辑合法性：当前 executable profile 固定为两层 ME-TCF + COO leaf；
2. 层次语义：BLK_M 行窗口、Y_s 窗口内列凝聚、BLK_K TC block、局部 COO
   恢复和 MMA；
3. 物理候选：在已经打通的 `16x8` 与 `16x16` 间选择；
4. 特征含义：padded COO work、TC block/metadata、Tensor Core 空算、B index
   padding、列复用和窗口负载倾斜；
5. 后端先验：历史 26 数据集上 8 比 16 更常获胜，块数减半不是选择 16 的
   充分条件；
6. 不确定性：静态特征不能替代后续正确性、真实计时与 NCU 闭环。

模型输出除 DSL 外还必须包含：

- 至少 5 条可由程序逐值核验的特征证据，且覆盖两个候选；
- 中文格式语义解释；
- `16x8` 和 `16x16` 的独立收益/代价分析；
- 最终选择理由、总体 rationale、权衡与风险；
- confidence。

所有模型 JSON 都经过确定性规范化；原始回答不会被覆盖。格式名、完整轴树、
COO leaf、符号参数和候选范围在进入 RTS 前再次验证。

## 当前部分执行状态

两次外层工具中断没有留下后台进程。当前 run 已有：

- `ASIC_680k`：正式 DeepSeek 输出通过质量门，最终候选 `16x8`，输入 7012
  tokens，输出 3943 tokens，已经写入 `formats/ASIC_680k.format.json`；
- `DD`：该次回答未得到可用 JSON，保存了完整审计并进入 fallback，不作为最终
  格式；resume 时只会重做 DD，不会重做 ASIC。

这两个调用都属于正式数据集设计，不是额外冒烟测试。

## 完成后的检查

正常完成应满足：

```text
format_design_summary.json: ok=26, failed=0
formats/*.format.json:      26 files
datasets/*/format_decision.json: 26 final decisions
```

`token_usage_total.json` 从全部 `audit.json` 重新聚合，因此解析失败、fallback
和历史 attempt 的真实 API token 也会计入，不会只统计最后一次成功结果。

## 费用与重试策略

- 不为 prompt 单独发送冒烟矩阵；API 只用于正式格式设计与未来算子生成。
- 连通性测试是最小 32-token 请求，可用 `SKIP_CONNECTIVITY=1` 避免重复。
- 默认每个数据集一次正式调用；网络或解析失败由用户根据日志决定是否 resume。
- 不在后台脚本中覆盖模型的 `max_tokens`，避免无意截断完整 DSL 与依据。
- 所有失败先看已经落盘的完整响应，再决定是否付费重试。

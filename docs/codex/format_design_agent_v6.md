# Format Design Agent v6：合法、可发散、源码可追溯的层次化 DSL

## 目标

v6 修复 2026-07-18 v5 批次暴露的输出契约问题，同时保留“研究设计不受当前
后端能力限制”的原则：

```text
v3 matrix features + hardware + source-derived format vocabulary
  -> DeepSeek JSON Output
  -> non-structural scalar canonicalization
  -> deterministic DSL validation
  -> independent backend capability classification
  -> audited research artifact
```

格式智能体不是在已知名称中分类。它可以复用、删减、增加或混合层次、轴、
重排和 atomic leaf；但输出必须先是一个语法和语义合法的 Format DSL。

## 固定性能目标与计时边界

2026-07-18 起，格式智能体、Codex、后续策略智能体和性能报告统一使用独立版本
的评价契约：

```text
sparsene.kernel_only_post_conversion.v1
```

主指标是格式转换、metadata/padding 构造和行列重排全部完成后，CUDA event 内
warmed、重复执行的 GPU SpMM 计算 kernel 平均时间。当前 DTC 代码与该口径一致：

- `main.cu::buildMETCFMetadata()`、checkpoint 读写、旧 metadata 到生成 ABI 的
  转换、`cudaMalloc/cudaMemcpy` 都发生在 `mykernel_time` event 之前；
- `mykernel_time` event 只包围 `generated_kernel<<<...>>>()` 的重复 launch；
- cuSPARSE 的内存准备和 `cusparseSpMM_preprocess()` 发生在 event 之前，
  `cusparse_time` event 只包围重复 `cusparseSpMM()`。

因此主指标排除格式转换、重排及正确性用逆置换、checkpoint、内存申请与传输、
代码生成/编译、结果验证和 cuSPARSE preprocess。格式智能体不得因为这些离线
步骤耗时更长而降低格式的 kernel 性能排名。

但“计时外”不等于“不需要设计”：置换/逆映射语义、metadata、额外存储、缓存
身份和 host/backend 实现必须完整记录并保证正确。另一方面，凡是实际位于计时
kernel 内的 metadata 读取、B gather、atomic restore/scatter、padding 零工作、
同步、MMA/累加和 C 写回均属于主指标。批次 manifest、每个格式 artifact 和中文
依据文件都会持久化该固定契约，防止后续阶段改变口径。

## v5 失败证据与根因

失败运行：

```text
/home/huoda/sparsene/logs/e2e_runs/
  20260718_152039_deepseek_v4_pro_format26_v5_flexible
```

该 run 只完成9个数据集，随后在 `consph` 开始后退出，没有
`format_design_summary.json`。9个已完成请求中：

- 6个达到 `completion_tokens=8192` 且 `finish_reason=length`；thinking token
  与最终 JSON 共用预算，响应在 JSON 中间被截断；
- 3个 JSON 完整，但提示词示例把 `level` 写成带引号的占位字符串，模型返回
  `"1"`/`"2"`，校验器严格期待整数 `1`/`2`；
- 完整响应还出现 `ceil(M/BLK_M)`、`varies per row_window`、未声明的
  `tc_blocks_per_window` 等长度。原因是提示词解释了 varlen 概念，却没有讲清
  `Axis.length` 是逻辑上界/split factor，而实际变长 cardinality 来自元数据。

这说明主要失败点是 prompt/schema/validator 契约，而不是模型无法产生多样设计。
完整响应已经出现 MCO、固定活跃列块、row sorting 等不同方向。

## v6 JSON 与 DSL 合法性契约

### JSON 输出

- OpenAI-compatible DeepSeek 请求设置
  `response_format={"type":"json_object"}`。
- system/user prompt 同时明确要求 JSON，并提供类型正确的 JSON 示例。
- `LLMTextResponse`、Agent audit 和格式决策记录 `finish_reason`。
- `finish_reason=length` 一律标记为截断失败，即使局部文本偶然可解析，也不能
  进入普通 DSL 校验。
- reasoning token 兼容读取
  `completion_tokens_details.reasoning_tokens` 和
  `output_tokens_details.reasoning_tokens`，批次汇总不再把已返回的 thinking
  使用量错误记为0。

DeepSeek V4 Pro 当前官方文档说明 JSON Output 支持 `json_object`，thinking 与
最终内容共享 output budget；当前模型页给出的最大输出远高于原配置的8192。
本项目选择32768作为格式设计默认上限：它是防截断余量，不会要求模型实际
生成32768 tokens，计费仍以真实生成量为准。

### 标量与表达式

- `level` 必须是无引号、1-based、连续的 JSON 整数。
- `params` 必须是具体正 JSON 整数。
- `length` 只允许整数、`M/K/N`、已声明参数、括号和 `+ - * /`。
- 禁止 `ceil/floor/%//`、点路径、自然语言和未声明符号。
- `sv/dv` 的实际 per-parent cardinality 由变长元数据表达；`length` 仍写合法
  逻辑上界。例如 `sv(Y_s,COL,K)`，不能写 `varies per window`。
- 源码通过 `X_o(M)` 与 `X_i(BLK_M)` 表达行切分，不写
  `ceil(M/BLK_M)`；`K/BLK_K` 是合法块序长度。
- 为兼容模型偶发的 JSON 标量序列化，normalizer 仅把纯数字字符串 level 和
  params 转为整数并记录 action；它不修补层次、轴、atomic 或非法表达式。

### 输出长度控制

模型仍输出完整设计依据，但避免重复：

- 6至8条精确 feature evidence；
- 恰好3个完整格式备选；
- transformation、axis purpose、中文语义和实现要求保持简洁；
- 深入推理留在 thinking，最终 JSON 只保存可审计结论。

## 源码派生的格式知识

权威机器可读 catalog：

```text
python/sparsene/formats/design_catalog.py
```

人类可读说明：

```text
python/sparsene/formats/README.md
```

catalog 会随每次格式设计请求注入，覆盖本目录所有当前声明：

| Catalog id | 源码 | 合法结构/叶子 | 智能体必须理解的关键点 |
|---|---|---|---|
| `ME_TCF` | `DTC_SpMM.py` | 2层列凝聚 TCF + COO | 行窗口内列 union/排序不是全局重排 |
| `BIT_TCF_ACC` | `Acc_SpMM.py` | 同层次 + MCO | mask+compact values；常量名与 `Format.name` 有历史命名差异 |
| `CSR_TCF` | `CSR_TCF.py` | 同层次 + CSR | row_ptr/col_idx/value 是专用 payload，不是 COO 改名 |
| `ELL_TCF` | `ELL_TCF.py` | 同层次 + ELL | 局部行长度规则性对比 padding/sentinel 成本 |
| `DIA_TCF` | `DIA_TCF.py` | 同层次 + DIA | 局部 tile 对角结构才决定收益 |
| `SR_BCRS` | `SR_BCRS.py` | 同层次 + dense | active block 内零填充/零计算与无 restore 元数据的权衡 |
| `BIT_BSR_SPADEN` | `Spaden.py` | 1层活跃固定块 + MCO | 保存稀疏活跃块选择，不遍历全 dense grid |
| `HME_TCF` | `HME_TCF.py` | 5层多尺度 + COO | 外层 tile 内再次稀疏化/子分块；源码记录 RTS 不稳定 |
| `ROW_REORDER_SR_BCRS` | `ROW_REORDER_SR_BCRS.py` | 稀疏行域+行块+active block+dense | 源码 `Axis.reorder` 实际为 None；稀疏行轴不等于稳定行置换 |
| `DENSE_AND_DENSE_T` | `dense.py` | 稠密轴序，无 atomic child | 是 operand layout 构件，不是可原样返回的完整稀疏格式 |

catalog 将“分组、列表示、层次深度、显式 reorder、atomic 编码”拆成可组合
设计维度。已知 capability 只是上下文，不是搜索空间边界。

## 特征证据

提示词仍提供 v3 digest 的所有候选信息，并生成精确 dotted-path/value map。
`feature_evidence.feature` 必须原样复制一个 map key，`value` 必须匹配；不允许
合并路径或使用类似 `row_nnz.mean / col_nnz.mean` 的自造表达式。候选统计对
ME-style 行窗口列凝聚是精确量，对不同 hierarchy/reorder 只能作为 proxy。

## 运行与 schema 隔离

当前 schema/profile：

```text
sparsene.format_design.v6
hierarchical_sparse_format_research_v2
```

批处理拒绝把 v6 resume 到 v5/v4 run。失败的 v5 目录应只保留为诊断证据。
重新运行必须使用新目录，例如：

```bash
cd /home/huoda/sparsene
bash scripts/run_format_design_26_background.sh \
  /home/huoda/sparsene/logs/e2e_runs/<new_run_id>_deepseek_v4_pro_format26_v6
```

wrapper 自动加载：

```bash
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
```

默认 `MAX_ATTEMPTS=1`、`MAX_TOKENS=32768`，不发送额外付费连通性请求。
如需显式覆盖，可在启动命令前设置 `MAX_TOKENS`，但不应再降回8192。

## 修改范围

- `python/sparsene/formats/design_catalog.py`：源码派生 hierarchy/format catalog。
- `python/sparsene/formats/README.md`：所有现有声明及发散维度说明。
- `python/sparsene/op_gen/format_design_agent.py`：v6 prompt/schema、长度契约、
  catalog/feature map 注入、标量 canonicalization、JSON/truncation处理。
- `python/sparsene/performance_contract.py`：格式/策略/性能结果共享的固定
  kernel-only 评价契约。
- `python/sparsene/op_gen/strategy_agent.py`、`end2end_pipeline.py`：策略请求和
  闭环性能结果携带同一契约，避免后续阶段改成端到端转换耗时。
- `.agent/skills/format-design/SKILL.md`：同步运行时格式知识和合法性清单。
- `python/sparsene/agent/client.py`、`runtime.py`：JSON Output、finish reason 和
  reasoning token 审计。
- `python/test/run_format_design_batch.py`、background wrapper：v6隔离、有效
  max token、finish reason、固定 kernel-only 契约和 manifest 选项。

## 本地验证

没有调用付费模型，也没有运行 GPU。实际执行：

```bash
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
python -m py_compile \
  python/sparsene/formats/design_catalog.py \
  python/sparsene/op_gen/format_design_agent.py \
  python/sparsene/agent/client.py \
  python/sparsene/agent/runtime.py \
  python/test/run_format_design_batch.py \
  python/test/test_format_design_agent.py \
  python/test/test_agent_runtime_audit.py

PYTHONPATH=python python -m unittest \
  python/test/test_format_design_agent.py \
  python/test/test_agent_runtime_audit.py
```

结果：18项测试全部通过。额外结构检查验证9个带 atomic 的源码 catalog
设计全部通过 v6 DSL validator：ME-TCF、BIT-TCF、CSR/ELL/DIA-TCF、
SR-BCRS、BIT-BSR、HME-TCF、ROW-REORDER-SR-BCRS。`bash -n` 也验证了
background wrapper。

另用本地构造的 strategy request 验证：请求 payload 与渲染 prompt 均包含
`sparsene.kernel_only_post_conversion.v1`，且明确禁止把转换/重排耗时作为
策略排名惩罚；该检查不调用模型。

使用一个真实 v3 feature 文件渲染后，完整 prompt（含 runtime skill）约
68,358字符，包含全部 catalog id 和655个可引用标量。它仍远低于模型上下文，
但后续如果希望进一步降低输入费用，可在不删除设计信号的前提下研究候选特征
选择/压缩，而不能重新退化为固定 ME-TCF 提示词。

全仓 `git diff --check` 仍报告用户已有的 `python/me_tcf_kernel3.inc` 尾随空格；
本任务没有修改或清理该生成文件。

## 尚未验证

- 没有为了测试 prompt 额外消费 DeepSeek API；JSON Output 的真实首个 v6
  请求应通过正式数据集设计验证。
- v6 只保证研究 DSL 前端合法与 capability 分离，不会让 HME、row reorder、
  新 hierarchy 或所有 atomic 自动获得 host/backend/数值正确性。
- 下一最高价值检查是仅运行新的26数据集正式 batch，并首先观察第1个数据集的
  `finish_reason`、`dsl_validation` 和 quality gate，再决定是否继续整批。

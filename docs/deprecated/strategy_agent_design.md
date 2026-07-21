# ME_TCF 第二智能体设计

## 目标

当前先固定稀疏格式为 `ME_TCF`，只引入“第二个智能体”来参与代码生成阶段的关键性能策略选择，而不直接改变稀疏格式本身。这样可以在保持已有 correctness/performance 基线的前提下，把后续大模型/agent 最需要控制的决策点从硬编码逻辑中剥离出来。

最初阶段只做一次性闭环：

1. 读取硬件信息、稀疏格式信息、外部张量摘要和已知约束；
2. 智能体输出一份结构化策略；
3. codegen 读取该策略并生成 `.inc`；
4. 记录 request / prompt / response，便于后续分析；
5. 再走 correctness + 性能验证。

后续如果加入“第一个智能体”，它可以先根据输入矩阵设计新的层次化混合稀疏格式，然后把格式描述和候选 lowering 约束继续传给当前这个“策略智能体”。

当前实现已经进一步扩展为“带反馈的多轮闭环”：

1. 先用 `heuristic` 生成并测量一个 `iteration 0` 基线；
2. 将硬件信息、格式信息、历史策略、最近 correctness/perf、`ncu` profile 摘要，以及当前 `.inc` 代码回灌给在线智能体；
3. 智能体输出下一轮结构化策略；
4. 使用 `manual_json` 路径重新生成 `.inc`，避免绕开已验证的 codegen 主路径；
5. 重新编译 generated binary，跑 correctness、轻量性能和 `ncu` profile；
6. 记录 `performance_curve.csv` / `summary.json` / 每轮日志，继续下一轮。

## 当前接入位置

- `python/test/test_generate.py`
  - 新增智能体 provider / manual json / subprocess command / log 目录参数
- `python/test/run_strategy_feedback_loop.py`
  - 默认 5 轮 feedback loop 驱动脚本
  - 自动跑 `iteration 0` heuristic 基线
  - 每轮组织 prompt、调用 agent、生成 `.inc`、编译运行、采集 `ncu`、记录性能曲线
- `python/sparsene/op_gen/strategy_agent.py`
  - 定义策略 schema、prompt、provider 和日志落盘
  - 支持多轮 feedback context
  - prompt 可带入历史性能、`ncu` 摘要和当前 `.inc` 代码
- `python/sparsene/op_gen/nvir/generate.py`
  - 在 `generate_nvir()` 内部调用智能体
  - 将策略接入张量放置与 `cp.async` 相关决策
- `python/sparsene/op_gen/nvir/compiler_driver.py`
  - 将策略接入 pipeline shift 配置
- `python/sparsene/op_gen/nvir/codegen.py`
  - 将策略接入 shared-memory 对齐和 `wait/commit/sync` 生成行为

## 智能体要看到的输入

### 1. 硬件信息

当前通过 `nvidia-smi` 和环境变量收集：

- `gpu_name`
- `driver_version`
- `cuda_arch`
- `total_memory_mb`
- `warp_size`

如果后续扩展，还可以加入：

- `sm_count`
- `shared_memory_per_sm`
- `registers_per_sm`
- `ldmatrix/mma` 支持矩阵
- `cp.async` 支持级别

### 2. 稀疏格式说明

当前对 `ME_TCF` 提供一份简要格式说明，重点不是论文式介绍，而是告诉智能体：

- 这是分层混合稀疏的 SpMM lowering
- 路径里有 `val_len / val_sidx / val_coo_idx / val_coo_val / B_val / C_val`
- 关键性能瓶颈集中在：
  - `COO idx/value load`
  - `B-side array_ref`
  - `coo_atomic_val_restore`
  - `ldmatrix + mma` 之前的 shared-memory 组织

### 3. 外部张量摘要

智能体可以看到每个 gmem tensor 的：

- `name`
- `dtype`
- `shape`
- `role`

这比只看 op 名更容易做稳定决策。

### 4. 当前任务与约束

当前 prompt 明确告诉智能体：

- 只能在结构化 schema 内输出
- 优先 correctness
- 可以是一次性决策，也可以是带历史反馈的 autotuning 循环
- 需要考虑 `cp.async` 对齐和 fallback
- 需要留下输入输出日志

在 feedback loop 模式下，还会额外给智能体：

- `iteration / max_iterations`
- 最近若干轮的策略与性能历史
- 当前最佳轮次及其关键指标
- 最近一轮 correctness 结果
- 最近一轮 `mykernel_ms / cusparse_ms / ratio_vs_cusparse`
- `ncu` 摘要
  - `waves_per_sm`
  - `registers_per_thread`
  - `memory_throughput_pct`
  - `compute_throughput_pct`
  - `l1tex_hit_rate_pct`
  - `l2_hit_rate_pct`
  - `mem_pipes_busy_pct`
  - `duration_ns`
- 当前生成的 `.inc` 内核代码

## 智能体要输出什么

当前输出 schema 固定为两部分：

### 1. `tensor_placement`

- `materialize_b_array_ref_to_smem`
- `materialize_val_sidx_to_smem`
- `enable_b_array_ref_cp_async`
- `b_array_ref_mode`
- `coo_idx_output_mem`
- `coo_val_output_mem`
- `coo_load_mode`
- `align_cp_async_smem_outputs`

这部分直接影响：

- `array_ref` 是否物化到 `smem`
- `coo_atomic_format_load_idx/val` 是否走 `cp.async`
- `smem` buffer 是否需要 16B 对齐

### 2. `pipeline`

- `enable_async_pipeline`
- `sync_threads_on_smem_input`
- `max_wait_prior`
- `stage_shifts`
- `rationale`

这部分直接影响：

- 是否发 `__pipeline_wait_prior`
- 是否发 `__pipeline_commit`
- 是否对 `smem` consumer 强制 `__syncthreads`
- 相邻 stage 的 shift 设置

## 当前 provider 设计

为了先跑通，当前实现了 3 类 provider：

### 1. `heuristic`

默认 provider，不依赖外部 API。

它返回的是当前已经验证过的安全基线策略：

- 保持 `B-side` gather 落到 `smem`
- 保持 `COO idx/value` 走 `cp.async + aligned fallback`
- 保持 `cp.async` shared 输出 16B 对齐
- 保持保守的 pipeline

### 2. `manual_json`

从本地 JSON 文件读取一份结构化策略。适合：

- 手工做消融实验
- 复现实验配置
- 暂时不用真实大模型，也能稳定控制策略

### 3. `subprocess_json`

通过外部命令接第三方 agent / LLM 服务。

约定：

- `stdin` 输入 `{"request": ..., "prompt": ...}`
- `stdout` 输出结构化 JSON response

这个接口比把 OpenAI/HTTP SDK 直接写进项目更通用，也方便之后切换任意 API。

## Prompt 设计原则

当前 prompt 不是让模型自由发挥写 CUDA，而是要求它只做“策略选择器”。

核心原则：

1. 只给性能相关上下文，不让它碰 IR 细节语法
2. 只允许输出严格 schema
3. 明确 correctness-first
4. 明确 `cp.async` 需要考虑对齐和 fallback
5. 如果处于 feedback loop，就明确要求它利用历史性能与 profile 做小步调整
6. 必要时把当前 `.inc` 代码也提供给它，让它从真实 CUDA/CuTe 生成物理解当前实现
7. 如果服务端因为 prompt 过重而超时，自动降级为不附带完整 `.inc` 的重试请求

这样做的目的是把大模型从“编译器后端生成器”降级成“结构化策略提供者”，避免输出不可控。

## 每次执行如何留痕

当前每次运行 `python/test/test_generate.py`，都会在 `docs/agent_runs/` 下生成一份目录，包含：

- `request.json`
- `prompt.txt`
- `response.json`

这样后续分析时可以精确回答：

- 当时给了智能体什么上下文
- prompt 长什么样
- 智能体返回了什么策略
- 最终生成代码和性能表现如何

这个目录已经加入 `.gitignore`，避免频繁污染版本库。

如果运行多轮闭环脚本 `python/test/run_strategy_feedback_loop.py`，则会在 `docs/agent_feedback_runs/` 下生成一份 run 目录，包含：

- `run_config.json`
- `summary.json`
- `performance_curve.csv`
- `baseline_heuristic/`
- `iter_01/ ... iter_05/`

每轮目录下又包含：

- `strategy.json`
- `strategy_for_codegen.json`
- `me_tcf_kernel3.inc`
- `validate.log`
- `perf.log`
- `ncu.csv`
- `iteration_summary.json`
- `agent/request.json`
- `agent/prompt.txt`
- `agent/response.json`

## 当前已经接入的代码决策点

### 1. 张量放置

在 `nvir/generate.py` 中，`array_ref` 现在不再完全由硬编码规则控制，而是读取策略：

- `B_val` gather 是否 materialize 到 `smem`
- `val_sidx` 是否 materialize 到 `smem`

### 2. `cp.async`

当前已接入：

- `B-side array_ref` 是否启用 `cp.async`
- `coo_atomic_format_load_idx/val` 走 `cp_async_x4_aligned_fallback` 还是 `sync_scalar`
- `cp.async` 输出是否要求 `align_16`

### 3. 流水线

当前已接入：

- `enable_async_pipeline`
- `sync_threads_on_smem_input`
- `max_wait_prior`
- `stage_shifts`

这意味着后续智能体已经可以直接参与 `wait / commit / sync / shift` 的设计。

## 为什么这样设计

当前项目最适合智能体接管的，不是“怎么把 IR 翻译成 CUDA 语法”，而是 lowering 过程中的性能策略选择。因为这些选择：

- 对性能高度敏感
- 和输入矩阵特征、硬件特征密切相关
- 适合结构化输出
- 适合之后做性能反馈闭环

所以当前设计把智能体定位成：

- 输入：硬件 + 格式 + 张量摘要 + 约束
- 输出：结构化策略
- 消费者：确定性的 codegen

而不是让模型直接写 kernel。

## 下一步扩展

后续可以继续扩展两层能力：

### 1. 引入矩阵特征

当前只把格式与张量信息给智能体，之后可以增加：

- 行 nnz 分布
- block density
- `sidx` 局部性
- `coo range` 长度分布
- 16B 对齐比例
- tile-level reuse

这样第二个智能体才真正变成“按输入矩阵动态给策略”。

### 2. 引入性能反馈闭环

这一层现在已经落地到可运行版本，默认流程是：

1. 运行 `heuristic` 基线
2. 调用在线 agent 生成第 1 轮策略
3. 生成 `.inc`
4. 跑 correctness
5. 跑轻量性能
6. 跑 `ncu`
7. 把历史、profile 和当前 `.inc` 回灌给下一轮

当前默认迭代次数是 `5`，可通过脚本参数调整。

现阶段仍然有两个边界：

- agent 输出的策略空间仍然受限于当前 schema，不能直接改变 tile shape 或重写 CUDA 结构
- profile 反馈主要来自单个 generated kernel 的 `ncu` 摘要，还没有扩展到多数据集统计或更细粒度的 source-level hotspot 对齐

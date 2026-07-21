# 基于 Sparsene 的 LLM / Agent 稀疏格式设计与优化方案

本文给你一个可落地的研发方案，目标是在当前项目基础上，引入大模型或 Agent 来做三件事：

1. 为输入矩阵设计合适的层次化稀疏格式
2. 在代码生成前做张量放置、流水线、负载均衡等策略设计
3. 形成一个“分析 -> 生成 -> 编译 -> 测试 -> 反馈”的自动优化闭环

方案设计遵循一个原则：

- 不让 LLM 直接“瞎写 CUDA”，而是让它在 Sparsene 已经提供的结构化抽象层上做决策

这点非常关键。

## 1. 为什么这个项目很适合接 LLM

Sparsene 已经天然把问题拆成了几层：

- 稀疏格式层
- lowering / 变换层
- 元数据结构层
- pipeline 规划层
- backend codegen 层

这意味着 LLM 不需要从零端到端输出一个 CUDA kernel。

更合理的方式是：

- LLM 负责提出结构化决策
- 项目已有代码负责把这些决策变成实际代码

也就是：

- LLM 决策
- 规则系统约束
- 编译器后端执行
- 评测系统验证

这比“直接让 LLM 生成整段 CUDA”更靠谱，也更适合长期迭代。

## 2. 总体架构

建议整体分成 6 个模块：

1. `Matrix Feature Extractor`
2. `Format Candidate Generator`
3. `Pipeline / Placement Optimizer`
4. `Cost Model + Retrieval Memory`
5. `Compile & Validation Runner`
6. `Experiment Database`

对应的数据流：

```text
输入矩阵
  -> 特征提取
  -> 候选格式生成
  -> 候选 lowering / pipeline / placement 方案生成
  -> 编译与运行
  -> 结果回写数据库
  -> 反哺 LLM/Agent 检索与策略更新
```

## 3. 模块一：矩阵特征提取

这是你后续工作的第一优先级，因为没有稳定特征，LLM 就无法做出靠谱决策。

## 3.1 必须提取的基础统计特征

对输入稀疏矩阵 `A`，至少提取：

- `M, K, nnz`
- 总体 sparsity
- 每行 nnz 分布：
  - 均值
  - 方差
  - 最大值
  - 分位数
- 每列 nnz 分布：
  - 均值
  - 方差
  - 最大值
  - 分位数
- 行窗口级统计：
  - 每个 row window 的 unique column 数
  - 每个 row window 的 TC block 数
  - 每个 row window 的 nnz 数
- padding 相关统计：
  - 如果用 `BLK_M x BLK_K` 划分，padding 比例是多少
- block density 相关统计：
  - 固定块大小下的块占用率分布

## 3.2 对 LLM 特别重要的结构特征

除了基础统计，还建议加入“更能帮助格式选择”的特征：

- 行间相似度：
  - 相邻行非零列集合 Jaccard 相似度
  - row window 内平均列重用率
- 列聚集度：
  - 非零是否集中在少数列段
- 分块稳定性：
  - 改变 `BLK_M` / `BLK_K` 时，TC block 数是否稳定
- 层次化压缩收益：
  - 从 CSR 到 `row window + condensed cols + TC block` 后，元数据膨胀/压缩比例
- 负载均衡难度：
  - row window 工作量变异系数
  - top-k hot windows 占总工作量比例

这些特征会直接影响：

- 用不用 `ME_TCF`
- 要不要变种格式
- 是 normal binding / multi-binding / strict LB
- `BLK_M/BLK_K/TILE_B` 应怎么设

## 3.3 工程建议

建议把这部分做成独立模块，例如：

- `python/sparsene/analysis/matrix_features.py`

输出统一 JSON：

```json
{
  "shape": {"M": 0, "K": 0, "nnz": 0},
  "global_stats": {...},
  "row_stats": {...},
  "col_stats": {...},
  "row_window_stats": {...},
  "padding_estimation": {...},
  "block_density": {...},
  "load_balance": {...}
}
```

这样后续：

- LLM 提示词
- 检索系统
- 实验数据库

都能直接用同一份结构化输入。

## 4. 模块二：格式设计 Agent

目标不是让 LLM 自由发挥，而是让它在一个“受限设计空间”里做选择。

## 4.1 设计空间怎么定义

建议先不要让 LLM 直接 invent 新 DSL 语法，而是让它在这些维度里选择：

- 格式大类：
  - `ME_TCF`
  - `BIT_TCF`
  - `SR_BCRS`
  - `BIT_BSR`
  - 新增候选变体
- 轴层次：
  - 是否保留 `row window`
  - 是否把 `K` 方向 split 成两层
  - 原子格式选 `COO` / `MCO` / bitmask
- 参数：
  - `BLK_M`
  - `BLK_K`
  - `TILE_B`
  - 是否 reorder
  - 是否 densify 某层 sparse axis

也就是说，让 LLM 输出的不是 free-form code，而是一个结构化 proposal：

```json
{
  "format_family": "ME_TCF",
  "params": {
    "BLK_M": 16,
    "BLK_K": 8,
    "TILE_B": 64
  },
  "atomic_format": "coo",
  "transform_hints": {
    "reorder_k": true,
    "prefer_varlen_lowering": true
  },
  "reasoning": [...]
}
```

## 4.2 先做“推荐”，不要一开始就做“直接修改代码”

建议分两阶段：

### 阶段 A：recommendation mode

LLM 只输出：

- 建议格式
- 建议参数
- 原因

真正执行代码生成由现有 Python 代码完成。

### 阶段 B：constrained synthesis mode

等 recommendation 跑稳后，再让 LLM 输出：

- `Format(...)`
- 参数组合
- 某些变换 hint

然后由一个 parser/validator 检查后再交给生成器。

## 4.3 设计建议

建议实现一个 `FormatDesignerAgent`，输入：

- 矩阵特征
- 目标 GPU 架构
- 任务约束
  - 追求吞吐
  - 追求显存节省
  - 追求低开发风险

输出：

- top-k 格式候选
- 每个候选的理由
- 不确定性评分

## 5. 模块三：张量放置与流水线 Agent

这是你提到的第二个重点：代码生成之前的一些优化策略，也要让大模型参与。

这里建议把问题拆开，不要让一个 agent 一次决定所有东西。

### 5.1 Tensor Placement Agent

负责决定：

- 哪些中间张量放在 `rmem`
- 哪些放在 `smem`
- 哪些直接从 `gmem` 读
- 哪些路径需要 `ldmatrix`
- 哪些路径需要 double buffer

输入：

- `opir` / `nvir` 中的 op graph
- 张量 shape
- producer-consumer 关系
- GPU 架构信息

输出：

- placement proposal

例如：

```json
{
  "tensor_placements": {
    "tileA": "smem",
    "tileB": "smem",
    "coo_idx": "rmem",
    "coo_val": "smem",
    "C_tile": "rmem"
  },
  "rationale": [...]
}
```

### 5.2 Pipeline Planner Agent

负责决定：

- stage 划分
- 是否 overlap
- shift offset
- buffer 数量

它不需要直接替代 `compiler_driver.py`，更好的方式是：

- 它生成 plan 候选
- 现有 simulator / profiler / 编译器去评估

也就是：

- LLM 负责 proposal
- 现有系统负责 verification

### 5.3 Load Balance Agent

负责判断：

- normal binding
- multi-binding
- strict load balance

建议输入的关键特征：

- row window 级工作量分布
- top-heavy 程度
- small-window 占比
- block splitting 可能的 atomic 开销

输出：

- 负载均衡策略
- 阈值建议
- 是否需要原子写回

## 6. 模块四：检索记忆与经验数据库

如果你只用 LLM 直接从 prompt 推理，效果会很不稳定。

更建议做一个“经验检索层”。

## 6.1 要存什么

建议为每次实验存：

- 矩阵特征
- 选用的格式
- 生成参数
- pipeline plan
- placement plan
- 编译是否成功
- 运行是否成功
- 性能指标
- 正确性结果
- 失败日志摘要

## 6.2 检索怎么用

每次来一个新矩阵时：

1. 先提取特征
2. 在历史数据库里做相似矩阵检索
3. 把 top-k 相似样本交给 LLM
4. 让 LLM 在“相似案例 + 当前特征”的上下文里做决策

这样 LLM 会从：

- 空想式推理

变成：

- case-based reasoning

实际效果会稳定很多。

## 6.3 数据库存储建议

建议最开始不要搞太复杂，先用：

- `sqlite` 或 `duckdb`
- 外加 JSON artifact 文件

表结构可以有：

- `matrix_features`
- `format_trials`
- `pipeline_trials`
- `compile_failures`
- `runtime_failures`

## 7. 模块五：自动编译与验证闭环

这个模块是把 LLM 变成真正 agent 的关键。

建议统一一个 runner：

1. 读矩阵
2. 提特征
3. 调 LLM/Agent 产出候选
4. 调生成器出 `.inc`
5. 编译
6. 跑 correctness
7. 跑性能
8. 写回数据库

可以把它做成：

- `python/sparsene/agent/runner.py`

## 7.1 correctness 先于 performance

务必让闭环顺序是：

- compile pass
- correctness pass
- performance pass

不要反过来。

因为当前项目本身就暴露了一个重要现实：

- 有些候选根本还没达到 compile-correctness

## 7.2 对 LLM 的奖励信号

如果你要做更自动化的 agent，可以把以下结果作为 reward：

- 编译成功：+1
- correctness 通过：+5
- 相对 baseline 提升：按 speedup 加分
- 显存超限：负分
- 编译失败：大负分

这样甚至可以和 bandit / Bayesian optimization / RLHF 式流程结合。

## 8. 模块六：数据集与实验组织

你已经给了一个数据集目录：

- `/mnt/lustre-client/zhangkaige/selected_mtx`

建议把它分三类使用：

### 8.1 训练/经验积累集

用途：

- 构建经验数据库
- 让 agent 学到哪些矩阵特征对应哪些格式更靠谱

### 8.2 验证集

用途：

- 调整特征、prompt、策略阈值

### 8.3 保留测试集

用途：

- 最终汇报和论文结果

不要所有矩阵都混在一起，否则很难知道 agent 是否真的泛化了。

## 9. 推荐的工程落地顺序

建议按下面顺序推进。

### 阶段 1：先把基础设施补齐

目标：

- 稳定提取矩阵特征
- 能把一次试验完整记录下来

输出：

- `matrix_features.py`
- `experiment_db.py`
- `runner.py`

### 阶段 2：先做规则系统，再接 LLM

目标：

- 用人工规则先完成一个 baseline 决策器

例如：

- 行窗口工作量方差特别大 -> 优先 strict LB
- padding 比率过高 -> 降低 `BLK_K` 或考虑 bitmask 方案
- row window 内 unique col 高复用 -> 优先 ME_TCF

这样做的好处是：

- 有一个强 baseline
- 也能给 LLM 提供 teacher signal

### 阶段 3：做 recommendation-style LLM

目标：

- LLM 只推荐格式和参数，不直接改代码

### 阶段 4：做 placement/pipeline multi-agent

建议拆成三个 agent：

- `FormatAgent`
- `PlacementAgent`
- `PipelineAgent`

再加一个 `Coordinator` 做仲裁。

### 阶段 5：做自动搜索闭环

目标：

- top-k 候选自动编译、自动验证、自动回写结果

## 10. 你后续具体可以新增哪些目录/模块

建议目录结构：

```text
python/sparsene/
  analysis/
    matrix_features.py
    dataset_loader.py
  agent/
    prompts.py
    format_agent.py
    placement_agent.py
    pipeline_agent.py
    coordinator.py
    runner.py
  knowledge/
    format_space.py
    heuristics.py
  storage/
    experiment_db.py
```

另外建议在仓库根目录增加：

```text
exp_agent/
  configs/
  logs/
  reports/
```

## 11. 提示词设计建议

不要只把原始稀疏矩阵甩给 LLM。

应该给它的是：

- 结构化特征
- 候选空间
- 明确目标
- 历史相似案例

一个推荐 prompt 的内容应该包含：

- 当前矩阵特征摘要
- 当前 GPU 架构
- 允许的候选格式空间
- 近邻历史案例 top-k
- 当前优化目标
- 输出必须是 JSON

例如输出限制成：

```json
{
  "format": "...",
  "params": {...},
  "load_balance": "...",
  "placement_hint": {...},
  "pipeline_hint": {...},
  "confidence": 0.0
}
```

## 12. 对你这个方向最重要的风险提醒

### 风险 1：backend 不稳定会污染 agent 学习

如果 codegen 还经常 compile fail，agent 会学到很多“假失败”。

所以当前项目最优先还是要先把：

- 生成代码可编译
- correctness 可验证

这两件事打稳。

### 风险 2：特征维度太少，LLM 决策会漂

如果只有：

- `M/K/nnz/sparsity`

这种粗特征，LLM 基本不可能稳定决定格式。

必须增加：

- row window 统计
- padding 统计
- load-balance 统计
- block density 统计

### 风险 3：直接让 LLM 生成 DSL / CUDA，失败率会很高

建议始终把 LLM 放在：

- 结构化 proposal

而不是：

- 直接写最终 kernel

## 13. 我最推荐你当前立刻开工的三个子任务

### 任务 A：矩阵特征提取器

这是整个 LLM 方案的地基。

### 任务 B：实验数据库

没有历史结果积累，agent 很难持续变强。

### 任务 C：规则版格式推荐器

先不用 LLM，先写一个规则系统跑通 end-to-end。

等这三件事完成后，再接入 LLM，研发效率会高很多。

## 14. 最后一句建议

你这个方向最值得做的，不是“让 LLM 直接生成 CUDA”，而是：

- 让 LLM 学会在 Sparsene 已有的结构化编译框架里做高价值决策

也就是：

- 选格式
- 选参数
- 选张量放置
- 选 pipeline
- 选负载均衡

一旦这条线做通，你得到的会不是一个 demo，而是一个真正可扩展的“稀疏格式与 SpMM 优化 agent 系统”。

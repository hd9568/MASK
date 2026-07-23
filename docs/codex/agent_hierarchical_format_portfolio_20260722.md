# AI 层次化格式性能诊断与可审计 portfolio 闭环

日期：2026-07-22  
范围：`20260718_165146_deepseek_v4_pro_format26_v6`、A800/SM80、N=512、FP32 输入输出与 TF32 MMA。  
主指标：`sparsene.kernel_only_post_conversion.v1`；格式转换、重排和正确性逆置换不计时，所有 kernel 内 metadata、restore、padding、MMA、reduction 都计时。

## 结论

v6 的 26 个 Agent 设计并非 26 条独立的已优化 CUDA 路径：25 个最终落到两层
column-condensed `TCF + MCO`，`dc2` 落到一层 `BIT-BSR + MCO`。历史 AI 表中 MCO
仅在 Yeast 快于已优化 SR-BCRS；而 `SR_BCRS 16x8` 在 25/26 项更快。这不是单纯
LLM“选错格式”，而是设计依据、原子叶子执行语义、硬件 tile 对齐和 host 调度之间有
两个未建模的断层。

1. MCO 的存储大小被误当成运行时间。当前 SM80 MCO backend 对每个活跃 tile 仍要
   gather 同样的 B 行、恢复完整物理 dense A tile、执行完整 TF32 MMA；MCO 不会跳过
   零位置的 MMA 或 B gather。它只压缩 A payload，却额外增加 mask/offset、popcount
   rank 和 restore 成本。
2. `BLK_M=8` 的逻辑窗口在本 backend 必须物理补成 16 行。Agent 用小窗口降低行长
   偏斜时没有把这部分 kernel 内填零、restore、MMA 和 store 成本纳入决策。
3. 通用 custom runner 过去没有把 dense backend 的运行时自动分段/任务映射 ABI 传给
   testbed；因此其 dense sibling 即使生成正确，也会退化为固定 32-tile 分段和 partial-C
   reduction。内建 SR-BCRS runner 已经在用 `max-tiles=0 + execution_policy=auto`，能对
   ASIC 选择 `strict_lb`。

## 现场证据

ASIC_680k 的 Agent 原案为 `TCF+MCO, BLK_M=8, BLK_K=16`。其旧记录是 10.2432 ms。
以新代码生成并完整验证的两个 dense sibling 如下；均为 warmup=3、repeat=32，单进程
串行运行：

| candidate | 物理 tile | 运行时调度 | generated ms | correctness |
| --- | --- | --- | ---: | --- |
| `atomic_dense_sibling` | 16x16（逻辑 8x16） | 固定 32-tile segment | 25.4389 | PASS, 0 mismatch |
| `dense_16x8_sibling` | 16x8 | 固定 32-tile segment | 28.7540 | PASS, 0 mismatch |
| `dense_16x8_sibling` | 16x8 | 自动 segment + `auto`→`strict_lb` | 4.89523 | PASS, 0 mismatch |

最后一行与原有内建 `SR_BCRS 16x8` 的 ASIC 记录 4.90611 ms 一致到测量噪声范围，证明
层次格式本身可以通过共享生成链得到高性能；此前 portfolio 的主要退化点是 host scheduling
contract 丢失，而不是 dense sibling 生成器或数值语义错误。

## 代码修改与合同

- `python/sparsene/op_gen/format_design_agent.py`
  - 向格式设计 prompt 注入可执行 SM80 成本模型，明确 MCO/dense 的相同 MMA/B-gather
    工作、MCO restore 成本、M=8 物理 padding，以及 storage estimate 不能替代 kernel
    time。
  - 不限制研究 DSL；该模型只约束 Agent 在引用现有可执行 TCF backend 时不得做错误的
    性能推断。
- `python/sparsene/op_gen/format_portfolio.py`
  - 保留 Agent 原 DSL 不变，产生带 provenance 的执行 measurement sibling。
  - 两层 TCF 可产生原子 `MCO↔dense` sibling；另提供小型硬件对齐 dense tile portfolio：
    `16x8`、`16x16`、`32x8`。BIT-BSR 不会被伪造为 dense，因为没有对应的 host/backend
    ABI。
- `python/test/generate_custom_format_operators.py`
  - `--candidate-id` 和 `--worktree-label` 为每个 sibling 生成独立
    `IR -> include -> manifest` 树；manifest 记录 candidate source、atomic、format SHA256
    和 backend。不覆盖 Agent 原始产物。
- `python/test/run_custom_format_e2e.py`
  - manifest-owned work directory 支持独立 sibling build/run。
  - 新增 `--runtime-max-tiles-per-segment`、`--execution-policy`、
    `--task-group-tiles`，并把运行时调度写入 result JSON 和结果文件名，防止不同调度条件
    覆盖同一结果。
- `python/test/run_atomic_portfolio_e2e.py`
  - 严格串行 (`--jobs 1`) 地测 Agent 原案和 atomic siblings；默认 runtime segment=0，
    即自动负载均衡。`--include-tile-siblings` 显式加入三个硬件对齐 dense tile sibling。
  - 输出 `atomic_portfolio_e2e_summary.{json,csv}`，每行标注 `agent_proposal` 或
    `deterministic_*_sibling`，不得将后者宣称为 Agent 原始选择。

## 推荐执行方式

先用一个数据集闭环确认，再扩大。以下命令不会调用付费模型，GPU 评测始终串行：

```bash
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
cd /home/huoda/sparsene
PYTHONPATH=python python3 python/test/run_atomic_portfolio_e2e.py \
  logs/e2e_runs/20260718_165146_deepseek_v4_pro_format26_v6 \
  --datasets ASIC_680k --n 512 --warmup 3 --repeat 32 --jobs 1 \
  --config-file sota_results/summary/a800_n512_fp32_final_config.json \
  --include-tile-siblings
```

26 数据集运行前必须确认不存在其他 GPU benchmark；不要并行启动该命令。只把
`correctness_passed=true` 的候选加入每数据集 portfolio 选择。选择表需要同时保留
Agent 原案、所有 sibling、生成 include SHA256、runtime execution policy 和计时参数，
以便把“Agent 设计质量”与“后端/调度可实现性”分开反馈。

## 已完成验证与限制

- Python syntax：新 portfolio、generator、两个 runner 和测试模块已通过 `py_compile`。
- Agent prompt 的 13 项 `unittest` 通过；v6 全部 26 导出 DSL 的结构检查表明 25 个
  two-level `MCO/dense` pair 和 1 个 BSR-only 设计。
- ASIC dense sibling 已完成 Format→RTS→Computent→OPIR→NVIR→CUDA、NVCC host ABI、
  真实矩阵全输出正确性和 GPU timing 验证；生成物路径位于
  `datasets/ASIC_680k/operator_generation/portfolio/`。
- venv 未安装 `pytest`，因此 `test_custom_format_runtime.py` 的 pytest 收集未运行；其
  26-format portfolio structure assertions 已用等价的 venv Python 结构检查执行。
- 本次尚未跑完整 26×portfolio sweep，不能把 ASIC 的结果外推为 26 数据集结论；下一步
  应按上述 runner 串行收集真实 feedback，再把每数据集最佳 correct candidate 作为
  format agent 下一轮 prompt 的校准数据。

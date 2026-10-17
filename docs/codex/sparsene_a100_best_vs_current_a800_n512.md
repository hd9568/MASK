# 历史 Sparsene A100 最佳格式与当前 A800 生成版本对比

日期：2026-07-19  
N/精度：N=512，FP32 输入输出，tensor-core 路径使用 TF32 MMA

## 1. 输入与假设

历史输入：
`sota_results/summary/a100_n512_fp32_sparsene_historical_all.csv`，包含 26 个数据集、
每项 9 个 Sparsene 历史格式/负载均衡变体，共 234 条。

当前输入：
`logs/e2e_runs/20260718_165146_deepseek_v4_pro_format26_v6/operator_e2e_summary.csv`，
包含当前 v6 版本 26 个正确性通过的最终 A800 结果。

按用户要求，把单卡 A100 与 A800 视为可近似比较，不做硬件归一化。该比较仍是跨机器、
跨历史运行的工程近似，不具有同卡 A/B 的严格性。

## 2. 历史最佳选择规则

每个数据集在 `anomaly` 为空的历史记录中选最小 `milliseconds`。OVCAR-8H、Yeast、
YeastH 各有一条 `0.000084/0.000087/0.000093 ms` 的 ACC 记录，已由原始整理器标记为
`implausibly_small_time_lt_0.01ms`，因此不参与审计后的最佳选择。

输出 `sparsene_a100_n512.csv` 保留数据集、最佳格式、最佳时间、参与选择的变体数和排除
异常数。历史原始输入不改写。

## 3. 对比结果

定义：

`historical_over_current_speedup = historical_A100_best_ms / current_A800_generated_ms`

大于 1 表示当前版本更快。结果为：

- 当前版本超过历史最佳：1/26，唯一数据集是 Yeast；
- Yeast：历史 5.15845 ms，当前 4.93918 ms，名义加速 1.044394x；
- 10% 以内的接近项：OVCAR-8H、Yeast、consph；
- 26 项几何平均 `historical/current=0.533922x`，即当前时间约为历史最佳的 1.873x；
- 中位数 `historical/current=0.514494x`，即当前时间约为历史最佳的 1.944x；
- 如果机械地把三条已标记异常的 ACC 值也参与最小值，当前版本为 0/26。

历史最佳格式分布为：SR-BCRS-16x8 19 项、SR-BCRS-16x8-multi-bind 3 项、
SR-BCRS-16x8-strict-lb 3 项、ACC-BIT-TCF 1 项。也就是说，25/26 的历史最佳来自当前
v6 column-condensed/active-fixed-block 路径没有复现的 SR-BCRS 执行家族；差距不能简单
归因于某一个 CUDA 参数。

Yeast 的优势只有约 4.4%，而验证规范要求几百分点差异做噪声控制；再考虑 A100/A800
跨卡，该项应表述为“名义超过”，不是严格同卡胜出结论。

## 4. 正确性和来源限制

- 当前 26 项均有完整输出检查和 TF32 误差指标；
- 历史 Sparsene 日志只有 `check nan` 哨兵，没有当前逐元素/relative-L2 证据；
- 两侧都按 kernel-only 语义理解，但历史 NVCC、host、timer 边界和当前实现并非同一二进制；
- 因此本表适合定位当前相对历史 Sparsene 的性能缺口，不应替代同卡、同代码、同验证的复测。

## 5. 产物与复现

- `sota_results/summary/sparsene_a100_n512.csv`：历史逐数据集最佳简表；
- `sota_results/summary/sparsene_current_vs_a100_n512.csv`：26 项逐项比较；
- `sota_results/summary/sparsene_current_vs_a100_n512_audit.json`：规则、统计和路径审计；
- `sota_results/summary/compare_current_to_sparsene_a100.py`：可重复生成脚本。

复现命令：

```bash
PYTHONPATH=python /mnt/lustre-client/huoda/venv/llm-sog/bin/python3 \
  sota_results/summary/compare_current_to_sparsene_a100.py
```

本次只读取已有 CSV，没有重新运行 GPU。

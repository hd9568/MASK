# A100 / N=512 / FP32 原始结果整理

本目录由 `build_a100_n512_fp32_summary.py` 从上一级原始文件可重复生成。原始文件
不改写，所有时间单位均为 ms。

- `a100_n512_fp32_operator_long.csv`：算子级全部原始记录，保留同名方法的不同来源。
- `a100_n512_fp32_operator_wide_ms.csv`：26 个数据集横向查看表。
- `a100_n512_fp32_sparsene_historical_all.csv`：旧 Sparsene 的 9 种实现/负载均衡组合。
- `paper_reported_a100_n512_fp32_aggregate.csv`：论文表格给出的平均加速，不从逐项数据反推。
- `paper_a100_n512_fp32_sparsene_per_dataset.csv`：从论文 A100/N=512/FP32
  vector PDF 的橙色 Sparsene 柱高除以绿色 cuSPARSE 单位柱高得到的26项发表值。
- `paper_a100_n512_fp32_sparsene_validation.csv`：逐项对照论文柱状图、最接近的原始
  Sparsene 变体和“九种变体取最快”的历史上包络，避免混淆论文选择与事后 best-of-all。
- `a100_hidden512_gcn_e2e_long.csv`：GCN hidden=512 的端到端结果，和算子 kernel-only 表分开。
- `audit.json`：覆盖数和异常值。

重要口径：

1. 论文写明 A100 80GB、NVCC 12.6 `-O3`、FP32、1000 次均值；当前 A800
   复现实验不能冒充这组历史 A100 数据。
2. `DTCSpMM_exe_time_and_throughput.csv` 每个数据集/N 有三行，但没有表头说明
   GPU、variant 或重复次序，因此只标为 `unlabeled-observation-{1,2,3}`，不擅自挑选。
3. 旧 `sparsene_fp32_N512.log` 只有 `check nan` 哨兵，没有当前要求的完整逐元素
   验证；其中 OVCAR-8H、Yeast、YeastH 的 ACC 时间小于 0.01 ms，已标为异常，
   不能用于排名。
4. 不同项目原始文件里的 cuSPARSE 计时可能使用不同版本、预处理或 timer 边界，
   所以 `direct-log` 和 `RoDe-source` 保持为两个来源。
5. 当前生成算子的统一主指标是 `sparsene.kernel_only_post_conversion.v1`；只有新运行
   中完整正确性通过的结果才能进入当前版本最终表。
6. 论文 vector 柱状图恢复出的26项算术平均为 2.1869x，四舍五入后严格复现正文表格
   的 2.19x。每项都能在原始日志中找到误差不超过5%的对应 Sparsene 变体；但
   `sparsene_a100_n512.csv` 是九种变体事后取最快的历史上包络，算术平均为2.2324x，
   不是论文逐项选择的精确复刻，主要差异来自 `reddit` 和 `mip1`。

## 当前 A800 复现（与上面的 A100 历史数据分表）

- `a800_n512_fp32_final_config.json`：26 个数据集逐项固化的 Tile-N/执行分段配置，
  明确要求按数据集严格串行运行。
- `build_a800_n512_fp32_final_summary.py`：检查 26/26 覆盖、配置匹配、统一 timing
  重复次数、单 warp launch ABI 和正确性后才允许出表。
- `sparsene_a800_n512_fp32_final.csv`：当前生成算子的完整机器可读表。
- `sparsene_a800_n512_fp32_final.md`：便于查看的最终性能表和总体统计。
- `sparsene_a800_n512_fp32_final_audit.json`：结果路径、误差、打包和性能审计信息。
- `sparsene_a100_n512.csv`：从 9 个历史 Sparsene 变体中排除已标记异常后，逐数据集
  选择最小时间的 26 行简表。
- `sparsene_current_vs_a100_n512.csv`：当前 A800 生成版本与上述 A100 历史最佳的逐项
  近似比较；`historical_over_current_speedup > 1` 表示当前更快。
- `sparsene_current_vs_a100_n512_audit.json`：跨卡近似假设、异常规则和聚合统计。

这组新结果来自 NVIDIA A800 80GB PCIe，输入/输出为 FP32，生成 kernel 内使用
SM80 TF32 tensor-core MMA。它不能与上面的 A100 原始记录合并后直接排名。
在用户明确允许 A100/A800 单卡近似比较后，当前版本名义上超过历史最佳 1/26
（Yeast）；历史最佳中的 25/26 来自 SR-BCRS 家族。

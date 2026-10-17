# v6 自定义格式：26 数据集 N=512 FP32 生成算子闭环

日期：2026-07-19  
格式运行：`20260718_165146_deepseek_v4_pro_format26_v6`  
实测硬件：NVIDIA A800 80GB PCIe（SM80）

## 1. 范围和结论边界

本轮把原先只验证前三个数据集的链路扩展为 26/26：

`Format JSON -> 可导入 DSL -> RTS -> Computent -> OPIR -> lowering -> NVIR -> CUDA include -> NVCC -> 真实矩阵 -> cuSPARSE 对照`

目标是“全部可生成、可编译、可执行、数值正确并有稳定性能记录”。“全部跑通”不等于
“每项都快于 cuSPARSE”；最终表同时保留快项和慢项，禁止选择性删除。

最终稳定复测为 26/26 PASS，12/26 快于同次运行的 cuSPARSE，26 项
`cuSPARSE/generated` 几何平均为 1.0385x。最大 relative-L2 为 `2.41863e-4`，
最大归一化绝对误差为 `2.65992e-4`，都明显低于固定 TF32 阈值。

历史论文/SOTA 数据来自 A100，而本轮机器是 A800。两者分别放表，不能把不同硬件的
时间混为同一排名。输入/输出和累加为 FP32，生成 kernel 使用 SM80 TF32 tensor-core
MMA，因此误差标准同时记录严格逐元素 mismatch、非有限值、relative-L2 和归一化最大
绝对误差。

## 2. SOTA 原始数据整理

`sota_results/summary` 中新增可重复生成的 A100/N=512/FP32 汇总：

- `a100_n512_fp32_operator_long.csv`：594 条算子记录、23 个方法/来源标签；
- `a100_n512_fp32_operator_wide_ms.csv`：26 数据集宽表；
- `a100_n512_fp32_sparsene_historical_all.csv`：Sparsene 9 个历史变体、234 条；
- `a100_hidden512_gcn_e2e_long.csv`：GCN hidden=512 端到端结果，和 kernel-only 分离；
- `paper_reported_a100_n512_fp32_aggregate.csv`：论文直接报告的聚合加速；
- `audit.json`：覆盖、缺项和异常审计。

审计没有擅自补语义：DTC 的三行重复值保持为三个未标注 observation；RoDe/Sputnik/
SparseTIR 的部分源只有 25 个数据集；旧 Sparsene 日志只有 `check nan` 哨兵；OVCAR、
Yeast、YeastH 的三条小于 0.01 ms 的旧 ACC 值标为异常，不能用于当前排名。

## 3. 26 个格式如何落到通用后端

`backend_registry.py` 按 hierarchy、axis type/direction、atomic、layout、reorder 和参数匹配，
不读取数据集名，也不依赖 Agent 给出的描述性 `format_name`。本轮 26 个格式全部落入：

1. 两层 column-condensed TCF + row-major MCO：25 个数据集，生成 `BIT_TCF`；
2. 一层 active fixed blocks + row-major MCO：`dc2`，生成 `BIT_BSR`。

SM80 后端支持逻辑 `BLK_M={8,16,32}`、`BLK_K` 为 8 的倍数。逻辑 M=8 保持 8 行凝聚
语义，只在物理 MMA 中补到 16 行；BIT-BSR 的逻辑 K=16 精确细分成相邻的物理 K=8
micro-block，避免旧 restore swizzle 跨多个 MMA-K 片段。`row_rcm`、`first_col_sort` 和
可扩展 minhash-bucketed `jaccard_greedy` 都在 host 转换阶段真实执行，并在计时外恢复
原始输出行。

## 4. 主要错误与闭环修复

### 4.1 K>8 只计算首个 fragment

旧 ACC 特化把 A/B 寄存器形状的 K 维压掉，因为旧 testbed 只实例化 K=8。K=16/32
能够编译，但后续 K fragments 从未 `ldmatrix`/MMA。现在寄存器形状保留 K 维，A/B load
和 TF32 MMA 显式循环全部 K fragments。

### 4.2 M=32 只写前 16 行

旧 store 固定读取 `m_iter=0`，导致 BLK_M=32 的后半输出丢失。row-major 和旧 DTC
布局的 store 都改为遍历全部 M fragments，`cant/YeastH/pdb1HYS` 已覆盖该路径。

### 4.3 BIT-BSR 网格轴和 K=16 restore 错误

BIT-BSR 的 OPIR 将 row block 映射到 `grid.x`、N tile 映射到 `grid.y`，与 TCF 相反；
旧统一 launch 导致严重错位。host bridge 现在按 packing mode 选择轴。`dc2` 的逻辑
16x16 block 又与已验证的 K=8 restore 路径不兼容，改为物理 16x8 的精确细分后，由
数百万错误降为 TF32 误差范围内并通过完整输出验证。

### 4.4 大网格、长尾窗口与执行分段

ASIC 等数据超过 SM80 `grid.y=65535`，TCF launch 被切为多个不重叠 chunk。对单行窗口
含大量 sparse tile 的长尾，转换阶段建立 execution segments；每段先生成 partial C，
随后 GPU reduction 合并。计算和 reduction 都在 CUDA event 内，转换不计时。

初始上限 32 对常规矩阵有利，但会让 `protein/reddit` 等产生大量额外段。实测逐级扩大
32/64/128/256/512/1024，最终让 protein 的 8284 row windows 对应 8284 execution
segments，完全取消归约；reddit 只剩很少额外段。

### 4.5 重排和正确性验证

实现了确定性 RCM、首列排序和可扩展 Jaccard 近邻排序；逆映射不计入 kernel 时间。
验证扫描所有 `M*N` 输出，不再只打印前若干错误；任何 nonfinite 直接失败。TF32 路径
预先固定通过标准为：严格逐元素全过，或同时满足 relative-L2 `<=5e-4`、归一化最大
绝对误差 `<=5e-3`。所有原始指标和最大误差位置都写入 JSON。

### 4.6 生成算子 launch ABI

当前生成 kernel 每 block 恰好一个 32-thread warp。生成物现在显式声明 ABI，将流水线
里的 block barrier 降为具有 warp 内存序保证的 barrier；testbed 在编译期拒绝缺少
该声明的旧 include。DD A/B 表明收益约 0.3%，不是主要性能来源，但消除了无必要的
block-wide 同步并让约束可审计。

## 5. 性能调优与最终配置

所有 GPU benchmark 严格按数据集串行执行，runner 默认 `jobs=1`，没有并发跑算子。
调优观察：

- `tile_n=32` 在代表性 K=16/32 数据集上普遍更差；
- `tile_n=64` 是 K=16/32 的稳定选择；
- 所有 K=8 数据集用 `tile_n=128` 都有收益，减少了 N 方向 block 和重复稀疏元数据工作；
- 长尾矩阵单独调 execution segment 上限，避免 partial-C reduction 膨胀；
- 单 warp barrier 降低正确但只有噪声量级性能改善，不把它夸大为核心优化。

最终逐项选择固化在
`sota_results/summary/a800_n512_fp32_final_config.json`，统一串行命令为：

```bash
PYTHONPATH=python /mnt/lustre-client/huoda/venv/llm-sog/bin/python3 \
  python/test/run_custom_format_e2e.py \
  logs/e2e_runs/20260718_165146_deepseek_v4_pro_format26_v6 \
  --n 512 \
  --config-file sota_results/summary/a800_n512_fp32_final_config.json \
  --warmup 10 --repeat 100 --jobs 1
```

## 6. 最终结果和产物

最终性能表由脚本强制检查 26/26 覆盖、所选配置、N、性能合同、统一 warmup/repeat、
launch ABI 和 correctness 后生成：

- `sota_results/summary/sparsene_a800_n512_fp32_final.md`；
- `sota_results/summary/sparsene_a800_n512_fp32_final.csv`；
- `sota_results/summary/sparsene_a800_n512_fp32_final_audit.json`；
- 每个 tuning observation：`datasets/*/operator_generation/e2e_results/`；
- 每个最终生成 include、manifest、IR、编译日志和运行日志均留在原 run 目录。

最终聚合：26/26 correctness PASS，12/26 超过 cuSPARSE，几何平均加速 1.0385x；
所有数据集均为 warmup=10、repeat=100。代表性快项包括 consph 3.4218x、pwtk
2.7112x、cop20k_A 2.7026x、pdb1HYS 2.2961x；ASIC/Ga/Si/scircuit 等慢项也完整
保留在表中。

当前性能合同为 `sparsene.kernel_only_post_conversion.v1`：格式转换、重排/逆映射、
allocation/copy 和 cuSPARSE preprocess 不计时；生成 kernel 的全部 launch、物理补零工作和
必要 reduction 均计时。最终表的 `cuSPARSE/generated` 大于 1 才表示生成算子更快。

## 7. 尚存性能瓶颈

正确性和可执行覆盖已闭环，但部分格式仍慢于 cuSPARSE，主要集中在逻辑 M=8 的物理
补零、K=32 的寄存器/共享内存压力、极不规则列 gather 和格式本身的低有效 tensor-core
密度。下一阶段若追求逐项胜出，应做 NCU 驱动的多 warp/共享 A 设计、M=8 相邻窗口融合、
occupancy 与 pipeline depth 联合搜索，以及让 format/strategy Agent 在当前真实性能反馈下
重新选择格式；不能通过改计时边界或放宽正确性来制造加速。

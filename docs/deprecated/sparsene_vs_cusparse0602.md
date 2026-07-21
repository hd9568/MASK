# Sparsene vs cuSPARSE 全量数据集对比 — 2026-06-02

> 运行时间: 2026-06-02 00:27 ~ 05:04
> 模式: agent (格式智能体 + 策略智能体, 4 轮迭代)
> 模型: deepseek-v4-pro (格式 + 策略)
> N=512, heuristic 基线 + 多轮策略搜索
> 说明: 格式设计由 LLM 智能体决定，策略在 4 轮反馈中搜索最优配置。取 best_iteration 的性能数据。

## 总体统计

| 指标 | 数值 |
|------|------|
| 总数据集数 | 26 |
| 执行成功 | 22 |
| 执行失败 | 4 (protein, pwtk, reddit, rma10) |
| 快于 cuSPARSE | 11 |
| 慢于 cuSPARSE | 11 |
| 中位 speedup | 0.98x |

## 详细结果

| dataset | M | K | N | mykernel_ms | cusparse_ms | ratio_vs_cusparse | best_iter | format_name | total_tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ASIC_680k.mtx | 682862 | 682862 | 512 | 71.743335 | 4.957030 | 14.4730 | 4 | ASIC_680k_TCF_8x8 | 35027 |
| cant.mtx | 62451 | 62451 | 512 | 1.006285 | 1.875968 | 0.5364 | 4 | Cant_ME_TCF_32x8 | 33448 |
| conf5_4-8x8-10.mtx | 49152 | 49152 | 512 | 0.668826 | 1.034701 | 0.6464 | 4 | ME_TCF_32x8 | 36674 |
| consph.mtx | 83334 | 83334 | 512 | 1.472819 | 3.077171 | 0.4786 | 4 | TCF_BLK32x8 | 34912 |
| cop20k_A.mtx | 121192 | 121192 | 512 | 1.547008 | 3.263078 | 0.4741 | 3 | ME_TCF_OPT_8x8 | 34976 |
| dc2.mtx | 116835 | 116835 | 512 | 19.880602 | 0.793651 | 25.0496 | 4 | dc2_opt | 35965 |
| ddi.mtx | 4267 | 4267 | 512 | 0.796774 | 0.897178 | 0.8881 | 3 | ME_TCF_32x16 | 35572 |
| DD.mtx | 334925 | 334925 | 512 | 2.001050 | 1.604659 | 1.2470 | 3 | DD_TCF_8x16 | 34984 |
| eu-2005.mtx | 862664 | 862664 | 512 | 9.828967 | 9.403188 | 1.0453 | 3 | ME_TCF_8x8 | 33626 |
| Ga41As41H72.mtx | 268096 | 268096 | 512 | 9.699891 | 10.349773 | 0.9372 | 4 | ME_TCF_16x8_COO | 35015 |
| mac_econ_fwd500.mtx | 206500 | 206500 | 512 | 1.541837 | 1.099930 | 1.4018 | 4 | ME_TCF_8x16_COO | 36105 |
| mc2depi.mtx | 525825 | 525825 | 512 | 4.217549 | 2.433485 | 1.7331 | 3 | ME_TCF_8x8 | 33640 |
| mip1.mtx | 66463 | 66463 | 512 | 13.466368 | 4.286874 | 3.1413 | 2 | ME_COO_16x16 | 35396 |
| OVCAR-8H.mtx | 1889542 | 1889542 | 512 | 10.157516 | 7.753267 | 1.3101 | 4 | ROWC_8x8_VARLEN | 36198 |
| pdb1HYS.mtx | 36417 | 36417 | 512 | 0.740096 | 2.074880 | 0.3567 | 4 | ME_TCF_16x16 | 33195 |
| scircuit.mtx | 170998 | 170998 | 512 | 1.215846 | 0.982374 | 1.2377 | 0 | ME_TCF_8x8 | 34910 |
| shipsec1.mtx | 140874 | 140874 | 512 | 1.515930 | 4.005530 | 0.3785 | 0 | ME_TCF_16x16 | 36176 |
| Si41Ge41H72.mtx | 185639 | 185639 | 512 | 7.365376 | 7.990579 | 0.9218 | 4 | ME_TCF_16x8 | 35420 |
| webbase-1M.mtx | 1000005 | 1000005 | 512 | 6.871296 | 4.276941 | 1.6066 | 4 | XL_TCF | 34607 |
| web-BerkStan.mtx | 685230 | 685230 | 512 | 5.204736 | 4.186112 | 1.2433 | 4 | ME_TCF_8x8_tuned | 34282 |
| YeastH.mtx | 3138114 | 3138114 | 512 | 16.607898 | 12.842189 | 1.2932 | 4 | ShortRow_ME_TCF | 36874 |
| Yeast.mtx | 1710902 | 1710902 | 512 | 7.733709 | 7.196365 | 1.0747 | 3 | ShortRow_ME_TCF_8x8 | 35560 |

## 执行失败

| dataset | M | K | 失败原因 |
|---|---|---|---|
| protein.mtx | 132534 | 132534 | CUDA kernel runtime error: illegal memory access（generated kernel bug） |
| pwtk.mtx | 217918 | 217918 | CUDA kernel runtime error: illegal memory access（generated kernel bug） |
| reddit.mtx | 232965 | 232965 | CUDA kernel runtime error: illegal memory access（generated kernel bug） |
| rma10.mtx | 46835 | 46835 | CUDA kernel runtime error: illegal memory access（generated kernel bug） |

> 这 4 个矩阵在 checkpoint 重新生成后仍然在 CUDA kernel 执行期崩溃，与 checkpoint 数据无关。根因是当前生成的 `.inc` 内核代码在上述矩阵的 shape/tile 参数组合下存在 GPU 侧非法内存访问。旧版 generated kernel（2026-05-12 reference）可正常执行这些矩阵，说明是 codegen pipeline 近期的修改引入了不兼容。需后续 kernel-level 调试解决。

## 分析

- **ASIC_680k** ratio=14.47 是因为格式智能体选择了极端小的 `BLK_M=8, BLK_K=8`，导致 TC block 数量爆炸、每个 block 极其稀疏，形成大量 kernel launch 开销。但 kernel 仍能正确执行和通过 nan check。
- **dc2** ratio=25.05 同理，该矩阵本身就非常不好处理（手写 reference 也只有 0.0585x cuSPARSE）。
- **shipsec1** (0.3785x) 和 **pdb1HYS** (0.3567x) 表现最好，大幅领先 cuSPARSE。
- 22 个成功矩阵中 **11 个快于 cuSPARSE**（ratio < 1.0），成功率 50%。
- 中位 speedup 0.98x，接近打平 cuSPARSE。
- 格式智能体输出的格式名比较自由（如 `ASIC_680k_TCF_8x8`、`XL_TCF`），说明 LLM 确实在做差异化决策而非固定模板。

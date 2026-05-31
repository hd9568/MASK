# SPARSENE vs cuSPARSE (2026-05-31)

## 测试说明

- 对象: 当前最新生成的 `ME_TCF` kernel，不使用智能体，仅默认 heuristic 策略。
- 数据集: `/mnt/lustre-client/zhangkaige/selected_mtx` 下 26 个 `.mtx` 数据集。
- 命令: `MODE=generated SKIP_GENERATE=1 RESULT_ROOT=/home/huoda/sparsene/examples/src_fp32/dtc/testbed/build_perf_logs SPARSENE_DISABLE_MTX_CACHE=1 CUDA_ARCH=sm_80 bash scripts/benchmark_inc_multi_dataset.sh`
- 参数: `N=512`, `warmup=20`, `repeat=100`。
- 原始结果: `/home/huoda/sparsene/examples/src_fp32/dtc/testbed/build_perf_logs/multi_dataset_generated_20260531_182938/summary.csv`

## 总体结论

- 成功率: `26/26`，本轮无 correctness error、无 crash、无 timeout。
- 中位 `speedup_vs_cusparse`: `0.8724x`。
- 平均 `speedup_vs_cusparse`: `1.1515x`。
- 快于 cuSPARSE 的数据集个数: `11/26`。
- 最优数据集: `pdb1HYS.mtx`，`speedup_vs_cusparse=2.7195x`。
- 最弱数据集: `dc2.mtx`，`speedup_vs_cusparse=0.0353x`。

## 结果表

| dataset | M | K | N | mykernel_ms | cusparse_ms | speedup_vs_cusparse | status |
|---|---:|---:|---:|---:|---:|---:|---|
| ASIC_680k.mtx | 682862 | 682862 | 512 | 78.470225 | 4.957932 | 0.0632 | ok |
| DD.mtx | 334925 | 334925 | 512 | 2.011904 | 1.604403 | 0.7975 | ok |
| Ga41As41H72.mtx | 268096 | 268096 | 512 | 10.106511 | 10.337607 | 1.0229 | ok |
| OVCAR-8H.mtx | 1889542 | 1889542 | 512 | 10.239344 | 7.753472 | 0.7572 | ok |
| Si41Ge41H72.mtx | 185639 | 185639 | 512 | 7.665377 | 7.989832 | 1.0423 | ok |
| Yeast.mtx | 1710902 | 1710902 | 512 | 7.746837 | 7.196907 | 0.9290 | ok |
| YeastH.mtx | 3138114 | 3138114 | 512 | 16.720824 | 12.841748 | 0.7680 | ok |
| cant.mtx | 62451 | 62451 | 512 | 1.041510 | 1.875456 | 1.8007 | ok |
| conf5_4-8x8-10.mtx | 49152 | 49152 | 512 | 0.678502 | 1.033871 | 1.5238 | ok |
| consph.mtx | 83334 | 83334 | 512 | 1.469389 | 3.076424 | 2.0937 | ok |
| cop20k_A.mtx | 121192 | 121192 | 512 | 1.604567 | 3.260139 | 2.0318 | ok |
| dc2.mtx | 116835 | 116835 | 512 | 22.507468 | 0.793405 | 0.0353 | ok |
| ddi.mtx | 4267 | 4267 | 512 | 0.838380 | 0.897720 | 1.0708 | ok |
| eu-2005.mtx | 862664 | 862664 | 512 | 10.154384 | 9.482363 | 0.9338 | ok |
| mac_econ_fwd500.mtx | 206500 | 206500 | 512 | 1.541171 | 1.102746 | 0.7155 | ok |
| mc2depi.mtx | 525825 | 525825 | 512 | 4.237148 | 2.432276 | 0.5740 | ok |
| mip1.mtx | 66463 | 66463 | 512 | 14.610873 | 4.328909 | 0.2963 | ok |
| pdb1HYS.mtx | 36417 | 36417 | 512 | 0.762255 | 2.072955 | 2.7195 | ok |
| protein.mtx | 132534 | 132534 | 512 | 49.885020 | 39.080571 | 0.7834 | ok |
| pwtk.mtx | 217918 | 217918 | 512 | 2.310912 | 5.647073 | 2.4437 | ok |
| reddit.mtx | 232965 | 232965 | 512 | 87.292070 | 68.291494 | 0.7823 | ok |
| rma10.mtx | 46835 | 46835 | 512 | 0.595302 | 1.194220 | 2.0061 | ok |
| scircuit.mtx | 170998 | 170998 | 512 | 1.216686 | 0.982804 | 0.8078 | ok |
| shipsec1.mtx | 140874 | 140874 | 512 | 1.602437 | 4.007854 | 2.5011 | ok |
| web-BerkStan.mtx | 685230 | 685230 | 512 | 5dddddddddddddddddddddddddddddsd.198828 | 4.240968 | 0.8158 | ok |
| webbase-1M.mtx | 1000005 | 1000005 | 512 | 6.864363 | 4.276296 | 0.6230 | ok |

## 观察

- 当前版本保持 `26/26` 全量正确，并明显优于上一版 `2026-05-31` 早些时候的回退结果。
- `shipsec1.mtx` 目前为 `1.602437 ms`，已经重新追回并略好于 `2026-05-29` 文档中的 `1.605171 ms`。
- 本轮真正可主线化的收益来自两点：
  - 恢复 `val_sidx` 路径的 128-bit G2S 异步搬运，并显式补齐 `cp_async` 元信息，让 `commit/wait` 正确生成。
  - 将 `load_offset_1` 改成 codegen 级局部标量内联：一次计算 `ll/rr`，供两个 coo loader 复用，减少独立 op 的模板与中转开销。

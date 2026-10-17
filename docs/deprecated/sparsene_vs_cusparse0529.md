# SPARSENE vs cuSPARSE (2026-05-29)

## 测试说明

- 对象: 当前最新生成的 `ME_TCF` kernel，不使用智能体，仅默认 heuristic 策略。
- 数据集: `/mnt/lustre-client/zhangkaige/selected_mtx` 下 26 个 `.mtx` 数据集。
- 命令: `RESULT_ROOT=/home/huoda/sparsene/examples/src_fp32/dtc/testbed/build_perf_logs SPARSENE_DISABLE_MTX_CACHE=1 CUDA_ARCH=sm_80 bash scripts/benchmark_inc_multi_dataset.sh`
- 参数: `N=512`, `warmup=20`, `repeat=100`。
- 原始结果: `/home/huoda/sparsene/examples/src_fp32/dtc/testbed/build_perf_logs/multi_dataset_generated_20260529_213506/summary.csv`

## 总体结论

- 成功率: `26/26`，本轮无 correctness error、无 crash、无 timeout。
- 中位 `speedup_vs_cusparse`: `0.8700x`。
- 平均 `speedup_vs_cusparse`: `1.1519x`。
- 快于 cuSPARSE 的数据集个数: `11/26`。
- 最优数据集: `pdb1HYS.mtx`，`speedup_vs_cusparse=2.7230x`。
- 最弱数据集: `dc2.mtx`，`speedup_vs_cusparse=0.0354x`。

## 结果表

| dataset | M | K | N | mykernel_ms | cusparse_ms | speedup_vs_cusparse | status |
|---|---:|---:|---:|---:|---:|---:|---|
| ASIC_680k.mtx | 682862 | 682862 | 512 | 78.088120 | 4.958392 | 0.0635 | ok |
| DD.mtx | 334925 | 334925 | 512 | 2.011249 | 1.604321 | 0.7977 | ok |
| Ga41As41H72.mtx | 268096 | 268096 | 512 | 10.084066 | 10.350265 | 1.0264 | ok |
| OVCAR-8H.mtx | 1889542 | 1889542 | 512 | 10.318623 | 7.753308 | 0.7514 | ok |
| Si41Ge41H72.mtx | 185639 | 185639 | 512 | 7.645276 | 7.990149 | 1.0451 | ok |
| Yeast.mtx | 1710902 | 1710902 | 512 | 7.761940 | 7.196478 | 0.9271 | ok |
| YeastH.mtx | 3138114 | 3138114 | 512 | 16.849531 | 12.841370 | 0.7621 | ok |
| cant.mtx | 62451 | 62451 | 512 | 1.039391 | 1.874975 | 1.8039 | ok |
| conf5_4-8x8-10.mtx | 49152 | 49152 | 512 | 0.677386 | 1.033820 | 1.5262 | ok |
| consph.mtx | 83334 | 83334 | 512 | 1.466962 | 3.076362 | 2.0971 | ok |
| cop20k_A.mtx | 121192 | 121192 | 512 | 1.598413 | 3.261092 | 2.0402 | ok |
| dc2.mtx | 116835 | 116835 | 512 | 22.394194 | 0.793334 | 0.0354 | ok |
| ddi.mtx | 4267 | 4267 | 512 | 0.835707 | 0.896092 | 1.0723 | ok |
| eu-2005.mtx | 862664 | 862664 | 512 | 10.144922 | 9.482086 | 0.9347 | ok |
| mac_econ_fwd500.mtx | 206500 | 206500 | 512 | 1.539512 | 1.101609 | 0.7156 | ok |
| mc2depi.mtx | 525825 | 525825 | 512 | 4.237363 | 2.432379 | 0.5740 | ok |
| mip1.mtx | 66463 | 66463 | 512 | 14.573055 | 4.330455 | 0.2972 | ok |
| pdb1HYS.mtx | 36417 | 36417 | 512 | 0.761303 | 2.072996 | 2.7230 | ok |
| protein.mtx | 132534 | 132534 | 512 | 49.735034 | 39.098010 | 0.7861 | ok |
| pwtk.mtx | 217918 | 217918 | 512 | 2.313953 | 5.646920 | 2.4404 | ok |
| reddit.mtx | 232965 | 232965 | 512 | 87.000703 | 68.357847 | 0.7857 | ok |
| rma10.mtx | 46835 | 46835 | 512 | 0.594196 | 1.191209 | 2.0047 | ok |
| scircuit.mtx | 170998 | 170998 | 512 | 1.216973 | 0.983142 | 0.8079 | ok |
| shipsec1.mtx | 140874 | 140874 | 512 | 1.605171 | 4.007076 | 2.4964 | ok |
| web-BerkStan.mtx | 685230 | 685230 | 512 | 5.217925 | 4.241091 | 0.8128 | ok |
| webbase-1M.mtx | 1000005 | 1000005 | 512 | 6.870221 | 4.276204 | 0.6224 | ok |

## 观察

- 当前新 kernel 相比 `docs/ref_vs_generate.md` 中最初版本已有明显提升，尤其在 `shipsec1.mtx`、`pwtk.mtx`、`cop20k_A.mtx`、`consph.mtx`、`pdb1HYS.mtx` 上能稳定快于 cuSPARSE。
- 仍明显偏慢的数据集集中在 `ASIC_680k.mtx`、`dc2.mtx`、`mip1.mtx`、`YeastH.mtx`、`webbase-1M.mtx`，这些点后续更适合单点 profile 分析。

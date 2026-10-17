# Sparsene generated operators: A800, N=512, FP32

This is the correctness-closed final version selected by `a800_n512_fp32_final_config.json`. GPU benchmarks ran strictly serially.
Format conversion, reorder/inverse mapping, and cuSPARSE preprocess are outside the timing contract; all generated compute/reduction launches are inside.

- Correctness: 26/26 PASS
- Faster than cuSPARSE: 12/26
- Geometric-mean cuSPARSE/generated speedup: 1.0385x
- Warmup/repeat: 10/100

| dataset | logical tile | tile N | segment | generated ms | cuSPARSE ms | speedup | correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASIC_680k | 8x16 | 64 | 32 | 10.2866 | 4.9611 | 0.4823x | PASS |
| DD | 16x16 | 64 | 32 | 2.08319 | 1.62134 | 0.7783x | PASS |
| Ga41As41H72 | 8x32 | 64 | 32 | 21.9329 | 10.5267 | 0.4799x | PASS |
| OVCAR-8H | 16x8 | 128 | 32 | 6.43563 | 7.73835 | 1.2024x | PASS |
| Si41Ge41H72 | 8x32 | 64 | 32 | 17.7688 | 8.12898 | 0.4575x | PASS |
| Yeast | 16x8 | 128 | 32 | 4.93918 | 7.17712 | 1.4531x | PASS |
| YeastH | 32x8 | 128 | 32 | 12.4118 | 12.8467 | 1.0350x | PASS |
| cant | 32x32 | 64 | 32 | 2.73674 | 1.90305 | 0.6954x | PASS |
| conf5_4-8x8-10 | 16x16 | 64 | 32 | 0.581673 | 1.0371 | 1.7830x | PASS |
| consph | 16x8 | 128 | 32 | 0.906721 | 3.10266 | 3.4218x | PASS |
| cop20k_A | 16x8 | 128 | 128 | 1.20623 | 3.25989 | 2.7026x | PASS |
| dc2 | 16x16 | 128 | 32 | 1.47242 | 0.791286 | 0.5374x | PASS |
| ddi | 16x16 | 64 | 64 | 0.550574 | 0.908503 | 1.6501x | PASS |
| eu-2005 | 16x16 | 64 | 32 | 13.3257 | 9.6844 | 0.7267x | PASS |
| mac_econ_fwd500 | 16x16 | 64 | 32 | 1.90772 | 1.11542 | 0.5847x | PASS |
| mc2depi | 16x32 | 64 | 32 | 4.35836 | 2.45195 | 0.5626x | PASS |
| mip1 | 8x16 | 64 | 512 | 2.69732 | 4.37074 | 1.6204x | PASS |
| pdb1HYS | 32x16 | 64 | 32 | 0.91008 | 2.08965 | 2.2961x | PASS |
| protein | 16x16 | 64 | 1024 | 40.7365 | 39.9543 | 0.9808x | PASS |
| pwtk | 16x16 | 64 | 32 | 2.11023 | 5.72116 | 2.7112x | PASS |
| reddit | 16x32 | 64 | 1024 | 90.791 | 70.1146 | 0.7723x | PASS |
| rma10 | 16x16 | 64 | 32 | 0.54999 | 1.2083 | 2.1970x | PASS |
| scircuit | 16x16 | 64 | 32 | 1.92382 | 0.985272 | 0.5121x | PASS |
| shipsec1 | 16x8 | 128 | 32 | 1.89398 | 4.03581 | 2.1309x | PASS |
| web-BerkStan | 16x8 | 128 | 32 | 5.8348 | 4.32507 | 0.7413x | PASS |
| webbase-1M | 16x16 | 64 | 32 | 7.91522 | 4.29918 | 0.5432x | PASS |

Hardware note: these newly measured results are from an NVIDIA A800 80GB PCIe. Historical paper/SOTA tables in this directory are A100 results and must not be mixed into a same-hardware ranking.

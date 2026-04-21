# End2End Smoke Test (DTC)

Use one command block; do not put environment variables on separate lines.

Default backend policy is now no CSR fallback. Recommended TILE_B is auto (runtime chooses 16/32/64 with automatic pad/unpad on N).

```bash
cd /workspace/sparsene/end2end
SPARSENE_SPMM_BACKEND=dtc \
SPARSENE_DTC_VARIANT=base \
SPARSENE_DTC_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/dtc/testbed \
SPARSENE_DTC_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_DTC_TILE_B=auto \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/baselines/dtc_datasets \
    --datasets DD \
    --hidden-list 64 \
    --layer-list 3 \
    --epochs 1 \
    --warmup-epochs 0 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function dtc_spmm \
    --output-csv result/gcn_e2e_dtc_smoke.csv
```

## Build src_fp32 DTC Python Extension

The src_fp32 testbed now includes a concrete pybind extension in
[sparsene/examples/src_fp32/dtc/testbed/DTCSpMM.cpp](sparsene/examples/src_fp32/dtc/testbed/DTCSpMM.cpp) and
[sparsene/examples/src_fp32/dtc/testbed/DTCSpMM_kernel.cu](sparsene/examples/src_fp32/dtc/testbed/DTCSpMM_kernel.cu).

Build once:

```bash
cd /workspace/sparsene/examples/src_fp32/dtc/testbed
python3 setup.py build_ext --inplace
```

Then run one-click validation:

```bash
cd /workspace/sparsene/end2end
SPARSENE_DTC_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/dtc/testbed \
SPARSENE_DTC_MODULE_NAME=DTCSpMM \
SPARSENE_DTC_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_DTC_TILE_B=auto \
bash scripts/smoke_test_dtc.sh
```

## Build src_fp32 SR-BCRS Python Extension

Build once:

```bash
cd /workspace/sparsene/examples/src_fp32/sr_bcrs/testbed
python3 setup.py build_ext --inplace
```

## GCN Variant Test Commands (DTC / multi-bind / strict lb)

In this repo, multi-bind corresponds to `SPARSENE_DTC_VARIANT=multi_binding`.

### 1) DTC base

```bash
cd /workspace/sparsene/end2end
SPARSENE_SPMM_BACKEND=dtc \
SPARSENE_DTC_VARIANT=base \
SPARSENE_DTC_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/dtc/testbed \
SPARSENE_DTC_MODULE_NAME=DTCSpMM \
SPARSENE_DTC_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_DTC_TILE_B=auto \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/baselines/dtc_datasets \
    --datasets DD \
    --hidden-list 128,256,512 \
    --layer-list 3 \
    --epochs 1 \
    --warmup-epochs 0 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function dtc_spmm \
    --output-csv result/gcn_e2e_dtc_base_smoke.csv
```

### 2) DTC multi-bind (`multi_binding`)

```bash
cd /workspace/sparsene/end2end
SPARSENE_SPMM_BACKEND=dtc \
SPARSENE_DTC_VARIANT=multi_binding \
SPARSENE_DTC_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/dtc/testbed \
SPARSENE_DTC_MODULE_NAME=DTCSpMM \
SPARSENE_DTC_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_DTC_TILE_B=auto \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/baselines/dtc_datasets \
    --datasets DD \
    --hidden-list 64 \
    --layer-list 3 \
    --epochs 1 \
    --warmup-epochs 0 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function dtc_spmm \
    --output-csv result/gcn_e2e_dtc_multibind_smoke.csv
```

### 3) DTC strict lb (`strict_lb`)

```bash
cd /workspace/sparsene/end2end
SPARSENE_SPMM_BACKEND=dtc \
SPARSENE_DTC_VARIANT=strict_lb \
SPARSENE_DTC_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/dtc/testbed \
SPARSENE_DTC_MODULE_NAME=DTCSpMM \
SPARSENE_DTC_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_DTC_TILE_B=auto \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/baselines/dtc_datasets \
    --datasets DD \
    --hidden-list 64 \
    --layer-list 3 \
    --epochs 1 \
    --warmup-epochs 0 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function dtc_spmm \
    --output-csv result/gcn_e2e_dtc_strict_lb_smoke.csv
```

### Quick smoke script by variant

```bash
cd /workspace/sparsene/end2end
SPARSENE_DTC_VARIANT=base SPARSENE_DTC_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/dtc/testbed SPARSENE_DTC_MODULE_NAME=DTCSpMM SPARSENE_DTC_ALLOW_NON_SRC_FP32=0 SPARSENE_ALLOW_CSR_FALLBACK=0 SPARSENE_DTC_TILE_B=auto bash scripts/smoke_test_dtc.sh
SPARSENE_DTC_VARIANT=multi_binding SPARSENE_DTC_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/dtc/testbed SPARSENE_DTC_MODULE_NAME=DTCSpMM SPARSENE_DTC_ALLOW_NON_SRC_FP32=0 SPARSENE_ALLOW_CSR_FALLBACK=0 SPARSENE_DTC_TILE_B=auto bash scripts/smoke_test_dtc.sh
SPARSENE_DTC_VARIANT=strict_lb SPARSENE_DTC_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/dtc/testbed SPARSENE_DTC_MODULE_NAME=DTCSpMM SPARSENE_DTC_ALLOW_NON_SRC_FP32=0 SPARSENE_ALLOW_CSR_FALLBACK=0 SPARSENE_DTC_TILE_B=auto bash scripts/smoke_test_dtc.sh
```

More details and API notes are in [sparsene/examples/src_fp32/dtc/testbed/README_python_bind.md](sparsene/examples/src_fp32/dtc/testbed/README_python_bind.md).

## Fair Timing Command (Segmented + Repeat + Preprocess Excluded)

The benchmark script now supports segmented timing and repeated runs:

- `--repeat-runs`: run multiple measured trials and report mean/std.
- `--segment-timing 1`: report forward/backward/optimizer breakdown.
- `--exclude-preprocess 1`: prewarm backend preprocess before timed epochs.
- `--backend-warmup-iters`: number of backend prewarm iterations.

Example (DTC, same setting for fair comparison):

```bash
cd /workspace/sparsene/end2end
SPARSENE_SPMM_BACKEND=dtc \
SPARSENE_DTC_VARIANT=base \
SPARSENE_DTC_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/dtc/testbed \
SPARSENE_DTC_MODULE_NAME=DTCSpMM \
SPARSENE_DTC_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_DTC_TILE_B=auto \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/baselines/dtc_datasets \
    --datasets DD \
    --hidden-list 64 \
    --layer-list 3 \
    --epochs 30 \
    --warmup-epochs 10 \
    --repeat-runs 5 \
    --segment-timing 1 \
    --exclude-preprocess 1 \
    --backend-warmup-iters 2 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function dtc_spmm \
    --output-csv result/gcn_e2e_dtc_fair_timing.csv
```

Example (torch baseline, same timing knobs):

```bash
cd /workspace/sparsene/end2end
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/baselines/dtc_datasets \
    --datasets DD \
    --hidden-list 64 \
    --layer-list 3 \
    --epochs 30 \
    --warmup-epochs 10 \
    --repeat-runs 5 \
    --segment-timing 1 \
    --exclude-preprocess 0 \
    --backend-warmup-iters 0 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend torch \
    --output-csv result/gcn_e2e_torch_fair_timing.csv
```

## All-26-Matrix Commands (DTC / DTC multi-bind / DTC strict-lb / SR-BCRS / SR-BCRS-16x8 / FlashSparse)

Use the matrix list in `/workspace/scripts/sparsene_test_mtx_list.txt` directly.

For current DTC tests, it is recommended to skip `ASIC_680k` first (to avoid the known slow/unstable case while validating other datasets):

```bash
cd /workspace/sparsene/end2end
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' /workspace/scripts/sparsene_test_mtx_list.txt | grep -v '^ASIC_680k$' | paste -sd, -)
echo "$DATASETS" | tr ',' '\n' | wc -l  # expect: 25
```

change the dataset dir: from `/workspace/baselines/dtc_datasets` to `/workspace/scripts/selected_npz` 

### 1) DTC base (all 26)

```bash
cd /workspace/sparsene/end2end
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' /workspace/scripts/sparsene_test_mtx_list.txt | grep -v '^ASIC_680k$' | paste -sd, -)
SPARSENE_SPMM_BACKEND=dtc \
SPARSENE_DTC_VARIANT=base \
SPARSENE_DTC_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/dtc/testbed \
SPARSENE_DTC_MODULE_NAME=DTCSpMM \
SPARSENE_DTC_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_DTC_TILE_B=auto \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/scripts/selected_npz \
    --datasets "$DATASETS" \
    --hidden-list 128,256,512 \
    --layer-list 3 \
    --epochs 100 \
    --warmup-epochs 30 \
    --repeat-runs 2 \
    --segment-timing 1 \
    --exclude-preprocess 1 \
    --backend-warmup-iters 2 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function dtc_spmm \
    --output-csv result/gcn_e2e_dtc_base_26_128-512.csv
```

### 2) DTC multi-bind (`multi_binding`, all 26)

```bash
cd /workspace/sparsene/end2end
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' /workspace/scripts/sparsene_test_mtx_list.txt | grep -v '^ASIC_680k$' | paste -sd, -)
SPARSENE_SPMM_BACKEND=dtc \
SPARSENE_DTC_VARIANT=multi_binding \
SPARSENE_DTC_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/dtc/testbed \
SPARSENE_DTC_MODULE_NAME=DTCSpMM \
SPARSENE_DTC_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_DTC_TILE_B=auto \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/scripts/selected_npz \
    --datasets "$DATASETS" \
    --hidden-list 128 \
    --layer-list 3 \
    --epochs 100 \
    --warmup-epochs 30 \
    --repeat-runs 2 \
    --segment-timing 1 \
    --exclude-preprocess 1 \
    --backend-warmup-iters 2 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function dtc_spmm \
    --output-csv result/gcn_e2e_dtc_multibind_26_128-512.csv
```

### 3) DTC strict-lb (`strict_lb`, all 26)

Note: strict-lb is currently unstable in one-process multi-dataset runs (may hit CUDA illegal memory access).
Use isolated per-dataset processes and then merge CSV rows:

```bash
cd /workspace/sparsene/end2end
set -euo pipefail
mapfile -t DATASETS_ARR < <(sed -E 's#.*/##; s#\.mtx$##' /workspace/scripts/sparsene_test_mtx_list.txt | grep -v '^ASIC_680k$')

rm -f result/_tmp_dtc_strict_lb_*.csv result/gcn_e2e_dtc_strict_lb_26.csv

for ds in "${DATASETS_ARR[@]}"; do
    echo "[RUN] ${ds}"
    SPARSENE_SPMM_BACKEND=dtc \
    SPARSENE_DTC_VARIANT=strict_lb \
    SPARSENE_DTC_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/dtc/testbed \
    SPARSENE_DTC_MODULE_NAME=DTCSpMM \
    SPARSENE_DTC_ALLOW_NON_SRC_FP32=0 \
    SPARSENE_ALLOW_CSR_FALLBACK=0 \
    SPARSENE_DTC_TILE_B=auto \
    python3 gcn/eva_gcn_sparsene.py \
            --dataset-dir /workspace/scripts/selected_npz \
            --datasets "${ds}" \
            --hidden-list 128 \
            --layer-list 3 \
            --epochs 100 \
            --warmup-epochs 30 \
            --repeat-runs 2 \
            --segment-timing 1 \
            --exclude-preprocess 1 \
            --backend-warmup-iters 2 \
            --feature-dim 128 \
            --num-classes 16 \
            --device cuda:0 \
            --backend external \
            --external-module external_backend_spmm \
            --external-function dtc_spmm \
            --output-csv "result/_tmp_dtc_strict_lb_${ds}.csv"
done

first=1
for ds in "${DATASETS_ARR[@]}"; do
    f="result/_tmp_dtc_strict_lb_${ds}.csv"
    if [[ $first -eq 1 ]]; then
        cat "$f" > result/gcn_e2e_dtc_strict_lb_26_128-512.csv
        first=0
    else
        tail -n +2 "$f" >> result/gcn_e2e_dtc_strict_lb_26_128-512.csv
    fi
done

wc -l result/gcn_e2e_dtc_strict_lb_26_128-512.csv
```


**nsys**
```bash
cd /workspace/sparsene/end2end
SPARSENE_SPMM_BACKEND=dtc \
SPARSENE_DTC_VARIANT=strict_lb \
SPARSENE_DTC_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/dtc/testbed \
SPARSENE_DTC_MODULE_NAME=DTCSpMM \
SPARSENE_DTC_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_DTC_TILE_B=auto \
SPARSENE_ENABLE_NVTX=1 \
nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --force-overwrite=true \
    --output result/nsys_dtc_strictlb_mip1_n128 \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/scripts/selected_npz \
    --datasets mip1 \
    --hidden-list 128 \
    --layer-list 3 \
    --epochs 3 \
    --warmup-epochs 0 \
    --repeat-runs 1 \
    --segment-timing 1 \
    --exclude-preprocess 1 \
    --backend-warmup-iters 1 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function dtc_spmm \
    --output-csv result/gcn_e2e_dtc_strictlb_profile_mip1.csv 

nsys stats \
    --report cuda_api_sum,cuda_gpu_kern_sum,cuda_kern_exec_sum \
    --format csv \
    --force-export=true \
    --output result/nsys_dtc_strictlb_mip1_n128 \
    result/nsys_dtc_strictlb_mip1_n128.nsys-rep
```

### 4) SR-BCRS base (all 26)

```bash
cd /workspace/sparsene/end2end
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' /workspace/scripts/sparsene_test_mtx_list.txt | paste -sd, -)
SPARSENE_SPMM_BACKEND=sr_bcrs \
SPARSENE_SRBCRS_VARIANT=base \
SPARSENE_SRBCRS_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/sr_bcrs/testbed \
SPARSENE_SRBCRS_MODULE_NAME=SRBCRSSpMM \
SPARSENE_SRBCRS_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_SRBCRS_TILE_B=auto \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/scripts/selected_npz \
    --datasets "$DATASETS" \
    --hidden-list 128 \
    --layer-list 3 \
    --epochs 100 \
    --warmup-epochs 30 \
    --repeat-runs 2 \
    --segment-timing 1 \
    --exclude-preprocess 1 \
    --backend-warmup-iters 2 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function srbcrs_spmm \
    --output-csv result/gcn_e2e_srbcrs_base_26_128-512.csv
```

### 5) SR-BCRS 16x8 (all 26)

```bash
cd /workspace/sparsene/end2end
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' /workspace/scripts/sparsene_test_mtx_list.txt | paste -sd, -)
SPARSENE_SPMM_BACKEND=sr_bcrs \
SPARSENE_SRBCRS_VARIANT=16x8 \
SPARSENE_SRBCRS_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/sr_bcrs/testbed \
SPARSENE_SRBCRS_MODULE_NAME=SRBCRSSpMM \
SPARSENE_SRBCRS_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_SRBCRS_TILE_B=auto \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/scripts/selected_npz \
    --datasets "$DATASETS" \
    --hidden-list 128 \
    --layer-list 3 \
    --epochs 100 \
    --warmup-epochs 30 \
    --repeat-runs 2 \
    --segment-timing 1 \
    --exclude-preprocess 1 \
    --backend-warmup-iters 2 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function srbcrs_spmm \
    --output-csv result/gcn_e2e_srbcrs_16x8_26_128-512.csv
```

### 5.1) SR-BCRS 16x8 multi-bind / strict-lb (all 26)

`external_backend_spmm` now accepts the following SR-BCRS variants for end2end routing:

- `SPARSENE_SRBCRS_VARIANT=16x8_multi_bind`
- `SPARSENE_SRBCRS_VARIANT=16x8_strict_lb`

Example (16x8 multi-bind):

```bash
cd /workspace/sparsene/end2end
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' /workspace/scripts/sparsene_test_mtx_list.txt | paste -sd, -)
SPARSENE_SPMM_BACKEND=sr_bcrs \
SPARSENE_SRBCRS_VARIANT=16x8_multi_bind \
SPARSENE_SRBCRS_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/sr_bcrs/testbed \
SPARSENE_SRBCRS_MODULE_NAME=SRBCRSSpMM \
SPARSENE_SRBCRS_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_SRBCRS_TILE_B=auto \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/scripts/selected_npz \
    --datasets "$DATASETS" \
    --hidden-list 128 \
    --layer-list 3 \
    --epochs 100 \
    --warmup-epochs 30 \
    --repeat-runs 2 \
    --segment-timing 1 \
    --exclude-preprocess 1 \
    --backend-warmup-iters 2 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function srbcrs_spmm \
    --output-csv result/gcn_e2e_srbcrs_16x8_multibind_26_128-512.csv
```

Example (16x8 strict-lb):

```bash
cd /workspace/sparsene/end2end
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' /workspace/scripts/sparsene_test_mtx_list.txt | paste -sd, -)
SPARSENE_SPMM_BACKEND=sr_bcrs \
SPARSENE_SRBCRS_VARIANT=16x8_strict_lb \
SPARSENE_SRBCRS_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/sr_bcrs/testbed \
SPARSENE_SRBCRS_MODULE_NAME=SRBCRSSpMM \
SPARSENE_SRBCRS_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_SRBCRS_TILE_B=auto \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/scripts/selected_npz \
    --datasets "$DATASETS" \
    --hidden-list 128 \
    --layer-list 3 \
    --epochs 100 \
    --warmup-epochs 30 \
    --repeat-runs 2 \
    --segment-timing 1 \
    --exclude-preprocess 1 \
    --backend-warmup-iters 2 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function srbcrs_spmm \
    --output-csv result/gcn_e2e_srbcrs_16x8_strict_lb_26_128.csv
```

**SRBCRS 16x8 nsys**

```bash
cd /workspace/sparsene/end2end
SPARSENE_SPMM_BACKEND=sr_bcrs \
SPARSENE_SRBCRS_VARIANT=16x8 \
SPARSENE_SRBCRS_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/sr_bcrs/testbed \
SPARSENE_SRBCRS_MODULE_NAME=SRBCRSSpMM \
SPARSENE_SRBCRS_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_SRBCRS_TILE_B=auto \
SPARSENE_ENABLE_NVTX=1 \
nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --force-overwrite=true \
    --output result/nsys_srbcrs16x8_pdb1HYS_n128 \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/scripts/selected_npz \
    --datasets pdb1HYS \
    --hidden-list 128 \
    --layer-list 3 \
    --epochs 3 \
    --warmup-epochs 0 \
    --repeat-runs 1 \
    --segment-timing 1 \
    --exclude-preprocess 1 \
    --backend-warmup-iters 1 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function srbcrs_spmm \
    --output-csv result/gcn_e2e_srbcrs16x8_profile_pdb1HYS.csv 

nsys stats \
    --report cuda_api_sum,cuda_gpu_kern_sum,cuda_kern_exec_sum \
    --format csv \
    --force-export=true \
    --output result/nsys_srbcrs16x8_pdb1HYS_n128 \
    result/nsys_srbcrs16x8_pdb1HYS_n128.nsys-rep
```

```bash
cd /workspace/sparsene/end2end
SPARSENE_EXTERNAL_RESET_PER_CASE=1 \
SPARSENE_SPMM_BACKEND=sr_bcrs \
SPARSENE_SRBCRS_VARIANT=16x8 \
SPARSENE_SRBCRS_SOURCE_ROOT=/workspace/sparsene/examples/src_fp32/sr_bcrs/testbed \
SPARSENE_SRBCRS_MODULE_NAME=SRBCRSSpMM \
SPARSENE_SRBCRS_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_SRBCRS_TILE_B=auto \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/scripts/selected_npz \
    --datasets pdb1HYS \
    --hidden-list 128 \
    --layer-list 3 \
    --epochs 100 \
    --warmup-epochs 30 \
    --repeat-runs 2 \
    --segment-timing 1 \
    --exclude-preprocess 1 \
    --backend-warmup-iters 2 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function srbcrs_spmm \
    --output-csv result/gcn_e2e_srbcrs16x8_stable_pdb1HYS.csv
```


### 6) FlashSparse (all 26)

Note: FlashSparse may be unstable in one-process multi-dataset runs (can hit CUDA illegal memory access).
Use isolated per-dataset processes and then merge CSV rows:

```bash
cd /workspace/sparsene/end2end
set -euo pipefail
mapfile -t DATASETS_ARR < <(sed -E 's#.*/##; s#\.mtx$##' /workspace/scripts/sparsene_test_mtx_list.txt)

rm -f result/_tmp_flashsparse_*.csv result/gcn_e2e_flashsparse_26.csv

for ds in "${DATASETS_ARR[@]}"; do
    echo "[RUN] ${ds}"
    SPARSENE_SPMM_BACKEND=flashsparse \
    SPARSENE_FLASHSPARSE_VARIANT=tf32_balance \
    SPARSENE_FLASHSPARSE_SOURCE_ROOT=/workspace/baselines/FlashSparse/FlashSparse \
    SPARSENE_FLASHSPARSE_ALLOW_NON_DEFAULT_ROOT=1 \
    python3 gcn/eva_gcn_sparsene.py \
            --dataset-dir /workspace/scripts/selected_npz \
            --datasets "${ds}" \
            --hidden-list 128 \
            --layer-list 3 \
            --epochs 100 \
            --warmup-epochs 30 \
            --repeat-runs 2 \
            --segment-timing 1 \
            --exclude-preprocess 1 \
            --backend-warmup-iters 2 \
            --feature-dim 128 \
            --num-classes 16 \
            --device cuda:0 \
            --backend external \
            --external-module external_backend_spmm \
            --external-function flashsparse_spmm \
            --output-csv "result/_tmp_flashsparse_${ds}.csv"
done

first=1
for ds in "${DATASETS_ARR[@]}"; do
    f="result/_tmp_flashsparse_${ds}.csv"
    if [[ $first -eq 1 ]]; then
        cat "$f" > result/gcn_e2e_flashsparse_26_128-512.csv
        first=0
    else
        tail -n +2 "$f" >> result/gcn_e2e_flashsparse_26_128-512.csv
    fi
done

wc -l result/gcn_e2e_flashsparse_26_128-512.csv
```

**Flash Sparse nsys**

```bash
cd /workspace/sparsene/end2end
SPARSENE_SPMM_BACKEND=flashsparse \
SPARSENE_FLASHSPARSE_VARIANT=tf32_balance \
SPARSENE_FLASHSPARSE_SOURCE_ROOT=/workspace/baselines/FlashSparse/FlashSparse \
SPARSENE_FLASHSPARSE_ALLOW_NON_DEFAULT_ROOT=1 \
SPARSENE_ENABLE_NVTX=1 \
nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --force-overwrite=true \
    --output result/nsys_flashsparse_mip1_n128 \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/scripts/selected_npz \
    --datasets mip1 \
    --hidden-list 128 \
    --layer-list 3 \
    --epochs 3 \
    --warmup-epochs 0 \
    --repeat-runs 1 \
    --segment-timing 1 \
    --exclude-preprocess 1 \
    --backend-warmup-iters 1 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function flashsparse_spmm \
    --output-csv result/gcn_e2e_flashsparse_profile_mip1.csv

nsys stats \
    --report cuda_api_sum,cuda_gpu_kern_sum,cuda_kern_exec_sum \
    --format csv \
    --force-export=true \
    --output result/nsys_flashsparse_mip1_n128 \
    result/nsys_flashsparse_mip1_n128.nsys-rep
```

```bash
cd /workspace/sparsene/end2end
SPARSENE_EXTERNAL_RESET_PER_CASE=1 \
SPARSENE_SPMM_BACKEND=flashsparse \
SPARSENE_FLASHSPARSE_VARIANT=tf32_balance \
SPARSENE_FLASHSPARSE_SOURCE_ROOT=/workspace/baselines/FlashSparse/FlashSparse \
SPARSENE_FLASHSPARSE_ALLOW_NON_DEFAULT_ROOT=1 \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/scripts/selected_npz \
    --datasets pdb1HYS \
    --hidden-list 128 \
    --layer-list 3 \
    --epochs 100 \
    --warmup-epochs 30 \
    --repeat-runs 2 \
    --segment-timing 1 \
    --exclude-preprocess 1 \
    --backend-warmup-iters 2 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function flashsparse_spmm \
    --output-csv result/gcn_e2e_flashsparse_stable_pdb1HYS.csv
```

### DTC-SpMM (Origin Paper)

Use the original ASPLOS'24 DTC extension as baseline in current end2end pipeline.

Build once:

```bash
cd /workspace/baselines/DTC-SpMM_ASPLOS24
source init_dtc.sh
source third_party/init_sputnik.sh
# First-time dependency build (skip if libs already exist):
# source third_party/build_sputnik.sh
cd DTC-SpMM
python3 setup.py build_ext --inplace
```

Run all datasets except `ASIC_680k` (recommended for current DTC origin test):

```bash
cd /workspace/sparsene/end2end
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' /workspace/scripts/sparsene_test_mtx_list.txt | grep -v '^ASIC_680k$' | paste -sd, -)
SPARSENE_SPMM_BACKEND=dtc \
SPARSENE_DTC_VARIANT=base \
SPARSENE_DTC_SOURCE_ROOT=/workspace/baselines/DTC-SpMM_ASPLOS24/DTC-SpMM \
SPARSENE_DTC_MODULE_NAME=DTCSpMM \
SPARSENE_DTC_ALLOW_NON_SRC_FP32=1 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_DTC_TILE_B=auto \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/scripts/selected_npz \
    --datasets "$DATASETS" \
    --hidden-list 128 \
    --layer-list 3 \
    --epochs 100 \
    --warmup-epochs 30 \
    --repeat-runs 2 \
    --segment-timing 1 \
    --exclude-preprocess 1 \
    --backend-warmup-iters 2 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function dtc_spmm \
    --output-csv result/gcn_e2e_dtc_origin_26_n128.csv
```

### PyG

Install dependencies if needed:

```bash
pip install torch_geometric
```

Run all 26 datasets with native PyG layers:

```bash
cd /workspace/sparsene/end2end
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' /workspace/scripts/sparsene_test_mtx_list.txt | paste -sd, -)
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/scripts/selected_npz \
    --datasets "$DATASETS" \
    --hidden-list 128 \
    --layer-list 3 \
    --epochs 100 \
    --warmup-epochs 30 \
    --repeat-runs 2 \
    --segment-timing 1 \
    --exclude-preprocess 0 \
    --backend-warmup-iters 0 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend pyg \
    --output-csv result/gcn_e2e_pyg_26_n128.csv
```

### DGL

Recommended: use a dedicated venv for CUDA DGL (validated in this workspace).

```bash
python3 -m pip install -U virtualenv
python3 -m virtualenv /workspace/venv_dgl
source /workspace/venv_dgl/bin/activate

# Install a CUDA-enabled PyTorch stack first.
python -m pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install CUDA DGL (nightly wheel index used by current container setup).
python -m pip install --pre dgl -f https://data.dgl.ai/wheels-test/torch-2.4/cu124/repo.html
```

Verify DGL CUDA support first (required for `--device cuda:0`):

```bash
/workspace/venv_dgl/bin/python - <<'PY'
import torch
import dgl
import dgl.nn.pytorch as dglnn

print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'cuda_available:', torch.cuda.is_available())
print('dgl:', dgl.__version__)

dev = torch.device('cuda:0')
g = dgl.graph((torch.tensor([0], device=dev), torch.tensor([0], device=dev)), num_nodes=1, device=dev)
x = torch.ones((1, 1), device=dev)
conv = dglnn.GraphConv(1, 1, norm='none', weight=False, bias=False, allow_zero_in_degree=True).to(dev)
_ = conv(g, x)
print('DGL CUDA check: OK')
PY
```

If it fails with messages like `Device API cuda is not enabled` or `SpMM does not support cuda device`, your current DGL is CPU-only. Install a CUDA-enabled DGL build matching your CUDA/PyTorch setup, or run DGL baseline with `--device cpu`.

Quick smoke test (DD, 1 epoch):

```bash
cd /workspace/sparsene/end2end
/workspace/venv_dgl/bin/python gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/scripts/selected_npz \
    --datasets DD \
    --hidden-list 128 \
    --layer-list 3 \
    --epochs 1 \
    --warmup-epochs 0 \
    --repeat-runs 1 \
    --segment-timing 1 \
    --exclude-preprocess 0 \
    --backend-warmup-iters 0 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend dgl \
    --output-csv result/_tmp_dgl_cuda_venv_smoke.csv
```

Run all 26 datasets with native DGL layers:

```bash
cd /workspace/sparsene/end2end
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' /workspace/scripts/sparsene_test_mtx_list.txt | paste -sd, -)
/workspace/venv_dgl/bin/python gcn/eva_gcn_sparsene.py \
    --dataset-dir /workspace/scripts/selected_npz \
    --datasets "$DATASETS" \
    --hidden-list 128 \
    --layer-list 3 \
    --epochs 100 \
    --warmup-epochs 30 \
    --repeat-runs 2 \
    --segment-timing 1 \
    --exclude-preprocess 0 \
    --backend-warmup-iters 0 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend dgl \
    --output-csv result/gcn_e2e_dgl_26_n128.csv
```
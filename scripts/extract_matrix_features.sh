#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTBED="${ROOT}/examples/src_fp32/dtc/testbed"
INPUT_DIR="${1:-${INPUT_DIR:-/mnt/lustre-client/zhangkaige/selected_mtx}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-/mnt/lustre-client/huoda/selected_mtx}}"
CUDA_ARCH="${CUDA_ARCH:-sm_80}"
HOST_CXX="${NVCC_CCBIN:-/opt/rh/gcc-toolset-13/root/usr/bin/g++}"
TIMEOUT_SEC="${FEATURE_TIMEOUT_SEC:-1800}"
BUILD_DIR="${ROOT}/examples/src_fp32/dtc/testbed/build_features"
BIN="${BUILD_DIR}/featurize_mtx_v3"

mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs"

nvcc -O3 -std=c++17 -arch="${CUDA_ARCH}" -ccbin "${HOST_CXX}" \
    "-I${TESTBED}" "-I${TESTBED}/src" \
    "${TESTBED}/featurize_mtx.cu" "${TESTBED}/src/mmio.cu" \
    -o "${BIN}"

mapfile -t matrices < <(find "${INPUT_DIR}" -maxdepth 1 -type f -name '*.mtx' -print | sort)
if [[ ${#matrices[@]} -eq 0 ]]; then
    echo "No .mtx files found in ${INPUT_DIR}" >&2
    exit 1
fi

manifest="${OUTPUT_DIR}/features_manifest.csv"
echo "dataset,input_path,feature_path,sha256,status" > "${manifest}"

for matrix in "${matrices[@]}"; do
    name="$(basename "${matrix}" .mtx)"
    output="${OUTPUT_DIR}/${name}.features.v3.json"
    temporary="${output}.tmp.$$"
    stderr_log="${OUTPUT_DIR}/logs/${name}.stderr.log"
    echo "[feature] ${name}"
    status="ok"
    if timeout "${TIMEOUT_SEC}" "${BIN}" "${matrix}" > "${temporary}" 2> "${stderr_log}" && \
       python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["schema_version"] == "sparsene.matrix_features.v3"; assert p["data_quality"]["invalid_entries"] == 0; assert len(p["hierarchical_candidates"]) == 9' "${temporary}"; then
        mv -f "${temporary}" "${output}"
        digest="$(sha256sum "${output}" | awk '{print $1}')"
    else
        status="failed"
        digest=""
        rm -f "${temporary}"
    fi
    echo "${name},${matrix},${output},${digest},${status}" >> "${manifest}"
done

failed="$(awk -F, 'NR > 1 && $5 != "ok" {count++} END {print count+0}' "${manifest}")"
echo "Feature extraction complete: total=${#matrices[@]} failed=${failed} manifest=${manifest}"
[[ "${failed}" -eq 0 ]]

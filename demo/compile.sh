#!/bin/bash

pushd /work/sparsene/examples
rm -rf build
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
make demo-ablation demo-cusp-fp16 -j
pwd
ls -la bin
popd
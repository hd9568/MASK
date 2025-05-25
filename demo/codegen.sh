#!/bin/bash

python /work/sparsene/examples/src_fp16/acc/testbed/acc_ops.py
python /work/sparsene/examples/src_fp16/acc/testbed/acc_ops_nopipe.py
python /work/sparsene/examples/src_fp16/acc/testbed/acc_ops_nopipe_nopre.py
python /work/sparsene/examples/src_fp16/acc/testbed/acc_ops_nopipe_nopre_noswz.py
python /work/sparsene/examples/src_fp16/bitbsr/testbed/bitbsr_ops.py
python /work/sparsene/examples/src_fp16/bitbsr/testbed/bitbsr_ops_nopipe.py
python /work/sparsene/examples/src_fp16/bitbsr/testbed/bitbsr_ops_nopipe_nopre.py
python /work/sparsene/examples/src_fp16/bitbsr/testbed/bitbsr_ops_nopipe_nopre_noswz.py
python /work/sparsene/examples/src_fp16/dtc/testbed/dtc_ops.py
python /work/sparsene/examples/src_fp16/dtc/testbed/dtc_ops_nopipe.py
python /work/sparsene/examples/src_fp16/dtc/testbed/dtc_ops_nopipe_nopre.py
python /work/sparsene/examples/src_fp16/dtc/testbed/dtc_ops_nopipe_nopre_noswz.py
python /work/sparsene/examples/src_fp16/sr_bcrs/testbed/sr_bcrs_ops.py
python /work/sparsene/examples/src_fp16/sr_bcrs/testbed/sr_bcrs_ops_nopipe.py
python /work/sparsene/examples/src_fp16/sr_bcrs/testbed/sr_bcrs_ops_nopipe_nopre.py
python /work/sparsene/examples/src_fp16/sr_bcrs/testbed/sr_bcrs_ops_nopipe_nopre_noswz.py

ls -la /work/sparsene/examples/src_fp16/acc/testbed/kernel.inc
ls -la /work/sparsene/examples/src_fp16/acc/testbed/kernel_nopipe.inc
ls -la /work/sparsene/examples/src_fp16/acc/testbed/kernel_nopipe_nopre.inc
ls -la /work/sparsene/examples/src_fp16/acc/testbed/kernel_nopipe_nopre_noswz.inc
ls -la /work/sparsene/examples/src_fp16/bitbsr/testbed/kernel.inc
ls -la /work/sparsene/examples/src_fp16/bitbsr/testbed/kernel_nopipe.inc
ls -la /work/sparsene/examples/src_fp16/bitbsr/testbed/kernel_nopipe_nopre.inc
ls -la /work/sparsene/examples/src_fp16/bitbsr/testbed/kernel_nopipe_nopre_noswz.inc
ls -la /work/sparsene/examples/src_fp16/dtc/testbed/kernel.inc
ls -la /work/sparsene/examples/src_fp16/dtc/testbed/kernel_nopipe.inc
ls -la /work/sparsene/examples/src_fp16/dtc/testbed/kernel_nopipe_nopre.inc
ls -la /work/sparsene/examples/src_fp16/dtc/testbed/kernel_nopipe_nopre_noswz.inc
ls -la /work/sparsene/examples/src_fp16/sr_bcrs/testbed/kernel.inc
ls -la /work/sparsene/examples/src_fp16/sr_bcrs/testbed/kernel_nopipe.inc
ls -la /work/sparsene/examples/src_fp16/sr_bcrs/testbed/kernel_nopipe_nopre.inc
ls -la /work/sparsene/examples/src_fp16/sr_bcrs/testbed/kernel_nopipe_nopre_noswz.inc
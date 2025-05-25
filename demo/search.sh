#!/bin/bash

pushd /work/sparsene/examples/src_fp16/dtc/testbed
python scripts/plan_searcher.py

ls plans plans.txt
popd
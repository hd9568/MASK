# Matrix Feature Extraction v3

## Goal

The feature contract must describe costs that the current hierarchical-format
compiler and host preprocessing actually create. In particular, ME-TCF first
groups rows, condenses the union of columns inside each row window, and only
then partitions the condensed axis into TC blocks. Statistics over regular
`(row / BLK_M, col / BLK_K)` blocks do not model this layout.

## Design

`examples/src_fp32/dtc/testbed/featurize_mtx.cu` remains a CUDA executable.
CUDA builds the expanded row and column histograms; format-aware hierarchy,
locality, reorder proxies, and storage estimates are derived on the host from
the same expanded COO structure. This hybrid split avoids materializing an
`M*K` grid and keeps memory proportional to NNZ.

The v3 schema contains:

- exact `M`, `K`, file NNZ, symmetry-expanded NNZ, structural NNZ, invalid and
  duplicate entry counts;
- high-precision density/sparsity;
- row/column distributions with tail percentiles, CV, Gini, empty fraction,
  and top-1-percent share;
- row spans, column gaps, normalized bandwidth, and diagonal-band fraction;
- adjacent-row Jaccard and a deterministic first-column reorder proxy;
- exact ME-TCF simulations for `BLK_M={8,16,32}` and
  `BLK_K={8,16,32}`;
- per-candidate condensed unique columns, TC blocks, useful density, Tensor
  Core zero work, COO padding-to-8, padded B indices, column reuse, load
  balance, and estimated COO/mask/dense storage.

The MatrixMarket reader also accepts `real`, `integer`, `pattern`, and
`complex` coordinate fields. This matters for `mc2depi`: the previous shared
reader returned success for `integer` input without filling its coordinate
arrays.

`candidate_block_stats` is retained as a compatibility alias, but its contents
now use condensed hierarchical semantics. New consumers should use
`hierarchical_candidates`.

## Interpretation

- `tensorcore_zero_fraction` estimates arithmetic redundancy after column
  condensation, not sparsity in the original coordinate grid.
- `coo_padding_ratio` follows the host's per-TC-block multiple-of-eight COO
  padding.
- `b_index_padding_ratio` estimates useless `val_sidx` lanes and gathered B
  rows caused by an incomplete final TC block.
- storage numbers are comparable estimates, not guarantees of the final
  codegen ABI for every experimental atomic format.
- reorder metrics are decision features only. Reordering remains unavailable
  in the current integrated generated-kernel route.

## Batch workflow

Run:

```bash
bash scripts/extract_matrix_features.sh \
  /mnt/lustre-client/zhangkaige/selected_mtx \
  /mnt/lustre-client/huoda/selected_mtx
```

The persistent store is `/mnt/lustre-client/huoda/selected_mtx` with filenames
`<dataset>.features.v3.json`, a `features_manifest.csv`, and per-dataset stderr
logs. Outputs are written through a temporary file, JSON/schema validated,
then atomically renamed. Batch validation also rejects nonzero invalid-entry
counts and requires all nine hierarchy candidates, preventing a parser failure
from being persisted as a plausible all-zero feature file.

## Validation requirements

For extractor changes, record NVCC compilation, a small general matrix, a
symmetric matrix, an empty-row case, duplicate coordinates, schema validation,
and the 26-dataset manifest. Compare at least one candidate against
`buildMETCFMetadata()` totals when a GPU testbed is available.

## Validation evidence (2026-07-15)

- NVCC compilation succeeded for `sm_80` with GCC 13.
- CUDA execution succeeded on an NVIDIA A800 80GB PCIe (compute capability
  8.0).
- Synthetic tests covered a rectangular general matrix, an empty row, a
  duplicate coordinate, and symmetry expansion; all JSON assertions passed.
- All 26 selected matrices produced schema-v3 JSON. The manifest contains
  26 `ok` rows, and every recorded SHA256 matches its output file.
- For every one of the 26 matrices, candidate `16x8` exactly matches the
  historical ME-TCF `tbk8.log` triple
  `(total_TCBlock_num, nnz, nnz_pad)`. This validates row-window condensation,
  symmetry expansion, and per-block padding semantics against the actual host
  preprocessing path.
- Final feature payload: 26 files, 756,510 bytes total, stored at
  `/mnt/lustre-client/huoda/selected_mtx`.

## Known limitations

- Matrix numerical values are intentionally ignored because the current format
  design is structural.
- The deterministic reorder proxy is not an expensive graph/hypergraph
  partitioner and should not be interpreted as the best achievable reorder.
- Feature extraction does not include dense-B column count `N`; that is a
  workload feature and must be supplied separately to strategy decisions.

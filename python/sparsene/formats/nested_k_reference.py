"""Reference host ABI for window-condensed, sparsified inner-K tiles.

This is an offline specification and correctness oracle, not a GPU backend.
The active-child decision is made *after* splitting the parent tile into
fixed-width children. This is deliberately distinct from sparsifying scalar
columns and then splitting the compressed scalar sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class NestedKPayload:
    shape: tuple[int, int]
    blk_m: int
    blk_k: int
    blk_k_i: int
    leaf: Literal['dense', 'coo']
    window_block_ptr: np.ndarray
    block_child_ptr: np.ndarray
    block_column_ids: np.ndarray
    child_index: np.ndarray
    values: np.ndarray
    child_value_ptr: np.ndarray | None = None
    local_rows: np.ndarray | None = None
    local_cols: np.ndarray | None = None


def pack_nested_k(matrix, *, blk_m: int, blk_k: int, blk_k_i: int,
                  leaf: Literal['dense', 'coo'] = 'dense') -> NestedKPayload:
    """Pack A; stable row-window and ascending global-column order.

    Each active child covers a fixed range in its parent's condensed columns;
    nonexistent boundary columns are -1. Duplicates are summed by CSR, and
    explicit zero values are removed. No matrix-wide dense array is constructed.
    """
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
           for v in (blk_m, blk_k, blk_k_i)) or blk_k % blk_k_i:
        raise ValueError('tile sizes must be positive integers and blk_k divisible by blk_k_i')
    if leaf not in ('dense', 'coo'):
        raise ValueError('supported reference leaf encodings are dense and coo')
    a = sparse.csr_matrix(matrix, copy=True)
    a.sum_duplicates(); a.eliminate_zeros(); a.sort_indices()
    if not np.issubdtype(a.dtype, np.number):
        raise ValueError('matrix must contain numeric values')
    nrows, ncols = a.shape
    window_ptr = [0]; child_ptr = [0]; columns = []; child_ids = []
    dense_values = []; coo_values = []; coo_rows = []; coo_cols = []; value_ptr = [0]
    for begin in range(0, nrows, blk_m):
        window = a[begin:min(begin + blk_m, nrows)].tocoo()
        unique_cols = np.unique(window.col)
        positions = np.searchsorted(unique_cols, window.col)
        for base in range(0, len(unique_cols), blk_k):
            col_ids = np.full(blk_k, -1, dtype=np.int64)
            valid_cols = unique_cols[base:base + blk_k]
            col_ids[:len(valid_cols)] = valid_cols
            columns.append(col_ids)
            mask = (positions >= base) & (positions < base + blk_k)
            r, c, v = window.row[mask], positions[mask] - base, window.data[mask]
            for child in range(blk_k // blk_k_i):
                in_child = (c >= child * blk_k_i) & (c < (child + 1) * blk_k_i)
                if not np.any(in_child):
                    continue
                child_ids.append(child)
                rr = r[in_child].astype(np.int32, copy=False)
                cc = (c[in_child] - child * blk_k_i).astype(np.int32, copy=False)
                vv = v[in_child]
                if leaf == 'dense':
                    dense = np.zeros((blk_m, blk_k_i), dtype=a.dtype)
                    dense[rr, cc] = vv
                    dense_values.append(dense)
                else:
                    coo_rows.extend(rr); coo_cols.extend(cc); coo_values.extend(vv)
                    value_ptr.append(len(coo_values))
            child_ptr.append(len(child_ids))
        window_ptr.append(len(columns))
    return NestedKPayload(
        shape=(nrows, ncols), blk_m=blk_m, blk_k=blk_k, blk_k_i=blk_k_i,
        leaf=leaf, window_block_ptr=np.asarray(window_ptr, dtype=np.int64),
        block_child_ptr=np.asarray(child_ptr, dtype=np.int64),
        block_column_ids=np.asarray(columns, dtype=np.int64).reshape(-1, blk_k),
        child_index=np.asarray(child_ids, dtype=np.int32),
        values=(np.asarray(dense_values, dtype=a.dtype).reshape(-1, blk_m, blk_k_i)
                if leaf == 'dense' else np.asarray(coo_values, dtype=a.dtype)),
        child_value_ptr=None if leaf == 'dense' else np.asarray(value_ptr, dtype=np.int64),
        local_rows=None if leaf == 'dense' else np.asarray(coo_rows, dtype=np.int32),
        local_cols=None if leaf == 'dense' else np.asarray(coo_cols, dtype=np.int32),
    )


def unpack_nested_k(p: NestedKPayload):
    """Reconstruct a CSR matrix, rejecting invalid metadata and padding."""
    nw = (p.shape[0] + p.blk_m - 1) // p.blk_m
    nb = len(p.block_column_ids); nc = len(p.child_index)
    if (len(p.window_block_ptr) != nw + 1 or len(p.block_child_ptr) != nb + 1
        or p.window_block_ptr[0] != 0 or p.window_block_ptr[-1] != nb
        or p.block_child_ptr[0] != 0 or p.block_child_ptr[-1] != nc
        or np.any(np.diff(p.window_block_ptr) < 0) or np.any(np.diff(p.block_child_ptr) < 0)
        or p.block_column_ids.shape != (nb, p.blk_k)):
        raise ValueError('invalid nested offsets or column metadata')
    if p.leaf == 'dense':
        if p.values.shape != (nc, p.blk_m, p.blk_k_i):
            raise ValueError('invalid dense leaf shape')
    elif p.leaf == 'coo':
        if (p.child_value_ptr is None or p.local_rows is None or p.local_cols is None
            or len(p.child_value_ptr) != nc + 1 or p.child_value_ptr[0] != 0
            or p.child_value_ptr[-1] != len(p.values)
            or np.any(np.diff(p.child_value_ptr) < 0)
            or len(p.local_rows) != len(p.values) or len(p.local_cols) != len(p.values)):
            raise ValueError('invalid COO leaf metadata')
    else:
        raise ValueError('unsupported leaf')
    out_r = []; out_c = []; out_v = []
    for w in range(nw):
        for b in range(p.window_block_ptr[w], p.window_block_ptr[w + 1]):
            previous = -1
            for child in range(p.block_child_ptr[b], p.block_child_ptr[b + 1]):
                idx = int(p.child_index[child])
                if idx <= previous or not 0 <= idx < p.blk_k // p.blk_k_i:
                    raise ValueError('invalid or unordered child index')
                previous = idx
                if p.leaf == 'dense':
                    rr, cc = np.nonzero(p.values[child]); vv = p.values[child, rr, cc]
                else:
                    lo, hi = p.child_value_ptr[child:child + 2]
                    rr, cc, vv = p.local_rows[lo:hi], p.local_cols[lo:hi], p.values[lo:hi]
                if np.any(rr < 0) or np.any(rr >= p.blk_m) or np.any(cc < 0) or np.any(cc >= p.blk_k_i):
                    raise ValueError('leaf coordinate out of bounds')
                global_r = w * p.blk_m + rr
                global_c = p.block_column_ids[b, idx * p.blk_k_i + cc]
                if np.any(global_r >= p.shape[0]) or np.any(global_c < 0) or np.any(global_c >= p.shape[1]):
                    raise ValueError('nonzero in padded or out-of-range position')
                out_r.extend(global_r); out_c.extend(global_c); out_v.extend(vv)
    return sparse.coo_matrix((np.asarray(out_v, dtype=p.values.dtype), (out_r, out_c)), shape=p.shape).tocsr()

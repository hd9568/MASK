"""CPU specification for recursively split, sparsified K-block hierarchies.

Reference data construction and decoding only. This module does not select a
GPU backend, implement NVIR lowering, or promise Tensor Core compatibility.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class MultiLevelKPayload:
    shape: tuple[int, int]
    blk_m: int
    widths: tuple[int, ...]
    leaf: Literal['dense', 'coo']
    window_root_ptr: np.ndarray
    root_column_ids: np.ndarray
    # For level d, ptrs[d][parent:parent+2] indexes indices[d].
    # The final level's nodes index the leaf data directly.
    child_ptrs: tuple[np.ndarray, ...]
    child_indices: tuple[np.ndarray, ...]
    values: np.ndarray
    leaf_value_ptr: np.ndarray | None = None
    local_rows: np.ndarray | None = None
    local_cols: np.ndarray | None = None


def pack_multilevel_k(matrix, *, blk_m: int, widths: tuple[int, ...],
                      leaf: Literal['dense', 'coo'] = 'dense') -> MultiLevelKPayload:
    """Condense each row window, recursively split, then prune empty children.

    widths[0] is the parent width and each later width divides its parent. All
    coordinates refer to a single fixed column order within each row window;
    each leaf stores columns relative to its own block. Values at duplicate
    coordinates are added and zeros removed before structure is determined.
    """
    widths = tuple(widths)
    if (isinstance(blk_m, bool) or not isinstance(blk_m, int) or blk_m <= 0
        or not widths or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in widths)
        or any(a % b for a, b in zip(widths, widths[1:]))):
        raise ValueError('positive integer tile widths must divide their parents')
    if leaf not in ('dense', 'coo'):
        raise ValueError('reference supports dense and coo leaves')
    a = sparse.csr_matrix(matrix, copy=True)
    a.sum_duplicates(); a.eliminate_zeros(); a.sort_indices()
    roots = []; root_cols = []; window_ptr = [0]
    for row0 in range(0, a.shape[0], blk_m):
        window = a[row0:min(row0 + blk_m, a.shape[0])].tocoo()
        cols = np.unique(window.col)
        coords = np.searchsorted(cols, window.col)
        for off in range(0, len(cols), widths[0]):
            mask = (coords >= off) & (coords < off + widths[0])
            ids = np.full(widths[0], -1, dtype=np.int64)
            ids[:len(cols[off:off + widths[0]])] = cols[off:off + widths[0]]
            root_cols.append(ids)
            roots.append((window.row[mask], coords[mask] - off, window.data[mask]))
        window_ptr.append(len(roots))
    current = roots
    ptrs = []; indices = []
    for width in widths[1:]:
        following = []; ptr = [0]; idx = []
        for rows, cols, vals in current:
            # Sorted unique block ids give deterministic child ordering.
            for i in np.unique(cols // width):
                take = (cols // width == i)
                following.append((rows[take], cols[take] - i * width, vals[take]))
                idx.append(i)
            ptr.append(len(following))
        ptrs.append(np.asarray(ptr, dtype=np.int64))
        indices.append(np.asarray(idx, dtype=np.int32))
        current = following
    out_vals = []; out_rows = []; out_cols = []; val_ptr = [0]
    for rows, cols, vals in current:
        if leaf == 'dense':
            tile = np.zeros((blk_m, widths[-1]), dtype=a.dtype)
            tile[rows, cols] = vals
            out_vals.append(tile)
        else:
            out_rows.extend(rows); out_cols.extend(cols); out_vals.extend(vals)
            val_ptr.append(len(out_vals))
    return MultiLevelKPayload(
        shape=tuple(a.shape), blk_m=blk_m, widths=widths, leaf=leaf,
        window_root_ptr=np.asarray(window_ptr, dtype=np.int64),
        root_column_ids=np.asarray(root_cols, dtype=np.int64).reshape(-1, widths[0]),
        child_ptrs=tuple(ptrs), child_indices=tuple(indices),
        values=(np.asarray(out_vals, dtype=a.dtype).reshape(-1, blk_m, widths[-1])
                if leaf == 'dense' else np.asarray(out_vals, dtype=a.dtype)),
        leaf_value_ptr=None if leaf == 'dense' else np.asarray(val_ptr, dtype=np.int64),
        local_rows=None if leaf == 'dense' else np.asarray(out_rows, dtype=np.int32),
        local_cols=None if leaf == 'dense' else np.asarray(out_cols, dtype=np.int32))


def unpack_multilevel_k(p: MultiLevelKPayload):
    """Decode and validate a payload, returning its original mathematical A."""
    if (not p.widths or p.blk_m <= 0 or any(x <= 0 for x in p.widths)
        or any(a % b for a, b in zip(p.widths, p.widths[1:]))):
        raise ValueError('invalid tile widths')
    nw = (p.shape[0] + p.blk_m - 1) // p.blk_m
    nr = len(p.root_column_ids)
    if (p.root_column_ids.shape != (nr, p.widths[0]) or len(p.window_root_ptr) != nw + 1
        or p.window_root_ptr[0] != 0 or p.window_root_ptr[-1] != nr
        or np.any(np.diff(p.window_root_ptr) < 0)
        or len(p.child_ptrs) != len(p.widths) - 1
        or len(p.child_indices) != len(p.widths) - 1):
        raise ValueError('invalid root or level metadata')
    count = nr
    for level, (ptr, idx) in enumerate(zip(p.child_ptrs, p.child_indices)):
        if (len(ptr) != count + 1 or ptr[0] != 0 or ptr[-1] != len(idx)
            or np.any(np.diff(ptr) < 0)
            or np.any(idx < 0) or np.any(idx >= p.widths[level] // p.widths[level + 1])):
            raise ValueError('invalid child offsets or indices')
        for parent in range(count):
            if np.any(np.diff(idx[ptr[parent]:ptr[parent + 1]]) <= 0):
                raise ValueError('child indices must be ordered and unique')
        count = len(idx)
    if p.leaf == 'dense':
        if p.values.shape != (count, p.blk_m, p.widths[-1]):
            raise ValueError('invalid dense leaf shape')
    elif p.leaf == 'coo':
        if (p.leaf_value_ptr is None or p.local_rows is None or p.local_cols is None
            or len(p.leaf_value_ptr) != count + 1 or p.leaf_value_ptr[0] != 0
            or p.leaf_value_ptr[-1] != len(p.values) or np.any(np.diff(p.leaf_value_ptr) < 0)
            or len(p.local_rows) != len(p.values) or len(p.local_cols) != len(p.values)):
            raise ValueError('invalid COO payload')
    else:
        raise ValueError('invalid leaf encoding')
    out_r = []; out_c = []; out_v = []
    def visit(window, root, node, level, offset):
        if level < len(p.child_ptrs):
            ptr = p.child_ptrs[level]; idx = p.child_indices[level]
            for j in range(ptr[node], ptr[node + 1]):
                visit(window, root, j, level + 1, offset + idx[j] * p.widths[level + 1])
            return
        if p.leaf == 'dense':
            rr, cc = np.nonzero(p.values[node]); vv = p.values[node, rr, cc]
        else:
            lo, hi = p.leaf_value_ptr[node:node + 2]
            rr, cc, vv = p.local_rows[lo:hi], p.local_cols[lo:hi], p.values[lo:hi]
        if np.any(rr < 0) or np.any(rr >= p.blk_m) or np.any(cc < 0) or np.any(cc >= p.widths[-1]):
            raise ValueError('invalid leaf coordinate')
        global_r = window * p.blk_m + rr
        global_c = p.root_column_ids[root, offset + cc]
        if np.any(global_r >= p.shape[0]) or np.any(global_c < 0) or np.any(global_c >= p.shape[1]):
            raise ValueError('value in padding or outside matrix')
        out_r.extend(global_r); out_c.extend(global_c); out_v.extend(vv)
    for w in range(nw):
        for root in range(p.window_root_ptr[w], p.window_root_ptr[w + 1]):
            visit(w, root, root, 0, 0)
    return sparse.coo_matrix((np.asarray(out_v, dtype=p.values.dtype), (out_r, out_c)), shape=p.shape).tocsr()

"""Reference ABI for condense -> parent split -> scalar sparsify -> inner split.

This model is separate from child-block-sparsification: metadata indexes each
selected scalar column, and only then groups the selected sequence into tiles.
It does not claim equivalence to unvalidated compiler middle-end metadata.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from .nested_k_reference import NestedKPayload, pack_nested_k


@dataclass(frozen=True)
class ScalarCompactKPayload:
    nested: NestedKPayload
    parent_scalar_ptr: np.ndarray
    scalar_parent_positions: np.ndarray


def pack_scalar_compact_k(matrix, *, blk_m: int, blk_k: int, blk_k_i: int) -> ScalarCompactKPayload:
    """Preserve stable scalar coordinates prior to splitting compact sequence."""
    p = pack_nested_k(matrix, blk_m=blk_m, blk_k=blk_k, blk_k_i=blk_k_i, leaf='dense')
    mask = p.block_column_ids >= 0
    counts = mask.sum(axis=1, dtype=np.int64)
    ptr = np.concatenate((np.zeros(1,dtype=np.int64),np.cumsum(counts,dtype=np.int64)))
    positions = np.nonzero(mask)[1].astype(np.int32)
    # Column condensation guarantees all nonpadding columns are active and
    # densely ordered; therefore earlier child packing has the same leaf values.
    if len(p.child_index) != int(np.sum((counts+blk_k_i-1)//blk_k_i)):
        raise ValueError('leaf packing violates scalar-compacted tile counts')
    return ScalarCompactKPayload(p,ptr,positions)


def unpack_scalar_compact_k(p: ScalarCompactKPayload):
    """Decode through scalar index metadata, not child-block coordinate ids."""
    n=p.nested; nb=len(n.block_column_ids); nw=len(n.window_block_ptr)-1
    if len(p.parent_scalar_ptr)!=nb+1 or p.parent_scalar_ptr[0]!=0 or p.parent_scalar_ptr[-1]!=len(p.scalar_parent_positions) or np.any(np.diff(p.parent_scalar_ptr)<0):
        raise ValueError('invalid scalar pointers')
    if len(n.block_child_ptr)!=nb+1 or n.block_child_ptr[0]!=0 or n.block_child_ptr[-1]!=len(n.values) or np.any(np.diff(n.block_child_ptr)<0):
        raise ValueError('invalid tile pointers')
    rr=[];cc=[];vv=[]
    for w in range(nw):
        for b in range(n.window_block_ptr[w], n.window_block_ptr[w+1]):
            off,end=p.parent_scalar_ptr[b:b+2]; pos=p.scalar_parent_positions[off:end]
            if np.any(pos<0) or np.any(pos>=n.blk_k) or np.any(np.diff(pos)<=0):
                raise ValueError('invalid scalar mapping')
            left,right=n.block_child_ptr[b:b+2]
            if right-left!=(len(pos)+n.blk_k_i-1)//n.blk_k_i:
                raise ValueError('invalid scalar-to-tile length')
            for leaf in range(left,right):
                rows,cols=np.nonzero(n.values[leaf]); compact=(leaf-left)*n.blk_k_i+cols
                if np.any(compact>=len(pos)):
                    raise ValueError('nonzero stored in padding')
                cols_global=n.block_column_ids[b,pos[compact]]; rows_global=w*n.blk_m+rows
                if np.any(cols_global<0) or np.any(cols_global>=n.shape[1]) or np.any(rows_global>=n.shape[0]):
                    raise ValueError('coordinate outside matrix')
                rr.extend(rows_global);cc.extend(cols_global);vv.extend(n.values[leaf,rows,cols])
    return sparse.coo_matrix((np.asarray(vv,dtype=n.values.dtype),(rr,cc)),shape=n.shape).tocsr()

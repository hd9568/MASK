"""Experimental host ABI for historical 4-level OPIR/NVIR.

Outer offsets are parent-block units (matching the existing outer fusion);
inner offsets are padded scalar slots (so NVIR must retain division by K_I).
The composed B column tiles contain -1 padding, requiring masked B loads.
Not connected to the production runner or sufficient to execute its CUDA.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sparsene.formats.scalar_compact_k_reference import ScalarCompactKPayload
from .composed_sparse_indices import compose_nested_columns

@dataclass(frozen=True)
class ScalarCompactHostABI:
    logical_shape: tuple[int,int]
    padded_rows: int
    window_block_offsets: np.ndarray
    padded_scalar_offsets: np.ndarray
    global_column_tiles: np.ndarray
    local_position_tiles: np.ndarray
    values: np.ndarray
    true_scalar_counts: np.ndarray


def make_scalar_compact_host_abi(p: ScalarCompactKPayload) -> ScalarCompactHostABI:
    n=p.nested
    if n.leaf!='dense': raise ValueError('only dense historical leaf supported')
    c=compose_nested_columns(n.block_column_ids,p.parent_scalar_ptr,p.scalar_parent_positions,
                             inner_width=n.blk_k_i,ncols=n.shape[1])
    if not np.array_equal(c.padded_scalar_ptr//n.blk_k_i,n.block_child_ptr):
        raise ValueError('padded scalar offsets and leaf offsets disagree')
    if n.values.shape!=(len(c.global_column_tiles),n.blk_m,n.blk_k_i):
        raise ValueError('value leaf shape disagrees with composed indices')
    padding=c.global_column_tiles<0
    if np.any(n.values.transpose(0,2,1)[padding]!=0):
        raise ValueError('padded scalar lane contains nonzero values')
    limit=np.iinfo(np.int32).max
    if (n.window_block_ptr.size and int(n.window_block_ptr[-1])>limit) or (c.padded_scalar_ptr.size and int(c.padded_scalar_ptr[-1])>limit):
        raise OverflowError('OPIR int32 offset overflow')
    if n.shape[1]>limit: raise OverflowError('OPIR int32 column overflow')
    return ScalarCompactHostABI(n.shape,((n.shape[0]+n.blk_m-1)//n.blk_m)*n.blk_m,
        n.window_block_ptr.astype(np.int32),c.padded_scalar_ptr.astype(np.int32),
        c.global_column_tiles.astype(np.int32),c.local_position_tiles.astype(np.int32),
        n.values,c.scalar_counts)

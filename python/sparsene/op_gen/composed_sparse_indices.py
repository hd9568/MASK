"""Compose nested sparse index maps, with per-parent inner-tile padding.

This is a host-side IR representation bridge. It is not yet wired to the
production host runner or an executable NVIR backend.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ComposedSparseIndices:
    # Prefix sums in scalar slots, including padding (always divisible by tile).
    padded_scalar_ptr: np.ndarray
    # Per-parent true lengths, independent of padding.
    scalar_counts: np.ndarray
    # A row per compact inner tile, -1 for nonexistent lanes.
    global_column_tiles: np.ndarray
    local_position_tiles: np.ndarray


def compose_nested_columns(parent_columns, scalar_ptr, scalar_positions, *, inner_width: int, ncols: int) -> ComposedSparseIndices:
    """Compute global_column=parent_columns[parent, scalar_positions[index]].

    Selection is stable within each parent. Both the real scalar counts and
    padded scalar offsets are retained so that a backend cannot confuse scalar
    offsets with leaf-tile offsets. `-1` in composed tiles denotes padding.
    """
    cols=np.asarray(parent_columns);ptr=np.asarray(scalar_ptr);pos=np.asarray(scalar_positions)
    if cols.ndim!=2 or ptr.ndim!=1 or pos.ndim!=1:
        raise ValueError('parent columns and scalar index maps have invalid rank')
    if any(not np.issubdtype(x.dtype,np.integer) for x in (cols,ptr,pos)):
        raise ValueError('indices and offsets must be integers')
    if isinstance(inner_width,bool) or not isinstance(inner_width,int) or inner_width<=0 or ncols<0:
        raise ValueError('invalid inner width or matrix column count')
    parents,width=cols.shape
    if len(ptr)!=parents+1 or ptr[0]!=0 or ptr[-1]!=len(pos) or np.any(np.diff(ptr)<0):
        raise ValueError('invalid scalar parent offsets')
    counts=np.diff(ptr).astype(np.int64,copy=False)
    if np.any(counts>width): raise ValueError('selection exceeds parent width')
    if np.any(pos<0) or np.any(pos>=width): raise ValueError('local position outside parent')
    parent_of_scalar=np.repeat(np.arange(parents,dtype=np.int64),counts)
    if len(pos):
        # Do not compare entries across boundaries, including empty parents.
        same_parent=parent_of_scalar[1:]==parent_of_scalar[:-1]
        if np.any(same_parent & (np.diff(pos)<=0)):
            raise ValueError('local positions must be unique and stable-ordered')
        global_cols=cols[parent_of_scalar,pos]
        if np.any(global_cols<0) or np.any(global_cols>=ncols):
            raise ValueError('selected position is padding or outside matrix')
    else:
        global_cols=np.empty(0,dtype=cols.dtype)
    padded_counts=((counts+inner_width-1)//inner_width)*inner_width
    padded_ptr=np.concatenate((np.zeros(1,dtype=np.int64),np.cumsum(padded_counts,dtype=np.int64)))
    parent_local_index=np.arange(len(pos),dtype=np.int64)-np.repeat(ptr[:-1],counts)
    slot=np.repeat(padded_ptr[:-1],counts)+parent_local_index
    global_flat=np.full(int(padded_ptr[-1]),-1,dtype=np.int64)
    local_flat=np.full(int(padded_ptr[-1]),-1,dtype=np.int32)
    global_flat[slot]=global_cols;local_flat[slot]=pos
    return ComposedSparseIndices(padded_ptr,counts,global_flat.reshape(-1,inner_width),local_flat.reshape(-1,inner_width))

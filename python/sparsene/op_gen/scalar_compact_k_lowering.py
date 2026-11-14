"""Strict host plan for the historical scalar-sparsify-before-split DSL.

It does not register an executable GPU backend or assert NVIR ABI support.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from typing import Any, Mapping
from sympy import Symbol, simplify
from sparsene.formats.dsl import build_format_from_spec,parse_length_expr,validate_format_spec
from sparsene.transform.rts import derive_rts

@dataclass(frozen=True)
class ScalarCompactKPlan:
    schema_version: str
    source_sha256: str
    blk_m: int
    blk_k: int
    blk_k_i: int
    backend_status: str = 'host_reference_only'
    def to_dict(self) -> dict[str,Any]:
        out=asdict(self)
        out['metadata_abi']={'window_block_ptr':'int64[num_windows+1]',
            'block_column_ids':f'int64[num_blocks,{self.blk_k}]',
            'parent_scalar_ptr':'int64[num_blocks+1]',
            'scalar_parent_positions':'int32[num_selected_scalars]',
            'block_tile_ptr':'int64[num_blocks+1]',
            'values':f'float[num_tiles,{self.blk_m},{self.blk_k_i}]'}
        return out


def lower_scalar_compact_k_spec(spec: Mapping[str,Any]) -> ScalarCompactKPlan:
    report=validate_format_spec(spec)
    if report['status']!='valid': raise ValueError('invalid DSL: '+'; '.join(report['errors']))
    lev=spec['levels']; params=spec['params']; atom=spec['atomic']
    if len(lev)!=4 or atom['type']!='dense' or atom.get('layout','row_major')!='row_major':
        raise ValueError('requires the supported four-level, row-major dense profile')
    if any(x.get('reorder') is not None for l in lev for x in l['axes']): raise ValueError('reorder unsupported')
    for name in ('BLK_M','BLK_K','BLK_K_I'):
        if name not in params: raise ValueError('required parameter missing: '+name)
    m,k,i=(params[x] for x in ('BLK_M','BLK_K','BLK_K_I'))
    if k%i: raise ValueError('BLK_K_I must divide BLK_K')
    kinds=[[('df','ROW'),('sv','COL'),('df','ROW')],
           [('df','COL'),('df','COL'),('df','ROW')],
           [('sv','COL'),('df','ROW')],
           [('df','COL'),('df','COL'),('df','ROW')]]
    lengths=[['M','K','BLK_M'],['K/BLK_K','BLK_K','BLK_M'],
             ['BLK_K','BLK_M'],['BLK_K/BLK_K_I','BLK_K_I','BLK_M']]
    names=['M','K','N',*params]; substitutes={Symbol(k):v for k,v in params.items()}
    for level,expected,lens in zip(lev,kinds,lengths):
        axes=level['axes']
        if len(axes)!=len(expected): raise ValueError('unsupported axis count')
        for x,(kind,direction),length in zip(axes,expected,lens):
            if x['type']!=kind or x['direction']!=direction: raise ValueError('unsupported axis semantics')
            delta=(parse_length_expr(x['length'],names)-parse_length_expr(length,names)).subs(substitutes)
            if simplify(delta)!=0: raise ValueError('unsupported axis length')
    if atom['x_axis']!=lev[-1]['axes'][2]['name'] or atom['y_axis']!=lev[-1]['axes'][1]['name']:
        raise ValueError('atomic axes do not match leaf')
    seq=[x for x in derive_rts(build_format_from_spec(spec)).sequence if x.name in ('split','sparsify')]
    expected=[('split','BLK_M'),('sparsify',None),('split','BLK_K'),('sparsify',None),('split','BLK_K_I')]
    if len(seq)!=len(expected): raise ValueError('unsupported transform sequence')
    for op,(name,size) in zip(seq,expected):
        if op.name!=name or (size is not None and simplify(op.split_size-Symbol(size))!=0):
            raise ValueError('requires scalar sparsification before inner split')
    digest=sha256(json.dumps(spec,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return ScalarCompactKPlan('sparsene.scalar_compact_k_host_plan.v1',digest,m,k,i)

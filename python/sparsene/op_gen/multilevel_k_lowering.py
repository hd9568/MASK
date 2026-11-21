"""Strict lowering contract for a bounded family of nested-K format DSLs.

This plans host metadata only. It must not be presented as an executable GPU
backend until the plan is consumed by matching generated code and runtime.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from typing import Any, Mapping
from sympy import Symbol, simplify
from sparsene.formats.dsl import build_format_from_spec, parse_length_expr, validate_format_spec
from sparsene.transform.rts import derive_rts


@dataclass(frozen=True)
class MultiLevelKPlan:
    schema_version: str
    source_sha256: str
    blk_m: int
    widths: tuple[int, ...]
    leaf: str
    row_reorder: None = None
    backend_status: str = 'host_reference_only'

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result['metadata_abi'] = {
            'index_dtype': 'int64', 'child_index_dtype': 'int32',
            'window_root_ptr': ['num_windows + 1'],
            'root_column_ids': ['num_roots', self.widths[0]],
            'child_ptrs': ['num_parents[level] + 1' for _ in self.widths[1:]],
            'child_indices': ['num_children[level]' for _ in self.widths[1:]],
            'leaf': self.leaf,
            'value_shape': (['num_leaves', self.blk_m, self.widths[-1]]
                            if self.leaf == 'dense' else ['nnz']),
            'coo_metadata': [] if self.leaf == 'dense' else
                ['leaf_value_ptr:int64', 'local_rows:int32', 'local_cols:int32'],
        }
        return result


def lower_multilevel_k_spec(spec: Mapping[str, Any]) -> MultiLevelKPlan:
    """Accept only a proven structural profile; never physicalize a proposal.

    L1: ROW windows plus condensed sparse COL, L2: fixed K parent,
    subsequent levels: COL block index sparsified *after* split, fixed inner
    COL coordinate, same dense ROW. Axis lengths and the derived RTS are
    checked; valid-but-unsupported proposals fail instead of being replaced.
    """
    report = validate_format_spec(spec)
    if report['status'] != 'valid':
        raise ValueError('invalid format DSL: ' + '; '.join(report['errors']))
    levels = spec['levels']; params = spec['params']; atomic = spec['atomic']
    if len(levels) < 3:
        raise ValueError('nested-K reference requires at least three Format levels')
    if atomic['type'] not in ('dense', 'coo') or atomic.get('layout', 'row_major') != 'row_major':
        raise ValueError('only row-major dense or COO leaves are supported')
    if any(axis.get('reorder') is not None for lev in levels for axis in lev['axes']):
        raise ValueError('nested-K reference does not support axis reordering')
    names = ['M', 'K', 'N', *params]
    def eq_expr(value, want):
        return simplify((parse_length_expr(value, names) - parse_length_expr(want, names)).subs({Symbol(k): v for k, v in params.items()})) == 0
    def check(level, kinds, lengths):
        axes = level['axes']
        if len(axes) != len(kinds):
            raise ValueError('unsupported axis count')
        for axis, (kind, direction), length in zip(axes, kinds, lengths):
            if axis['type'] != kind or axis['direction'] != direction or not eq_expr(axis['length'], length):
                raise ValueError('axis kind, order, direction or length does not match nested-K profile')
    check(levels[0], [('df','ROW'),('sv','COL'),('df','ROW')], ['M','K','BLK_M'])
    check(levels[1], [('df','COL'),('df','COL'),('df','ROW')], ['K/BLK_K','BLK_K','BLK_M'])
    if 'BLK_M' not in params or 'BLK_K' not in params:
        raise ValueError('BLK_M and BLK_K parameters are required')
    widths = [params['BLK_K']]; split_exprs = []
    for lev in levels[2:]:
        axes = lev['axes']
        if len(axes) != 3:
            raise ValueError('inner level must contain a sparse block axis, dense inner K and dense row')
        inner = axes[1]['length']
        inner_expr = parse_length_expr(inner, names)
        if not inner_expr.free_symbols or not inner_expr.free_symbols.issubset({Symbol(k) for k in params}):
            raise ValueError('inner K width must depend only on declared parameters')
        val = inner_expr.subs({Symbol(k): v for k,v in params.items()})
        if val.is_integer is not True or val <= 0:
            raise ValueError('inner K width must resolve to a positive integer')
        parent = widths[-1]; width = int(val)
        if parent % width:
            raise ValueError('inner width must evenly divide parent width')
        expected_outer = str(parent // width)
        check(lev, [('sv','COL'),('df','COL'),('df','ROW')],
              [expected_outer, inner, 'BLK_M'])
        widths.append(width); split_exprs.append(inner_expr)
    deepest = levels[-1]['axes']
    if atomic['x_axis'] != deepest[2]['name'] or atomic['y_axis'] != deepest[1]['name']:
        raise ValueError('atomic axes must refer to deepest row and inner-K axes')
    # Check emitted RTS ordering rather than relying on level names or prose.
    ops = derive_rts(build_format_from_spec(spec)).sequence
    meaningful = [op for op in ops if op.name in ('split', 'sparsify')]
    wanted = [('split', Symbol('BLK_M')), ('sparsify', None), ('split', Symbol('BLK_K'))]
    for inner_expr in split_exprs:
        wanted.extend([('split', inner_expr), ('sparsify', None)])
    if len(meaningful) != len(wanted):
        raise ValueError('RTS does not match recursive split-then-sparsify profile')
    for op, (name, expr) in zip(meaningful, wanted):
        if op.name != name or (expr is not None and simplify(op.split_size - expr) != 0):
            raise ValueError('RTS does not match recursive split-then-sparsify profile')
    source = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return MultiLevelKPlan('sparsene.multilevel_k_host_plan.v1', sha256(source).hexdigest(),
                           params['BLK_M'], tuple(widths), atomic['type'])

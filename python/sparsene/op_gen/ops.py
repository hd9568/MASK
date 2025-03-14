from __future__ import annotations

from sparsene.op_gen.rts_parser import computent_from_rts, Computent
from sparsene.model.stage import Stage
from sparsene.model.format import Format, Expr, Symbol, Number
from sparsene.op_gen.rts_parser import ArrayDefCollection
from typing import Any, List, Optional, Dict, Literal, Tuple, Sequence
from sparsene.op_gen.rts_parser import (
    Schedule,
    DenseAxisIterator,
    SparseAxisIterator,
    SparseAxisSlicer,
    AtomicFormatOp,
    AtomicFormatType,
)
from dataclasses import dataclass
from sparsene.op_gen.op_ir import (
    Op,
    DeviceOp,
    OpBuilder,
    SymbolTable,
    ForLoopOp,
    Value,
    IndexType,
    context,
)


@dataclass
class SparseOffsetLoadOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value]):
        super().__init__(
            mem=mem,
            operands=operands,
            results=[
                Value(
                    name=f"l",
                    type=context.get_type(IndexType),
                    defining_op_or_owner=self,
                    result_idx_in_owner=0,
                ),
                Value(
                    name=f"r",
                    type=context.get_type(IndexType),
                    defining_op_or_owner=self,
                    result_idx_in_owner=1,
                ),
            ],
        )


@dataclass
class GetFromRangeOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value]):
        super().__init__(
            mem=mem,
            operands=operands,
            results=[],
        )


@dataclass
class CooAtomicFormatLoadOffOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value]):
        super().__init__(
            mem=mem,
            operands=operands,
            results=[],
        )


@dataclass
class CooAtomicFormatLoadIdxOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value]):
        super().__init__(
            mem=mem,
            operands=operands,
            results=[],
        )


@dataclass
class CooAtomicFormatLoadValOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value]):
        super().__init__(
            mem=mem,
            operands=operands,
            results=[],
        )


@dataclass
class ValSidxLoadOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value]):
        super().__init__(
            mem=mem,
            operands=operands,
            results=[],
        )


@dataclass
class BValLoadOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value]):
        super().__init__(
            mem=mem,
            operands=operands,
            results=[],
        )


@dataclass
class CooAtomicValRestoreOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value]):
        super().__init__(
            mem=mem,
            operands=operands,
            results=[],
        )


@dataclass
class MmaOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value]):
        super().__init__(
            mem=mem,
            operands=operands,
            results=[],
        )


@dataclass
class CValStoreOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value]):
        super().__init__(
            mem=mem,
            operands=operands,
            results=[],
        )

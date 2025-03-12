from __future__ import annotations

from sparsene.op_gen.rts_parser import computent_from_rts, Computent
from sparsene.model.stage import Stage
from sparsene.model.format import Format, Expr, Symbol, Number
from sparsene.op_gen.rts_parser import ArrayDefCollection
from typing import Any, List, Optional, Dict, Literal, Tuple
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
    Index,
)


@dataclass
class SparseOffsetLoadOp(DeviceOp):
    pass


@dataclass
class GetFromRangeOp(DeviceOp):
    pass


@dataclass
class CooAtomicFormatLoadOffOp(DeviceOp):
    pass


@dataclass
class CooAtomicFormatLoadIdxOp(DeviceOp):
    pass


@dataclass
class CooAtomicFormatLoadValOp(DeviceOp):
    pass


@dataclass
class ValSidxLoadOp(DeviceOp):
    pass


@dataclass
class BValLoadOp(DeviceOp):
    pass


@dataclass
class CooAtomicValRestoreOp(DeviceOp):
    pass


@dataclass
class MmaOp(Op):
    pass

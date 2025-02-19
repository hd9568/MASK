from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List


@dataclass(kw_only=True)
class Variable:
    symbol: str


@dataclass(kw_only=True)
class SymbolicConstant:
    symbol: str


@dataclass(kw_only=True)
class NumericConstant:
    value: int


Constant = SymbolicConstant | NumericConstant


class AtomicFormatType(Enum):
    DENSE = 0
    COO = 1
    MASK = 2


@dataclass(kw_only=True)
class Slice:
    start: Index
    end: Index


Index = Variable | Constant


@dataclass(kw_only=True)
class OpArg:
    array: str
    indices: List[Index | Slice]


@dataclass(kw_only=True)
class AtomicFormatOp:
    type: AtomicFormatType
    args: List[OpArg]


@dataclass(kw_only=True)
class SvIterator:
    axis: str
    body: Schedule


@dataclass(kw_only=True)
class DfIterator:
    axis: str
    body: Schedule


class ArrayType(Enum):
    VAL = 0
    INDEX = 1
    MASK = 2
    OFFSET = 3


@dataclass(kw_only=True)
class ArrayDim:
    axis: str
    length: Variable | Constant


@dataclass(kw_only=True)
class ArrayDef:
    name: str
    dims: list[ArrayDim]
    type: ArrayType


Schedule = DfIterator | SvIterator | AtomicFormatOp


@dataclass(kw_only=True)
class Computent:
    name: str
    schedule: dict
    array_defs: dict


DENSE_SCHEDULE: Schedule = AtomicFormatOp(
    type=AtomicFormatType.DENSE,
    args=[
        OpArg(array="val", indices=[]),
    ],
)

DENSE_ARRAY_DEFS = {
    "val": ArrayDef(
        name="val",
        dims=[
            ArrayDim(
                axis="X",
                length=Variable(symbol="M"),
            ),
            ArrayDim(
                axis="Y",
                length=Variable(symbol="N"),
            ),
        ],
        type=ArrayType.VAL,
    )
}

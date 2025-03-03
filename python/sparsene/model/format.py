from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from abc import ABC, abstractmethod
from sympy import Symbol, Expr, sympify


class Direction(Enum):
    ROW = 0
    COL = 1


VARLEN = Symbol("_")


@dataclass(kw_only=True)
class Axis:
    name: str
    direction: Direction
    length: Expr = VARLEN
    is_sparse: bool
    is_varlen: bool
    reorder: Optional[str] = None

    def __init__(
        self,
        name: str,
        direction: Direction,
        length: int | str | Expr = VARLEN,
        is_sparse: bool = False,
        is_varlen: bool = False,
        reorder: Optional[str] = None,
    ):
        self.name = name
        self.direction = direction
        if isinstance(length, Expr):
            self.length = length
        else:
            self.length = sympify(length)
        self.is_sparse = is_sparse
        self.is_varlen = is_varlen
        self.reorder = reorder

    def is_df(self) -> bool:
        return not self.is_sparse and not self.is_varlen

    def is_dv(self) -> bool:
        return not self.is_sparse and self.is_varlen

    def is_sf(self) -> bool:
        return self.is_sparse and not self.is_varlen

    def is_sv(self) -> bool:
        return self.is_sparse and self.is_varlen


@dataclass(kw_only=True)
class FormatBase(ABC):
    name: Optional[str] = None
    axes: List[Axis]

    @property
    @abstractmethod
    def is_atomic(self) -> bool:
        pass

    def is_dense(self) -> bool:
        return all(axis.is_df() for axis in self.axes)


@dataclass(kw_only=True)
class AtomicFormat(FormatBase):
    atomic_type: str = "dense"

    @property
    def is_atomic(self) -> bool:
        return True


@dataclass(kw_only=True)
class DenseAtomicFormat(AtomicFormat):
    atomic_type: str = "dense"


@dataclass(kw_only=True)
class CooAtomicFormat(AtomicFormat):
    atomic_type: str = "coo"


@dataclass(kw_only=True)
class McoAtomicFormat(AtomicFormat):
    atomic_type: str = "mco"


@dataclass(kw_only=True)
class Format(FormatBase):
    child: Optional[Format | AtomicFormat] = None

    @property
    def is_atomic(self) -> bool:
        return False


def sv_axis(
    name: str,
    direction: Direction,
    length: int | str | Expr = VARLEN,
    reorder: Optional[str] = None,
) -> Axis:
    return Axis(
        name=name,
        direction=direction,
        length=length,
        is_sparse=True,
        is_varlen=True,
        reorder=reorder,
    )


def sf_axis(
    name: str,
    direction: Direction,
    length: int | str | Expr = VARLEN,
    reorder: Optional[str] = None,
) -> Axis:
    return Axis(
        name=name,
        direction=direction,
        length=length,
        is_sparse=True,
        is_varlen=False,
        reorder=reorder,
    )


def dv_axis(
    name: str,
    direction: Direction,
    length: int | str | Expr = VARLEN,
    reorder: Optional[str] = None,
) -> Axis:
    return Axis(
        name=name,
        direction=direction,
        length=length,
        is_sparse=False,
        is_varlen=True,
        reorder=reorder,
    )


def df_axis(
    name: str,
    direction: Direction,
    length: int | str | Expr = VARLEN,
    reorder: Optional[str] = None,
) -> Axis:
    return Axis(
        name=name,
        direction=direction,
        length=length,
        is_sparse=False,
        is_varlen=False,
        reorder=reorder,
    )


def coo_atomic_format(
    x_name: str, y_name: str, col_major: bool = False
) -> AtomicFormat:
    return CooAtomicFormat(
        name="coo",
        axes=[
            Axis(
                name=x_name,
                direction=Direction.COL if col_major else Direction.ROW,
                is_sparse=False,
                is_varlen=False,
            ),
            Axis(
                name=y_name,
                direction=Direction.ROW if col_major else Direction.COL,
                is_sparse=False,
                is_varlen=False,
            ),
        ],
    )


def mco_atomic_format(
    x_name: str, y_name: str, col_major: bool = False
) -> AtomicFormat:
    return McoAtomicFormat(
        name="coo",
        axes=[
            Axis(
                name=x_name,
                direction=Direction.COL if col_major else Direction.ROW,
                is_sparse=False,
                is_varlen=False,
            ),
            Axis(
                name=y_name,
                direction=Direction.ROW if col_major else Direction.COL,
                is_sparse=False,
                is_varlen=False,
            ),
        ],
    )


def dense_atomic_format(
    x_name: str, y_name: str, col_major: bool = False
) -> AtomicFormat:
    return DenseAtomicFormat(
        name="coo",
        axes=[
            Axis(
                name=x_name,
                direction=Direction.COL if col_major else Direction.ROW,
                is_sparse=False,
                is_varlen=False,
            ),
            Axis(
                name=y_name,
                direction=Direction.ROW if col_major else Direction.COL,
                is_sparse=False,
                is_varlen=False,
            ),
        ],
    )


def atomic_format(
    atomic_type: str, x_name: str, y_name: str, col_major: bool = False
) -> AtomicFormat:
    match atomic_type:
        case "coo":
            return coo_atomic_format(x_name, y_name, col_major)
        case "mco":
            return mco_atomic_format(x_name, y_name, col_major)
        case "dense":
            return dense_atomic_format(x_name, y_name, col_major)
        case _:
            raise ValueError(f"Unknown atomic type: {atomic_type}")

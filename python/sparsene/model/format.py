from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from abc import ABC, abstractmethod


class Direction(Enum):
    ROW = 0
    COL = 1


@dataclass(kw_only=True)
class Axis:
    name: str
    direction: Direction
    length: Optional[int] = None
    is_sparse: bool
    is_varlen: bool
    reorder: Optional[str] = None


@dataclass(kw_only=True)
class FormatBase(ABC):
    name: Optional[str] = None
    axes: List[Axis]

    @property
    @abstractmethod
    def is_atomic(self) -> bool:
        pass


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
class MaskAtomicFormat(AtomicFormat):
    atomic_type: str = "mask"


@dataclass(kw_only=True)
class Format(FormatBase):
    child: Optional[Format | AtomicFormat] = None

    @property
    def is_atomic(self) -> bool:
        return False


def sv_axis(
    name: str,
    direction: Direction,
    length: Optional[int] = None,
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
    length: Optional[int] = None,
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
    length: Optional[int] = None,
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
    length: Optional[int] = None,
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


def mask_atomic_format(
    x_name: str, y_name: str, col_major: bool = False
) -> AtomicFormat:
    return MaskAtomicFormat(
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
        case "mask":
            return mask_atomic_format(x_name, y_name, col_major)
        case "dense":
            return dense_atomic_format(x_name, y_name, col_major)
        case _:
            raise ValueError(f"Unknown atomic type: {atomic_type}")

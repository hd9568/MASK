from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class Direction(Enum):
    ROW = 0
    COL = 1


@dataclass(kw_only=True)
class SimpleAxis:
    name: str
    direction: Direction
    length: Optional[int] = None


@dataclass(kw_only=True)
class Axis:
    name: str
    direction: Direction
    length: Optional[int] = None
    is_sparse: bool
    is_varlen: bool
    reorder: Optional[str] = None


@dataclass(kw_only=True)
class CooFormat:
    name: Optional[str] = None
    axes: List[SimpleAxis]


@dataclass(kw_only=True)
class MaskFormat:
    name: Optional[str] = None
    axes: List[SimpleAxis]


@dataclass(kw_only=True)
class Format:
    name: Optional[str] = None
    axes: List[Axis]
    child: Optional["Format | CooFormat"] = None


def sv_axis(name: str,
            direction: Direction,
            length: Optional[int] = None,
            reorder: Optional[str] = None) -> Axis:
    return Axis(
        name=name,
        direction=direction,
        length=length,
        is_sparse=True,
        is_varlen=True,
        reorder=reorder,
    )


def sf_axis(name: str,
            direction: Direction,
            length: Optional[int] = None,
            reorder: Optional[str] = None) -> Axis:
    return Axis(
        name=name,
        direction=direction,
        length=length,
        is_sparse=True,
        is_varlen=False,
        reorder=reorder,
    )


def dv_axis(name: str,
            direction: Direction,
            length: Optional[int] = None,
            reorder: Optional[str] = None) -> Axis:
    return Axis(
        name=name,
        direction=direction,
        length=length,
        is_sparse=False,
        is_varlen=True,
        reorder=reorder,
    )


def df_axis(name: str,
            direction: Direction,
            length: Optional[int] = None,
            reorder: Optional[str] = None) -> Axis:
    return Axis(
        name=name,
        direction=direction,
        length=length,
        is_sparse=False,
        is_varlen=False,
        reorder=reorder,
    )


def coo_format(x_name: str, y_name: str, col_major: bool = False) -> CooFormat:
    return CooFormat(
        name="coo",
        axes=[
            SimpleAxis(name=x_name, direction=Direction.COL if col_major else Direction.ROW),
            SimpleAxis(name=y_name, direction=Direction.ROW if col_major else Direction.COL),
        ],
    )

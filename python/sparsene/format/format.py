from __future__ import annotations
from sparsene.utils.source import indent
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from abc import ABC, abstractmethod
from sympy import Symbol, Expr, Number, Function


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
        is_sparse: bool,
        is_varlen: bool,
        length: int | str | Expr = VARLEN,
        reorder: Optional[str] = None,
    ):
        self.name = name
        self.direction = direction
        if isinstance(length, Expr):
            self.length = length
        elif isinstance(length, str):
            self.length = Symbol(length)
        else:
            self.length = Number(length)
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

    def set_varlen(self):
        self.length = Function("varlen")(self.length)

    def __str__(self, indent_width: int = 4) -> str:
        return (
            f"{self.__class__.__name__}("
            + f"name={self.name}, "
            + f"direction={self.direction.name}, "
            + f"length={self.length}, "
            + f"is_sparse={self.is_sparse}, "
            + f"is_varlen={self.is_varlen}, "
            + f"reorder={self.reorder})"
        )


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

    def __str__(self, indent_width: int = 4) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            + indent(
                f"axes=(\n"
                + indent(
                    "\n".join(map(str, self.axes)),
                    indent_width,
                )
                + "\n)",
                indent_width,
            )
            + "\n"
            + indent(
                f"atomic_type={self.atomic_type}",
                indent_width,
            )
            + "\n)"
        )

    def get_axis(self, name: str) -> Axis:
        for axis in self.axes:
            if axis.name == name:
                return axis
        raise ValueError(f"Axis {name} not found in {self}")


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

    def __str__(self, indent_width: int = 4) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            + indent(
                f"axes=(\n"
                + indent(
                    "\n".join(map(str, self.axes)),
                    indent_width,
                )
                + "\n)",
                indent_width,
            )
            + "\n"
            + (
                indent(
                    f"child={self.child.__str__(indent_width)}",
                    indent_width,
                )
                if self.child is not None
                else ""
            )
            + "\n)"
        )

    def get_axis(self, name: str) -> Axis:
        for axis in self.axes:
            if axis.name == name:
                return axis
        if self.child is not None:
            return self.child.get_axis(name)
        raise ValueError(f"Axis {name} not found in {self}")


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
            Axis(
                name=f"{x_name}_{y_name}_coo",
                direction=Direction.COL if col_major else Direction.ROW,
                is_sparse=True,
                is_varlen=True,
            )
        ],
    )


def mco_atomic_format(
    x_name: str, y_name: str, col_major: bool = False
) -> AtomicFormat:
    return McoAtomicFormat(
        name="mco",
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
            Axis(
                name=f"{x_name}_{y_name}_mco",
                direction=Direction.COL if col_major else Direction.ROW,
                is_sparse=True,
                is_varlen=True,
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

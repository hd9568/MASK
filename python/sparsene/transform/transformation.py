from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import chain
from typing import List, Tuple, Optional
from sympy import Expr, Number, Symbol


@dataclass(kw_only=True)
class Transformation(ABC):
    name: str
    axes: Tuple[int, ...]

    @property
    @abstractmethod
    def naxes(self) -> int:
        pass


@dataclass(kw_only=True)
class NullaryTransformation(Transformation, ABC):
    @abstractmethod
    def __init__(self, name: Optional[str] = None):
        self.axes = ()
        if name is not None:
            self.name = name

    @property
    def naxes(self) -> int:
        return 0


@dataclass(kw_only=True)
class UnaryTransformation(Transformation, ABC):
    @abstractmethod
    def __init__(self, i: int, name: Optional[str] = None):
        self.axes = (i,)
        if name is not None:
            self.name = name

    @property
    def naxes(self) -> int:
        return 1


@dataclass(kw_only=True)
class BinaryTransformation(Transformation, ABC):
    @abstractmethod
    def __init__(self, i: int, j: int, name: Optional[str] = None):
        self.axes = (i, j)
        if name is not None:
            self.name = name

    @property
    def naxes(self) -> int:
        return 2


@dataclass(kw_only=True)
class SplitTransformation(UnaryTransformation):
    split_size: Expr

    def __init__(
        self, axis: int, split_size: int | str | Expr, name: Optional[str] = None
    ):
        self.axes = (axis,)
        if name is not None:
            self.name = name

        if isinstance(split_size, str):
            self.split_size = Symbol(split_size)
        elif isinstance(split_size, int):
            self.split_size = Number(split_size)
        else:
            self.split_size = split_size

    name: str = "split"


@dataclass(kw_only=True)
class MergeTransformation(BinaryTransformation):
    def __init__(self, axis_i: int, axis_j: int, name: Optional[str] = None):
        self.axes = (axis_i, axis_j)
        if name is not None:
            self.name = name

    name: str = "merge"


@dataclass(kw_only=True)
class SparsifyTransformation(UnaryTransformation):
    def __init__(self, axis: int, name: Optional[str] = None):
        self.axes = (axis,)
        if name is not None:
            self.name = name

    name: str = "sparsify"


@dataclass(kw_only=True)
class DensifyTransformation(UnaryTransformation):
    def __init__(self, axis: int, name: Optional[str] = None):
        self.axes = (axis,)
        if name is not None:
            self.name = name

    name: str = "densify"


@dataclass(kw_only=True)
class SwapTransformation(BinaryTransformation):
    def __init__(self, axis_i: int, axis_j: int, name: Optional[str] = None):
        self.axes = (axis_i, axis_j)
        if name is not None:
            self.name = name

    name: str = "swap"


@dataclass(kw_only=True)
class SpawnTransformation(NullaryTransformation):
    def __init__(self, name: Optional[str] = None):
        self.axes = ()
        if name is not None:
            self.name = name

    name: str = "spawn"


@dataclass(kw_only=True)
class GoUpTransformation(NullaryTransformation):
    def __init__(self, name: Optional[str] = None):
        self.axes = ()
        if name is not None:
            self.name = name

    name: str = "go_up"


@dataclass(kw_only=True)
class CooizeTransformation(NullaryTransformation):
    def __init__(self, name: Optional[str] = None):
        self.axes = ()
        if name is not None:
            self.name = name

    name: str = "cooize"


@dataclass(kw_only=True)
class McoizeTransformation(NullaryTransformation):
    def __init__(self, name: Optional[str] = None):
        self.axes = ()
        if name is not None:
            self.name = name

    name: str = "mcoize"


@dataclass(kw_only=True)
class TransformationSequence:
    sequence: List[Transformation] = field(default_factory=list)

    def append(self, transformation: Transformation):
        self.sequence.append(transformation)

    def prepend(self, transformation: Transformation):
        self.sequence.insert(0, transformation)

    def __str__(self) -> str:
        return (
            "TansformationSequence(\n  "
            + ",\n  ".join(map(lambda x: str(x), self.sequence))
            + "\n)"
        )

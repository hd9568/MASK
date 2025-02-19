from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import chain
from typing import List, Tuple, Optional


@dataclass(kw_only=True)
class Transformation(ABC):
    name: str
    args: Tuple

    @property
    @abstractmethod
    def nargs(self) -> int:
        pass


@dataclass(kw_only=True)
class NullaryTransformation(Transformation, ABC):
    @abstractmethod
    def __init__(self, name: Optional[str] = None):
        self.args = ()
        if name is not None:
            self.name = name

    @property
    def nargs(self) -> int:
        return 0


@dataclass(kw_only=True)
class UnaryTransformation(Transformation, ABC):
    @abstractmethod
    def __init__(self, i: int, name: Optional[str] = None):
        self.args = (i,)
        if name is not None:
            self.name = name

    @property
    def nargs(self) -> int:
        return 1


@dataclass(kw_only=True)
class BinaryTransformation(Transformation, ABC):
    @abstractmethod
    def __init__(self, i: int, j: int, name: Optional[str] = None):
        self.args = (i, j)
        if name is not None:
            self.name = name

    @property
    def nargs(self) -> int:
        return 2


@dataclass(kw_only=True)
class SplitTransformation(UnaryTransformation):
    def __init__(self, i: int, name: Optional[str] = None):
        self.args = (i,)
        if name is not None:
            self.name = name

    name: str = "split"


@dataclass(kw_only=True)
class MergeTransformation(BinaryTransformation):
    def __init__(self, i: int, j: int, name: Optional[str] = None):
        self.args = (i, j)
        if name is not None:
            self.name = name

    name: str = "merge"


@dataclass(kw_only=True)
class SparsifyTransformation(UnaryTransformation):
    def __init__(self, i: int, name: Optional[str] = None):
        self.args = (i,)
        if name is not None:
            self.name = name

    name: str = "sparsify"


@dataclass(kw_only=True)
class DensifyTransformation(UnaryTransformation):
    def __init__(self, i: int, name: Optional[str] = None):
        self.args = (i,)
        if name is not None:
            self.name = name

    name: str = "densify"


@dataclass(kw_only=True)
class SwapTransformation(BinaryTransformation):
    def __init__(self, i: int, j: int, name: Optional[str] = None):
        self.args = (i, j)
        if name is not None:
            self.name = name

    name: str = "swap"


@dataclass(kw_only=True)
class SpawnTransformation(NullaryTransformation):
    def __init__(self, name: Optional[str] = None):
        self.args = ()
        if name is not None:
            self.name = name

    name: str = "spawn"


@dataclass(kw_only=True)
class GoUpTransformation(NullaryTransformation):
    def __init__(self, name: Optional[str] = None):
        self.args = ()
        if name is not None:
            self.name = name

    name: str = "go_up"


@dataclass(kw_only=True)
class CooizeTransformation(NullaryTransformation):
    def __init__(self, name: Optional[str] = None):
        self.args = ()
        if name is not None:
            self.name = name

    name: str = "cooize"


@dataclass(kw_only=True)
class MaskizeTransformation(NullaryTransformation):
    def __init__(self, name: Optional[str] = None):
        self.args = ()
        if name is not None:
            self.name = name

    name: str = "maskize"


@dataclass(kw_only=True)
class TransformationSequence:
    sequence: List[Transformation] = field(default_factory=list)

    def append(self, transformation: Transformation):
        self.sequence.append(transformation)

    def prepend(self, transformation: Transformation):
        self.sequence.insert(0, transformation)

    def __str__(self) -> str:
        return "TansformationSequence(\n  " + ",\n  ".join(map(lambda x: str(x), self.sequence)) + "\n)"

from enum import Enum
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


class Direction(Enum):
    ROW = 0
    COL = 1


@dataclass(kw_only=True)
class CooAxis:
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


@dataclass(kw_only=True)
class CooFormat:
    name: Optional[str] = None
    axes: List[CooAxis]


@dataclass(kw_only=True)
class Format:
    name: Optional[str] = None
    axes: List[Axis]
    child: Optional["Format | CooFormat"] = None

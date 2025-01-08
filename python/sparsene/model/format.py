from enum import Enum
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


class Direction(Enum):
    ROW = 0
    COL = 1


@dataclass
class Axis:
    name: str
    direction: Direction
    is_sparse: bool
    is_varlen: bool


@dataclass
class Format:
    name: Optional[str] = None
    axes: List[Axis] = field(default_factory=lambda: [])
    child: Optional[Tuple[Tuple[int, int], "Format"]] = None
    impl: Optional[str] = None

from typing import List, Iterable
from transformation import Transformation
from axis import Axis


class Format:
    def __init__(self, axes: Iterable[Axis] = []) -> None:
        self._axes: List[Axis] = list(axes)

    def transform(self, transformation: Transformation):
        return transformation.transform(self)

    @property
    def axes(self) -> List[Axis]:
        return self._axes

from format.format import Format
from typing import Optional, AnyStr, Iterable


class Axis:
    def __init__(self, name: AnyStr) -> None:
        self._name = name


class PlainAxis(Axis):
    def __init__(
        self,
        name: AnyStr,
        is_sparse: bool,
        parent_axis: Optional["Axis"] = None,
    ) -> None:
        super().__init__(name)
        self._name = name
        self._is_sparse = is_sparse
        self._parent_axis = parent_axis


class NestedFormatAxis(Axis):
    def __init__(
        self,
        name: AnyStr,
        format: Format,
    ) -> None:
        super().__init__(name)
        self._format = format

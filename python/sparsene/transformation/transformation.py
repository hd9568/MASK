# from typing import Iterable, Tuple
# from sparsene.model.format import Format
# from sparsene.model.axis import NestedFormatAxis
#
#
# class Transformation:
#     def __init__(self) -> None:
#         pass
#
#     def transform(self, format: Format) -> Format:
#         raise NotImplementedError
#
#
# class Sequential(Transformation):
#     def __init__(self, transformations: Iterable[Transformation]) -> None:
#         super().__init__()
#         self._transformations = list(transformations)
#
#     def transform(self, format: Format) -> Format:
#         for transformation in self._transformations:
#             format = transformation.transform(format)
#         return format
#
#
# class Nest(Transformation):
#     def __init__(self, begin_axis, end_axis) -> None:
#         super().__init__()
#         self._axis_range: Tuple[int, int] = begin_axis, end_axis
#
#     def transform(self, format: Format) -> Format:
#         nested_format = Format(format.axes[self._axis_range[0] : self._axis_range[1]])
#         new_axis = NestedFormatAxis("", nested_format)
#         return Format(
#             [
#                 *format.axes[: self._axis_range[0]],
#                 new_axis,
#                 *format.axes[self._axis_range[1] :],
#             ]
#         )

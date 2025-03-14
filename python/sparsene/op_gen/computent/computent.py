from __future__ import annotations
from abc import ABC, abstractmethod
from typing import overload
from copy import deepcopy
from sparsene.utils.source import indent
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, List, Optional, Dict, Tuple, Iterable, Sequence, Any
from sparsene.transform.transformation import (
    Transformation,
    TransformationSequence,
    SplitTransformation,
    SwapTransformation,
    SparsifyTransformation,
    SpawnTransformation,
    CooizeTransformation,
    McoizeTransformation,
)
from sparsene.format.format import (
    Direction,
    Format,
    AtomicFormat,
    Axis,
    df_axis,
    sv_axis,
    coo_atomic_format,
    mco_atomic_format,
    Expr,
    Number,
    Symbol,
    VARLEN,
)
from sparsene.formats.dense import get_dense_format

from sparsene.logging import get_logger

logger = get_logger(__name__)


class NameManager:
    def __init__(self):
        self.counter = 0

    def new_name(self) -> str:
        self.counter += 1
        return f"v{self.counter}"


name_manager = NameManager()


class AtomicFormatType(Enum):
    DENSE = 0
    COO = 1
    MCO = 2


@dataclass(kw_only=True)
class ArrayRef:
    array: str
    indices: List[Variable]

    def replace_symbol(self, old: str, new: str) -> None:
        for i in range(len(self.indices)):
            index = self.indices[i]
            if isinstance(index, ArrayRef):
                index.replace_symbol(old, new)
            elif index == old:
                self.indices[i] = new

    def replace_index(self, old: str, new: List[Variable]) -> None:
        for i in range(len(self.indices)):
            index = self.indices[i]
            if isinstance(index, ArrayRef):
                index.replace_index(old, new)
        for i in range(len(self.indices)):
            if self.indices[i] == old:
                self.indices = self.indices[:i] + new + self.indices[i + 1 :]
                break

    def append_ref_index(self, new_index: Variable) -> None:
        nested = False
        for i in range(len(self.indices)):
            index = self.indices[i]
            if isinstance(index, ArrayRef):
                index.append_ref_index(new_index)
                nested = True
        if not nested:
            self.indices.append(new_index)

    def pop_ref_index(self) -> None:
        nested = False
        for i in range(len(self.indices)):
            index = self.indices[i]
            if isinstance(index, ArrayRef):
                index.pop_ref_index()
                nested = True
        if not nested:
            self.indices.pop()

    def __str__(self, indent_width: int = 4) -> str:
        return f"{self.array}[{', '.join(map(str, self.indices))}]"


Variable = ArrayRef | str


@dataclass(kw_only=True)
class Schedule(ABC):
    parent: Optional[AxisVisitor]

    @abstractmethod
    def is_leaf(self) -> bool: ...

    @abstractmethod
    def get_child(self) -> Optional[Schedule]: ...

    @property
    def child(self) -> Optional[Schedule]:
        return self.get_child()

    @abstractmethod
    def refers_to(self, axis: str) -> bool: ...

    @abstractmethod
    def replace_symbol(self, old: str, new: str) -> None: ...

    @abstractmethod
    def replace_all_array_ref_idx(self, old: str, new: List[Variable]) -> None: ...

    def get_parent(self) -> Optional[AxisVisitor]:
        return self.parent

    def find_axis_ref(self, axis: str) -> Optional[Schedule]:
        if self.refers_to(axis):
            return self
        child = self.get_child()
        if child is None:
            return None
        return child.find_axis_ref(axis)

    @abstractmethod
    def __str__(self, indent_width: int = 4) -> str: ...

    @abstractmethod
    def replace_array_ref_idx(
        self, array_name: str, old: str, new: Variable
    ) -> None: ...


@dataclass(kw_only=True)
class AtomicFormatOp(Schedule):
    axes: Tuple[str, str]
    type: AtomicFormatType
    args: List[Variable]

    def is_leaf(self) -> bool:
        return True

    def get_child(self) -> Optional[Schedule]:
        return None

    def refers_to(self, axis: str) -> bool:
        return axis in self.axes

    def replace_symbol(self, old: str, new: str) -> None:
        for i in range(len(self.args)):
            arg = self.args[i]
            if isinstance(arg, ArrayRef):
                arg.replace_symbol(old, new)
            elif arg == old:
                self.args[i] = new

    def replace_all_array_ref_idx(self, old: str, new: List[Variable]) -> None:
        for i in range(len(self.args)):
            arg = self.args[i]
            if isinstance(arg, ArrayRef):
                arg.replace_index(old, new)

    def __str__(self, indent_width: int = 4) -> str:
        return (
            f"AtomicFormatOp[{self.type.name}]({self.axes[0]}, {self.axes[1]})"
            + "{\n"
            + indent(
                f"args=(\n"
                + indent(str("\n".join(map(str, self.args))), indent_width)
                + "\n)"
            )
            + "\n}"
        )

    def replace_array_ref_idx(self, array_name: str, old: str, new: Variable) -> None:
        for i in range(len(self.args)):
            arg = self.args[i]
            if isinstance(arg, ArrayRef) and arg.array == array_name:
                for j in range(len(arg.indices)):
                    idx = arg.indices[j]
                    if idx == old:
                        arg.indices[j] = new

    def append_array_ref_idx(self, array_name: str, new: Variable) -> None:
        for i in range(len(self.args)):
            arg = self.args[i]
            if isinstance(arg, ArrayRef) and arg.array == array_name:
                arg.indices.append(new)


@dataclass(kw_only=True)
class AxisVisitor(Schedule, ABC):
    name: str = ""
    axis: str
    body: Schedule
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.name == "":
            self.name = name_manager.new_name()

    def is_leaf(self) -> bool:
        return False

    def get_child(self) -> Schedule | None:
        return self.body

    def refers_to(self, axis: str) -> bool:
        return axis == self.axis

    def replace_symbol(self, old: str, new: str) -> None:
        if self.name == old:
            raise ValueError(
                f"Cannot replace the name of an {self.__class__.__name__} itself: {self}"
            )
        self.body.replace_symbol(old, new)

    def replace_all_array_ref_idx(self, old: str, new: List[Variable]) -> None:
        self.body.replace_all_array_ref_idx(old, new)

    def replace_array_ref_idx(self, array_name: str, old: str, new: Variable):
        self.body.replace_array_ref_idx(array_name, old, new)

    @property
    @abstractmethod
    def is_sparse(self) -> bool: ...

    def __str__(self, indent_width: int = 4) -> str:
        return (
            f"for {self.name} in {self.__class__.__name__}({self.axis})"
            + "{\n"
            + indent(f"{self.body.__str__(indent_width)}", indent_width)
            + "\n}"
        )


@dataclass(kw_only=True)
class DenseAxisIterator(AxisVisitor):
    @property
    def is_sparse(self) -> bool:
        return False


@dataclass(kw_only=True)
class SparseAxisVisitor(AxisVisitor):
    offset_array: str

    @property
    def is_sparse(self) -> bool:
        return True

    def __str__(self, indent_width: int = 4) -> str:
        return (
            f"for {self.name} in {self.__class__.__name__}({self.axis})(soff={self.offset_array})"
            + "{\n"
            + indent(f"{self.body.__str__(indent_width)}", indent_width)
            + "\n}"
        )


@dataclass(kw_only=True)
class SparseAxisIterator(SparseAxisVisitor):
    """
    This is a schedule that iterates over a range of an array according to the offset
    array in a sparse format and provide the index to be used in the body.
    """


@dataclass(kw_only=True)
class SparseAxisSlicer(SparseAxisVisitor):
    """
    This is a schedule that slices an array according to the offset array in a sparse
    format and provide the slice to be used in the body.
    """


class ArrayType(Enum):
    VAL = 0
    IDX = 1
    OFFSET = 2
    B_VAL = 3
    COO_IDX = 4
    MASK = 5


class ArrayDef:
    name: str
    axes: List[str]
    dims: List[Expr]
    type: ArrayType

    def __init__(
        self,
        name: str,
        axes: List[str],
        dims: Sequence[Expr | int | str],
        type: ArrayType,
    ):
        self.name = name
        self.axes = axes
        self.dims = []
        for dim in dims:
            if isinstance(dim, int):
                self.dims.append(Number(dim))
            elif isinstance(dim, str):
                self.dims.append(Symbol(dim))
            elif isinstance(dim, Expr):
                self.dims.append(dim)
            else:
                raise ValueError(f"Unsupported dimension type: {dim}")
        self.type = type

    def index_of(self, axis: str) -> Optional[int]:
        for i, dim in enumerate(self.axes):
            if dim == axis:
                return i
        return None

    def __str__(self) -> str:
        return f"ArrayDef({self.name}, {self.axes}, {self.dims}, {self.type})"


@dataclass(kw_only=True)
class ArrayDefCollection:
    array_defs: Dict[str, ArrayDef]

    def __init__(self, **kwargs: ArrayDef):
        self.array_defs = deepcopy(kwargs)

    def __iter__(self) -> Iterator[ArrayDef]:
        return iter(self.array_defs.values())

    @overload
    def __getitem__(self, key: str) -> ArrayDef: ...

    @overload
    def __getitem__(self, key: ArrayType) -> List[ArrayDef]: ...

    def __getitem__(self, key: str | ArrayType) -> ArrayDef | List[ArrayDef]:
        if isinstance(key, ArrayType):
            return [
                array_def
                for array_def in self.array_defs.values()
                if array_def.type == key
            ]
        return self.array_defs[key]

    def __setitem__(self, key: str, value: ArrayDef) -> None:
        assert key == value.name
        self.array_defs[key] = value

    @overload
    def add(self, array_def: ArrayDef) -> None: ...

    @overload
    def add(self, array_def: Iterable[ArrayDef]) -> None: ...

    def add(self, array_def: ArrayDef | Iterable[ArrayDef]) -> None:
        if isinstance(array_def, ArrayDef):
            self.array_defs[array_def.name] = array_def
        else:
            for ad in array_def:
                self.array_defs[ad.name] = ad

    def items(self) -> Iterator[Tuple[str, ArrayDef]]:
        return iter(self.array_defs.items())

    def replace(self, old: List[str], new: List[ArrayDef]) -> None:
        for old_def_name in old:
            if old_def_name in self.array_defs:
                del self.array_defs[old_def_name]
        for new_def in new:
            self.array_defs[new_def.name] = new_def

    def __str__(self, indent_width: int = 4) -> str:
        return (
            f"ArrayDefCollection(\n"
            + indent(
                "\n".join(
                    [
                        f"  {name}: {array_def}"
                        for name, array_def in self.array_defs.items()
                    ]
                ),
                indent_width,
            )
            + "\n)"
        )


@dataclass(kw_only=True)
class Computent:
    name: str
    schedule: Schedule
    array_defs: ArrayDefCollection
    format: Format

    def __str__(self, indent_width: int = 4) -> str:
        return (
            f"Computent(\n"
            + indent(f"name={self.name},", indent_width)
            + "\n"
            + indent(
                f"schedule=\n"
                + indent(self.schedule.__str__(indent_width), indent_width)
                + ",",
                indent_width,
            )
            + "\n"
            + indent(
                f"array_defs={self.array_defs.__str__(indent_width)},", indent_width
            )
            + "\n"
            + indent(f"format={self.format.__str__(indent_width)}", indent_width)
            + "\n)"
        )


DENSE_SCHEDULE: Schedule = AtomicFormatOp(
    axes=("X", "Y"),
    type=AtomicFormatType.DENSE,
    args=[
        ArrayRef(array="val", indices=[]),
        ArrayRef(array="B_val", indices=[]),
    ],
    parent=None,
)


DENSE_ARRAY_DEFS = ArrayDefCollection(
    val=ArrayDef(
        name="val",
        axes=["X", "Y"],
        dims=["m", "n"],
        type=ArrayType.VAL,
    ),
    B_val=ArrayDef(
        name="B_val",
        axes=["Y"],
        dims=["n"],
        type=ArrayType.B_VAL,
    ),
)

DENSE_FORMAT = get_dense_format()


def apply_split_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    format: Format,
    root_format: Format,
    transform: SplitTransformation,
) -> Tuple[Schedule, ArrayDefCollection, Format]:
    schedule, array_defs, format = (
        deepcopy(schedule),
        deepcopy(array_defs),
        deepcopy(format),
    )

    axis_idx = transform.axes[0]

    length_before = format.axes[axis_idx].length
    new_length_outer = length_before / transform.split_size  # type: ignore
    new_length_inner = transform.split_size

    # 1. Handle format
    axis_to_split = format.axes[axis_idx]
    new_axis_outer = Axis(
        name=axis_to_split.name + "_o",
        direction=axis_to_split.direction,
        length=new_length_outer,
        is_sparse=axis_to_split.is_sparse,
        is_varlen=axis_to_split.is_varlen,
    )
    new_axis_inner = Axis(
        name=axis_to_split.name + "_i",
        direction=axis_to_split.direction,
        length=new_length_inner,
        is_sparse=axis_to_split.is_sparse,
        is_varlen=axis_to_split.is_varlen,
    )
    format.axes = (
        format.axes[:axis_idx]
        + [new_axis_outer]
        + [new_axis_inner]
        + format.axes[axis_idx + 1 :]
    )

    # 2. Handle array_defs
    for array_def in array_defs:
        idx_dim = array_def.index_of(axis_to_split.name)

        if idx_dim is None:
            continue

        if (
            array_def.type == ArrayType.B_VAL
            and root_format.get_axis(axis_to_split.name).is_sparse
        ):
            continue

        array_def.axes[idx_dim] = new_axis_outer.name
        array_def.axes.insert(idx_dim + 1, new_axis_inner.name)

        array_def.dims[idx_dim] = new_length_outer
        array_def.dims.insert(idx_dim + 1, new_length_inner)
    # 3. Handle schedule
    axis_ref = schedule.find_axis_ref(axis_to_split.name)
    if isinstance(axis_ref, (DenseAxisIterator, SparseAxisIterator)):
        new_inner_axis_ref = DenseAxisIterator(
            axis=new_axis_inner.name,
            body=axis_ref.body,
            parent=axis_ref,
        )
        axis_ref.axis = new_axis_outer.name
        axis_ref.body = new_inner_axis_ref

        new_inner_axis_ref.body.replace_all_array_ref_idx(
            axis_ref.name, [axis_ref.name, new_inner_axis_ref.name]
        )
    elif isinstance(axis_ref, SparseAxisSlicer):
        new_axis_ref = SparseAxisIterator(
            name=axis_ref.name,
            axis=new_axis_outer.name,
            body=axis_ref.body,
            offset_array=axis_ref.offset_array,
            parent=axis_ref.get_parent(),
        )
        assert isinstance(axis_ref.body, AtomicFormatOp)
        axis_ref.body.parent = new_axis_ref
        if axis_ref.body.axes[0] == axis_to_split.name:
            axis_ref.body.axes = new_axis_inner.name, axis_ref.body.axes[1]
        elif axis_ref.body.axes[1] == axis_to_split.name:
            axis_ref.body.axes = axis_ref.body.axes[0], new_axis_inner.name
        else:
            raise ValueError(
                f"None of the axes match {axis_to_split.name}: {axis_ref.body.axes}"
            )

        parent_axis_ref = axis_ref.get_parent()
        if parent_axis_ref is not None:
            assert isinstance(
                parent_axis_ref, AxisVisitor
            ), f"Parent axis reference should be an AxisVisitor, got {parent_axis_ref}"
            parent_axis_ref.body = new_axis_ref
        else:
            schedule = new_axis_ref
    elif isinstance(axis_ref, AtomicFormatOp):
        if axis_ref.axes[0] == axis_to_split.name:
            # axis_ref is the first axis of the atomic format op
            # we need to create a dense axis iterator for the new outer axis
            # and replace the first axis of the atomic format op with the new inner axis
            # (the first axis of the atomic format op is split across the boundary)
            axis_ref.axes = new_axis_inner.name, axis_ref.axes[1]
            new_axis_ref = DenseAxisIterator(
                axis=new_axis_outer.name,
                body=axis_ref,
                parent=axis_ref.get_parent(),
            )

            parent_axis_ref = axis_ref.get_parent()
            if parent_axis_ref is not None:
                assert isinstance(
                    parent_axis_ref, AxisVisitor
                ), f"Parent axis reference should be an AxisVisitor, got {parent_axis_ref}"
                parent_axis_ref.body = new_axis_ref
            else:
                schedule = new_axis_ref
            axis_ref.parent = new_axis_ref

            # Insert a new dense dim index of x_o into array refs
            # Maybe always append to the tail of the refs?
            for arg in axis_ref.args:
                if isinstance(arg, ArrayRef):
                    if array_defs[arg.array].index_of(new_axis_ref.axis) is not None:
                        arg.append_ref_index(new_axis_ref.name)
        elif axis_ref.axes[1] == axis_to_split.name:
            # TODO: This case is actually illegal, but we preserve it for now

            # axis_ref is the second axis of the atomic format op
            # we need to move the first axis out of the atomic format op
            # and use the split two new axes as the new two axes of the atomic format op
            new_outer_dense_axis_iter = DenseAxisIterator(
                axis=axis_ref.axes[0],
                body=axis_ref,
                parent=axis_ref.get_parent(),
            )
            axis_ref.axes = new_axis_outer.name, new_axis_inner.name
            for arg in axis_ref.args:
                if isinstance(arg, ArrayRef):
                    if (
                        array_defs[arg.array].index_of(new_outer_dense_axis_iter.axis)
                        is not None
                    ):
                        arg.append_ref_index(new_outer_dense_axis_iter.name)

            parent_axis_ref = axis_ref.get_parent()
            if parent_axis_ref is not None:
                assert isinstance(
                    parent_axis_ref, AxisVisitor
                ), f"Parent axis reference should be an AxisVisitor, got {parent_axis_ref}"
                parent_axis_ref.body = new_outer_dense_axis_iter
            else:
                schedule = new_outer_dense_axis_iter
            axis_ref.parent = new_outer_dense_axis_iter
        else:
            raise ValueError(
                f"None of the axes match {axis_to_split.name}: {axis_ref.axes}"
            )

    else:
        raise ValueError(f"Unknown axis reference: {axis_ref}")

    return schedule, array_defs, format


def apply_swap_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    format: Format,
    root_format: Format,
    transform: SwapTransformation,
) -> Tuple[Schedule, ArrayDefCollection, Format]:
    schedule, array_defs, format = (
        deepcopy(schedule),
        deepcopy(array_defs),
        deepcopy(format),
    )

    axis0_name = format.axes[transform.axes[0]].name
    axis1_name = format.axes[transform.axes[1]].name

    # 1. Handle format
    format.axes[transform.axes[0]], format.axes[transform.axes[1]] = (
        format.axes[transform.axes[1]],
        format.axes[transform.axes[0]],
    )

    # 2. Handle array_defs
    for array_def in array_defs:
        idx_dim_0 = array_def.index_of(axis0_name)
        idx_dim_1 = array_def.index_of(axis1_name)
        if idx_dim_0 is None or idx_dim_1 is None:
            continue

        array_def.axes[idx_dim_0], array_def.axes[idx_dim_1] = (
            array_def.axes[idx_dim_1],
            array_def.axes[idx_dim_0],
        )
        array_def.dims[idx_dim_0], array_def.dims[idx_dim_1] = (
            array_def.dims[idx_dim_1],
            array_def.dims[idx_dim_0],
        )

    # 3. Handle schedule
    axis0_ref = schedule.find_axis_ref(axis0_name)
    axis1_ref = schedule.find_axis_ref(axis1_name)
    if axis0_ref is None or axis1_ref is None:
        raise ValueError(f"Unknown axis reference: {axis0_ref} or {axis1_ref}")

    if axis0_ref is axis1_ref:
        assert isinstance(axis0_ref, AtomicFormatOp)
        # they are in the same atomic op, just swap the axes
        axis0_ref.axes = axis0_ref.axes[1], axis0_ref.axes[0]
    elif axis0_ref.get_child() is axis1_ref:
        assert isinstance(axis0_ref, DenseAxisIterator)
        if isinstance(axis1_ref, DenseAxisIterator):
            # they are in the same dense axis iterator, just swap the axes
            axis0_ref.axis, axis1_ref.axis = axis1_ref.axis, axis0_ref.axis
            # we do not need to swap the indices of the array refs since they are
            # already swapped with regard to the swapping of the axes
        elif isinstance(axis1_ref, AtomicFormatOp):
            # axis0_ref is a dense axis iterator,
            # axis1_ref belongs to the first axis of the atomic format op
            assert (
                axis1_ref.axes[0] == axis1_name
            ), f"{axis1_name} must be the first axis of the atomic format op"
            axis0_ref.axis = axis1_name
            axis1_ref.axes = axis0_name, axis1_ref.axes[1]

            # SPECIAL: we need to handle B_val indices following the rules below:
            # 1. If an axis is swapped into atomic from outside and it belongs to B_val's
            #    array def, we need to remove the iter var name from the array ref index
            # 2. If an axis is swapped from atomic to outside and it belongs to B_val's
            #    array def, we need to append the iter var name to the array ref index
            for arg in axis1_ref.args:
                if isinstance(arg, ArrayRef):
                    if (
                        array_defs[arg.array].index_of(axis0_name) is not None
                        and axis0_ref.name in arg.indices
                    ):
                        arg.pop_ref_index()
                    if (
                        array_defs[arg.array].index_of(axis1_name) is not None
                        and axis0_ref.name not in arg.indices
                    ):
                        arg.append_ref_index(axis0_ref.name)
        else:
            raise ValueError(
                f"Unknown axis reference: {axis1_ref} for swap({transform.axes[0]}, {transform.axes[1]})"
            )
    else:
        raise ValueError(
            f"Axis references do not match: {axis0_ref} and {axis1_ref} for "
            f"swap({transform.axes[0]}, {transform.axes[1]})"
        )

    return schedule, array_defs, format


def apply_sparsify_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    format: Format,
    root_format: Format,
    transform: SparsifyTransformation,
) -> Tuple[Schedule, ArrayDefCollection, Format]:
    new_schedule, new_array_defs, new_format = (
        deepcopy(schedule),
        ArrayDefCollection(),
        deepcopy(format),
    )

    axis_gid = transform.axes[0]
    prev_axis_name = format.axes[axis_gid].name
    new_axis_name = prev_axis_name + "_s"

    # 1. Handle format
    axis = new_format.axes[axis_gid]
    axis.is_sparse = True
    axis.is_varlen = True
    axis.name = new_axis_name

    # 2. Handle array_defs
    soff_array_name = None
    for array_def in array_defs:
        if array_def.type == ArrayType.B_VAL:
            idx = array_def.index_of(prev_axis_name)
            if idx is not None:
                array_def.axes[idx] = new_axis_name
            new_array_defs.add(array_def)
            continue

        idx = array_def.index_of(prev_axis_name)
        if idx is None:
            new_array_defs.add(array_def)
            continue

        idx_p = idx - 1
        assert idx_p >= 0

        sparse_val_def = ArrayDef(
            name=array_def.name,
            axes=[axis.name] + array_def.axes[idx + 1 :],
            dims=[axis.length] + array_def.dims[idx + 1 :],
            type=ArrayType.VAL,
        )

        sparse_idx_def = ArrayDef(
            name=array_def.name + "_sidx",
            axes=array_def.axes[: idx - 1] + [axis.name],
            dims=[axis.length] + array_def.dims[: idx - 1],
            type=ArrayType.IDX,
        )

        sparse_off_def = ArrayDef(
            name=array_def.name + "_soff",
            axes=array_def.axes[:idx],
            dims=array_def.dims[:idx],
            type=ArrayType.OFFSET,
        )
        soff_array_name = sparse_off_def.name

        new_array_defs.replace(
            [array_def.name], [sparse_val_def, sparse_idx_def, sparse_off_def]
        )

    # 3. Handle schedule
    axis_ref = new_schedule.find_axis_ref(prev_axis_name)
    assert axis_ref is not None

    if isinstance(axis_ref, DenseAxisIterator):
        parent_axis_ref = axis_ref.get_parent()
        assert isinstance(
            parent_axis_ref, AxisVisitor
        ), f"Parent axis reference should be an AxisVisitor, got {parent_axis_ref}"
        sparse_axis_iter = SparseAxisIterator(
            axis=axis.name,
            body=axis_ref.body,
            name=axis_ref.name,
            offset_array=soff_array_name or "<ILLEGAL>",
            parent=parent_axis_ref,
        )
        parent_axis_ref.body = sparse_axis_iter

        axis_ref.body.replace_array_ref_idx(
            array_name=array_defs[ArrayType.B_VAL][0].name,
            old=axis_ref.name,
            new=ArrayRef(array=sparse_idx_def.name, indices=[sparse_axis_iter.name]),
        )

    elif isinstance(axis_ref, AtomicFormatOp):
        parent_axis_ref = axis_ref.get_parent()
        assert isinstance(
            parent_axis_ref, AxisVisitor
        ), f"Parent axis reference should be an AxisVisitor, got {parent_axis_ref}"
        sparse_axis_slicer = SparseAxisSlicer(
            axis=axis.name,
            body=axis_ref,
            offset_array=soff_array_name or "<ILLEGAL>",
            parent=parent_axis_ref,
        )
        parent_axis_ref.body = sparse_axis_slicer
        axis_ref.replace_symbol(parent_axis_ref.name, sparse_axis_slicer.name)
        axis_ref.append_array_ref_idx(
            array_name=array_defs[ArrayType.B_VAL][0].name,
            new=ArrayRef(array=sparse_idx_def.name, indices=[sparse_axis_slicer.name]),
        )
        if axis_ref.axes[0] == prev_axis_name:
            axis_ref.axes = axis.name, axis_ref.axes[1]
        elif axis_ref.axes[1] == prev_axis_name:
            axis_ref.axes = axis_ref.axes[0], axis.name
        else:
            raise ValueError(
                f"None of the axes match {prev_axis_name}: {axis_ref.axes}"
            )
    else:
        raise ValueError(
            f"Unknown axis reference: {axis_ref} for sparsify({axis.name})"
        )

    return new_schedule, new_array_defs, new_format


def apply_spawn_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    format: Format,
    root_format: Format,
    transform: SpawnTransformation,
) -> Tuple[Schedule, ArrayDefCollection, Format]:
    schedule, array_defs, format = (
        deepcopy(schedule),
        deepcopy(array_defs),
        deepcopy(format),
    )

    # 1. Handle format
    parent = format
    while isinstance(parent.child, Format):
        parent = parent.child
    assert parent.child is None

    parent.child = Format(
        axes=[
            df_axis(
                name=parent.axes[-2].name,
                direction=parent.axes[-2].direction,
                length=parent.axes[-2].length,
            ),
            df_axis(
                name=parent.axes[-1].name,
                direction=parent.axes[-1].direction,
                length=parent.axes[-1].length,
            ),
        ]
    )

    # 2. Handle array_defs
    # Nothing to do

    # 3. Handle schedule
    # Nothing to do

    return schedule, array_defs, format


def apply_cooize_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    format: Format,
    root_format: Format,
    transform: CooizeTransformation,
) -> Tuple[Schedule, ArrayDefCollection, AtomicFormat]:
    schedule, array_defs = (
        deepcopy(schedule),
        deepcopy(array_defs),
    )
    # 1. Handle format
    axis_x, axis_y = format.axes[0].name, format.axes[1].name
    assert len(format.axes) == 2 and format.is_dense()

    new_format = coo_atomic_format(
        x_name=axis_x,
        y_name=axis_y,
        col_major=format.axes[0].direction == Direction.COL,
    )
    new_format.axes[0].length = format.axes[0].length
    new_format.axes[1].length = format.axes[1].length

    # 2. Handle array_defs
    for val_def in array_defs[ArrayType.VAL]:
        idx_x, idx_y = val_def.index_of(axis_x), val_def.index_of(axis_y)
        # idx_x and idx_y should be the last two dimensions
        assert idx_x and idx_y and idx_x + 1 == idx_y and idx_y == len(val_def.axes) - 1
        coo_off_def = ArrayDef(
            name=val_def.name + "_coo_off",
            axes=val_def.axes[:-2],
            dims=val_def.dims[:-2],
            type=ArrayType.OFFSET,
        )
        coo_idx_def = ArrayDef(
            name=val_def.name + "_coo_idx",
            axes=[f"{axis_x}_{axis_y}_coo"],
            dims=[VARLEN],
            type=ArrayType.COO_IDX,
        )
        coo_val_def = ArrayDef(
            name=val_def.name + "_coo_val",
            axes=[f"{axis_x}_{axis_y}_coo"],
            dims=[VARLEN],
            type=ArrayType.VAL,
        )
        array_defs.replace([val_def.name], [coo_idx_def, coo_val_def, coo_off_def])

    # 3. Handle schedule
    axis_ref = schedule.find_axis_ref(axis_x)
    assert axis_ref is schedule.find_axis_ref(axis_y)
    assert isinstance(axis_ref, AtomicFormatOp)
    assert axis_ref.type == AtomicFormatType.DENSE

    new_axis_ref = AtomicFormatOp(
        axes=(axis_x, axis_y),
        type=AtomicFormatType.COO,
        args=[
            ArrayRef(array=val_def.name + "_coo_off", indices=[]),
            ArrayRef(array=val_def.name + "_coo_idx", indices=[]),
            ArrayRef(array=val_def.name + "_coo_val", indices=[]),
            next(
                filter(
                    lambda arg: isinstance(arg, ArrayRef) and arg.array == "B_val",
                    axis_ref.args,
                )
            ),
        ],
        parent=axis_ref.get_parent(),
    )

    parent_axis_ref = axis_ref.get_parent()
    if parent_axis_ref is not None:
        assert isinstance(parent_axis_ref, AxisVisitor)
        parent_axis_ref.body = new_axis_ref
    else:
        logger.warning(
            f"No parent axis reference found for {axis_ref} during cooize transformation. "
            f"Make sure this is what you want."
        )
        schedule = new_axis_ref

    return schedule, array_defs, new_format


def apply_mcoize_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    format: Format,
    root_format: Format,
    transform: McoizeTransformation,
) -> Tuple[Schedule, ArrayDefCollection, AtomicFormat]:
    schedule, array_defs = (
        deepcopy(schedule),
        deepcopy(array_defs),
    )
    # 1. Handle format
    axis_x, axis_y = format.axes[0].name, format.axes[1].name
    assert len(format.axes) == 2 and format.is_dense()

    new_format = mco_atomic_format(
        x_name=axis_x,
        y_name=axis_y,
        col_major=format.axes[0].direction == Direction.COL,
    )
    new_format.axes[0].length = format.axes[0].length
    new_format.axes[1].length = format.axes[1].length

    # 2. Handle array_defs
    for val_def in array_defs[ArrayType.VAL]:
        idx_x, idx_y = val_def.index_of(axis_x), val_def.index_of(axis_y)
        # idx_x and idx_y should be the last two dimensions
        assert idx_x and idx_y and idx_x + 1 == idx_y and idx_y == len(val_def.axes) - 1
        mco_mask_def = ArrayDef(
            name=val_def.name + "_mco_mask",
            axes=val_def.axes[:-2],
            dims=val_def.dims[:-2],
            type=ArrayType.MASK,
        )
        mco_val_def = ArrayDef(
            name=val_def.name + "_mco_val",
            axes=[f"{axis_x}_{axis_y}_mco"],
            dims=[VARLEN],
            type=ArrayType.VAL,
        )
        mco_off_def = ArrayDef(
            name=val_def.name + "_mco_off",
            axes=val_def.axes[:-2],
            dims=val_def.dims[:-2],
            type=ArrayType.OFFSET,
        )
        array_defs.replace([val_def.name], [mco_mask_def, mco_val_def, mco_off_def])

    # 3. Handle schedule
    axis_ref = schedule.find_axis_ref(axis_x)
    assert axis_ref is schedule.find_axis_ref(axis_y)
    assert isinstance(axis_ref, AtomicFormatOp)
    assert axis_ref.type == AtomicFormatType.DENSE

    new_axis_ref = AtomicFormatOp(
        axes=(axis_x, axis_y),
        type=AtomicFormatType.COO,
        args=[
            ArrayRef(array=val_def.name + "_mco_off", indices=[]),
            ArrayRef(array=val_def.name + "_mco_mask", indices=[]),
            ArrayRef(array=val_def.name + "_mco_val", indices=[]),
            next(
                filter(
                    lambda arg: isinstance(arg, ArrayRef) and arg.array == "B_val",
                    axis_ref.args,
                )
            ),
        ],
        parent=axis_ref.get_parent(),
    )

    parent_axis_ref = axis_ref.get_parent()
    if parent_axis_ref is not None:
        assert isinstance(parent_axis_ref, AxisVisitor)
        parent_axis_ref.body = new_axis_ref
    else:
        logger.warning(
            f"No parent axis reference found for {axis_ref} during cooize transformation. "
            f"Make sure this is what you want."
        )
        schedule = new_axis_ref

    return schedule, array_defs, new_format


def add_metadata(schedule: Optional[Schedule], format: Format) -> None:
    while schedule is not None:
        if (
            isinstance(schedule, AxisVisitor)
            and format.get_axis(schedule.axis).direction == Direction.ROW
        ):
            schedule.metadata = {
                "reduction_axis": True,
            }
            return
        schedule = schedule.child


def computent_from_rts(name: str, rts: TransformationSequence) -> Computent:
    format = deepcopy(DENSE_FORMAT)
    print(format)
    root_format = Format(axes=[], child=format)
    assert isinstance(root_format.child, Format)
    parent_format = root_format
    schedule = deepcopy(DENSE_SCHEDULE)
    array_defs = deepcopy(DENSE_ARRAY_DEFS)

    atomic_format_reached = False
    for transform in rts.sequence:
        assert (
            not atomic_format_reached
        ), f"Transforms remaining after atomic format reached: {transform}"
        logger.debug(f"Transform begins: {transform}")
        if isinstance(transform, SplitTransformation):
            schedule, array_defs, format = apply_split_transformation(
                schedule, array_defs, format, root_format.child, transform
            )
            parent_format.child = format
        elif isinstance(transform, SwapTransformation):
            schedule, array_defs, format = apply_swap_transformation(
                schedule, array_defs, format, root_format.child, transform
            )
            parent_format.child = format
        elif isinstance(transform, SparsifyTransformation):
            schedule, array_defs, format = apply_sparsify_transformation(
                schedule, array_defs, format, root_format.child, transform
            )
            parent_format.child = format
        elif isinstance(transform, SpawnTransformation):
            schedule, array_defs, format = apply_spawn_transformation(
                schedule, array_defs, format, root_format.child, transform
            )
            parent_format.child = format
            parent_format = format
            format = format.child
            assert isinstance(format, Format)
        elif isinstance(transform, CooizeTransformation):
            schedule, array_defs, atomic_format = apply_cooize_transformation(
                schedule, array_defs, format, root_format.child, transform
            )
            parent_format.child = atomic_format
            atomic_format_reached = True
        elif isinstance(transform, McoizeTransformation):
            schedule, array_defs, atomic_format = apply_mcoize_transformation(
                schedule, array_defs, format, root_format.child, transform
            )
            parent_format.child = atomic_format
            atomic_format_reached = True
        else:
            raise ValueError(f"Unknown transformation: {transform}")

        logger.debug(schedule)
        logger.debug(array_defs)
        logger.debug(root_format.child)
        logger.debug("-" * 100)

    assert isinstance(root_format.child, Format)
    add_metadata(schedule, root_format.child)
    return Computent(
        name=name,
        schedule=schedule,
        array_defs=array_defs,
        format=root_format.child,
    )

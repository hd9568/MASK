from __future__ import annotations
from abc import ABC, abstractmethod
from typing import overload
from copy import deepcopy
import copy
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

from sparsene.op_gen.computent.arraydef import *

class AtomicFormatType(Enum):
    DENSE = 0
    COO = 1
    MCO = 2

class NameManager:
    def __init__(self):
        self.counter = 0

    def new_name(self) -> str:
        self.counter += 1
        return f"i{self.counter}"


name_manager = NameManager()

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

    @abstractmethod
    def get_atomicformatop(
        self
    ) -> Schedule: ...

@dataclass(kw_only=True)
class AtomicFormatOp(Schedule):
    axes: Tuple[str, str]
    axes_len: Tuple[Expr, Expr]
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

    def append_symbol(self, arg_list: List[str], old: str, new: str) -> None:
        for argi in arg_list:
            for argj in self.args:
                if isinstance(argj, ArrayRef):
                    if argi == argj.array:
                        argj.append_symbol(old, new)
                elif argi == argj:
                    self.args.append(new)
        for argj in self.args:
            if not isinstance(argj, ArrayRef) and argj == old:
                self.args.append(new)

    def replace_all_array_ref_idx(self, old: str, new: List[Variable]) -> None:
        for i in range(len(self.args)):
            arg = self.args[i]
            if isinstance(arg, ArrayRef):
                arg.replace_index(old, new)

    def __str__(self, indent_width: int = 4) -> str:
        return (
            f"AtomicFormatOp[{self.type.name}]({self.axes[0]}(len={self.axes_len[0]}), {self.axes[1]}(len={self.axes_len[1]}))"
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
    
    def get_atomicformatop(self) -> Schedule:
        return self

@dataclass(kw_only=True)
class AxisVisitor(Schedule, ABC):
    induction_var: str = ""
    axis: str
    body: Schedule
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.induction_var == "":
            self.induction_var = name_manager.new_name()

    def is_leaf(self) -> bool:
        return False

    def get_child(self) -> Schedule | None:
        return self.body

    def refers_to(self, axis: str) -> bool:
        return axis == self.axis

    def replace_symbol(self, old: str, new: str) -> None:
        if self.induction_var == old:
            raise ValueError(
                f"Cannot replace the name of an {self.__class__.__name__} itself: {self}"
            )
        self.body.replace_symbol(old, new)

    def replace_all_array_ref_idx(self, old: str, new: List[Variable]) -> None:
        self.body.replace_all_array_ref_idx(old, new)

    def replace_array_ref_idx(self, array_name: str, old: str, new: Variable):
        self.body.replace_array_ref_idx(array_name, old, new)

    def get_atomicformatop(self) -> Schedule:
        return self.body.get_atomicformatop()

    @property
    @abstractmethod
    def is_sparse(self) -> bool: ...

    def __str__(self, indent_width: int = 4) -> str:
        return (
            f"for {self.induction_var} in {self.__class__.__name__}({self.axis})"
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
    splen_array: str
    splen: Expr
    

    @property
    def is_sparse(self) -> bool:
        return True

    def __str__(self, indent_width: int = 4) -> str:
        return (
            f"for {self.induction_var} in {self.__class__.__name__}({self.axis})(splen={self.splen})"
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

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

from sparsene.op_gen.computent.symbolTable import *

@dataclass(kw_only=True)
class ArrayRef:
    array: str
    indices: List[Variable]
    # arraydef: ArrayDef = None

    def replace_symbol(self, old: str, new: str) -> None:
        for i in range(len(self.indices)):
            index = self.indices[i]
            if isinstance(index, ArrayRef):
                index.replace_symbol(old, new)
            elif index == old:
                self.indices[i] = new
    
    # append new inductor to current indices
    def append_symbol(self, old: str, new: str) -> None: 
        for i in range(len(self.indices)):
            index = self.indices[i]
            if isinstance(index, ArrayRef):
                index.append_symbol(old, new)
            elif index == old:
                self.indices.append(new)

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
        # rewrite using symbol table
        st = SymbolTable.global_table("arraydef")
        assert st.get(self.array) is not None
        arraydef = st.get(self.array)
        assert isinstance(arraydef, ArrayDef)
        parts = []
        # assert(len(arraydef.axes) == len(self.indices))
        for i in range(len(arraydef.axes)):
            if i < len(self.indices):
                parts.append(str(self.indices[i]))
            else:
                parts.append("*")
        index_str = ", ".join(parts)
        indent = " " * indent_width
        return f"{self.array}[{index_str}]"


Variable = ArrayRef | str | Expr

class ArrayType(Enum):
    VAL = 0
    IDX = 1
    OFFSET = 2
    COO_IDX = 3
    MASK = 4
    B_VAL = 5
    C_VAL = 6
    SPLEN = 7

class DataType(Enum):
    FLOAT = 0
    INT = 1

class ArrayDef:
    name: str
    axes: List[str]
    dims: List[Expr]
    type: ArrayType
    datatype: DataType

    def __init__(
        self,
        name: str,
        axes: List[str],
        dims: Sequence[Expr | int | str],
        type: ArrayType,
        datatype: DataType,
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
        self.datatype = datatype

    def index_of(self, axis: str) -> Optional[int]:
        for i, dim in enumerate(self.axes):
            if dim == axis:
                return i
        return None

    def __str__(self) -> str:
        # for d in self.dims:
        #     print(d)
        dims_str = [str(d) for d in self.dims]  # 强制调用 NewExpr.__str__
        return f"ArrayDef({self.name}, {self.datatype}, {self.axes}, {dims_str}, {self.type})"

    def __repr__(self) -> str:
        return self.__str__()

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
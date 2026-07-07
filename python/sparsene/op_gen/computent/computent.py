from __future__ import annotations
from abc import ABC, abstractmethod
from typing import overload
from copy import deepcopy
import copy
from sympy import Array, Function, Symbol, Expr, Indexed
from sympy import IndexedBase, Idx, symbols, Integer
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
    CsrizeTransformation,
    EllizeTransformation,
    DiaizeTransformation,
)
from sparsene.format.format import (
    Direction,
    Format,
    AtomicFormat,
    Axis,
    atomic_format,
    df_axis,
    sv_axis,
    coo_atomic_format,
    mco_atomic_format,
    csr_atomic_format,
    ell_atomic_format,
    dia_atomic_format,
    Expr,
    Number,
    Symbol,
    VARLEN,
)
from sparsene.formats.dense import get_dense_format

from sparsene.logging import get_logger

from sparsene.op_gen.computent.symbolTable import *
from sparsene.op_gen.computent.schedule import *
from sparsene.op_gen.computent.arraydef import *



logger = get_logger(__name__)


class MyNameManager:
    _name_counts: dict[str, int] = {}

    @classmethod 
    def init_name(cls, name: str):
        if name not in cls._name_counts:
            cls._name_counts[name] = 0
        else:
            raise ValueError(
                f"MyNameManager init duplicated var name error"
            )    
    
    @classmethod
    def new_name(cls, name: str) -> str:
        if name not in cls._name_counts:
            cls._name_counts[name] = 0
            return f"{name}"
        cls._name_counts[name] += 1
        return f"{name}_{cls._name_counts[name]}"

    @classmethod
    def reset(cls):
        cls._name_counts.clear()

DENSE_ARRAY_DEFS = ArrayDefCollection(
    val=ArrayDef(
        name="val",
        axes=["X", "Y"],
        dims=["M", "K"],
        type=ArrayType.VAL,
        datatype=DataType.FLOAT,
    ),
    B_val=ArrayDef(
        name="B_val",
        axes=["Y", "N"],
        dims=["K", "N"],
        type=ArrayType.B_VAL,
        datatype=DataType.FLOAT,
    ),
    C_val=ArrayDef(
        name="C_val",
        axes=["X", "N"],
        dims=["M", "N"],
        type=ArrayType.C_VAL,
        datatype=DataType.FLOAT,
    ),
)

# st = SymbolTable.global_table("arraydef")
# st.add("val", DENSE_ARRAY_DEFS.__getitem__("val"))
# st.add("B_val", DENSE_ARRAY_DEFS.__getitem__("B_val"))
# st.add("C_val", DENSE_ARRAY_DEFS.__getitem__("C_val"))

DENSE_FORMAT = get_dense_format()
# st = SymbolTable.global_table("axis")
# for axis in DENSE_FORMAT.axes:
#     st.add(axis.name, axis)

DENSE_SCHEDULE: Schedule = AtomicFormatOp(
    axes=("X", "Y"),
    axes_len=(Symbol("M"), Symbol("K")),
    type=AtomicFormatType.DENSE,
    args=[
        ArrayRef(array="val", indices=["_", "_"]),
        ArrayRef(array="B_val", indices=["_", "_"]),
    ],
    parent=None,
)


@dataclass(kw_only=True)
class Computent:
    name: str
    schedule: Schedule
    array_defs: ArrayDefCollection
    format: Format
    varlen2LenArrayTable: Dict[str, ArrayDef] = field(default_factory=dict)

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

varlen2LenArrayTable = SymbolTable.global_table("varlen2LenArrayTable")

def update_st_array_def(array_defs: ArrayDefCollection):
    st = SymbolTable.global_table("arraydef")
    for array_def_str, array_def in array_defs.array_defs.items():
        st[array_def_str] = array_def

def update_st_axis(format: Format):
    st = SymbolTable.global_table("axis")
    for axis in format.axes:
        st[axis.name] = axis

def update_array_ref_recursive(arg: Variable, new_axis_ref, array_defs, axis_to_split_name, split_size):
    """
    递归更新 ArrayRef 及其嵌套的 indices
    """
    if not isinstance(arg, ArrayRef):
        return
    
    # 1. 获取当前数组对应的定义
    array_def = None
    try:
        array_def = array_defs[arg.array]
    except (KeyError, TypeError, AttributeError):
        pass
    if not array_def:
        # 如果找不到定义（比如某些临时中间变量），则只递归子项
        for index in arg.indices:
            update_array_ref_recursive(index, new_axis_ref, array_defs, axis_to_split_name, split_size)
        return

    # 2. 检查当前 ArrayRef 的 ArrayDef 中是否包含被 split 的轴
    target_idx_new = array_def.index_of(new_axis_ref.axis)
    target_idx_origin = array_def.index_of(axis_to_split_name)
    print("array_def", array_def, "target_idx_new", target_idx_new, "new_axis_ref.axis", new_axis_ref.axis)
    if target_idx_new is not None:
        # 执行你原来的“位置对齐与插入”逻辑
        new_len = len(array_def.axes)
        old_len = len(arg.indices)

        # 长度校验与补全逻辑
        if old_len == 0:
            arg.indices = ["_"] * new_len
        elif old_len < new_len - 1:
            # 如果差距过大，说明之前的轴也没对齐，先补齐到 new_len - 1
            arg.indices.extend(["_"] * (new_len - 1 - old_len))
        
        # 再次确认当前状态
        current_len = len(arg.indices)
        if current_len == new_len - 1:
            # 分裂情况：插入新的感应变量 (i2)
            arg.indices.insert(target_idx_new, new_axis_ref.induction_var)
        elif current_len == new_len:
            # 已经对齐：直接覆盖 (通常发生在多次变换重叠时)
            arg.indices[target_idx_new] = new_axis_ref.induction_var
        else:
            # 这里的逻辑可以根据你的严谨程度决定抛异常还是强制处理
            pass
    elif target_idx_origin is not None:
        # 保证 indices 长度
        while len(arg.indices) <= target_idx_origin:
            arg.indices.append("_")
        old_idx = arg.indices[target_idx_origin]

        # arg.indices[target_idx_origin] = str(new_axis_ref.induction_var + " * " + str(split_size) + " + " + str(old_idx))
        iv = Symbol(new_axis_ref.induction_var)
        sz = Integer(split_size) if isinstance(split_size, int) else Symbol(str(split_size))

        if old_idx == "_":
            base = Symbol("_")
        elif isinstance(old_idx, str):
            # 如果原本是字符串，可以尝试解析成 Sympy 表达式
            from sympy import sympify
            base = sympify(old_idx)
        else:
            base = old_idx

        arg.indices[target_idx_origin] = iv * sz + base
        # print(type(arg.indices[target_idx_origin]), "updated index:", arg.indices[target_idx_origin])

    # 3. 递归处理 indices 中的嵌套对象 (处理 B_val[val_sidx[...]] 的关键)
    for index in arg.indices:
        update_array_ref_recursive(index, new_axis_ref, array_defs, axis_to_split_name, split_size)

def apply_split_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    spformat: Format,
    root_format: Format,
    transform: SplitTransformation,
) -> Tuple[Schedule, ArrayDefCollection, Format]:
    schedule, array_defs, spformat = (
        deepcopy(schedule),
        deepcopy(array_defs),
        deepcopy(spformat),
    )
    update_st_array_def(array_defs)
    update_st_axis(spformat)

    axis_idx = transform.axes[0]

    length_before = spformat.axes[axis_idx].length
    new_length_outer = length_before / transform.split_size
    print("mydebug", new_length_outer, type(length_before), type(transform.split_size))
    new_length_inner = transform.split_size

    # 1. Handle format
    axis_to_split = spformat.axes[axis_idx]
    new_axis_outer = Axis(
        name=MyNameManager.new_name(axis_to_split.name + "_o"),
        direction=axis_to_split.direction,
        length=new_length_outer,
        is_sparse=axis_to_split.is_sparse,
        is_varlen=axis_to_split.is_varlen,
    )
    #! the inner axis always dense fixed
    new_axis_inner = Axis(
        name=MyNameManager.new_name(axis_to_split.name + "_i"),
        direction=axis_to_split.direction,
        length=new_length_inner,
        is_sparse=False,
        is_varlen=False,
    )
    spformat.axes = (
        spformat.axes[:axis_idx]
        + [new_axis_outer]
        + [new_axis_inner]
        + spformat.axes[axis_idx + 1 :]
    )
    update_st_axis(spformat)
    
    # 2. Handle array_defs
    for array_def in array_defs:
        idx_dim = array_def.index_of(axis_to_split.name)

        if idx_dim is None:
            continue
    
        if array_def.type == ArrayType.B_VAL:
            continue

        array_def.axes[idx_dim] = new_axis_outer.name
        array_def.axes.insert(idx_dim + 1, new_axis_inner.name)

        array_def.dims[idx_dim] = new_length_outer
        array_def.dims.insert(idx_dim + 1, new_length_inner)
        
    update_st_array_def(array_defs)

    # 3. Handle schedule
    print("---------")
    print(axis_to_split.name)
    print("---------")
    axis_ref = schedule.find_axis_ref(axis_to_split.name)
    if isinstance(axis_ref, (DenseAxisIterator, SparseAxisIterator)):
        print("mydebug: split transformation axis_ref is a iterator")
        new_inner_axis_ref = DenseAxisIterator(
            axis=new_axis_inner.name,
            body=axis_ref.body,
            parent=axis_ref
        )
        axis_ref.axis = new_axis_outer.name
        axis_ref.body = new_inner_axis_ref

        new_inner_axis_ref.body.replace_all_array_ref_idx(
            axis_ref.induction_var,
            [axis_ref.induction_var, new_inner_axis_ref.induction_var],
        )

    elif isinstance(axis_ref, AtomicFormatOp):
        print("mydebug: split transformation axis_ref is a atomic format op")
        print("axis_ref.axes[0]", axis_ref.axes[0], "axis_ref.axes[1]", axis_ref.axes[1])
        print("split direction", axis_to_split.direction)
        #> update atomic format op axes
        if axis_ref.axes[0] == axis_to_split.name:
            axis_ref.axes = new_axis_inner.name, axis_ref.axes[1]
        elif axis_ref.axes[1] == axis_to_split.name:
            axis_ref.axes = axis_ref.axes[0], new_axis_inner.name
        else:
            raise ValueError(
                f"None of the axes match {axis_to_split.name}: {axis_ref.axes}"
            )
        #> update atomic format op lens
        if axis_ref.axes[0] == new_axis_inner.name:
            axis_ref.axes_len = (new_length_inner, axis_ref.axes_len[1])
        elif axis_ref.axes[1] == new_axis_inner.name:
            axis_ref.axes_len = (axis_ref.axes_len[0], new_length_inner)
        else:
            raise ValueError(
                f"None of the axes match {new_axis_inner.name}: {axis_ref.axes}"
            )
        # if this axis is fixed, then we need to create a dense axis iterator for the new outer axis
        # else if this axis is varlen, then we need to create a sparse axis iterator for the outer axis
        # (the first axis of the atomic format op is split across the boundary)
        # axis_ref.axes = new_axis_inner.name, axis_ref.axes[1]
        if axis_to_split.is_varlen:
            # 1. 找到长度定义数组
            varlen_table = SymbolTable.global_table("varlen2LenArrayTable")
            sparse_len_def = varlen_table.get(str(axis_to_split.length))
            if sparse_len_def is None:
                raise ValueError(f"Could not find length array for varlen axis: {axis_to_split.length}")
            # 2. 构造 val_len 的访问索引
            # sparse_len_def.axes 可能是 ['X_o']
            # 我们需要从 parent 链条中找到这些轴对应的 induction_var
            len_indices = []
            current_p = axis_ref.get_parent()
            # 建立一个 轴名 -> 感应变量 的映射
            axis_to_var_map = {}
            while current_p is not None:
                axis_to_var_map[current_p.axis] = current_p.induction_var
                current_p = current_p.get_parent()

            sympy_indices = []
            for adef_axis in sparse_len_def.axes:
                var_name = axis_to_var_map.get(adef_axis, "_")
                sympy_indices.append(Symbol(var_name))
            
            val_len_indexed = None
            # 3. 使用 SymPy 构造 splen 表达式
            if (len(sympy_indices) == 0):
                val_len_indexed = Symbol(sparse_len_def.name)
            else:
                # 构造基础数组名
                val_len_base = IndexedBase(sparse_len_def.name)
                # 构造访问项：val_len[i1]
                # 如果有多个维度，IndexedBase 会自动处理为 val_len[i1, i2...]
                val_len_indexed = val_len_base[tuple(sympy_indices)]

            # 构造除法表达式
            # 如果 new_length_inner 是整数，使用 Integer 包装以获得更好的 SymPy 支持
            divisor = new_length_inner
            
            # 最终的 SymPy Expr: val_len[i1] / BLK_K
            splen_sympy_expr = val_len_indexed / divisor

            # 4. 创建新的 SparseAxisIterator
            # 注意：这里的 splen_array 现在持有一个 sympy.Expr 对象
            new_axis_ref = SparseAxisIterator(
                axis=new_axis_outer.name,
                body=axis_ref,
                parent=axis_ref.get_parent(),
                offset_array=None,
                splen_array=None,
                splen=splen_sympy_expr
            )
        else:
            new_axis_ref = DenseAxisIterator(
                axis=new_axis_outer.name,
                body=axis_ref,
                parent=axis_ref.get_parent()
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
        #> 如果arg.indices中含有AtomicFormatOp中的axis，则对应位置是"_"，如果包含new_axis_ref.axis，对应位置使用new_axis_ref.induction_var
        for arg in axis_ref.args:
            update_array_ref_recursive(arg, new_axis_ref, array_defs, axis_to_split.name, new_length_inner)
        
    else:
        raise ValueError(f"Unknown axis reference: {axis_ref}")

    return schedule, array_defs, spformat

def apply_swap_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    spformat: Format,
    root_format: Format,
    transform: SwapTransformation,
) -> Tuple[Schedule, ArrayDefCollection, Format]:
    schedule, array_defs, spformat = (
        deepcopy(schedule),
        deepcopy(array_defs),
        deepcopy(spformat),
    )
    update_st_array_def(array_defs)
    update_st_axis(spformat)

    axis0_name = spformat.axes[transform.axes[0]].name
    axis1_name = spformat.axes[transform.axes[1]].name

    # 1. Handle format
    spformat.axes[transform.axes[0]], spformat.axes[transform.axes[1]] = (
        spformat.axes[transform.axes[1]],
        spformat.axes[transform.axes[0]],
    )
    update_st_axis(spformat)

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

    update_st_array_def(array_defs)

    # 3. Handle schedule
    print("swap transformation: axis0 is " + axis0_name + " axis1 is " + axis1_name)
    axis0_ref = schedule.find_axis_ref(axis0_name)
    axis1_ref = schedule.find_axis_ref(axis1_name)
    if axis0_ref is None or axis1_ref is None:
        raise ValueError(f"Unknown axis reference: {axis0_ref} or {axis1_ref}")
    
    if axis0_ref is axis1_ref:
        print("swap transformation: axis0 is axis1")
        assert isinstance(axis0_ref, AtomicFormatOp)
        # they are in the same atomic op, just swap the axes and axes_lens
        axis0_ref.axes = axis0_ref.axes[1], axis0_ref.axes[0]
        axis0_ref.axes_len = axis0_ref.axes_len[1], axis0_ref.axes_len[0]
        #> 这里也需要swap一下AtomicFormatOp中的args内部的ArrayRef的indices。例如从val[x_idx, y_idx]变为val[y_idx, x_idx]，要实现这个功能需要建立起arrayref的indices和arraydef.axes之间的映射关系。但是由于AtomicForamtOp的axes对应在args中的indices是"_"，所以可以不考虑交换indices带来的问题
        for arg in axis0_ref.args:
            if isinstance(arg, ArrayRef):
                array_def = array_defs[arg.array]
                pos0 = array_def.index_of(axis0_name)
                pos1 = array_def.index_of(axis1_name)
                if pos0 is not None and pos1 is not None:
                    assert pos0 < len(arg.indices) and pos1 < len(arg.indices)
                    arg.indices[pos0], arg.indices[pos1] = arg.indices[pos1], arg.indices[pos0]


    else:
        raise ValueError(
            f"Axis references do not match: {axis0_ref} and {axis1_ref} for "
            f"swap({transform.axes[0]}, {transform.axes[1]})"
        )
    
    return schedule, array_defs, spformat

def apply_sparsify_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    spformat: Format,
    root_format: Format,
    transform: SparsifyTransformation,
) -> Tuple[Schedule, ArrayDefCollection, Format]:
    new_schedule, new_array_defs, new_format = (
        deepcopy(schedule),
        ArrayDefCollection(),
        deepcopy(spformat),
    )
    update_st_axis(new_format)

    axis_gid = transform.axes[0]
    prev_axis_name = spformat.axes[axis_gid].name
    new_axis_name = MyNameManager.new_name(prev_axis_name + "_s")

    # 1. Handle format
    axis = new_format.axes[axis_gid]
    axis.is_sparse = True
    axis.is_varlen = True
    axis.name = new_axis_name
    axis.set_varlen()

    update_st_axis(new_format)

    new_sparse_val_def = []

    # 2. Handle array_defs
    soff_array_name = None
    for array_def in array_defs:
        if array_def.type == ArrayType.B_VAL:
            # do not change B_val
            new_array_defs.add(array_def)
            continue

        idx = array_def.index_of(prev_axis_name)
        if idx is None:
            new_array_defs.add(array_def)
            continue

        def has_varlen(expr:Expr):
            from sympy import preorder_traversal, Function
            varlen = Function('varlen')
            for node in preorder_traversal(expr):
                if node.func == varlen:
                    return True
            return False
        #! 如果val修改了，还需要修改idx吗？可能需要改一下，一个direction留一个idx就行了
        if array_def.type == ArrayType.IDX:
            # 如果当前稀疏轴在sidx相同方向进行稀疏，则直接修改sidx矩阵
            # 例如val_sidx [Xo, Yso, Ysi]对Ysi稀疏化 -> val_sidx[Xo, Yso, Ysis],原来稀疏的是Ys,后来再对Ysi稀疏
            # 如果与当前sidx不同方向进行稀疏化,也直接修改sidx矩阵
            # 例如val_sidx [Xo, Yso, Ysi]对Xo稀疏化 -> val_sidx[Xos, Yso, Ysi]
            sparse_idx_def = ArrayDef(
                name=array_def.name,
                axes=array_def.axes[: idx] + [axis.name] + array_def.axes[idx + 1 : ],
                dims=array_def.dims[: idx] + [axis.length] + array_def.dims[idx + 1 : ],
                type=array_def.type,
                datatype=array_def.datatype,
            )
            new_array_defs.add(sparse_idx_def)
            # idx_array_direction = None
            # for i in len(array_def.dims):
            #     if idx_array_direction is not None:
            #         break
            #     if has_varlen(array_def.dims[i]):
            #         for j in len(spformat.axes):
            #             if idx_array_direction is not None:
            #                 break
            #             if spformat.axes[j].name == array_def.axes[i]:
            #                 idx_array_direction = spformat.axes[j].direction
            # assert idx_array_direction is not None
            # if idx_array_direction == axis.direction:
            #     # 在sidx的相同方向进行稀疏化，直接修改sidx矩阵
            #     # e.g. val_sidx [Xo, Yso, Ysi]对Ysi稀疏化 -> val_sidx[Xo, Yso, Ysis]

            # else:
            #     # 在与当前sidx不同方向进行稀疏化，不修改当前sidx矩阵
            #     # e.g. val_sidx [Xo, Yso, Ysi]对Xo稀疏化 -> val_sidx[Xos, Yso, Ysi]
        elif array_def.type == ArrayType.SPLEN:
            # 似乎和IDX矩阵相同,不管是否在SPLEN所负责的轴上稀疏化,结果仍然是只需要变化一个轴即可
            # 例如如果是val[Xo,Yo,Yis,Xi] & len[Xo,Yo]
            # 如果进一步在Yo稀疏化,得到len[Xo,Yos]
            # 如果进一步在Xo稀疏化,得到len[Xos,Yo]
            sparse_len_def = ArrayDef(
                name=array_def.name,
                axes=array_def.axes[: idx] + [axis.name] + array_def.axes[idx + 1 : ],
                dims=array_def.dims[: idx] + [axis.length] + array_def.dims[idx + 1 : ],
                type=array_def.type,
                datatype=array_def.datatype,
            )
            new_array_defs.add(sparse_len_def)
        elif array_def.type == ArrayType.VAL:
            sidx_gen_flag = 1
            print(array_def.dims)
            for i in range(len(array_def.dims)):
                if sidx_gen_flag == 0:
                    break
                if has_varlen(array_def.dims[i]):
                    for j in range(len(spformat.axes)):
                        if sidx_gen_flag == 0:
                            break
                        if spformat.axes[j].name == array_def.axes[i]:
                            if spformat.axes[j].direction == axis.direction:
                                sidx_gen_flag = 0

            

            sparse_val_def = ArrayDef(
                name=array_def.name,
                axes=array_def.axes[: idx] + [axis.name] + array_def.axes[idx + 1 : ],
                dims=array_def.dims[: idx] + [axis.length] + array_def.dims[idx + 1 : ],
                type=ArrayType.VAL,
                datatype=array_def.datatype
            )

            if sidx_gen_flag == 1:
                sparse_idx_def = ArrayDef(
                    name=MyNameManager.new_name(array_def.name + "_sidx"),
                    axes=array_def.axes[: idx] + [axis.name],
                    dims=array_def.dims[: idx] + [axis.length],
                    type=ArrayType.IDX,
                    datatype=DataType.INT,
                )

            sparse_len_def = ArrayDef(
                name=MyNameManager.new_name(array_def.name + "_len"),
                axes=array_def.axes[:idx],
                dims=array_def.dims[:idx],
                type=ArrayType.SPLEN,
                datatype=DataType.INT,
            )

            #! 将每个varlen和一个固定的arraydef绑定起来,便于之后找到特定的array
            varlen2LenArrayTable = SymbolTable.global_table("varlen2LenArrayTable")
            varlen2LenArrayTable[str(axis.length)] = sparse_len_def
        
            if sidx_gen_flag == 1:
                new_array_defs.replace(
                    [array_def.name], [sparse_val_def, sparse_idx_def, sparse_len_def]
                )
            else:
                new_array_defs.replace(
                    [array_def.name], [sparse_val_def, sparse_len_def]
                )
        elif array_def.type == ArrayType.C_VAL:
            sparse_c_def = ArrayDef(
                name=array_def.name,
                axes=array_def.axes[: idx] + [axis.name] + array_def.axes[idx + 1 : ],
                dims=array_def.dims[: idx] + [axis.length] + array_def.dims[idx + 1 : ],
                type=ArrayType.C_VAL,
                datatype=array_def.datatype
            )
            new_array_defs.add(sparse_c_def)
            # 记录下新的 C 定义，方便后续 schedule 处理
            # target_c_val_name = sparse_c_def.name

    update_st_array_def(new_array_defs)

    #! 3. Handle schedule
    axis_ref = new_schedule.find_axis_ref(prev_axis_name)
    assert axis_ref is not None

    if isinstance(axis_ref, DenseAxisIterator):
        # 1. 获取上下文映射 (axis name -> induction variable)
        loop_context = {}
        curr_parent = axis_ref.get_parent()
        while curr_parent is not None:
            loop_context[curr_parent.axis] = curr_parent.induction_var
            curr_parent = curr_parent.get_parent()
        
        # 当前轴的 induction var
        curr_var = axis_ref.induction_var

        varlen_table = SymbolTable.global_table("varlen2LenArrayTable")
        sparse_len_def = varlen_table.get(str(axis.length))
        if sparse_len_def:
            # 根据 val_len 的 axes ['X_o'] 找到对应的变量 i11
            len_indices = [Symbol(loop_context[ax]) for ax in sparse_len_def.axes if ax in loop_context]
            # splen = val_len[i11]
            # 注意：这里的长度是分块后的数量，如果 sparsify 作用于 Y_o，长度就是 val_len[i11]
            splen_expr = IndexedBase(sparse_len_def.name)[tuple(len_indices)]
        else:
            splen_expr = None
        
        # 3. 创建 SparseAxisIterator
        new_iterator = SparseAxisIterator(
            axis=axis.name,             # Y_o_s
            induction_var=curr_var,     # i12
            body=axis_ref.body,
            parent=axis_ref.get_parent(),
            splen=splen_expr,
            offset_array="None",
            splen_array="None",
        )
        # 更新树结构
        if axis_ref.get_parent():
            axis_ref.get_parent().body = new_iterator
        axis_ref.body.parent = new_iterator
        new_iterator.body = axis_ref.body
        new_iterator.parent = axis_ref.get_parent()
        axis_ref = new_iterator
        # 4. 深度递归更新内部 AtomicFormatOp 的 args
        atomic_format_op:AtomicFormatOp = axis_ref.get_atomicformatop()
        
        # 4.1 构造稀疏索引引用 (val_sidx[i11, i12])
        sidx_indices = []
        for s_axis in sparse_idx_def.axes:
            if s_axis == new_axis_name: # 当前正在稀疏的轴, Y_o_s
                sidx_indices.append(curr_var)
            elif s_axis in loop_context:
                sidx_indices.append(loop_context[s_axis])
            else:
                sidx_indices.append("_")

        # 创建嵌套的 ArrayRef：val_sidx[i11, i12]
        # TODO ArrayRef的indices只能是str|ArrayRef，i12 * BLK_K + _ 只能使用 str 表达，sidx[i11, i12] * BLK_K + _ 也必须使用str保存，非常不方便
        sidx_ref = ArrayRef(array=sparse_idx_def.name, indices=sidx_indices)
        # 4.2 遍历并替换args内部的符号
        matrix_B_val_name = new_array_defs[ArrayType.B_VAL][0].name
        for arg in atomic_format_op.args:
            if not isinstance(arg, ArrayRef):
                continue
            arg_adef = new_array_defs[arg.array]
            # case1: val
            if arg.array == sparse_val_def.name:
                new_indices = ["_"] * len(arg_adef.axes)
                full_context = {**loop_context, axis.name: curr_var}
                for i, adef_axis in enumerate(arg_adef.axes):
                    if adef_axis in full_context:
                        new_indices[i] = str(full_context[adef_axis])
                arg.indices = new_indices
            # case2: B_val
            elif arg.array == matrix_B_val_name:
                from sympy import sympify
                def to_sympy_expr(item):
                    """将字符串、ArrayRef 或现有 Expr 统一转换为 SymPy Expr"""
                    # 1. 处理基础类型
                    if isinstance(item, str):
                        return Symbol(item)
                    if isinstance(item, (int, float)):
                        return sympify(item)
                    if isinstance(item, Expr):
                        return item

                    # 2. 处理 ArrayRef 对象
                    if hasattr(item, 'array') and hasattr(item, 'indices'):
                        base = IndexedBase(item.array)
                        
                        # 使用 for 循环逐个转换索引项，构建一个新的列表
                        sym_indices = []
                        for idx in item.indices:
                            # 递归调用，确保嵌套的 ArrayRef 也能被转换
                            converted_idx = to_sympy_expr(idx)
                            sym_indices.append(converted_idx)
                        
                        # 核心修改点：
                        # 在 SymPy 中，base[tuple_obj] 等价于 base[i, j, k]
                        # 这样就避免了使用 *sym_indices 这种解包语法
                        return base[tuple(sym_indices)]

                    return sympify(item)
                def replace_symbol_in_arg(arg: ArrayRef, old_var: str, new_var_obj: any):
                    target_sym = Symbol(old_var)
                    
                    # 将要替换进去的对象（sidx_ref）转换为 SymPy 原生 Indexed 表达式
                    replacement_expr = to_sympy_expr(new_var_obj)

                    for i in range(len(arg.indices)):
                        idx_item = arg.indices[i]
                        
                        if hasattr(idx_item, 'array'): # 如果是嵌套的 ArrayRef
                            replace_symbol_in_arg(idx_item, old_var, new_var_obj)
                        else:
                            # 1. 将当前索引（如 "BLK_K*i12 + _"）转为 SymPy 表达式
                            expr = sympify(idx_item) if isinstance(idx_item, str) else idx_item
                            
                            # 2. 检查是否包含目标变量
                            if target_sym in expr.free_symbols:
                                # 3. 此时 replacement_expr 是 Indexed 类型，subs 不会报错
                                new_expr = expr.subs(target_sym, replacement_expr)
                                arg.indices[i] = new_expr
                print("before replacement, B_val arg indices:", arg.indices, "looking for var:", curr_var, type(curr_var), type(sidx_ref))
                replace_symbol_in_arg(arg, curr_var, sidx_ref)
        # parent_axis_ref = axis_ref.get_parent()
        # if parent_axis_ref is not None:
        #     assert isinstance(
        #         parent_axis_ref, AxisVisitor
        #     ), f"Parent axis reference should be an AxisVisitor, got {parent_axis_ref}"
        #     # splen type is Expr
        #     sparse_axis_iter = SparseAxisIterator(
        #         axis=axis.name,
        #         body=axis_ref.body,
        #         induction_var=axis_ref.induction_var,
        #         offset_array=f"OFFSET_ARRAY_NONE",
        #         splen_array=sparse_len_def.name or "<ILLEGAL>",
        #         parent=parent_axis_ref,
        #         splen=None,
        #     )
        #     parent_axis_ref.body = sparse_axis_iter

        #     """
        #     Transform dense iterator to sparse iterator.

        #     Example:
        #         from
        #             for i2 in DenseAxisIterator(Y_o) {
        #                 AtomicFormatOp[DENSE](X_i, Y_i) {
        #                     args=(
        #                         val[i1, i2, _, _]
        #                         B_val[i2 * BLK_K + _, _]
        #                     )
        #                 }
        #             }
        #         to
        #             for i2 in SparseAxisIterator(Y_o_s)(len=val_len[i1]) {
        #                 AtomicFormatOp[DENSE](X_i, Y_i) {
        #                     args=(
        #                         val[i1, i2, _, _]
        #                         B_val[val_idx[i1, i2] * BLK_K + _, _]
        #                     )
        #                 }
        #             }

        #     Here only B_val requires modification, since other arrays (like val) 
        #     are already transformed to sparse form.
        #     """
        #     #TODO:现在这里还是不太智能,对于多维数组的引用应该可以自动检测长度,需要induction_var的个数等
        #     print("asfadfadfadfadfadfadfa test point")
        #     matrix_B_val_name = array_defs[ArrayType.B_VAL][0].name
        #     idx = sparse_val_def.index_of(axis.name)
        #     atomic_format_op:AtomicFormatOp = axis_ref.get_atomicformatop()
        #     for i in range(len(atomic_format_op.args)):
        #         arg = atomic_format_op.args[i]
        #         if isinstance(arg, ArrayRef) and arg.array == sparse_val_def.name:
        #             previous_indices = arg.indices[:idx]
        #     for i in range(len(atomic_format_op.args)):
        #         arg = atomic_format_op.args[i]
        #         if isinstance(arg, ArrayRef) and arg.array == matrix_B_val_name:
        #             for j in range(len(arg.indices)):
        #                 idx = arg.indices[j]
        #                 if type(idx) is str:
        #                     if axis_ref.induction_var in idx:
        #                         idx.replace(axis_ref.induction_var, str(ArrayRef(array=sparse_idx_def.name, indices=previous_indices+[sparse_axis_iter.induction_var])))
        #                 elif type(idx) is ArrayRef:
        #                     if idx == axis_ref.induction_var:
        #                         arg.indices[j] = ArrayRef(array=sparse_idx_def.name, indices=previous_indices+[sparse_axis_iter.induction_var])


        # else:
        #     raise ValueError(
        #         f"This case sparsify on Dense Axis Iterator, which parent axis ref is none"
        #     )
    elif isinstance(axis_ref, SparseAxisIterator):
        # nonsense, this will never happen
        raise ValueError(
            f"Sparse on Sparse Iterator: {axis_ref} for sparsify({axis.name})"
        )
    elif isinstance(axis_ref, AtomicFormatOp):
        parent_axis_ref = axis_ref.get_parent()
        # 1. 建立完整的循环上下文映射 (Axis Name -> Induction Var)
        # 向上回溯所有父节点，捕获如 {"X_o": "i5", "Y_s_o": "i6"}
        loop_context = {}
        curr_p = axis_ref.get_parent()
        while curr_p is not None:
            if hasattr(curr_p, 'axis') and hasattr(curr_p, 'induction_var'):
                loop_context[curr_p.axis] = curr_p.induction_var
            curr_p = curr_p.get_parent()

        # 2. 准备 SymPy 索引
        # 根据 sparse_len_def.axes (如 ['X_o', 'Y_s_o']) 从上下文中提取变量
        len_indices = []
        for adef_axis in sparse_len_def.axes:
            if adef_axis in loop_context:
                len_indices.append(Symbol(loop_context[adef_axis]))
            else:
                # 容错处理：如果没找到，通常是由于该维度是全局或常量
                len_indices.append(Symbol("_"))

        # 3. 构造 Indexed 表达式
        # 如果 indices 为空（标量），直接用 Symbol；否则用 IndexedBase
        if not len_indices:
            new_len_expr = Symbol(sparse_len_def.name)
        else:
            new_len_expr = IndexedBase(sparse_len_def.name)[tuple(len_indices)]

        # 4. 更新 AtomicFormatOp 的轴和长度
        # 判定哪一个轴是当前正在被稀疏化的 prev_axis_name
        if axis_ref.axes[0] == prev_axis_name:
            axis_ref.axes = axis.name, axis_ref.axes[1]
            axis_ref.axes_len = (new_len_expr, axis_ref.axes_len[1])
        elif axis_ref.axes[1] == prev_axis_name:
            axis_ref.axes = axis_ref.axes[0], axis.name
            axis_ref.axes_len = (axis_ref.axes_len[0], new_len_expr)
        else:
            raise ValueError(
                f"Sparsify Transform: axis_ref {axis_ref.axes} don't match with {prev_axis_name}"
            )
        #! ===================================这里之前的len处理逻辑
        # parent_axis_ref = axis_ref.get_parent()
        # # print(parent_axis_ref)
        # if parent_axis_ref is not None:
        #     if axis_ref.axes[0] == prev_axis_name:
        #         axis_ref.axes = axis.name, axis_ref.axes[1]
        #         #! 这里不再使用lens了,首先通过AtomicFormatOp中的参与的axis可以从axis定义中找到他们的长度,如果长度中包含了varlen,使用SymbolTable.global_table("varlen2LenArrayTable")找到对应的array_def,根据induction_var获得最终的len
        #         axis_ref.axes_len = (
        #             IndexedBase(sparse_len_def.name)[Symbol(parent_axis_ref.induction_var)],
        #             axis_ref.axes_len[1]
        #         )
        #     elif axis_ref.axes[1] == prev_axis_name:
        #         axis_ref.axes = axis_ref.axes[0], axis.name
        #         axis_ref.axes_len = (
        #             axis_ref.axes_len[0],
        #             IndexedBase(sparse_len_def.name)[Symbol(parent_axis_ref.induction_var)],
        #         )
        #     else:
        #         raise ValueError(
        #             f"Sparsify Transform: axis_ref don't match with prev_axis_name (point 1)"
        #         )
        # else:
        #     if axis_ref.axes[0] == prev_axis_name:
        #         axis_ref.axes = axis.name, axis_ref.axes[1]
        #         axis_ref.axes_len = (Symbol(sparse_len_def.name), axis_ref.axes_len[1])
        #     elif axis_ref.axes[1] == prev_axis_name:
        #         axis_ref.axes = axis_ref.axes[0], axis.name
        #         axis_ref.axes_len = (axis_ref.axes_len[0], Symbol(sparse_len_def.name))
        #     else:
        #         raise ValueError(
        #             f"Saprsify Transform: axis_ref don't match with prev_axis_name (point 2)"
        #         )
        #! ===================================
        # TODO:我打算把所有ArrayRef替换为Expr,Expr的表达语义完全够用,而且特别灵活
        target_array = sparse_val_def.name
        sidx_array_name = sparse_idx_def.name
        print("target_array", target_array)
        print("sidx_array_name", sidx_array_name)
        print("prev_axis_name", prev_axis_name)
        outer_var = parent_axis_ref.induction_var if parent_axis_ref is not None else None
        
        # 1. 建立完整的轴名到感应变量的映射表 (Axis -> Inductor Map)
        axis_to_var_map = {}
        curr_parent = axis_ref.get_parent()
        while curr_parent is not None:
            # 记录每一层循环的轴名及其对应的感应变量
            # 例如: {"X_o": "i5", "Y_s_o": "i6"}
            axis_to_var_map[curr_parent.axis] = curr_parent.induction_var
            curr_parent = curr_parent.get_parent()

        # 2. 获取当前 Op 内部正在处理的轴 (这几个轴在 indices 中应保持为 "_")
        current_op_axes = set(axis_ref.axes)

        for arg in axis_ref.args:
            if not isinstance(arg, ArrayRef):
                continue

            arg_adef = new_array_defs[arg.array]
            
            # --- 情况 A: 处理 B_val 等引用了被稀疏轴的数组 ---
            old_axis_pos = arg_adef.index_of(prev_axis_name)
            if old_axis_pos is not None and arg.array != target_array:
                # 构造 sidx array_ref
                sidx_adef = new_array_defs[sidx_array_name]
                sidx_indices = []
                for s_axis in sidx_adef.axes:
                    if s_axis in axis_to_var_map:
                        sidx_indices.append(axis_to_var_map[s_axis])
                    else:
                        sidx_indices.append("_")
                sidx_proxy = ArrayRef(array=sidx_array_name, indices=sidx_indices)
                # 确保 B_val 长度对齐并填充
                while len(arg.indices) < len(arg_adef.axes):
                    arg.indices.append("_")
                arg.indices[old_axis_pos] = sidx_proxy
                pass 

            # --- 情况 B: 处理目标数组 val (重点修改这里) ---
            elif arg.array == target_array:
                new_len = len(arg_adef.axes)
                new_indices = ["_"] * new_len
                
                for i, axis_name in enumerate(arg_adef.axes):
                    # 优先从映射表中寻找该轴是否是外层循环轴
                    if axis_name in axis_to_var_map:
                        new_indices[i] = axis_to_var_map[axis_name]
                    # 如果该轴属于当前 AtomicFormatOp 正在处理的轴，或者是还没被绑定的轴
                    # 则保持为 "_"
                    elif axis_name in current_op_axes:
                        new_indices[i] = "_"
                    else:
                        # 其他情况，可以根据需要保留原有的索引或设为 "_"
                        # 如果 arg.indices 已经有值，可以尝试保留
                        if i < len(arg.indices):
                             # 只有当原先不是 "_" 且我们没找到新映射时才保留
                             # 但通常重新构建更安全
                             pass
                
                arg.indices = new_indices
    
        # for arg in axis_ref.args:
        #     if not isinstance(arg, ArrayRef):
        #         continue

        #     #! 这里存在一些问题，在当前对于Y的稀疏中，prev_axis_name是Y，但是例如对于Y_o的稀疏中，prev_axis_name是Y_o，无法找到old_axis_pos，如何修改B_val参与在schedule的索引？还是说在Y_o这种情况中，对B_val的索引一定要考虑Expr？
        #     arg_adef = array_defs[arg.array]
        #     # 情况 A：处理引用了旧轴 (prev_axis_name，如 "Y") 的数组（例如 B_val）
        #     # 注意：B_val 的 ArrayDef 不会变，它依然包含旧轴名
        #     old_axis_pos = arg_adef.index_of(prev_axis_name)

        #     if old_axis_pos is not None:
        #         if arg.array != target_array:
        #             # 对于 B_val，构造嵌套的间接寻址对象：val_sidx[i1, _]
        #             # 这里的 "_" 代表在 AtomicFormatOp 内部由当前稀疏轴填充
        #             sidx_indices = [outer_var, "_"] if outer_var else ["_"]
        #             sidx_proxy = ArrayRef(
        #                 array=sidx_array_name, 
        #                 indices=sidx_indices
        #             )
                    
        #             # 确保 indices 列表长度与 ArrayDef 对齐
        #             while len(arg.indices) < len(arg_adef.axes):
        #                 arg.indices.append("_")
                    
        #             # 在对应位置替换为代理对象
        #             arg.indices[old_axis_pos] = sidx_proxy
        #         else:
        #             # 情况 B：处理目标数组本身 (如 val)
        #             # 此时 val 的 ArrayDef 已经变为了 ['X_o', 'Y_s', 'X_i']
        #             # 我们需要重新对齐它的 indices
        #             new_len = len(arg_adef.axes)
                    
        #             # 简单的策略：创建一个全是 "_" 的新列表，然后把外层感应变量填回去
        #             new_indices = ["_"] * new_len
        #             for i, axis_name in enumerate(arg_adef.axes):
        #                 # 如果这个轴是外层轴 (如 X_o)，填入外层感应变量 (i1)
        #                 if parent_axis_ref and axis_name == parent_axis_ref.axis:
        #                     new_indices[i] = outer_var
        #                 # 如果是当前 Op 正在处理的轴 (Y_s 或 X_i)，保持为 "_"
                        
        #             arg.indices = new_indices
        
        
        # for arg in axis_ref.args:
        #     if isinstance(arg, ArrayRef):
        #         idx = array_defs[arg.array].index_of(prev_axis_name)
        #         if idx is not None:
        #             pass
        
    else:
        raise ValueError(
            f"Unknown axis reference: {axis_ref} for sparsify({axis.name})"
        )

    return new_schedule, new_array_defs, new_format 

def apply_spawn_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    spformat: Format,
    root_format: Format,
    transform: SpawnTransformation,
) -> Tuple[Schedule, ArrayDefCollection, Format]:
    schedule, array_defs, spformat = (
        deepcopy(schedule),
        deepcopy(array_defs),
        deepcopy(spformat),
    )

    update_st_array_def(array_defs)
    update_st_axis(spformat)

    # 1. Handle format
    parent = spformat
    while isinstance(parent.child, Format):
        parent = parent.child
    assert parent.child is None

    axes1_type = parent.axes[-2].is_df()
    parent.child = Format(
        axes=[
            Axis(
                name=parent.axes[-2].name,
                direction=parent.axes[-2].direction,
                length=parent.axes[-2].length,
                is_sparse=parent.axes[-2].is_sparse,
                is_varlen=parent.axes[-2].is_varlen
            ),
            Axis(
                name=parent.axes[-1].name,
                direction=parent.axes[-1].direction,
                length=parent.axes[-1].length,
                is_sparse=parent.axes[-1].is_sparse,
                is_varlen=parent.axes[-1].is_varlen,
            )
        ]
    )

    update_st_axis(spformat)

    # 2. Handle array_defs
    # Nothing to do

    # 3. Handle schedule
    # Nothing to do

    return schedule, array_defs, spformat

def apply_cooize_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    spformat: Format,
    root_format: Format,
    transform: CooizeTransformation,
    coo_storage_policy: str = "compact_prefix",
) -> Tuple[Schedule, ArrayDefCollection, AtomicFormat]:
    schedule, array_defs = (
        deepcopy(schedule),
        deepcopy(array_defs),
    )

    update_st_array_def(array_defs)

    # 1. Handle format
    axis_x, axis_y = spformat.axes[0].name, spformat.axes[1].name
    assert len(spformat.axes) == 2 and spformat.is_dense()

    new_format = coo_atomic_format(
        x_name=axis_x,
        y_name=axis_y,
        col_major=spformat.axes[0].direction == Direction.COL,
    )
    new_format.axes[0].length = spformat.axes[0].length
    new_format.axes[1].length = spformat.axes[1].length
    #! add x_i_y_s_i_coo axis
    merged_axis = new_format.axes[2]
    merged_axis.length = Function('varlen')(Symbol("nnz"))

    update_st_axis(new_format)

    # 2. Handle array_defs
    assert len(array_defs[ArrayType.VAL]) == 1
    for val_def in array_defs[ArrayType.VAL]:
        idx_x, idx_y = val_def.index_of(axis_x), val_def.index_of(axis_y)
        # idx_x and idx_y should be the last two dimensions
        assert idx_x and idx_y and idx_x + 1 == idx_y and idx_y == len(val_def.axes) - 1
        coo_len_def = None
        coo_off_def = None
        if coo_storage_policy == "padded_offset_pair":
            coo_off_def = ArrayDef(
                name=MyNameManager.new_name(val_def.name + "_coo_off"),
                axes=val_def.axes[:-2] + [f"{axis_x}_{axis_y}_coo_off_pair"],
                dims=val_def.dims[:-2] + [2],
                type=ArrayType.OFFSET,
                datatype=DataType.INT,
            )
        elif coo_storage_policy == "compact_prefix":
            coo_len_def = ArrayDef(
                name=MyNameManager.new_name(val_def.name + "_coo_len"),
                axes=val_def.axes[:-2],
                dims=val_def.dims[:-2],
                type=ArrayType.SPLEN,
                datatype=DataType.INT,
            )
        else:
            raise ValueError(f"Unsupported COO storage policy: {coo_storage_policy}")

        coo_idx_def = ArrayDef(
            name=MyNameManager.new_name(val_def.name + "_coo_idx"),
            axes=val_def.axes[:-2] + [f"{axis_x}_{axis_y}_coo"],
            dims=val_def.dims[:-2] + [merged_axis.length],
            type=ArrayType.COO_IDX,
            datatype=DataType.INT,
        )
        coo_val_def = ArrayDef(
            name=MyNameManager.new_name(val_def.name + "_coo_val"),
            axes=val_def.axes[:-2] + [f"{axis_x}_{axis_y}_coo"],
            dims=val_def.dims[:-2] + [merged_axis.length],
            type=ArrayType.VAL,
            datatype=val_def.datatype,
        )
        if coo_len_def is not None:
            varlen2LenArrayTable = SymbolTable.global_table("varlen2LenArrayTable")
            varlen2LenArrayTable[str(merged_axis.length)] = coo_len_def
            coo_metadata_def = coo_len_def
            replacement_defs = [coo_idx_def, coo_val_def, coo_len_def]
        else:
            assert coo_off_def is not None
            coo_metadata_def = coo_off_def
            replacement_defs = [coo_idx_def, coo_val_def, coo_off_def]

        array_defs.replace([val_def.name], replacement_defs)

    update_st_array_def(array_defs)

    # 3. Handle schedule
    # axis_ref = schedule.find_axis_ref(axis_x)
    # assert axis_ref is schedule.find_axis_ref(axis_y)
    # assert isinstance(axis_ref, AtomicFormatOp)
    # assert axis_ref.type == AtomicFormatType.DENSE

    # new_axis_ref = AtomicFormatOp(
    #     axes=(axis_x, axis_y),
    #     axes_len=(axis_ref.axes_len[0], axis_ref.axes_len[1]),
    #     type=AtomicFormatType.COO,
    #     args=[
    #         ArrayRef(array=coo_len_def.name, indices=[]),
    #         ArrayRef(array=coo_idx_def.name, indices=[]),
    #         ArrayRef(array=coo_val_def.name, indices=[]),
    #         next(
    #             filter(
    #                 lambda arg: isinstance(arg, ArrayRef) and arg.array == "B_val",
    #                 axis_ref.args,
    #             )
    #         ),
    #     ],
    #     parent=axis_ref.get_parent(),
    # )

    # parent_axis_ref = axis_ref.get_parent()
    # if parent_axis_ref is not None:
    #     assert isinstance(parent_axis_ref, AxisVisitor)
    #     parent_axis_ref.body = new_axis_ref
    # else:
    #     logger.warning(
    #         f"No parent axis reference found for {axis_ref} during cooize transformation. "
    #         f"Make sure this is what you want."
    #     )
    #     schedule = new_axis_ref

    
    
    axis_ref = schedule.find_axis_ref(axis_x)
    assert axis_ref is schedule.find_axis_ref(axis_y)
    assert isinstance(axis_ref, AtomicFormatOp)
    assert axis_ref.type == AtomicFormatType.DENSE

    # --- 核心修改：提取并继承索引 ---
    # 1. 在原 AtomicFormatOp 的参数中找到原始的 val 引用（即非 B_val 的那个）
    # 我们使用 val_def.name 来精准匹配
    orig_val_arg = next(
        filter(
            lambda arg: isinstance(arg, ArrayRef) and arg.array == val_def.name,
            axis_ref.args,
        )
    )
    
    # 2. 提取外层索引。由于 AtomicFormatOp 处理的是最后两个维度，
    # 对应的索引通常是 [..., _, _]。我们需要保留前面的部分（如 [i1, i2]）
    # 这里使用切片去掉最后两个位置
    inherited_indices = list(orig_val_arg.indices[:-2])
    
    # 3. 构造新的 COO AtomicFormatOp
    new_axis_ref = AtomicFormatOp(
        axes=(axis_x, axis_y),
        axes_len=(axis_ref.axes_len[0], axis_ref.axes_len[1]),
        type=AtomicFormatType.COO,
        args=[
            # compact policy uses coo_len[i1, i2]; padded policy uses
            # coo_off[i1, i2] and the pair dimension is addressed by the loader.
            ArrayRef(array=coo_metadata_def.name, indices=inherited_indices),
            
            # coo_idx 和 coo_val 需要外层索引 + 一个 COO 轴的占位符 "_"
            # val_coo_idx[i1, i2, _]
            ArrayRef(array=coo_idx_def.name, indices=inherited_indices + ["_"]),
            
            # val_coo_val[i1, i2, _]
            ArrayRef(array=coo_val_def.name, indices=inherited_indices + ["_"]),
            
            # 保持 B_val 不变
            next(
                filter(
                    lambda arg: isinstance(arg, ArrayRef) and arg.array == "B_val",
                    axis_ref.args,
                )
            ),
        ],
        parent=axis_ref.get_parent(),
    )
    # -----------------------------

    parent_axis_ref = axis_ref.get_parent()
    if parent_axis_ref is not None:
        assert isinstance(parent_axis_ref, AxisVisitor)
        parent_axis_ref.body = new_axis_ref
    else:
        # 如果没有父节点，说明当前节点就是根，直接更新 schedule 引用
        schedule = new_axis_ref

    return schedule, array_defs, new_format

def apply_mcoize_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    spformat: Format,
    root_format: Format,
    transform: McoizeTransformation,
) -> Tuple[Schedule, ArrayDefCollection, AtomicFormat]:
    schedule, array_defs = (
        deepcopy(schedule),
        deepcopy(array_defs),
    )
    # 1. Handle format
    axis_x, axis_y = spformat.axes[0].name, spformat.axes[1].name
    assert len(spformat.axes) == 2 and spformat.is_dense()

    new_format = mco_atomic_format(
        x_name=axis_x,
        y_name=axis_y,
        col_major=spformat.axes[0].direction == Direction.COL,
    )
    new_format.axes[0].length = spformat.axes[0].length
    new_format.axes[1].length = spformat.axes[1].length
    merged_axis = new_format.axes[2]
    merged_axis.length = Function("varlen")(Symbol("nnz"))
    update_st_axis(new_format)

    # 2. Handle array_defs
    assert len(array_defs[ArrayType.VAL]) == 1
    for val_def in array_defs[ArrayType.VAL]:
        idx_x, idx_y = val_def.index_of(axis_x), val_def.index_of(axis_y)
        # idx_x and idx_y should be the last two dimensions
        assert idx_x and idx_y and idx_x + 1 == idx_y and idx_y == len(val_def.axes) - 1
        mask_axis_name = f"{axis_x}_{axis_y}_mco_mask"
        mask_tile_shape = (val_def.dims[idx_x] * val_def.dims[idx_y]) / Number(64)
        mco_mask_def = ArrayDef(
            name=MyNameManager.new_name(val_def.name + "_mco_mask"),
            # Keep the mask-vector dimension explicit so the generated gmem
            # interface matches the handwritten kernels: [num_masks, nnz_block].
            axes=[mask_axis_name] + val_def.axes[:-2],
            dims=[mask_tile_shape] + val_def.dims[:-2],
            type=ArrayType.MASK,
            datatype=DataType.INT,
        )
        mco_val_def = ArrayDef(
            name=MyNameManager.new_name(val_def.name + "_mco_val"),
            axes=val_def.axes[:-2] + [f"{axis_x}_{axis_y}_mco"],
            dims=val_def.dims[:-2] + [merged_axis.length],
            type=ArrayType.VAL,
            datatype=val_def.datatype,
        )
        mco_len_def = ArrayDef(
            name=MyNameManager.new_name(val_def.name + "_mco_len"),
            axes=val_def.axes[:-2],
            dims=val_def.dims[:-2],
            type=ArrayType.SPLEN,
            datatype=DataType.INT,
        )
        varlen2LenArrayTable = SymbolTable.global_table("varlen2LenArrayTable")
        varlen2LenArrayTable[str(merged_axis.length)] = mco_len_def
        array_defs.replace([val_def.name], [mco_mask_def, mco_val_def, mco_len_def])

    update_st_array_def(array_defs)

    # 3. Handle schedule
    # axis_ref = schedule.find_axis_ref(axis_x)
    # assert axis_ref is schedule.find_axis_ref(axis_y)
    # assert isinstance(axis_ref, AtomicFormatOp)
    # assert axis_ref.type == AtomicFormatType.DENSE

    # new_axis_ref = AtomicFormatOp(
    #     axes=(axis_x, axis_y),
    #     type=AtomicFormatType.MCO,
    #     axes_len=(axis_ref.axes_len[0], axis_ref.axes_len[1]),
    #     args=[
    #         # TODO: ArrayRef改成Expr
    #         ArrayRef(array=mco_len_def.name, indices=[]),
    #         ArrayRef(array=mco_mask_def.name, indices=[]),
    #         ArrayRef(array=mco_val_def.name, indices=[]),
    #         next(
    #             filter(
    #                 lambda arg: isinstance(arg, ArrayRef) and arg.array == "B_val",
    #                 axis_ref.args,
    #             )
    #         ),
    #     ],
    #     parent=axis_ref.get_parent(),
    # )

    # parent_axis_ref = axis_ref.get_parent()
    # if parent_axis_ref is not None:
    #     assert isinstance(parent_axis_ref, AxisVisitor)
    #     parent_axis_ref.body = new_axis_ref
    # else:
    #     logger.warning(
    #         f"No parent axis reference found for {axis_ref} during cooize transformation. "
    #         f"Make sure this is what you want."
    #     )
    #     schedule = new_axis_ref

    axis_ref = schedule.find_axis_ref(axis_x)
    assert axis_ref is schedule.find_axis_ref(axis_y)
    assert isinstance(axis_ref, AtomicFormatOp)
    assert axis_ref.type == AtomicFormatType.DENSE

    # --- 核心修改：提取并继承索引 ---
    # 1. 在原 AtomicFormatOp 参数中定位原始 val 的引用 (依据 val_def.name)
    orig_val_arg = next(
        filter(
            lambda arg: isinstance(arg, ArrayRef) and arg.array == val_def.name,
            axis_ref.args,
        )
    )
    
    # 2. 提取外层索引前缀。去掉最后两个 "_" 占位符
    # 例如：从 [i3, i4, _, _] 提取出 [i3, i4]
    inherited_indices = list(orig_val_arg.indices[:-2])
    
    # 3. 构造新的 MCO AtomicFormatOp
    new_axis_ref = AtomicFormatOp(
        axes=(axis_x, axis_y),
        type=AtomicFormatType.MCO,
        axes_len=(axis_ref.axes_len[0], axis_ref.axes_len[1]),
        args=[
            # mco_len 和 mco_mask 仅保留外层索引: [i3, i4]
            ArrayRef(array=mco_len_def.name, indices=inherited_indices),
            ArrayRef(array=mco_mask_def.name, indices=inherited_indices),
            
            # mco_val 增加合并后的 MCO 轴占位符: [i3, i4, _]
            ArrayRef(array=mco_val_def.name, indices=inherited_indices + ["_"]),
            
            # 保持 B_val 引用及其复杂的索引逻辑不变
            next(
                filter(
                    lambda arg: isinstance(arg, ArrayRef) and arg.array == "B_val",
                    axis_ref.args,
                )
            ),
        ],
        parent=axis_ref.get_parent(),
    )
    # -----------------------------

    parent_axis_ref = axis_ref.get_parent()
    if parent_axis_ref is not None:
        assert isinstance(parent_axis_ref, AxisVisitor)
        parent_axis_ref.body = new_axis_ref
    else:
        # 如果没有父节点，说明是顶层，更新 schedule 引用
        schedule = new_axis_ref
    
    return schedule, array_defs, new_format


def apply_csrize_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    spformat: Format,
    root_format: Format,
    transform: CsrizeTransformation,
) -> Tuple[Schedule, ArrayDefCollection, AtomicFormat]:
    schedule, array_defs = (
        deepcopy(schedule),
        deepcopy(array_defs),
    )
    update_st_array_def(array_defs)

    # 1. Handle format
    axis_x, axis_y = spformat.axes[0].name, spformat.axes[1].name
    assert len(spformat.axes) == 2 and spformat.is_dense()

    new_format = csr_atomic_format(
        x_name=axis_x,
        y_name=axis_y,
        col_major=spformat.axes[0].direction == Direction.COL,
    )
    new_format.axes[0].length = spformat.axes[0].length
    new_format.axes[1].length = spformat.axes[1].length
    merged_axis = new_format.axes[2]
    merged_axis.length = Function('varlen')(Symbol("nnz"))
    update_st_axis(new_format)

    # 2. Handle array_defs
    assert len(array_defs[ArrayType.VAL]) == 1
    for val_def in array_defs[ArrayType.VAL]:
        idx_x, idx_y = val_def.index_of(axis_x), val_def.index_of(axis_y)
        assert idx_x and idx_y and idx_x + 1 == idx_y and idx_y == len(val_def.axes) - 1

        csr_len_def = ArrayDef(
            name=MyNameManager.new_name(val_def.name + "_csr_len"),
            axes=val_def.axes[:-2],
            dims=val_def.dims[:-2],
            type=ArrayType.SPLEN,
            datatype=DataType.INT,
        )
        csr_idx_def = ArrayDef(
            name=MyNameManager.new_name(val_def.name + "_csr_idx"),
            axes=val_def.axes[:-2] + [f"{axis_x}_{axis_y}_csr"],
            dims=val_def.dims[:-2] + [merged_axis.length],
            type=ArrayType.COO_IDX,
            datatype=DataType.INT,
        )
        csr_val_def = ArrayDef(
            name=MyNameManager.new_name(val_def.name + "_csr_val"),
            axes=val_def.axes[:-2] + [f"{axis_x}_{axis_y}_csr"],
            dims=val_def.dims[:-2] + [merged_axis.length],
            type=ArrayType.VAL,
            datatype=val_def.datatype,
        )
        varlen2LenArrayTable = SymbolTable.global_table("varlen2LenArrayTable")
        varlen2LenArrayTable[str(merged_axis.length)] = csr_len_def
        csr_row_ptr_def = ArrayDef(
            name=MyNameManager.new_name(val_def.name + "_csr_row_ptr"),
            axes=val_def.axes[:-2] + [axis_x],
            dims=val_def.dims[:-2] + [Symbol("BLK_M") + Number(1)],
            type=ArrayType.IDX,
            datatype=DataType.INT,
        )
        array_defs.replace([val_def.name], [csr_idx_def, csr_val_def, csr_len_def, csr_row_ptr_def])
        _captured_row_ptr_name = csr_row_ptr_def.name

    update_st_array_def(array_defs)

    # 3. Handle schedule
    axis_ref = schedule.find_axis_ref(axis_x)
    assert axis_ref is schedule.find_axis_ref(axis_y)
    assert isinstance(axis_ref, AtomicFormatOp)
    assert axis_ref.type == AtomicFormatType.DENSE

    orig_val_arg = next(
        filter(
            lambda arg: isinstance(arg, ArrayRef) and arg.array == val_def.name,
            axis_ref.args,
        )
    )
    inherited_indices = list(orig_val_arg.indices[:-2])

    new_axis_ref = AtomicFormatOp(
        axes=(axis_x, axis_y),
        axes_len=(axis_ref.axes_len[0], axis_ref.axes_len[1]),
        type=AtomicFormatType.CSR,
        args=[
            ArrayRef(array=csr_len_def.name, indices=inherited_indices),
            ArrayRef(array=csr_idx_def.name, indices=inherited_indices + ["_"]),
            ArrayRef(array=csr_val_def.name, indices=inherited_indices + ["_"]),
            next(
                filter(
                    lambda arg: isinstance(arg, ArrayRef) and arg.array == "B_val",
                    axis_ref.args,
                )
            ),
            ArrayRef(array=csr_row_ptr_def.name, indices=inherited_indices),
        ],
        parent=axis_ref.get_parent(),
    )

    parent_axis_ref = axis_ref.get_parent()
    if parent_axis_ref is not None:
        assert isinstance(parent_axis_ref, AxisVisitor)
        parent_axis_ref.body = new_axis_ref
    else:
        schedule = new_axis_ref

    return schedule, array_defs, new_format


def apply_ellize_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    spformat: Format,
    root_format: Format,
    transform: EllizeTransformation,
) -> Tuple[Schedule, ArrayDefCollection, AtomicFormat]:
    """ELL atomic format: packed 2D [BLK_M, max_nnz_per_row] with per-block ell_len scalar."""
    schedule, array_defs = (
        deepcopy(schedule),
        deepcopy(array_defs),
    )
    update_st_array_def(array_defs)

    # 1. Handle format
    axis_x, axis_y = spformat.axes[0].name, spformat.axes[1].name
    assert len(spformat.axes) == 2 and spformat.is_dense()

    new_format = ell_atomic_format(
        x_name=axis_x,
        y_name=axis_y,
        col_major=spformat.axes[0].direction == Direction.COL,
    )
    new_format.axes[0].length = spformat.axes[0].length
    new_format.axes[1].length = spformat.axes[1].length
    merged_axis = new_format.axes[2]
    merged_axis.length = Function('varlen')(Symbol("nnz"))
    update_st_axis(new_format)

    # 2. Handle array_defs
    assert len(array_defs[ArrayType.VAL]) == 1
    for val_def in array_defs[ArrayType.VAL]:
        idx_x, idx_y = val_def.index_of(axis_x), val_def.index_of(axis_y)
        assert idx_x and idx_y and idx_x + 1 == idx_y and idx_y == len(val_def.axes) - 1

        ell_len_def = ArrayDef(
            name=MyNameManager.new_name(val_def.name + "_ell_len"),
            axes=val_def.axes[:-2],
            dims=val_def.dims[:-2],
            type=ArrayType.SPLEN,
            datatype=DataType.INT,
        )
        ell_idx_def = ArrayDef(
            name=MyNameManager.new_name(val_def.name + "_ell_idx"),
            axes=val_def.axes[:-2] + [f"{axis_x}_{axis_y}_ell"],
            dims=val_def.dims[:-2] + [merged_axis.length],
            type=ArrayType.COO_IDX,
            datatype=DataType.INT,
        )
        ell_val_def = ArrayDef(
            name=MyNameManager.new_name(val_def.name + "_ell_val"),
            axes=val_def.axes[:-2] + [f"{axis_x}_{axis_y}_ell"],
            dims=val_def.dims[:-2] + [merged_axis.length],
            type=ArrayType.VAL,
            datatype=val_def.datatype,
        )
        varlen2LenArrayTable = SymbolTable.global_table("varlen2LenArrayTable")
        varlen2LenArrayTable[str(merged_axis.length)] = ell_len_def
        array_defs.replace([val_def.name], [ell_idx_def, ell_val_def, ell_len_def])

    update_st_array_def(array_defs)

    # 3. Handle schedule
    axis_ref = schedule.find_axis_ref(axis_x)
    assert axis_ref is schedule.find_axis_ref(axis_y)
    assert isinstance(axis_ref, AtomicFormatOp)
    assert axis_ref.type == AtomicFormatType.DENSE

    orig_val_arg = next(
        filter(
            lambda arg: isinstance(arg, ArrayRef) and arg.array == val_def.name,
            axis_ref.args,
        )
    )
    inherited_indices = list(orig_val_arg.indices[:-2])

    new_axis_ref = AtomicFormatOp(
        axes=(axis_x, axis_y),
        axes_len=(axis_ref.axes_len[0], axis_ref.axes_len[1]),
        type=AtomicFormatType.ELL,
        args=[
            ArrayRef(array=ell_len_def.name, indices=inherited_indices),
            ArrayRef(array=ell_idx_def.name, indices=inherited_indices + ["_"]),
            ArrayRef(array=ell_val_def.name, indices=inherited_indices + ["_"]),
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
        schedule = new_axis_ref

    return schedule, array_defs, new_format


def apply_diaize_transformation(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    spformat: Format,
    root_format: Format,
    transform: DiaizeTransformation,
) -> Tuple[Schedule, ArrayDefCollection, AtomicFormat]:
    """DIA atomic format: per-block diag_offsets[num_diags] + val[num_diags, BLK_K]."""
    schedule, array_defs = (
        deepcopy(schedule),
        deepcopy(array_defs),
    )
    update_st_array_def(array_defs)

    axis_x, axis_y = spformat.axes[0].name, spformat.axes[1].name
    assert len(spformat.axes) == 2 and spformat.is_dense()

    new_format = dia_atomic_format(
        x_name=axis_x,
        y_name=axis_y,
        col_major=spformat.axes[0].direction == Direction.COL,
    )
    new_format.axes[0].length = spformat.axes[0].length
    new_format.axes[1].length = spformat.axes[1].length
    merged_axis = new_format.axes[2]
    merged_axis.length = Function("varlen")(Symbol("nnz"))
    update_st_axis(new_format)

    assert len(array_defs[ArrayType.VAL]) == 1
    for val_def in array_defs[ArrayType.VAL]:
        idx_x, idx_y = val_def.index_of(axis_x), val_def.index_of(axis_y)
        assert idx_x and idx_y and idx_x + 1 == idx_y and idx_y == len(val_def.axes) - 1

        dia_len_def = ArrayDef(
            name=MyNameManager.new_name(val_def.name + "_dia_len"),
            axes=val_def.axes[:-2],
            dims=val_def.dims[:-2],
            type=ArrayType.SPLEN,
            datatype=DataType.INT,
        )
        dia_idx_def = ArrayDef(
            name=MyNameManager.new_name(val_def.name + "_dia_idx"),
            axes=val_def.axes[:-2] + [f"{axis_x}_{axis_y}_dia"],
            dims=val_def.dims[:-2] + [merged_axis.length],
            type=ArrayType.IDX,
            datatype=DataType.INT,
        )
        dia_val_def = ArrayDef(
            name=MyNameManager.new_name(val_def.name + "_dia_val"),
            axes=val_def.axes[:-2] + [f"{axis_x}_{axis_y}_dia"],
            dims=val_def.dims[:-2] + [merged_axis.length],
            type=ArrayType.VAL,
            datatype=val_def.datatype,
        )
        varlen2LenArrayTable = SymbolTable.global_table("varlen2LenArrayTable")
        varlen2LenArrayTable[str(merged_axis.length)] = dia_len_def
        array_defs.replace([val_def.name], [dia_idx_def, dia_val_def, dia_len_def])

    update_st_array_def(array_defs)

    axis_ref = schedule.find_axis_ref(axis_x)
    assert axis_ref is schedule.find_axis_ref(axis_y)
    assert isinstance(axis_ref, AtomicFormatOp)
    assert axis_ref.type == AtomicFormatType.DENSE

    orig_val_arg = next(
        filter(
            lambda arg: isinstance(arg, ArrayRef) and arg.array == val_def.name,
            axis_ref.args,
        )
    )
    inherited_indices = list(orig_val_arg.indices[:-2])

    new_axis_ref = AtomicFormatOp(
        axes=(axis_x, axis_y),
        axes_len=(axis_ref.axes_len[0], axis_ref.axes_len[1]),
        type=AtomicFormatType.DIA,
        args=[
            ArrayRef(array=dia_len_def.name, indices=inherited_indices),
            ArrayRef(array=dia_idx_def.name, indices=inherited_indices + ["_"]),
            ArrayRef(array=dia_val_def.name, indices=inherited_indices + ["_"]),
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
        schedule = new_axis_ref

    return schedule, array_defs, new_format


def add_metadata(schedule: Optional[Schedule], format: Format) -> None:
    found_reduction_axis = False
    found_block_idx_x_axis = False

    sch = schedule
    while sch is not None:
        if (
            isinstance(sch, AxisVisitor)
            and format.get_axis(sch.axis).direction == Direction.COL  # 找到第一个Y
        ):
            sch.metadata = {
                "reduction_axis": True,
            }
            found_reduction_axis = True
            break
        sch = sch.child

    sch = schedule
    while sch is not None:
        if isinstance(sch, AxisVisitor):
            if hasattr(sch, "metadata") and sch.metadata.get("reduction_axis"):
                raise ValueError(
                    f"Missing block_idx_x.axis outside of reduction axis {sch.axis}"
                )
            if format.get_axis(sch.axis).direction == Direction.ROW:  # 找到第一个X
                sch.metadata = {
                    "block_idx_x_axis": True,
                }
                found_block_idx_x_axis = True
                break
        sch = sch.child

    if not (found_reduction_axis and found_block_idx_x_axis):
        raise ValueError(
            f"Cannot find reduction axis or block_idx_x_axis in the schedule: {schedule}"
        )


def computent_from_rts(
    name: str,
    rts: TransformationSequence,
    coo_storage_policy: str = "auto",
) -> Computent:
    if coo_storage_policy == "auto":
        coo_storage_policy = "padded_offset_pair" if name == "ME_TCF" else "compact_prefix"
    if coo_storage_policy not in {"compact_prefix", "padded_offset_pair"}:
        raise ValueError(f"Unsupported COO storage policy: {coo_storage_policy}")

    spformat = deepcopy(DENSE_FORMAT)
    root_format = Format(axes=[], child=spformat)
    assert isinstance(root_format.child, Format)
    parent_format = root_format
    schedule = deepcopy(DENSE_SCHEDULE)
    array_defs = deepcopy(DENSE_ARRAY_DEFS)

    # add name manager
    MyNameManager.init_name("val")
    MyNameManager.init_name("B_val")
    MyNameManager.init_name("C_val")
    MyNameManager.init_name("X")
    MyNameManager.init_name("Y")
    MyNameManager.init_name("N")

    update_st_array_def(array_defs)
    update_st_axis(spformat)

    logger.debug("\n" + str(schedule))
    logger.debug("\n" + str(array_defs))
    logger.debug("\n" + str(root_format.child))
    logger.debug("-" * 100)

    atomic_format_reached = False
    for transform in rts.sequence:
        assert (
            not atomic_format_reached
        ), f"Transforms remaining after atomic format reached: {transform}"
        logger.debug(f"Transform begins: {transform}")
        if isinstance(transform, SplitTransformation):
            schedule, array_defs, spformat = apply_split_transformation(
                schedule, array_defs, spformat, root_format.child, transform
            )
            parent_format.child = spformat
        elif isinstance(transform, SwapTransformation):
            schedule, array_defs, spformat = apply_swap_transformation(
                schedule, array_defs, spformat, root_format.child, transform
            )
            parent_format.child = spformat
        elif isinstance(transform, SparsifyTransformation):
            schedule, array_defs, spformat = apply_sparsify_transformation(
                schedule, array_defs, spformat, root_format.child, transform
            )
            parent_format.child = spformat
        elif isinstance(transform, SpawnTransformation):
            schedule, array_defs, spformat = apply_spawn_transformation(
                schedule, array_defs, spformat, root_format.child, transform
            )
            parent_format.child = spformat
            parent_format = spformat
            spformat = spformat.child
            assert isinstance(spformat, Format)
        elif isinstance(transform, CooizeTransformation):
            schedule, array_defs, atomic_format = apply_cooize_transformation(
                schedule,
                array_defs,
                spformat,
                root_format.child,
                transform,
                coo_storage_policy=coo_storage_policy,
            )
            parent_format.child = atomic_format
            atomic_format_reached = True
        elif isinstance(transform, McoizeTransformation):
            schedule, array_defs, atomic_format = apply_mcoize_transformation(
                schedule, array_defs, spformat, root_format.child, transform
            )
            parent_format.child = atomic_format
            atomic_format_reached = True
        elif isinstance(transform, CsrizeTransformation):
            schedule, array_defs, atomic_format = apply_csrize_transformation(
                schedule, array_defs, spformat, root_format.child, transform
            )
            parent_format.child = atomic_format
            atomic_format_reached = True
        elif isinstance(transform, EllizeTransformation):
            schedule, array_defs, atomic_format = apply_ellize_transformation(
                schedule, array_defs, spformat, root_format.child, transform
            )
            parent_format.child = atomic_format
            atomic_format_reached = True
        elif isinstance(transform, DiaizeTransformation):
            schedule, array_defs, atomic_format = apply_diaize_transformation(
                schedule, array_defs, spformat, root_format.child, transform
            )
            parent_format.child = atomic_format
            atomic_format_reached = True
        else:
            raise ValueError(f"Unknown transformation: {transform}")

        logger.debug("\n" + str(schedule))
        logger.debug("\n" + str(array_defs))
        logger.debug("\n" + str(root_format.child))
        logger.debug("-" * 100)

    assert isinstance(root_format.child, Format)
    add_metadata(schedule, root_format.child)

    varlen2LenArrayTable = deepcopy(
        SymbolTable.global_table("varlen2LenArrayTable")._table
    )
    print("varlen2LenArrayTable:", varlen2LenArrayTable)


    MyNameManager.reset()
    SymbolTable.reset()
    print("varlen2LenArrayTable after reset:", varlen2LenArrayTable)
    return Computent(
        name=name,
        schedule=schedule,
        array_defs=array_defs,
        format=root_format.child,
        varlen2LenArrayTable=varlen2LenArrayTable,
    )

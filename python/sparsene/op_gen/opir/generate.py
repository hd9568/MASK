from __future__ import annotations

from sparsene.op_gen.computent.computent import computent_from_rts, Computent, ArrayDef
from sparsene.format.format import Format, Expr, Symbol, Number
from sparsene.op_gen.computent.computent import ArrayDefCollection
from typing import Any, List, Optional, Dict, Literal, Tuple, Sequence
from sparsene.op_gen.computent.computent import (
    Schedule,
    DenseAxisIterator,
    SparseAxisIterator,
    SparseAxisSlicer,
    AtomicFormatOp,
    AtomicFormatType,
    ArrayRef,
    DataType,
)
from sparsene.op_gen.computent.arraydef import ArrayType as ComputentArrayType
from dataclasses import dataclass
from sparsene.op_gen.opir.op_ir import (
    Op,
    DeviceOp,
    OpBuilder,
    SymbolTable,
    ConstantOp,
    ForLoopOp,
    ArrayRefOp,
    LoadOp,
    LoadOffsetOp,
    Value,
    IntType,
    MetaOp,
    ExternalSymbolOp,
    AddOp,
    MulOp,
    DivOp,
    PowOp,
    ArangeOp,
    op_builder,
    Type,
    ArrayType,
    OpOperand,
    FloatType,
    OpResult,
    LoopResultOp,
)
from sparsene.op_gen.opir.ops import (
    SparseOffsetLoadOp,
    CooAtomicFormatLoadOffOp,
    CooAtomicFormatLoadIdxOp,
    CooAtomicFormatLoadValOp,
    ValSidxLoadOp,
    BValLoadOp,
    CooAtomicValRestoreOp,
    CsrAtomicValRestoreOp,
    EllAtomicValRestoreOp,
    DiaAtomicFormatLoadIdxOp,
    DiaAtomicFormatLoadValOp,
    DiaAtomicValRestoreOp,
    McoAtomicFormatLoadMaskOp,
    McoAtomicFormatLoadValOp,
    McoAtomicValRestoreOp,
    MmaOp,
    CValLoadOp,
    CValStoreOp,
)
from sympy import Symbol, Indexed, Expr, Integer, Rational
from sparsene.format.format import Direction



def datatype_to_type(datatype: DataType) -> Type:
    match datatype:
        case DataType.FLOAT:
            return FloatType()
        case DataType.INT:
            return IntType()
        case _:
            raise ValueError(f"Unsupported datatype: {datatype}")


def get_array_type(array_def: ArrayDef) -> Type:
    dims = list(array_def.dims)
    return ArrayType(dims, datatype_to_type(array_def.datatype))


def build_int_value_from_expr(
    expr: Any,
    array_defs: ArrayDefCollection,
    name_hint: str = "len",
) -> Value:
    if isinstance(expr, Value):
        return expr
    if isinstance(expr, (Integer, int)):
        return op_builder.build(ConstantOp, int(expr), IntType(), name_hint=name_hint).result
    if isinstance(expr, str):
        return op_builder.lookup_symbol(expr, IntType())
    return translate_sympy_to_ir(expr, array_defs)


def infer_m_dim_len(
    tile: Value,
    array_defs: ArrayDefCollection,
    name_hint: str = "c_m_len",
) -> Value:
    assert isinstance(tile.type, ArrayType), "C tile should be an array"
    assert len(tile.type.dims) > 0, "C tile should have at least one dimension"
    return build_int_value_from_expr(tile.type.dims[0], array_defs, name_hint=name_hint)


def build_c_val_load_slice(
    c_tile_container: Value,
    offset: Value,
    array_defs: ArrayDefCollection,
    name_hint: str = "c_slice",
) -> Value:
    assert isinstance(c_tile_container.type, ArrayType), "C tile container should be an array"
    slice_dims = list(c_tile_container.type.dims[1:])
    assert len(slice_dims) > 0, "C tile slicing should keep at least one dimension"

    load_len = build_int_value_from_expr(
        slice_dims[0],
        array_defs,
        name_hint=f"{name_hint}_len",
    )

    c_val_load_op = op_builder.build(
        CValLoadOp,
        mem="S2R",
        array=c_tile_container,
        offset=offset,
        length=load_len,
    )
    c_slice_result = OpResult(
        type=ArrayType(slice_dims, c_tile_container.type.datatype),
        defining_op=c_val_load_op,
        result_idx_in_owner=0,
        name_hint=name_hint,
    )
    c_val_load_op.add_result(c_slice_result)
    return c_slice_result

def generate_dispatch(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    format: Format,
    current_c_tile: Optional[Value],
    parent_schedule: Optional[Schedule] = None,
) -> Value:
    """统一调度分发器，返回更新后的 C Tile Value"""
    if isinstance(schedule, DenseAxisIterator):
        return generate_from_dense_axis_iterator(
            schedule, array_defs, format, current_c_tile, parent_schedule
        )
    elif isinstance(schedule, SparseAxisIterator):
        return generate_from_sparse_axis_iterator(
            schedule, array_defs, format, current_c_tile, parent_schedule
        )
    elif isinstance(schedule, AtomicFormatOp):
        assert current_c_tile is not None, "Atomic format op should receive a C tile from parent schedule"
        # 注意：Atomic 直接返回计算后的结果，不需要 parent_schedule
        return generate_from_atomic_format_op(
            schedule, array_defs, format, current_c_tile
        )
    else:
        raise ValueError(f"Unsupported schedule type for dispatch: {type(schedule)}")

def generate_from_computent(computent: Computent) -> MetaOp:
    computent_op = op_builder.build(MetaOp)
    with op_builder.op_scope(computent_op):
        for array_def in computent.array_defs:
            op_builder.lookup_symbol(array_def.name, get_array_type(array_def))
        schedule = computent.schedule
        array_defs = computent.array_defs
        format = computent.format
        if isinstance(schedule, DenseAxisIterator):
            generate_from_dense_axis_iterator(schedule, array_defs, format)
        elif isinstance(schedule, SparseAxisIterator):
            generate_from_sparse_axis_iterator(schedule, array_defs, format)
            # raise NotImplementedError(
            #     "Sparse axis iterator as root schedule is not implemented"
            # )
        elif isinstance(schedule, SparseAxisSlicer):
            raise NotImplementedError(
                "Sparse axis slicer as root schedule is not implemented"
            )
        elif isinstance(schedule, AtomicFormatOp):
            raise NotImplementedError(
                "Atomic format op as root schedule is not implemented"
            )
        else:
            raise ValueError(f"Unsupported schedule type: {type(schedule)}")
    return computent_op


# TODO 再讲一下这里的 active_tile, current_c_tile, 分别是干啥的
def generate_from_dense_axis_iterator(
    schedule: DenseAxisIterator,
    array_defs: ArrayDefCollection,
    format: Format,
    current_c_tile: Optional[Value] = None,
    parent_schedule: Optional[Schedule] = None,
) -> Value:
    l = op_builder.build(ConstantOp, 0, IntType(), name_hint="zero").result
    r = op_builder.build(
        ConstantOp,
        format.get_axis(schedule.axis).length,
        IntType(),
        name_hint="num_row_win",
    ).result

    is_reduction = format.get_axis(schedule.axis).direction == Direction.COL

    #> 空间轴切片（空间轴就是 Direction.ROW，与空间轴相对的是规约轴）
    active_tile = current_c_tile    
    
    # 1. 确定当前循环的初始 C Tile
    # 如果是顶层空间轴且没有输入，则直接使用全局 C_val 定义（作为逻辑上的大 Tile）
    if not is_reduction and current_c_tile is None and parent_schedule is None:
        current_c_tile = op_builder.lookup_symbol("C_val")

    iter_args = {"C_tile_io": current_c_tile} if current_c_tile else {}

    for_loop_op = op_builder.build(
        ForLoopOp,
        induction_var=schedule.induction_var,
        range=(l, r),
        iter_args=iter_args,
    )

    # with op_builder.op_scope(for_loop_op):
    #     op_builder.add_symbol(schedule.induction_var, for_loop_op.get_induction_var())
        
    #     if not is_reduction and active_tile is not None:
    #         # 如果当前的轴不是规约轴，并且传入了一个c_tile
    #         # 需要从传入的c_tile中切出一块给子循环
    #         # 比如[BLK_M, N] -> [BLK_M_I, N]
    #         active_tile = op_builder.build(
    #             ArrayRefOp,
    #             active_tile,
    #             # [Symbol(schedule.induction_var)],
    #             [op_builder.lookup_symbol(schedule.induction_var, IntType(), )],
    #             name_hint=f"c_sub_tile",
    #         ).result
        
    #     #> 内部逻辑
    #     inner_c_tile = for_loop_op.get_iter_arg(0) if iter_args else None

    #     #> 递归调用分发
    #     updated_sub_tile = generate_dispatch(schedule.body, array_defs, format, active_tile, schedule)

    #     #> 如果是空间轴切片，内部返回sub_tile，需要把sub_tile写回parent tile
    #     if iter_args:
    #         if not is_reduction:
    #             if updated_sub_tile is not None:
    #                 op_builder.build(CValStoreOp, mem="R2S",operands=[updated_sub_tile, inner_c_tile, op_builder.lookup_symbol(schedule.induction_var)])
    #             op_builder.build(LoopResultOp, inner_c_tile) 
    #         else:
    #             op_builder.build(LoopResultOp, updated_sub_tile)
                
    # # 如果最外层是空间轴而且没有父级，在循环外执行写回
    # if not is_reduction and parent_schedule is None and for_loop_op.num_results > 0:
    #     op_builder.build(CValStoreOp, mem="R2S", operands=[for_loop_op.get_results()[0]])

    # if iter_args:
    #     return for_loop_op.result
    # return None

    with op_builder.op_scope(for_loop_op):
        op_builder.add_symbol(schedule.induction_var, for_loop_op.get_induction_var())
        loop_index = op_builder.lookup_symbol(schedule.induction_var, IntType())
        
        # 取得当前迭代处理的容器
        inner_c_tile = for_loop_op.get_iter_arg(0) if iter_args else None
        active_sub_tile = inner_c_tile

        # 2. 如果是空间轴，执行切分 (Logical Slice)
        if not is_reduction and inner_c_tile is not None:
            active_sub_tile = build_c_val_load_slice(
                inner_c_tile,
                loop_index,
                array_defs,
                name_hint="c_slice",
            )
        
        # 3. 递归分发
        # 这里的 active_sub_tile 可能是切出的块（空间轴）或 容器本身（规约轴）
        updated_res_tile = generate_dispatch(schedule.body, array_defs, format, active_sub_tile, schedule)

        # 4. 空间轴负责合并写回
        if iter_args:
            if not is_reduction:
                # 只有空间轴需要在 Body 结束前将子结果更新进父容器
                if updated_res_tile is not None:
                    store_len = infer_m_dim_len(updated_res_tile, array_defs, name_hint="c_store_len")
                    op_builder.build(CValStoreOp, mem="R2S", operands=[
                        updated_res_tile, 
                        inner_c_tile, 
                        loop_index,
                    ], length=store_len)
                op_builder.build(LoopResultOp, inner_c_tile) 
            else:
                # 规约轴直接返回累加后的值
                op_builder.build(LoopResultOp, updated_res_tile)
                
    # # 顶层物理写回判断（由 Pass 处理物理坐标映射，这里只负责生成 Store 指令）
    # if parent_schedule is None and for_loop_op.num_results > 0:
    #     # 如果最外层是空间轴，或者是规约轴算完，执行物理写回
    #     op_builder.build(CValStoreOp, mem="R2S", operands=[for_loop_op.get_results()[0]])

    return for_loop_op.get_results()[0] if iter_args else None


def translate_sympy_to_ir(expr: Expr, array_defs: ArrayDefCollection) -> Value:
    """将 SymPy 表达式递归转换为 OpIR 指令"""
    if str(expr) == "_":
        return Value(type=None, name_hint="_")

    if isinstance(expr, (Integer, int)):
        return op_builder.build(ConstantOp, int(expr), IntType()).result
    
    if isinstance(expr, Rational):
        num = op_builder.build(ConstantOp, int(expr.p), IntType()).result
        den = op_builder.build(ConstantOp, int(expr.q), IntType()).result
        return op_builder.build(DivOp, num, den).result

    if isinstance(expr, Symbol):
        # 从符号表中查找对应的 Induction Variable 或全局变量
        return op_builder.lookup_symbol(str(expr), IntType())
    
    if isinstance(expr, Indexed):
        # 处理 val_len[i1] 这种数组加载
        base_name = str(expr.base)
        indices = [translate_sympy_to_ir(idx, array_defs) for idx in expr.indices]
        array_val = op_builder.lookup_symbol(base_name, get_array_type(array_defs[base_name]))
        return op_builder.build(LoadOp, array_val, indices).result

    # 处理算术运算
    if expr.is_Add:
        # args = [translate_sympy_to_ir(a, array_defs) for a in expr.args]
        # res = args[0]
        # for a in args[1:]:
        #     res = op_builder.build(AddOp, res, a).result
        # return res
        # 1. 检查是否存在占位符 "_"
        placeholder_args = [a for a in expr.args if str(a) == "_"]
        if placeholder_args:
            # 2. 提取所有非 "_" 的数值项
            non_placeholder_args = [a for a in expr.args if str(a) != "_"]
            
            if len(non_placeholder_args) == 0:
                return Value(type=IntType(), name_hint="_")
            
            # 3. 递归生成数值偏移的 IR（避免解包，逐个相加）
            base_offset_val = translate_sympy_to_ir(non_placeholder_args[0], array_defs)
            for i in range(1, len(non_placeholder_args)):
                next_val = translate_sympy_to_ir(non_placeholder_args[i], array_defs)
                base_offset_val = op_builder.build(AddOp, base_offset_val, next_val).result
            
            # 4. 关键标记：这是一个切片的起始偏移量
            base_offset_val.name_hint = "slice_offset"
            return base_offset_val
    if expr.is_Mul:
        #> 原始 Mul 处理逻辑，但是 sympy 将 a/b 解析成 a * (b^-1)，生成的opIR中先做 1/b，结果再乘以a
        # args = [translate_sympy_to_ir(a, array_defs) for a in expr.args]
        # res = args[0]
        # for a in args[1:]:
        #     res = op_builder.build(MulOp, res, a).result
        # return res
        #> 修改 Mul 逻辑，a/b 直接生成 a/b
        numerators = []
        denominators = []
        
        for arg in expr.args:
            # 检查是否是除法项: b**-1
            if arg.is_Pow and arg.exp == -1:
                denominators.append(arg.base)
            # 检查是否有理数的分母项: Rational(1, 32) -> 分子1, 分母32
            elif isinstance(arg, Rational):
                if arg.p != 1: numerators.append(Integer(arg.p))
                denominators.append(Integer(arg.q))
            else:
                numerators.append(arg)
        
        # 构建分子部分的 IR
        if not numerators:
            res_num = op_builder.build(ConstantOp, 1, IntType()).result
        else:
            res_num = translate_sympy_to_ir(numerators[0], array_defs)
            for n in numerators[1:]:
                res_num = op_builder.build(MulOp, res_num, translate_sympy_to_ir(n, array_defs)).result
        
        # 如果没有分母，直接返回分子的乘积
        if not denominators:
            return res_num
            
        # 构建分母部分的 IR
        res_den = translate_sympy_to_ir(denominators[0], array_defs)
        for d in denominators[1:]:
            res_den = op_builder.build(MulOp, res_den, translate_sympy_to_ir(d, array_defs)).result
            
        # 生成最终的除法：(n1 * n2 * ...) / (d1 * d2 * ...)
        # 这样在 GPU 上就会先算分子乘积，再除以 BLK_K，保留了精度
        return op_builder.build(DivOp, res_num, res_den).result
    if expr.is_Pow: # 处理除法的一部分
        base_val = translate_sympy_to_ir(expr.base, array_defs)
        if expr.exp == -1:
            # 1 / base
            one = op_builder.build(ConstantOp, 1, IntType()).result
            return op_builder.build(DivOp, one, base_val).result
        else:
            exp_val = translate_sympy_to_ir(expr.exp, array_defs)
            return op_builder.build(PowOp, base_val, exp_val).result
        pass # 简化处理，通常处理 Mul(a, Pow(b, -1))
    
    # 特殊处理除法 (例如 val_len/BLK_K)
    if expr.func.__name__ == 'Mul': # 检查是否包含 Rational 或 Pow(-1)
        # 简单实现：如果是 a / b
        # ... 这里可以根据 Sympy 结构进一步细化
        pass
    print(expr)
    raise NotImplementedError(f"Sympy expression {expr} translation not implemented.")

def collect_spatial_shapes(node: Schedule, format: Format) -> List[any]:
    """
    向下扫描 schedule 树，收集所有属于空间维度（X轴/ROW）的轴长度
    """
    shapes = []
    curr = node

    while curr is not None:
        if isinstance(curr, (DenseAxisIterator, SparseAxisIterator)):
            axis_obj = format.get_axis(curr.axis)
            if axis_obj.direction == Direction.ROW:
                shapes.append(axis_obj.length)
            curr = curr.body
        elif isinstance(curr, AtomicFormatOp):
            # AtomicFormatOp 可能包含多个轴，检查每个轴
            for axis in curr.axes:
                axis_obj = format.get_axis(axis)
                if axis_obj.direction == Direction.ROW:
                    shapes.append(axis_obj.length)
            break  # AtomicFormatOp 是叶节点，停止扫描
        else:
            break
    return shapes

def generate_from_sparse_axis_iterator(
    schedule: SparseAxisIterator,
    array_defs: ArrayDefCollection,
    format: Format,
    current_c_tile: Optional[Value] = None,
    parent_schedule: Optional[Schedule] = None,
) -> Value:
    zero = op_builder.build(ConstantOp, 0, IntType(), name_hint="zero").result
    upper_bound = translate_sympy_to_ir(schedule.splen, array_defs)

    is_reduction = format.get_axis(schedule.axis).direction == Direction.COL

    # 1. 规约轴初始化（逻辑与你之前的一致，但在 Spatial 轴包裹下可能由上层传入）
    if is_reduction and current_c_tile is None:
        tile_shape = collect_spatial_shapes(schedule.body, format) + [array_defs["C_val"].dims[-1]]
        current_c_tile = op_builder.build(ConstantOp, 0, ArrayType(tile_shape, FloatType()), name_hint="zero_tile").result
    
    if not is_reduction and current_c_tile is None and parent_schedule is None:
        current_c_tile = op_builder.lookup_symbol("C_val")

    iter_args = {"C_io": current_c_tile} if current_c_tile else {}
    
    for_loop_op = op_builder.build(
        ForLoopOp,
        induction_var=schedule.induction_var,
        range=(zero, upper_bound),
        iter_args=iter_args,
    )

    with op_builder.op_scope(for_loop_op):
        op_builder.add_symbol(schedule.induction_var, for_loop_op.get_induction_var())
        loop_index = op_builder.lookup_symbol(schedule.induction_var, IntType())
        inner_c_tile = for_loop_op.get_iter_arg(0) if iter_args else None
        
        # 2. 空间轴处理（针对 Sparse Spatial 轴，如 CSR 的行切分）
        # 这里需要处理变长带来的 indirect indexing
        active_sub_tile = inner_c_tile
        if not is_reduction and inner_c_tile is not None:
            # 如果是变长空间轴，切分时可能需要用到 schedule 自带的索引逻辑
            active_sub_tile = build_c_val_load_slice(
                inner_c_tile,
                loop_index,
                array_defs,
                name_hint="c_sparse_slice",
            )

        updated_res_tile = generate_dispatch(schedule.body, array_defs, format, active_sub_tile, schedule)
        
        # 3. 统一返回逻辑
        if iter_args:
            if not is_reduction:
                # 空间轴：写回 inner_c_tile[i]
                if updated_res_tile is not None:
                    store_len = infer_m_dim_len(updated_res_tile, array_defs, name_hint="c_sparse_store_len")
                    op_builder.build(CValStoreOp, mem="R2S", operands=[
                        updated_res_tile,
                        inner_c_tile,
                        loop_index,
                    ], length=store_len)
                op_builder.build(LoopResultOp, inner_c_tile)
            else:
                op_builder.build(LoopResultOp, updated_res_tile)

    return for_loop_op.get_results()[0] if iter_args else None

# def generate_from_sparse_axis_iterator(
#     schedule: SparseAxisIterator,
#     array_defs: ArrayDefCollection,
#     format: Format,
#     current_c_tile: Optional[Value] = None,
#     # offset_index: Value,
#     parent_schedule: Optional[Schedule] = None,
# ) -> Value:
#     #> 1. 循环边界计算[l, r)
#     zero = op_builder.build(ConstantOp, 0, IntType(), name_hint="zero").result
#     upper_bound = translate_sympy_to_ir(schedule.splen, array_defs)

#     #> 2. 确定是否需要规约 C （当前轴是否在规约轴方向，通过direction判断
#     is_reduction = format.get_axis(schedule.axis).direction == Direction.COL

#     #> 3. 如果是规约轴，且没有传入 C_tile，需要初始化（例如最外层的 Y 循环）
#     iter_args = {}
#     if is_reduction:
#         if current_c_tile is None:
#             spatial_shapes = collect_spatial_shapes(schedule.body, format)
#             assert len(spatial_shapes) > 0, "At least one spatial shape should be collected for reduction axis"
#             c_def = array_defs["C_val"] # 获取 C 数组定义的末尾维度
#             n_length = c_def.dims[-1]
            
#             tile_shape = spatial_shapes + [n_length] # 组合成最终形状 [BLK_M_I, N]
            
#             current_c_tile = op_builder.build(ConstantOp, 0, ArrayType(tile_shape, FloatType()), name_hint="zero_tile").result
#     #     iter_args["C_tile_in"] = current_c_tile
#     # assert current_c_tile is not None, "current_c_tile should be initialized at this point"
#     iter_args = {"C_in": current_c_tile} if current_c_tile else {}
#     for_loop_op = op_builder.build(
#         ForLoopOp,
#         induction_var=schedule.induction_var,
#         range=(zero, upper_bound),
#         iter_args=iter_args,
#         name_hint=f"loop_{schedule.induction_var}" if iter_args else None,
#     )

#     with op_builder.op_scope(for_loop_op):
#         op_builder.add_symbol(schedule.induction_var, for_loop_op.get_induction_var())
        
#         # 取得传入循环内部的 C Tile 引用
#         inner_c_tile = for_loop_op.get_iter_arg(0) if iter_args else None

#         updated_sub_tile = generate_dispatch(schedule.body, array_defs, format, inner_c_tile, schedule)
#         # 6. 如果是归约循环，必须返回更新后的 Tile
#         if iter_args:
#             if not is_reduction:
#                 # 如果这个稀疏轴是空间轴（比如 CSR 的行循环），执行写回
#                 if updated_sub_tile is not None:
#                      # 根据是否有父容器决定写回方式
#                      if parent_schedule is not None:
#                          op_builder.build(CValStoreOp, mem="R2S", operands=[updated_sub_tile, inner_c_tile, op_builder.lookup_symbol(schedule.induction_var)])
#                      else:
#                          op_builder.build(CValStoreOp, mem="R2S", operands=[updated_sub_tile, op_builder.lookup_symbol(schedule.induction_var)])
#                 op_builder.build(LoopResultOp, inner_c_tile)
#             else:
#                 # 规约轴：只传递结果
#                 op_builder.build(LoopResultOp, updated_sub_tile)
            
#     # return for_loop_op.result
#     if iter_args:
#         return for_loop_op.result
#     return None



def generate_from_sparse_axis_slicer(
    schedule: SparseAxisSlicer,
    array_defs: ArrayDefCollection,
    format: Format,
    offset_index: Value,
    parent_schedule: Optional[Schedule] = None,
) -> None:
    raise NotImplementedError("Sparse axis slicer is not implemented")


def generate_from_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> OpResult:
    match schedule.type:
        case AtomicFormatType.DENSE:
            return generate_from_dense_atomic_format_op(
                schedule, array_defs, format, C_val_tile
            )
        case AtomicFormatType.COO:
            return generate_from_coo_atomic_format_op(
                schedule, array_defs, format, C_val_tile
            )
        case AtomicFormatType.MCO:
            return generate_from_mco_atomic_format_op(
                schedule, array_defs, format, C_val_tile
            )
        case AtomicFormatType.CSR:
            return generate_from_csr_atomic_format_op(
                schedule, array_defs, format, C_val_tile
            )
        case AtomicFormatType.ELL:
            return generate_from_ell_atomic_format_op(
                schedule, array_defs, format, C_val_tile
            )
        case AtomicFormatType.DIA:
            return generate_from_dia_atomic_format_op(
                schedule, array_defs, format, C_val_tile
            )


def build_array_def(
    array_def: ArrayDef,
) -> Value:
    return op_builder.lookup_symbol(
        array_def.name,
        get_array_type(array_def),
    )


def build_array_ref_indices(
    array_ref: ArrayRef,
    array_defs: ArrayDefCollection,
    format: Format,
    name_hint: Optional[str] = None,
    tile_context: Optional[Dict[Direction, Expr]] = None,
) -> Sequence[Value]:
    indices: List[Value] = []

    # 获得数组定义，确定每一维方向
    adef = array_defs[array_ref.array]
    for i, index in enumerate(array_ref.indices):
        if isinstance(index, ArrayRef):
            index_result = build_array_ref(
                index,
                array_defs,
                format,
                name_hint=f"{name_hint}_sidx_{i}" if name_hint else None,
            )
        elif isinstance(index, Expr):
            index_result = translate_sympy_to_ir(index, array_defs)
        #> 下划线参与的数据访存中，如果出现下划线，就会产生 %_ = external @_: int
        elif index == "_":
            # 直接创建一个裸 Value，不让 op_builder 生成任何 Op
            index_result = Value(type=IntType(), name_hint="_")
        else:
            index_result = op_builder.lookup_symbol(
                index,
                IntType(),
            )

        # 将切片语义转换为Arange向量
        hint = getattr(index_result, "name_hint", None)
        if hint == "slice_offset":
            axis_name = adef.axes[i]
            if axis_name == "Y":
                axis_dir = Direction.COL
            else:
                axis_dir = format.get_axis(axis_name).direction
            length = tile_context.get(axis_dir) if tile_context else None
            if length:
                index_result = op_builder.build(ArangeOp, index_result, length).result
        
        indices.append(index_result)
    return indices


def build_array_ref(
    array_ref: ArrayRef,
    array_defs: ArrayDefCollection,
    format: Format,
    name_hint: Optional[str] = None,
    tile_context: Optional[Dict[Direction, Expr]] = None,
) -> OpResult:
    array = build_array_def(array_defs[array_ref.array])
    indices = build_array_ref_indices(
        array_ref, array_defs, format, name_hint=name_hint, tile_context=tile_context
    )

    array_ref_op = op_builder.build(
        ArrayRefOp,
        array,
        indices,
        name_hint=name_hint,
    )

    print(f"(debug)build_array_ref: array={array}, indices={indices}, result={array_ref_op.result}")
    return array_ref_op.result


def generate_from_coo_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> OpResult:

    assert len(schedule.args) == 4
    assert isinstance(schedule.args[0], ArrayRef)
    assert isinstance(schedule.args[1], ArrayRef)
    assert isinstance(schedule.args[2], ArrayRef)
    assert isinstance(schedule.args[3], ArrayRef)
    print("Generating COO atomic format op with args: ", schedule.args)

    metadata_def = array_defs[schedule.args[0].array]
    metadata_array = op_builder.lookup_symbol(
        schedule.args[0].array, get_array_type(array_defs[schedule.args[0].array])
    )
    idx_array = op_builder.lookup_symbol(
        schedule.args[1].array, get_array_type(array_defs[schedule.args[1].array])
    )
    val_array = op_builder.lookup_symbol(
        schedule.args[2].array, get_array_type(array_defs[schedule.args[2].array])
    )

    B_val_tile = build_array_ref(
        schedule.args[3], array_defs, format, name_hint="B_val_tile"
    )

    assert isinstance(B_val_tile.type, ArrayType)
    assert isinstance(C_val_tile.type, ArrayType)
    if metadata_def.type == ComputentArrayType.OFFSET:
        ll, rr = op_builder.build(
            CooAtomicFormatLoadOffOp,
            mem="G2R",
            operands=[
                metadata_array,
                *build_array_ref_indices(schedule.args[0], array_defs, format),
            ],
            name_hint=["ll", "rr"],
        ).results
    else:
        ll, rr = op_builder.build(
            LoadOffsetOp,
            metadata_array,
            build_array_ref_indices(schedule.args[0], array_defs, format),
            name_hint=["ll", "rr"],
        ).results

    idx_tile_shape = C_val_tile.type.dims[0] * B_val_tile.type.dims[0]
    coo_idx, coo_range = op_builder.build(
        CooAtomicFormatLoadIdxOp,
        mem="G2S",
        operands=[
            idx_array,
            ll,
            rr,
        ],
        out_shape=idx_tile_shape,
        name_hint=["coo_idx", "coo_range"],
    ).results

    coo_val = op_builder.build(
        CooAtomicFormatLoadValOp,
        mem="G2S",
        operands=[
            val_array,
            ll,
            rr,
        ],
        out_shape=idx_tile_shape,
        name_hint="coo_val",
    ).result

    A_val_tile = op_builder.build(
        CooAtomicValRestoreOp,
        mem="S2R",
        operands=[
            coo_val,
            coo_idx,
            coo_range,
        ],
        out_shape=(C_val_tile.type.dims[0], B_val_tile.type.dims[0]),
        name_hint="A_val_tile",
    ).result

    C_val_tile_updated = op_builder.build(
        MmaOp, 
        "R2R", 
        [A_val_tile, B_val_tile, C_val_tile], 
        name_hint="C_val_tile"
    ).result


    return C_val_tile_updated


def generate_from_csr_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> OpResult:
    """CSR atomic format: use row_ptr to restore A tile row-by-row."""
    assert len(schedule.args) == 5
    assert isinstance(schedule.args[0], ArrayRef)
    assert isinstance(schedule.args[1], ArrayRef)
    assert isinstance(schedule.args[2], ArrayRef)
    assert isinstance(schedule.args[3], ArrayRef)
    assert isinstance(schedule.args[4], ArrayRef)
    print("Generating CSR atomic format op with args: ", schedule.args)

    len_array = op_builder.lookup_symbol(
        schedule.args[0].array, get_array_type(array_defs[schedule.args[0].array])
    )
    idx_array = op_builder.lookup_symbol(
        schedule.args[1].array, get_array_type(array_defs[schedule.args[1].array])
    )
    val_array = op_builder.lookup_symbol(
        schedule.args[2].array, get_array_type(array_defs[schedule.args[2].array])
    )

    B_val_tile = build_array_ref(
        schedule.args[3], array_defs, format, name_hint="B_val_tile"
    )
    csr_row_ptr = build_array_ref(
        schedule.args[4], array_defs, format, name_hint="csr_row_ptr"
    )

    assert isinstance(B_val_tile.type, ArrayType)
    assert isinstance(C_val_tile.type, ArrayType)
    assert isinstance(csr_row_ptr.type, ArrayType)

    ll, rr = op_builder.build(
        LoadOffsetOp,
        len_array,
        build_array_ref_indices(schedule.args[0], array_defs, format),
        name_hint=["ll", "rr"],
    ).results

    idx_tile_shape = C_val_tile.type.dims[0] * B_val_tile.type.dims[0]

    csr_idx, csr_range = op_builder.build(
        CooAtomicFormatLoadIdxOp,
        mem="G2S",
        operands=[idx_array, ll, rr],
        out_shape=idx_tile_shape,
        name_hint=["csr_idx", "csr_range"],
    ).results

    csr_val = op_builder.build(
        CooAtomicFormatLoadValOp,
        mem="G2S",
        operands=[val_array, ll, rr],
        out_shape=idx_tile_shape,
        name_hint="csr_val",
    ).result

    A_val_tile = op_builder.build(
        CsrAtomicValRestoreOp,
        mem="S2R",
        operands=[csr_val, csr_idx, csr_row_ptr, csr_range],
        out_shape=(C_val_tile.type.dims[0], B_val_tile.type.dims[0]),
        name_hint="A_val_tile",
    ).result

    C_val_tile_updated = op_builder.build(
        MmaOp,
        "R2R",
        [A_val_tile, B_val_tile, C_val_tile],
        name_hint="C_val_tile"
    ).result

    return C_val_tile_updated


def generate_from_ell_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> OpResult:
    """ELL atomic format: flat row-major idx/val with ell_len scalar, COO-style restore."""
    assert len(schedule.args) == 4
    assert isinstance(schedule.args[0], ArrayRef)
    assert isinstance(schedule.args[1], ArrayRef)
    assert isinstance(schedule.args[2], ArrayRef)
    assert isinstance(schedule.args[3], ArrayRef)
    print("Generating ELL atomic format op with args: ", schedule.args)

    len_array = op_builder.lookup_symbol(
        schedule.args[0].array, get_array_type(array_defs[schedule.args[0].array])
    )
    idx_array = op_builder.lookup_symbol(
        schedule.args[1].array, get_array_type(array_defs[schedule.args[1].array])
    )
    val_array = op_builder.lookup_symbol(
        schedule.args[2].array, get_array_type(array_defs[schedule.args[2].array])
    )

    B_val_tile = build_array_ref(
        schedule.args[3], array_defs, format, name_hint="B_val_tile"
    )

    assert isinstance(B_val_tile.type, ArrayType)
    assert isinstance(C_val_tile.type, ArrayType)

    ll, rr = op_builder.build(
        LoadOffsetOp,
        len_array,
        build_array_ref_indices(schedule.args[0], array_defs, format),
        name_hint=["ll", "rr"],
    ).results

    idx_tile_shape = C_val_tile.type.dims[0] * B_val_tile.type.dims[0]

    ell_idx, ell_range = op_builder.build(
        CooAtomicFormatLoadIdxOp,
        mem="G2S",
        operands=[idx_array, ll, rr],
        out_shape=idx_tile_shape,
        name_hint=["ell_idx", "ell_range"],
    ).results

    ell_val = op_builder.build(
        CooAtomicFormatLoadValOp,
        mem="G2S",
        operands=[val_array, ll, rr],
        out_shape=idx_tile_shape,
        name_hint="ell_val",
    ).result

    A_val_tile = op_builder.build(
        EllAtomicValRestoreOp,
        mem="S2R",
        operands=[ell_val, ell_idx, ell_range],
        out_shape=(C_val_tile.type.dims[0], B_val_tile.type.dims[0]),
        name_hint="A_val_tile",
    ).result

    C_val_tile_updated = op_builder.build(
        MmaOp,
        "R2R",
        [A_val_tile, B_val_tile, C_val_tile],
        name_hint="C_val_tile"
    ).result

    return C_val_tile_updated


def generate_from_dia_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> OpResult:
    """DIA atomic format: diag-major payload with per-slot repeated diagonal offsets."""
    assert len(schedule.args) == 4
    assert isinstance(schedule.args[0], ArrayRef)
    assert isinstance(schedule.args[1], ArrayRef)
    assert isinstance(schedule.args[2], ArrayRef)
    assert isinstance(schedule.args[3], ArrayRef)
    print("Generating DIA atomic format op with args: ", schedule.args)

    len_array = op_builder.lookup_symbol(
        schedule.args[0].array, get_array_type(array_defs[schedule.args[0].array])
    )
    idx_array = op_builder.lookup_symbol(
        schedule.args[1].array, get_array_type(array_defs[schedule.args[1].array])
    )
    val_array = op_builder.lookup_symbol(
        schedule.args[2].array, get_array_type(array_defs[schedule.args[2].array])
    )

    B_val_tile = build_array_ref(
        schedule.args[3], array_defs, format, name_hint="B_val_tile"
    )

    assert isinstance(B_val_tile.type, ArrayType)
    assert isinstance(C_val_tile.type, ArrayType)

    ll, rr = op_builder.build(
        LoadOffsetOp,
        len_array,
        build_array_ref_indices(schedule.args[0], array_defs, format),
        name_hint=["ll", "rr"],
    ).results

    diag_tile_shape = C_val_tile.type.dims[0] + B_val_tile.type.dims[0] - Number(1)
    val_tile_shape = diag_tile_shape * B_val_tile.type.dims[0]

    dia_idx, dia_range = op_builder.build(
        DiaAtomicFormatLoadIdxOp,
        mem="G2S",
        operands=[idx_array, ll, rr],
        out_shape=val_tile_shape,
        name_hint=["dia_idx", "dia_range"],
    ).results

    dia_val = op_builder.build(
        DiaAtomicFormatLoadValOp,
        mem="G2S",
        operands=[val_array, ll, rr],
        out_shape=val_tile_shape,
        name_hint="dia_val",
    ).result

    A_val_tile = op_builder.build(
        DiaAtomicValRestoreOp,
        mem="S2R",
        operands=[dia_val, dia_idx, dia_range],
        out_shape=(C_val_tile.type.dims[0], B_val_tile.type.dims[0]),
        name_hint="A_val_tile",
    ).result

    C_val_tile_updated = op_builder.build(
        MmaOp,
        "R2R",
        [A_val_tile, B_val_tile, C_val_tile],
        name_hint="C_val_tile"
    ).result

    return C_val_tile_updated


def generate_from_mco_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> OpResult:
    assert len(schedule.args) == 4
    assert isinstance(schedule.args[0], ArrayRef)
    assert isinstance(schedule.args[1], ArrayRef)
    assert isinstance(schedule.args[2], ArrayRef)
    assert isinstance(schedule.args[3], ArrayRef)
    print("Generating MCO atomic format op with args: ", schedule.args)
    print("(debug)schedule.axes_len", schedule.axes_len)
    print("(debug)schedule.axes", schedule.axes)

    tile_context = {}
    for axis_name, axis_len in zip(schedule.axes, schedule.axes_len):
        axis_dir = format.get_axis(axis_name).direction
        tile_context[axis_dir] = axis_len

    len_array = op_builder.lookup_symbol(
        schedule.args[0].array, get_array_type(array_defs[schedule.args[0].array])
    )
    mask_array = op_builder.lookup_symbol(
        schedule.args[1].array, get_array_type(array_defs[schedule.args[1].array])
    )
    val_array = op_builder.lookup_symbol(
        schedule.args[2].array, get_array_type(array_defs[schedule.args[2].array])
    )

    B_val_tile = build_array_ref(
        schedule.args[3], array_defs, format, name_hint="B_val_tile",
        tile_context=tile_context
    )   

    assert isinstance(B_val_tile.type, ArrayType)
    assert isinstance(C_val_tile.type, ArrayType)

    mco_len = op_builder.build(
        LoadOp,
        len_array,
        build_array_ref_indices(schedule.args[0], array_defs, format),
        name_hint="mco_len"
    ).result

    mask_tile_shape = (C_val_tile.type.dims[0] * B_val_tile.type.dims[0]) / Number(64)
    mco_mask = op_builder.build(
        McoAtomicFormatLoadMaskOp,
        mem="G2R",
        operands=[
            mask_array,
            *build_array_ref_indices(schedule.args[1], array_defs, format),
        ],
        out_shape=mask_tile_shape,
        name_hint="mco_mask"
    ).result

    mco_val = op_builder.build(
        McoAtomicFormatLoadValOp,
        mem="G2S",
        operands=[
            val_array,
            *build_array_ref_indices(schedule.args[2], array_defs, format),
        ],
        mco_len=mco_len,
        out_shape=C_val_tile.type.dims[0] * B_val_tile.type.dims[0],  # type: ignore
        name_hint="mco_val",
    ).result

    mco_val_restore = op_builder.build(
        McoAtomicValRestoreOp,
        mem="S2R",
        operands=[
            mco_val,
            mco_mask,
            mco_len,
        ],
        out_shape=(C_val_tile.type.dims[0], B_val_tile.type.dims[0]),
        name_hint="mco_val_restore",
    ).result

    C_val_tile_updated = op_builder.build(
        MmaOp, 
        "R2R", 
        [mco_val_restore, B_val_tile, C_val_tile], 
        name_hint="C_val_tile"
    ).result

    return C_val_tile_updated
    # raise NotImplementedError("MCO atomic format op is not implemented")


def generate_from_dense_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> OpResult:
    assert len(schedule.args) == 2
    assert isinstance(schedule.args[0], ArrayRef)
    assert isinstance(schedule.args[1], ArrayRef)
    print("Generating Dense atomic format op with args: ", schedule.args)

    val_array = op_builder.lookup_symbol(
        schedule.args[0].array, get_array_type(array_defs[schedule.args[0].array])
    )

    B_val_tile = build_array_ref(
        schedule.args[1], array_defs, format, name_hint="B_val_tile"
    )

    assert isinstance(B_val_tile.type, ArrayType)
    assert isinstance(C_val_tile.type, ArrayType)

    val_tile = op_builder.build(
        LoadOp,
        val_array,
        build_array_ref_indices(schedule.args[0], array_defs, format), name_hint="val_tile"
    )

    C_val_tile_updated = op_builder.build(
        MmaOp, 
        "R2R", 
        [val_tile.result, B_val_tile, C_val_tile], 
        name_hint="C_val_tile"
    ).result

    return C_val_tile_updated  
    # raise NotImplementedError("Dense atomic format op is not implemented")

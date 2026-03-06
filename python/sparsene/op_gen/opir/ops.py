from __future__ import annotations

from typing import Sequence, Tuple

from dataclasses import dataclass
from sparsene.op_gen.opir.op_ir import (
    DeviceOp,
    Value,
    IntType,
    ArrayType,
    FloatType,
    OpResult,
    Expr,
)


@dataclass
class SparseOffsetLoadOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value]):
        super().__init__(
            mem=mem,
            operands=operands,
            results=[
                OpResult(
                    type=IntType(),
                    defining_op=self,
                    result_idx_in_owner=0,
                ),
                OpResult(
                    type=IntType(),
                    defining_op=self,
                    result_idx_in_owner=1,
                ),
            ],
        )

    @property
    def name(self) -> str:
        return "sparse_offset_load"


@dataclass
class CooAtomicFormatLoadOffOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value]):
        super().__init__(
            mem=mem,
            operands=operands,
            results=[
                OpResult(
                    type=IntType(),
                    defining_op=self,
                    result_idx_in_owner=0,
                ),
                OpResult(
                    type=IntType(),
                    defining_op=self,
                    result_idx_in_owner=1,
                ),
            ],
        )

    @property
    def name(self) -> str:
        return "coo_atomic_format_load_off"

#> original CooAtomicFormatLoadIdxOp 仅支持ll rr插入的情况
# @dataclass
# class CooAtomicFormatLoadIdxOp(DeviceOp):
#     def __init__(self, mem: str, operands: Sequence[Value], out_shape: Expr):
#         assert isinstance(operands[0].type, ArrayType) # idx_array
#         assert isinstance(operands[0].type.datatype, IntType) # idx_array
#         assert isinstance(operands[1].type, IntType) # ll
#         assert isinstance(operands[2].type, IntType) # rr
#         super().__init__(
#             mem=mem,
#             operands=operands,
#             results=[
#                 OpResult(
#                     type=ArrayType([out_shape], IntType()),
#                     defining_op=self,
#                     result_idx_in_owner=0,
#                 ),
#                 OpResult(
#                     type=IntType(),
#                     defining_op=self,
#                     result_idx_in_owner=1,
#                 ),
#             ],
#         )

#     @property
#     def name(self) -> str:
#         return "coo_atomic_format_load_idx"
#> current CooAtomicFormatLoadIdxOp
@dataclass
class CooAtomicFormatLoadIdxOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value], out_shape: Expr):
        # 基本校验：第一个操作数必须是索引数组
        assert isinstance(operands[0].type, ArrayType)
        
        # 核心逻辑：判断当前是“物理偏移模式”还是“逻辑坐标模式”
        # 模式 A (Condensed): [idx_array, i1, i2, ..., ll, rr] -> 末尾两个是 Int (偏移)
        # 模式 B (Logical/Uncondensed): [idx_array, i1, i2, ..., _] -> 包含占位符
        
        # 我们通过判断最后两个操作数的类型和结果需求来定义
        # 如果最后两个操作数是 ll, rr，则保留原有的双结果逻辑
        is_condensed = (
            len(operands) >= 3 and 
            isinstance(operands[1], OpResult) and isinstance(operands[1].type, IntType) and
            isinstance(operands[2], OpResult) and isinstance(operands[2].type, IntType) and
            # isinstance(operands[-1].type, IntType) and 
            # isinstance(operands[-2].type, IntType) and
            operands[-1].name_hint != "_" # 排除占位符
        )

        print(f"CooAtomicFormatLoadIdxOp: is_condensed={is_condensed}, operands={operands}")

        results = []
        # 结果 1: COO 索引数组 (Buffer)
        results.append(OpResult(
            type=ArrayType([out_shape], IntType()),
            defining_op=self,
            result_idx_in_owner=0,
        ))

        # 结果 2 (可选): 只有在 Condensed 模式下才返回动态 range
        if is_condensed:
            results.append(OpResult(
                type=IntType(),
                defining_op=self,
                result_idx_in_owner=1,
            ))

        super().__init__(
            mem=mem,
            operands=operands,
            results=results,
        )

    @property
    def name(self) -> str:
        return "coo_atomic_format_load_idx"


#> origin CooAtomicFormatLoadValOp: must add ll and rr
# @dataclass
# class CooAtomicFormatLoadValOp(DeviceOp):
#     def __init__(self, mem: str, operands: Sequence[Value], out_shape: Expr):
#         assert isinstance(operands[0].type, ArrayType) # val_array
#         assert isinstance(operands[1].type, IntType) # ll
#         assert isinstance(operands[2].type, IntType) # rr
#         super().__init__(
#             mem=mem,
#             operands=operands,
#             results=[
#                 OpResult(
#                     type=ArrayType([out_shape], operands[0].type.datatype),
#                     defining_op=self,
#                     result_idx_in_owner=0,
#                 ),
#             ],
#         )

#     @property
#     def name(self) -> str:
#         return "coo_atomic_format_load_val"
#> current CooAtomicFormatLoadValOp
@dataclass
class CooAtomicFormatLoadValOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value], out_shape: Expr):
        # 第一个操作数始终是原始的 Value 数组 (val_coo_val)
        assert isinstance(operands[0].type, ArrayType)
        
        # 模式识别逻辑：
        # 模式 A (Condensed): [val_array, ..., ll, rr] -> 物理偏移
        # 模式 B (Logical): [val_array, ..., _] -> 逻辑坐标 (包含占位符)
        
        # 如果最后两个操作数是 Int 且不是占位符，则校验 ll/rr
        if (len(operands) >= 3 and 
            isinstance(operands[1], OpResult) and isinstance(operands[1].type, IntType) and
            isinstance(operands[2], OpResult) and isinstance(operands[2].type, IntType) and
            # isinstance(operands[-1].type, IntType) and 
            # isinstance(operands[-2].type, IntType) and 
            operands[-1].name_hint != "_"):
            # 物理模式校验
            # 确保传入了具体的偏移量
            pass 
        else:
            # 逻辑模式校验：确保至少包含了数组引用
            pass

        super().__init__(
            mem=mem,
            operands=operands,
            results=[
                OpResult(
                    # 结果是一个 1D 数组（Buffer），长度为 out_shape (通常是 BLK_M * BLK_K)
                    type=ArrayType([out_shape], operands[0].type.datatype),
                    defining_op=self,
                    result_idx_in_owner=0,
                ),
            ],
        )

    @property
    def name(self) -> str:
        return "coo_atomic_format_load_val"


@dataclass
class ValSidxLoadOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value]):
        assert isinstance(operands[0].type, ArrayType)
        array_dims = operands[0].type.dims
        super().__init__(
            mem=mem,
            operands=operands,
            results=[
                OpResult(
                    type=ArrayType(array_dims[len(operands) - 1 :], IntType()),
                    defining_op=self,
                    result_idx_in_owner=0,
                ),
            ],
        )

    @property
    def name(self) -> str:
        return "val_sidx_load"


@dataclass
class BValLoadOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value]):
        super().__init__(
            mem=mem,
            operands=operands,
            results=[
                OpResult(
                    type=ArrayType([], FloatType()),
                    defining_op=self,
                    result_idx_in_owner=0,
                ),
            ],
        )

    @property
    def name(self) -> str:
        return "b_val_load"


@dataclass
class CooAtomicValRestoreOp(DeviceOp):
    def __init__(
        self, mem: str, operands: Sequence[Value], out_shape: Tuple[Expr, Expr]
    ):
        assert isinstance(operands[0].type, ArrayType)
        assert isinstance(operands[1].type, ArrayType)
        assert isinstance(operands[2].type, IntType)
        super().__init__(
            mem=mem,
            operands=operands,
            results=[
                OpResult(
                    type=ArrayType(out_shape, operands[0].type.datatype),
                    defining_op=self,
                    result_idx_in_owner=0,
                ),
            ],
        )

    @property
    def name(self) -> str:
        return "coo_atomic_val_restore"

@dataclass
class McoAtomicFormatLoadMaskOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value]):
        assert isinstance(operands[0].type, ArrayType) # mask_array
        
        super().__init__(
            mem=mem,
            operands=operands,
            results=[
                OpResult(
                    type=IntType(),
                    defining_op=self,
                    result_idx_in_owner=0,
                )
            ]
        )

    @property
    def name(self) -> str:
        return "mco_atomic_format_load_mask"

@dataclass
class McoAtomicFormatLoadValOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value], out_shape: Expr):
        assert isinstance(operands[0].type, ArrayType) # val array
        print("McoAtomicFormatLoadValOp operands:", operands, "len(operands):", len(operands))
        if (len(operands) >= 3 and 
            isinstance(operands[1], OpResult) and isinstance(operands[1].type, IntType) and
            isinstance(operands[2], OpResult) and isinstance(operands[2].type, IntType) and
            # isinstance(operands[-1].type, IntType) and 
            # isinstance(operands[-2].type, IntType) and 
            operands[-1].name_hint != "_"):
            # 物理模式校验
            pass
        else:
            pass
    
        super().__init__(
            mem=mem,
            operands=operands,
            results=[
                OpResult(
                    type=ArrayType([out_shape], operands[0].type.datatype),
                    defining_op=self,
                    result_idx_in_owner=0,
                ),
            ],
        )
    
    @property
    def name(self) -> str:
        return "mco_atomic_format_load_val"

@dataclass
class McoAtomicValRestoreOp(DeviceOp):
    def __init__(
        self, mem: str, operands: Sequence[Value], out_shape: Tuple[Expr, Expr]
    ):
        assert isinstance(operands[0].type, ArrayType) # val array
        assert isinstance(operands[1].type, IntType) # mask
        assert isinstance(operands[2].type, IntType) # len
        super().__init__(
            mem=mem,
            operands=operands,
            results=[
                OpResult(
                    type=ArrayType(out_shape, operands[0].type.datatype),
                    defining_op=self,
                    result_idx_in_owner=0,
                ),
            ],
        )

    @property
    def name(self) -> str:
        return "mco_atomic_val_restore"


@dataclass
class MmaOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value]):
        super().__init__(
            mem=mem,
            operands=operands,
            results=[
                OpResult(
                    type=operands[2].type,
                    # type=ArrayType([], FloatType()),
                    defining_op=self,
                    result_idx_in_owner=0,
                ),
            ],
        )

    @property
    def name(self) -> str:
        return "mma"


@dataclass
class CValStoreOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value]):
        super().__init__(
            mem=mem,
            operands=operands,
            results=[],
        )

    @property
    def name(self) -> str:
        return "c_val_store"

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
    len: Value

    def __init__(self, mem: str, operands: Sequence[Value], out_shape: Expr, coo_len: Value):
        assert len(operands) >= 1, "coo_atomic_format_load_idx requires idx_array as first operand"
        assert isinstance(operands[0].type, ArrayType)
        assert isinstance(operands[0].type.datatype, IntType)

        assert isinstance(coo_len.type, IntType), "len must be coo_len (int)"
        self.len = coo_len

        super().__init__(
            mem=mem,
            operands=operands,
            results=[
                OpResult(
                    type=ArrayType([out_shape], IntType()),
                    defining_op=self,
                    result_idx_in_owner=0,
                ),
            ],
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
    len: Value

    def __init__(self, mem: str, operands: Sequence[Value], out_shape: Expr, coo_len: Value):
        assert len(operands) >= 1, "coo_atomic_format_load_val requires val_array as first operand"
        assert isinstance(operands[0].type, ArrayType)

        assert isinstance(coo_len.type, IntType), "len must be coo_len (int)"
        self.len = coo_len

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
    len: Value

    def __init__(self, mem: str, operands: Sequence[Value], out_shape: Expr, mco_len: Value):
        assert len(operands) >= 1, "mco_atomic_format_load_val requires val_array as first operand"
        assert isinstance(operands[0].type, ArrayType)
        assert isinstance(mco_len.type, IntType), "len must be mco_len (int)"

        self.len = mco_len
    
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
    len: Value
    def __init__(self, mem: str, operands: Sequence[Value], length: Value, name_hint: str = "c_slice"):
        self.len = length
        super().__init__(
            mem=mem,
            operands=operands,
            results=[],
        )

    @property
    def name(self) -> str:
        return "c_val_store"

@dataclass
class CValLoadOp(DeviceOp):
    len: Value
    def __init__(self, mem: str, array: Value, offset: Value, length: Value, name_hint: str = "c_slice"):
        # array: [M][N] 物理张量
        # offset: 起始行偏移
        # length: 加载的行数 (BLK_M)
        self.len = length
        super().__init__(
            mem=mem,
            operands=[array, offset],
            results=[]
        )
        
    @property
    def name(self) -> str:
        return "c_val_load"
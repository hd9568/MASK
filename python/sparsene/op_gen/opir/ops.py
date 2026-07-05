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
        assert len(operands) == 3, "coo_atomic_format_load_idx requires idx_array, ll, rr"
        assert isinstance(operands[0].type, ArrayType)
        assert isinstance(operands[0].type.datatype, IntType)
        assert isinstance(operands[1].type, IntType), "ll must be int"
        assert isinstance(operands[2].type, IntType), "rr must be int"

        super().__init__(
            mem=mem,
            operands=operands,
            results=[
                OpResult(
                    type=ArrayType([out_shape], IntType()),
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
        assert len(operands) == 3, "coo_atomic_format_load_val requires val_array, ll, rr"
        assert isinstance(operands[0].type, ArrayType)
        assert isinstance(operands[1].type, IntType), "ll must be int"
        assert isinstance(operands[2].type, IntType), "rr must be int"

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
class CsrAtomicValRestoreOp(DeviceOp):
    """CSR-specific restore: takes row_ptr (BLK_M+1 ints) + idx/val arrays.

    Unlike COO which takes a scalar coo_range (=nnz) and scatters flatly,
    CSR iterates row-by-row using the row_ptr to bound per-row ranges.
    operands = [csr_val, csr_idx, csr_row_ptr, csr_nnz]
    """
    def __init__(
        self, mem: str, operands: Sequence[Value], out_shape: Tuple[Expr, Expr]
    ):
        assert len(operands) == 4, "csr_atomic_val_restore requires [val, idx, row_ptr, nnz]"
        assert isinstance(operands[0].type, ArrayType)  # val array (smem)
        assert isinstance(operands[1].type, ArrayType)  # idx array (smem)
        assert isinstance(operands[2].type, ArrayType)  # row_ptr array (gmem/rmem)
        assert isinstance(operands[3].type, IntType)    # nnz (scalar range)
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
        return "csr_atomic_val_restore"


@dataclass
class EllAtomicValRestoreOp(DeviceOp):
    """ELL-specific restore: uses ell_len scalar + flat idx/val arrays.

    ELL stores data row-major packed across [BLK_M, max_nnz_per_row] 2D layout.
    The restore iterates flatly [0, BLK_M * ell_len) with per-row stride.
    operands = [ell_val, ell_idx, ell_len]  (3 operands, like COO)
    """
    def __init__(
        self, mem: str, operands: Sequence[Value], out_shape: Tuple[Expr, Expr]
    ):
        assert len(operands) == 3, "ell_atomic_val_restore requires [val, idx, ell_len]"
        assert isinstance(operands[0].type, ArrayType)  # val array (smem)
        assert isinstance(operands[1].type, ArrayType)  # idx array (smem)
        assert isinstance(operands[2].type, IntType)    # ell_len (scalar)
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
        return "ell_atomic_val_restore"


@dataclass
class DiaAtomicFormatLoadIdxOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value], out_shape: Expr):
        assert len(operands) == 3, "dia_atomic_format_load_idx requires idx_array, ll, rr"
        assert isinstance(operands[0].type, ArrayType)
        assert isinstance(operands[0].type.datatype, IntType)
        assert isinstance(operands[1].type, IntType)
        assert isinstance(operands[2].type, IntType)
        super().__init__(
            mem=mem,
            operands=operands,
            results=[
                OpResult(
                    type=ArrayType([out_shape], IntType()),
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
        return "dia_atomic_format_load_idx"


@dataclass
class DiaAtomicFormatLoadValOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value], out_shape: Expr):
        assert len(operands) == 3, "dia_atomic_format_load_val requires val_array, ll, rr"
        assert isinstance(operands[0].type, ArrayType)
        assert isinstance(operands[1].type, IntType)
        assert isinstance(operands[2].type, IntType)
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
        return "dia_atomic_format_load_val"


@dataclass
class DiaAtomicValRestoreOp(DeviceOp):
    """DIA-specific restore: uses diag_offsets[num_diags] + val[num_diags, BLK_K]."""

    def __init__(
        self, mem: str, operands: Sequence[Value], out_shape: Tuple[Expr, Expr]
    ):
        assert len(operands) == 3, "dia_atomic_val_restore requires [val, diag_offsets, num_diags]"
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
        return "dia_atomic_val_restore"


@dataclass
class McoAtomicFormatLoadMaskOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value], out_shape: Expr):
        assert isinstance(operands[0].type, ArrayType) # mask_array
        
        super().__init__(
            mem=mem,
            operands=operands,
            results=[
                OpResult(
                    type=ArrayType([out_shape], operands[0].type.datatype),
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
        assert isinstance(operands[1].type, ArrayType) # mask vector
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

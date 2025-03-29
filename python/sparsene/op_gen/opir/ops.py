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


@dataclass
class CooAtomicFormatLoadIdxOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value], out_shape: Expr):
        assert isinstance(operands[0].type, ArrayType) # idx_array
        assert isinstance(operands[0].type.datatype, IntType) # idx_array
        assert isinstance(operands[1].type, IntType) # ll
        assert isinstance(operands[2].type, IntType) # rr
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


@dataclass
class CooAtomicFormatLoadValOp(DeviceOp):
    def __init__(self, mem: str, operands: Sequence[Value], out_shape: Expr):
        assert isinstance(operands[0].type, ArrayType) # val_array
        assert isinstance(operands[1].type, IntType) # ll
        assert isinstance(operands[2].type, IntType) # rr
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

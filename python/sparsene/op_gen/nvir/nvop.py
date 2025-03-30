from __future__ import annotations

from typing import Literal, List, Dict, Union, Tuple, Any, Optional
from sparsene.op_gen.opir.op_ir import Op
from sparsene.op_gen.nvir.op_impl import NvOpImpl
from itertools import chain
from dataclasses import dataclass
from enum import Enum

IoMemType = Literal["gmem", "smem", "rmem"]
OpMemType = Union[
    Tuple[Literal["r"], Literal["s"]],
    Tuple[Literal["s"], Literal["r"]],
    Tuple[Literal["g"], Literal["s"]],
    Tuple[Literal["s"], Literal["g"]],
    Tuple[Literal["g"], Literal["r"]],
    Tuple[Literal["r"], Literal["g"]],
    Tuple[Literal["r"], Literal["r"]],
    Tuple[Literal["s"], Literal["s"]],
    Tuple[Literal["g"], Literal["g"]],
]


class ShapeEle:
    pass


class IntShape(ShapeEle):
    value: int

    def __init__(self, value: int):
        self.value = value


class ParamShape(ShapeEle):
    param: str

    def __init__(self, param: str):
        self.param = param


class MnkShape(ShapeEle):
    param: Literal["BLK_MNK", "MMA_MNK", "BLK_MMA_MNK"]
    mnk: Literal["m", "n", "k"]

    def __init__(
        self,
        param: Literal["BLK_MNK", "MMA_MNK", "BLK_MMA_MNK"],
        mnk: Literal["m", "n", "k"],
    ):
        self.param = param
        self.mnk = mnk


class VarlenShape(ShapeEle):
    name: str

    def __init__(self, name: str):
        self.name = name


class Shape(ShapeEle):
    elems: List[ShapeEle]

    def __init__(self, *elems: ShapeEle):
        self.elems = list(elems)

    def __iter__(self):
        return iter(self.elems)

    def __getitem__(self, idx: int):
        return self.elems[idx]

    def __len__(self):
        return len(self.elems)


class NvOpInput:
    idx: int
    shape: Shape
    mem: IoMemType
    name: str
    dtype: str
    source: Tuple[NvOp, int] | str

    def __init__(
        self,
        idx: int,
        shape: Shape,
        mem: IoMemType,
        name: str,
        dtype: str,
        source: Tuple[NvOp, int] | str,
    ):
        self.idx = idx
        self.shape = shape
        self.mem = mem
        self.name = name
        self.dtype = dtype
        self.source = source


class NvOpOutput:
    idx: int
    shape: Shape
    mem: IoMemType
    name: str
    dtype: str
    owning: bool
    source: Optional[Tuple[NvOp, int] | str]
    unique: bool # unique的output的nbuf维度永远为1
    attrs: Dict[str, Any]

    def __init__(
        self,
        idx: int,
        shape: Shape,
        mem: IoMemType,
        name: str,
        dtype: str,
        *,
        owning: bool = True,
        source: Optional[Tuple[NvOp, int] | str] = None,
        unique: bool = False,
        **kwargs: Any,
    ):
        self.idx = idx
        self.shape = shape
        self.mem = mem
        self.name = name
        self.dtype = dtype
        self.attrs = kwargs
        self.owning = owning
        self.source = source
        self.unique = unique
class GmemInout:
    shape: Shape
    name: str
    dtype: str

    def __init__(self, shape: Shape, name: str, dtype: str):
        self.shape = shape
        self.name = name
        self.dtype = dtype


@dataclass
class LayoutHint:
    # Layout hints是用来指导上游inputs的layout的
    # Layout hints 的传播规则：
    #  有hints的出发，先传播，每个op只被影响一次
    #  最终没有被hints影响的op，按照默认的copy layout计算方式得到layout
    #  layout hint会影响被传播op的ouput_shape和body实现，也会影响传播发起者op的input_shape
    P: int
    Tr: int
    Vr: int
    Tc: int
    Vc: int
    Prow: bool
    Trow: bool
    Vrow: bool

    layout: Tuple


class NvOpType(Enum):
    # 1d-copy: 稠密1D通用线性内存拷贝
    # 2d-copy: 稠密2D内存拷贝，用于矩阵
    # 2d-copy-sp: 稀疏2D内存拷贝，用于矩阵，允许行或列（目前暂时只支持行）离散
    # others: 其他
    COPY_1D = "copy_1d"
    COPY_2D = "copy_2d"
    COPY_2D_SP = "copy_2d_sp"
    LOAD_LR_OFF = "load_lr_off"  # 从一个offset数组中读取l,r
    LOAD_LR_PAIR = "load_lr_pair"  # 从一个同时存了l和r的offset数组中读取l,r
    OTHERS = "others"


class NvOp:
    name: str
    inputs: List[NvOpInput]
    outputs: List[NvOpOutput]
    backend_op: Op
    impl: NvOpImpl

    op_type: NvOpType
    mem_type: OpMemType

    parameters: Dict[str, int]

    attrs: Dict[str, Any]

    def __init__(
        self,
        name: str,
        inputs: List[NvOpInput],
        outputs: List[NvOpOutput],
        impl: NvOpImpl,
        mem_type: OpMemType,
        # backend_op: Op,
        parameters: Dict[str, int],
        **kwargs: Any,
    ):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.impl = impl
        self.mem_type = mem_type
        # self.backend_op = backend_op
        self.parameters = parameters
        self.attrs = kwargs

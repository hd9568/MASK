from __future__ import annotations

from typing import Literal, List, Dict, Union, Tuple, Any, Optional
from sparsene.op_gen.opir.op_ir import Op
from sparsene.op_gen.nvir.op_impl import NvOpImpl
from itertools import chain
from dataclasses import dataclass

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
    modify: bool

    def __init__(
        self,
        idx: int,
        shape: Shape,
        mem: IoMemType,
        name: str,
        dtype: str,
        source: Tuple[NvOp, int] | str,
        modify: bool = False,
    ):
        self.idx = idx
        self.shape = shape
        self.mem = mem
        self.name = name
        self.dtype = dtype
        self.source = source
        self.modify = modify


class NvOpOutput:
    idx: int
    shape: Shape
    mem: IoMemType
    name: str
    dtype: str
    attrs: Dict[str, Any]

    def __init__(
        self,
        idx: int,
        shape: Shape,
        mem: IoMemType,
        name: str,
        dtype: str,
        **kwargs: Any,
    ):
        self.idx = idx
        self.shape = shape
        self.mem = mem
        self.name = name
        self.dtype = dtype
        self.attrs = kwargs


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
    layout: Tuple


class NvOp:
    name: str
    op_type: str
    # op_type分类
    #  1d-copy: 稠密1D通用线性内存拷贝
    #  2d-copy: 稠密2D内存拷贝，用于矩阵
    #  2d-copy-sp: 稀疏2D内存拷贝，用于矩阵，允许行或列（目前暂时只支持行）离散
    #  others: 其他
    inputs: List[NvOpInput]
    outputs: List[NvOpOutput]
    backend_op: Op
    impl: NvOpImpl

    layout_hints: Dict[int, LayoutHint]  # Layout hints是用来指导上游inputs的layout的
    # Layout hints 的传播规则：
    #  有hints的出发，先传播，每个op只被影响一次
    #  最终没有被hints影响的op，按照默认的copy layout计算方式得到layout
    #  layout hint会影响被传播op的ouput_shape和body实现，也会影响传播发起者op的input_shape
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

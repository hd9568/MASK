from __future__ import annotations

from typing import Literal, List, Dict, Union, Tuple, Any, Optional
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


class NvOpImpl:
    code_template: str

    def __init__(self, code_template: str):
        self.code_template = code_template

    def nv_src(self) -> str:
        return self.code_template


class Copy1DNvOpImpl(NvOpImpl):
    layout_hint: LayoutHint

    def __init__(self, layout_hint: LayoutHint):
        self.layout_hint = layout_hint

    def nv_src(self) -> str:
        return ""


class ShapeEle:
    pass


class IntShape(ShapeEle):
    value: int

    def __init__(self, value: int):
        self.value = value

    def __repr__(self) -> str:
        return f"Int({self.value})"

    def __str__(self) -> str:
        return f"Int({self.value})"


class ParamShape(ShapeEle):
    param: str

    def __init__(self, param: str):
        self.param = param

    def __repr__(self) -> str:
        return f"Param({self.param})"

    def __str__(self) -> str:
        return f"Param({self.param})"


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

    def __str__(self) -> str:
        return f"Mnk({self.param}, {self.mnk})"

    def __repr__(self) -> str:
        return f"Mnk({self.param}, {self.mnk})"


class VarlenShape(ShapeEle):
    name: str

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return f"Varlen({self.name})"

    def __repr__(self) -> str:
        return f"Varlen({self.name})"


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

    def __str__(self) -> str:
        return "(" + ", ".join(str(elem) for elem in self.elems) + ")"

    def __repr__(self) -> str:
        return "(" + ", ".join(repr(elem) for elem in self.elems) + ")"


class NvOpInput:
    idx: int
    shape: Shape
    mem: IoMemType
    name: str
    dtype: str
    source: Tuple[NvOp, int] | str
    layout_hint: Optional[LayoutHint]

    def __init__(
        self,
        idx: int,
        shape: Shape,
        mem: IoMemType,
        name: str,
        dtype: str,
        source: Tuple[NvOp, int] | str,
        layout_hint: Optional[LayoutHint] = None,
    ):
        self.idx = idx
        self.shape = shape
        self.mem = mem
        self.name = name
        self.dtype = dtype
        self.source = source
        self.layout_hint = layout_hint

    def to_json(self):
        return {
            "idx": self.idx,
            "name": self.name,
            "shape": self.shape.__dict__,
            "mem": self.mem,
            "dtype": self.dtype,
            "source": (
                f'"{self.source}"'
                if isinstance(self.source, str)
                else f"{self.source[0].name}.output({self.source[1]})"
            ),
            "layout_hint": self.layout_hint.__dict__,
        }


class NvOpOutput:
    idx: int
    shape: Shape
    mem: IoMemType
    name: str
    dtype: str
    owning: bool
    source: Optional[Tuple[NvOp, int] | str]
    unique: bool  # unique的output的nbuf维度永远为1
    attrs: Dict[str, Any]
    consumers: List[NvOp]

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
        layout_hint: Optional[LayoutHint] = None,
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
        self.consumers = []
        self.layout_hint = layout_hint

    def to_json(self):
        output_dict = {
            "idx": self.idx,
            "name": self.name,
            "shape": self.shape.__dict__,
            "mem": self.mem,
            "dtype": self.dtype,
            "owning": self.owning,
        }
        if self.source is not None:
            output_dict["source"] = (
                f'"{self.source}"'
                if isinstance(self.source, str)
                else f"{self.source[0].name}.output({self.source[1]})"
            )
        if self.unique:
            output_dict["unique"] = self.unique
        if self.attrs:
            output_dict["attrs"] = self.attrs
        return output_dict


class GmemInout:
    shape: Shape
    name: str
    dtype: str

    def __init__(self, shape: Shape, name: str, dtype: str):
        self.shape = shape
        self.name = name
        self.dtype = dtype

    def __str__(self) -> str:
        return f"gmem({self.name}): {self.dtype}({self.shape})"

    def __repr__(self) -> str:
        return f"gmem({self.name}): {self.dtype}({self.shape})"


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
    P_major: Literal["row", "col"]
    T_major: Literal["row", "col"]
    V_major: Literal["row", "col"]


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
        parameters: Dict[str, int],
        **kwargs: Any,
    ):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.impl = impl
        self.mem_type = mem_type
        self.parameters = parameters
        self.attrs = kwargs

        for inp in self.inputs:
            if isinstance(inp.source, tuple):
                inp.source[0].outputs[inp.source[1]].consumers.append(self)

    def to_json(self):
        return {
            "name": self.name,
            "inputs": [inp.to_json() for inp in self.inputs],
            "outputs": [out.to_json() for out in self.outputs],
            "mem_type": self.mem_type,
            "parameters": self.parameters,
            "attrs": self.attrs,
        }


class ForLoopOp(NvOp):
    name: str
    iter_args: Dict[str, NvOpInput]
    loop_l: NvOpInput
    loop_r: NvOpInput
    body: NvOpSequence

    def __init__(
        self,
        name: str,
        iter_args: Dict[str, NvOpInput],
        loop_l: NvOpInput,
        loop_r: NvOpInput,
        loop_result: NvOpOutput,
        **kwargs: Any,
    ):
        super().__init__(name, [], [], **kwargs)
        self.iter_args = iter_args
        self.loop_l = loop_l
        self.loop_r = loop_r
        self.outputs = [loop_result]


class NvOpStage:
    ops: List[NvOp]

    def __init__(self, ops: List[NvOp]):
        self.ops = ops


class NvOpSequence:
    name: str
    stages: List[NvOpStage]
    shifts: List[int]
    prologue: List[NvOp]
    epilogue: List[NvOp]
    gmem_inouts: Dict[str, GmemInout]

    def __init__(
        self,
        name: str,
        stages: List[NvOpStage],
        shifts: List[int],
        prologue: List[NvOp],
        epilogue: List[NvOp],
        gmem_inouts: Dict[str, GmemInout],
    ):
        self.name = name
        self.stages = stages
        self.shifts = shifts
        self.prologue = prologue
        self.epilogue = epilogue
        self.gmem_inouts = {k: gmem_inouts[k] for k in sorted(gmem_inouts.keys())}

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def num_buf(self) -> int:
        return max(self.shifts) + 1

    @property
    def nbuf(self) -> int:
        return self.num_buf

    def all_parameters(self) -> List[str]:
        parameters = []
        for stage in self.stages:
            for op in stage.ops:
                parameters.extend(op.parameters)
        return sorted(dict.fromkeys(parameters))  # fromkeys trick to dedup

    def all_varlens(self) -> List[str]:
        varlens = []
        for stage in self.stages:
            for op in stage.ops:
                for inp in op.inputs:
                    for shape in inp.shape:
                        if isinstance(shape, VarlenShape):
                            varlens.append(shape.name)
                for out in op.outputs:
                    for shape in out.shape:
                        if isinstance(shape, VarlenShape):
                            varlens.append(shape.name)

        for io in self.gmem_inouts.values():
            for shape in io.shape:
                if isinstance(shape, VarlenShape):
                    varlens.append(shape.name)
        return sorted(dict.fromkeys(varlens))  # fromkeys trick to dedup

    def find_op_by_name(self, name: str) -> NvOp:
        for op in self.prologue:
            if op.name == name:
                return op
        for stage in self.stages:
            for op in stage.ops:
                if op.name == name:
                    return op
        for op in self.epilogue:
            if op.name == name:
                return op
        raise ValueError(f"NvOp {name} not found")

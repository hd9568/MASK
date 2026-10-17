from __future__ import annotations

from typing import Literal, List, Dict, Union, Tuple, Any, Optional, Sequence, Set
from dataclasses import dataclass
from enum import Enum
from itertools import chain
import uuid
from sparsene.logging import get_logger

logger = get_logger(__name__)

OpMemType = Union[
    Tuple[Literal["r"], Literal["s"]],  # rmem to smem
    Tuple[Literal["s"], Literal["r"]],  # smem to rmem
    Tuple[Literal["g"], Literal["s"]],  # gmem to smem
    Tuple[Literal["s"], Literal["g"]],  # smem to gmem
    Tuple[Literal["g"], Literal["r"]],  # gmem to rmem
    Tuple[Literal["r"], Literal["g"]],  # rmem to gmem
    Tuple[Literal["r"], Literal["r"]],  # rmem to rmem
    Tuple[Literal["s"], Literal["s"]],  # smem to smem
    Tuple[Literal["g"], Literal["g"]],  # gmem to gmem
    Tuple[Literal["x"], Literal["g"]],  # gmem without source
    Tuple[Literal["x"], Literal["s"]],  # smem without source
    Tuple[Literal["x"], Literal["r"]],  # rmem without source
    Tuple[Literal["x"], Literal["x"]],  # all unknown
]
IoMemType = Literal["gmem", "smem", "rmem", "xmem"]  # xmem means unspecified


@dataclass
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


@dataclass
class ShapeEle:
    pass


@dataclass
class IntShape(ShapeEle):
    value: int

    def __init__(self, value: int):
        self.value = value

    def __repr__(self) -> str:
        return f"Int({self.value})"

    def __str__(self) -> str:
        return f"Int({self.value})"


@dataclass
class ParamShape(ShapeEle):
    param: str

    def __init__(self, param: str):
        self.param = param

    def __repr__(self) -> str:
        return f"Param({self.param})"

    def __str__(self) -> str:
        return f"Param({self.param})"


@dataclass
class MnkShape(ShapeEle):
    param: Literal["BLK_MNK", "MMA_MNK", "BLK_MMA_MNK", "WARP_MNK"]
    mnk: Literal["m", "n", "k"]

    def __init__(
        self,
        param: Literal["BLK_MNK", "MMA_MNK", "BLK_MMA_MNK", "WARP_MNK"],
        mnk: Literal["m", "n", "k"],
    ):
        self.param = param
        self.mnk = mnk

    def __str__(self) -> str:
        return f"Mnk({self.param}, {self.mnk})"

    def __repr__(self) -> str:
        return f"Mnk({self.param}, {self.mnk})"


@dataclass
class VarlenShape(ShapeEle):
    name: str

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return f"Varlen({self.name})"

    def __repr__(self) -> str:
        return f"Varlen({self.name})"


@dataclass
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

    def all_parameters(self) -> Set[str]:
        params = set()
        for e in self.elems:
            if isinstance(e, ParamShape):
                params.add(e.param)
            elif isinstance(e, Shape):
                params.update(e.all_parameters())
        return params


@dataclass
class SwizzleLayout:
    b: int
    m: int
    s: int
    adaptive_blk_k: bool = False


@dataclass
class NvOpTensor:
    shape: Shape
    mem: IoMemType
    dtype: str
    source: Optional[
        NvOpOutput
        | ForLoopStart
        | ForLoopEnd
        | ForLoopCurrentIdx
        | ForLoopCurrentIter
        | str
    ]
    row_major: bool
    swizzle: Optional[SwizzleLayout]

    def __init__(
        self,
        shape: Shape,
        mem: IoMemType,
        dtype: str,
        *,
        source: Optional[
            NvOpOutput
            | ForLoopStart
            | ForLoopEnd
            | ForLoopCurrentIdx
            | ForLoopCurrentIter
            | str
        ] = None,
        row_major: bool = False,
        swizzle: Optional[SwizzleLayout] = None,
    ):
        self.shape = shape
        self.mem = mem
        self.dtype = dtype
        self.source = source
        self.row_major = row_major
        self.swizzle = swizzle


class NvOpInput:
    idx: int
    name: str
    tensor: NvOpTensor
    layout_hint: Optional[LayoutHint]

    def __init__(
        self,
        idx: int,
        name: str,
        tensor: NvOpTensor,
        layout_hint: Optional[LayoutHint] = None,
    ):
        self.idx = idx
        self.name = name
        self.tensor = tensor
        self.layout_hint = layout_hint


@dataclass
class ForLoopStart:
    for_loop: ForLoopNvOp


@dataclass
class ForLoopEnd:
    for_loop: ForLoopNvOp


@dataclass
class ForLoopCurrentIdx:
    for_loop: ForLoopNvOp


@dataclass
class ForLoopCurrentIter:
    for_loop: ForLoopNvOp


class NvOpOutput:
    idx: int
    name: str
    tensor: NvOpTensor
    owning: bool
    origin: Optional[NvOpOutput]
    unique: bool  # unique的output的nbuf维度永远为1
    attrs: Dict[str, Any]
    consumers: List[NvOp]
    op: Optional[NvOp]

    def __init__(
        self,
        idx: int,
        name: str,
        tensor: NvOpTensor,
        *,
        origin: Optional[NvOpOutput] = None,
        unique: Optional[bool] = None,
        layout_hint: Optional[LayoutHint] = None,
        **kwargs: Any,
    ):
        self.idx = idx
        self.tensor = tensor
        self.name = name
        self.attrs = kwargs
        self.origin = origin

        if self.origin is None:
            self.owning = True
            self.unique = unique is True
        else:
            self.owning = False
            self.unique = self.origin.unique
            if unique is not None:
                logger.warning(
                    f"NvOpOutput {self.name} unique={unique} ignored because origin {self.origin.name} already has unique={self.origin.unique}."
                )

        self.consumers = []
        self.op = None
        self.layout_hint = layout_hint

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "origin":
            self.owning = False
        super().__setattr__(name, value)


class GmemInout:
    shape: Shape
    name: str
    dtype: str
    tensor_str: str
    tensor_type_str: str
    parameters: Dict[str, int]

    def __init__(
        self,
        shape: Shape,
        name: str,
        dtype: str,
        tensor_str: str,
        tensor_type_str: str,
        parameters: Dict[str, int] = {},
    ):
        self.shape = shape
        self.name = name
        self.dtype = dtype
        self.tensor_str = tensor_str
        self.tensor_type_str = tensor_type_str
        self.parameters = parameters

    def __str__(self) -> str:
        return f"gmem({self.name}): {self.dtype}({self.shape})"

    def __repr__(self) -> str:
        return f"gmem({self.name}): {self.dtype}({self.shape})"


@dataclass
class LayoutHint:
    Pr: int
    Pc: int
    Tr: int
    Vr: int
    Tc: int
    Vc: int
    Pmajor: Literal["row", "col"]
    Tmajor: Literal["row", "col"]
    Vmajor: Literal["row", "col"]


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
    UNKNOWN = "unknown"


class NvOp:
    name: str
    inputs: List[NvOpInput]
    outputs: List[NvOpOutput]
    impl: NvOpImpl
    pipelined: bool

    op_type: NvOpType
    mem_type: OpMemType

    parameters: Dict[str, int]

    attrs: Dict[str, Any]

    def __init__(
        self,
        name: str,
        inputs: Sequence[NvOpInput],
        outputs: Sequence[NvOpOutput],
        impl: NvOpImpl,
        mem_type: OpMemType = ("x", "x"),
        parameters: Dict[str, int] = {},
        op_type: NvOpType = NvOpType.UNKNOWN,
        **kwargs: Any,
    ):
        self.name = name
        self.impl = impl
        self.op_type = op_type
        self.mem_type = mem_type
        self.parameters = parameters
        self.attrs = kwargs
        self.pipelined = False

        self.inputs = []
        self.outputs = []
        for inp in inputs:
            self.add_input(inp)
        for out in outputs:
            self.add_output(out)

    def add_input(self, input: NvOpInput):
        self.inputs.append(input)
        if isinstance(input.tensor.source, NvOpOutput):
            input.tensor.source.consumers.append(self)

    def add_output(self, output: NvOpOutput):
        self.outputs.append(output)
        output.op = self
        output.tensor.source = output

    def all_varlens(self) -> Set[str]:
        varlens = set()
        for inp in self.inputs:
            for shape in inp.tensor.shape:
                if isinstance(shape, VarlenShape):
                    varlens.add(shape.name)
        for out in self.outputs:
            for shape in out.tensor.shape:
                if isinstance(shape, VarlenShape):
                    varlens.add(shape.name)
        return varlens

    def all_parameters(self) -> Set[str]:
        # External symbol params (value == key, e.g. "K": "K") are passed
        # as constructor arguments, NOT template parameters.  Only template
        # params (value != key, e.g. "nbuf": 1) become template classes.
        return {k for k, v in self.parameters.items() if v != k}


class ConstantNvOp(NvOp):
    value: int | float | str

    def __init__(
        self,
        name: str,
        shape: Shape,
        dtype: str,
        value: int | float | str,
        **kwargs: Any,
    ):
        super().__init__(
            name=name or f"constant_{dtype}_{value}_{str(uuid.uuid4())[:8]}",
            inputs=[],
            outputs=[
                NvOpOutput(
                    idx=0,
                    name=f"{name}_out",
                    tensor=NvOpTensor(shape=shape, mem="rmem", dtype=dtype),
                    owning=True,
                    unique=True,
                )
            ],
            impl=NvOpImpl(code_template=""),
            op_type=NvOpType.UNKNOWN,
            parameters=kwargs.get('parameters', {}), 
        )
        self.value = value


class GmemTensorNvOp(NvOp):
    gmem_inout: GmemInout

    def __init__(self, gmem_inout: GmemInout, **kwargs: Any):
        super().__init__(
            name=gmem_inout.name + "Op",
            inputs=[],
            outputs=[
                NvOpOutput(
                    idx=0,
                    name=gmem_inout.name,
                    tensor=NvOpTensor(
                        shape=gmem_inout.shape,
                        mem="gmem",
                        dtype=gmem_inout.dtype,
                    ),
                    owning=True,
                    unique=True,
                ),
            ],
            impl=NvOpImpl(code_template=""),
            op_type=NvOpType.UNKNOWN,
            mem_type=("x", "g"),
            parameters=gmem_inout.parameters,
        )
        self.gmem_inout = gmem_inout
        for out in self.outputs:
            out.tensor.source = out


class ForLoopNvOp(NvOp):
    name: str
    iter_args: Dict[str, NvOpInput]
    loop_l: NvOpInput
    loop_r: NvOpInput
    body: NvOpSequence
    blk_idx_mapping: Optional[Literal["x", "y", "z"]]

    def __init__(
        self,
        name: str,
        loop_l: NvOpInput,
        loop_r: NvOpInput,
        loop_result: Optional[Sequence[NvOpOutput]] = None,
        iter_args: Optional[Dict[str, NvOpInput]] = None,
        body: Optional[NvOpSequence] = None,
        blk_idx_mapping: Optional[Literal["x", "y", "z"]] = None,
        **kwargs: Any,
    ):
        if iter_args is None:
            iter_args = {}
        if body is None:
            body = NvOpSequence()
        if loop_result is None:
            loop_result = []
        super().__init__(
            name=name,
            inputs=[loop_l, loop_r] + list(iter_args.values()),
            outputs=list(loop_result),
            impl=NvOpImpl(code_template=""),
            **kwargs,
        )
        self.iter_args = iter_args
        self.loop_l = loop_l
        self.loop_r = loop_r
        self.body = body
        self.blk_idx_mapping = blk_idx_mapping

    def append(self, op: NvOp):
        self.body.append(op)

    def extend(self, ops: Sequence[NvOp]):
        self.body.extend(ops)

    def find_op_by_name(self, name: str) -> NvOp:
        return self.body.find_op_by_name(name)

    def all_varlens(self) -> Set[str]:
        return self.body.all_varlens()

    def all_parameters(self) -> Set[str]:
        return self.body.all_parameters()

    def set_loop_result(self, idx: int, result: NvOpOutput):
        self.outputs[idx].origin = result


class MetaNvOp(NvOp):
    name: str
    body: NvOpSequence

    def __init__(self, name: str, body: NvOpSequence, **kwargs: Any):
        super().__init__(name, [], [], impl=NvOpImpl(""), **kwargs)
        self.body = body

    def append(self, op: NvOp):
        self.body.append(op)

    def extend(self, ops: Sequence[NvOp]):
        self.body.extend(ops)

    def find_op_by_name(self, name: str) -> NvOp:
        return self.body.find_op_by_name(name)

    def all_varlens(self) -> Set[str]:
        return self.body.all_varlens()

    def all_parameters(self) -> Set[str]:
        return self.body.all_parameters()


class NvOpSequence:
    name: str
    ops: List[NvOp]

    def __init__(self, *ops: NvOp, name: str = ""):
        self.name = name
        self.ops = list(ops)

    def append(self, op: NvOp):
        self.ops.append(op)

    def extend(self, ops: Sequence[NvOp]):
        self.ops.extend(ops)

    def all_varlens(self) -> Set[str]:
        varlens = set()
        for op in self.ops:
            varlens.update(op.all_varlens())
        return varlens

    def all_parameters(self) -> Set[str]:
        parameters = set()
        for op in self.ops:
            parameters.update(op.all_parameters())
        return parameters

    def find_op_by_name(self, name: str) -> NvOp:
        for op in self.ops:
            if isinstance(op, ForLoopNvOp):
                try:
                    return op.find_op_by_name(name)
                except ValueError:
                    pass
            elif op.name == name:
                return op
        raise ValueError(f"NvOp {name} not found")


class NvOpPipeline(NvOpSequence):
    stages: List[NvOpSequence]
    shifts: List[int]

    def __init__(
        self,
        name: str,
        stages: Sequence[NvOpSequence],
        shifts: Sequence[int],
    ):
        super().__init__(
            *chain(*(stage.ops for stage in stages)),
            name=name,
        )
        self.stages = list(stages)
        self.shifts = list(shifts)

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def max_shift(self) -> int:
        return max(self.shifts)

    @property
    def nbuf(self) -> int:
        return self.max_shift + 1

    def all_parameters(self) -> Set[str]:
        parameters = set()
        for stage in self.stages:
            parameters.update(stage.all_parameters())
        return parameters

    def all_varlens(self) -> Set[str]:
        varlens = set()
        for stage in self.stages:
            varlens.update(stage.all_varlens())
        return varlens

    def find_op_by_name(self, name: str) -> NvOp:
        for stage in self.stages:
            try:
                op = stage.find_op_by_name(name)
            except ValueError:
                pass
            else:
                return op
        raise ValueError(f"NvOp {name} not found")


class NvOpProgram:
    name: str
    meta_op: MetaNvOp
    gmem_inouts: Dict[str, GmemInout]
    gmem_tensor_ops: Dict[str, GmemTensorNvOp]
    attrs: Dict[str, Any]

    def __init__(
        self,
        name: str,
        gmem_inouts: Dict[str, GmemInout],
    ):
        self.name = name
        self.attrs = {}
        self.meta_op = MetaNvOp(
            name=name,
            body=NvOpSequence(),
        )
        self.gmem_inouts = {k: gmem_inouts[k] for k in sorted(gmem_inouts.keys())}

        self.gmem_tensor_ops = {}
        for gmem_inout in self.gmem_inouts.values():
            gmem_tensor_op = GmemTensorNvOp(gmem_inout)
            self.append_op(gmem_tensor_op)
            self.gmem_tensor_ops[gmem_inout.name] = gmem_tensor_op

    @property
    def ops(self) -> List[NvOp]:
        return self.meta_op.body.ops

    def append_op(self, op: NvOp):
        self.meta_op.append(op)

    def extend_ops(self, ops: Sequence[NvOp]):
        self.meta_op.extend(ops)

    def all_parameters(self) -> Set[str]:
        return self.meta_op.all_parameters()

    def all_varlens(self) -> Set[str]:
        varlens = set()

        varlens.update(self.meta_op.all_varlens())

        for io in self.gmem_inouts.values():
            for shape in io.shape:
                if isinstance(shape, VarlenShape):
                    varlens.add(shape.name)
        return varlens

    def find_op_by_name(self, name: str) -> NvOp:
        return self.meta_op.find_op_by_name(name)

    def add_gmem_inout(self, gmem_inout: GmemInout):
        self.gmem_inouts[gmem_inout.name] = gmem_inout
        self.gmem_tensor_ops[gmem_inout.name] = GmemTensorNvOp(gmem_inout)
        self.append_op(self.gmem_tensor_ops[gmem_inout.name])

    def dump_tree(
        self,
        *,
        indent_size: int = 2,
        show_io: bool = True,
        show_impl: bool = False,
        impl_max_lines: int = 8,
        debug_indent: bool = False,
    ) -> str:
        from sparsene.op_gen.nvir.printer import NvProgramPrinter

        printer = NvProgramPrinter(
            indent_size=indent_size,
            show_io=show_io,
            show_impl=show_impl,
            impl_max_lines=impl_max_lines,
            debug_indent=debug_indent,
        )
        return printer.dump_program(self)

    def print_tree(
        self,
        *,
        indent_size: int = 2,
        show_io: bool = True,
        show_impl: bool = False,
        impl_max_lines: int = 8,
        debug_indent: bool = False,
    ) -> None:
        print(
            self.dump_tree(
                indent_size=indent_size,
                show_io=show_io,
                show_impl=show_impl,
                impl_max_lines=impl_max_lines,
                debug_indent=debug_indent,
            )
        )

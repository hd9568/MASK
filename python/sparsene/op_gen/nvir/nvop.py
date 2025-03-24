from typing import Literal, List, Dict
from sparsene.op_gen.opir.op_ir import Op
from sparsene.op_gen.nvir.op_impl import NvOpImpl
from itertools import chain

IoMemType = Literal["gmem", "smem", "rmem"]
OpMemType = Literal[
    ("r", "s"),
    ("s", "r"),
    ("g", "s"),
    ("s", "g"),
    ("g", "r"),
    ("r", "g"),
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

    def __init__(self, param: Literal["BLK_MNK", "MMA_MNK", "BLK_MMA_MNK"], mnk: Literal["m", "n", "k"]):
        self.param = param
        self.mnk = mnk

class VarlenShape(ShapeEle):
    name: str

    def __init__(self, name: str):
        self.name = name

class Shape(ShapeEle):
    elems: List[ShapeEle]

    def __init__(self, *elems: ShapeEle):
        self.elems = elems

    def __iter__(self):
        return iter(self.elems)
    
    def __getitem__(self, idx: int):
        return self.elems[idx]
    
    def __len__(self):
        return len(self.elems)
    
    

class NvOpInput:
    idx: int
    shape: List[int | str]
    mem: IoMemType
    name: str
    dtype: str

    def __init__(
        self, idx: int, shape: List[int | str], mem: IoMemType, name: str, dtype: str
    ):
        self.idx = idx
        self.shape = shape
        self.mem = mem
        self.name = name
        self.dtype = dtype


class NvOpOutput:
    idx: int
    shape: Shape
    mem: IoMemType
    name: str
    dtype: str

    def __init__(
        self, idx: int, shape: List[int | str], mem: IoMemType, name: str, dtype: str
    ):
        self.idx = idx
        self.shape = shape
        self.mem = mem
        self.name = name
        self.dtype = dtype


class NvOp:
    name: str

    inputs: List[NvOpInput]
    outputs: List[NvOpOutput]
    backend_op: Op
    impl: NvOpImpl

    mem_type: OpMemType

    parameters: Dict[str, int]

    def __init__(
        self,
        name: str,
        inputs: List[NvOpInput],
        outputs: List[NvOpOutput],
        impl: NvOpImpl,
        mem_type: OpMemType,
        # backend_op: Op,
        parameters: List[str],
    ):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.impl = impl
        self.mem_type = mem_type
        # self.backend_op = backend_op
        self.parameters = parameters


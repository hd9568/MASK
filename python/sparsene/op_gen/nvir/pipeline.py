from sparsene.op_gen.nvir.nvop import (
    NvOp,
    NvOpInput,
    NvOpOutput,
    VarlenShape,
    IoMemType,
    OpMemType,
    Shape,
    GmemInout,
)
from typing import List, Dict


class NvOpStage:
    ops: List[NvOp]

    def __init__(self, ops: List[NvOp]):
        self.ops = ops


class NvOpPipeline:
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

    # def all_inputs_with_memtype(self, memtype: IoMemType) -> List[NvOpInput]:
    # 	inputs = []
    # 	for stage in self.stages:
    # 		for op in stage.ops:
    # 			for inp in op.inputs:
    # 				if inp.mem == memtype:
    # 					inputs.append(inp)
    # 	return sorted(inputs, key=lambda x: x.name)

    # def all_outputs_with_memtype(self, memtype: IoMemType) -> List[NvOpOutput]:
    # 	outputs = []
    # 	for stage in self.stages:
    # 		for op in stage.ops:
    # 			for out in op.outputs:
    # 				if out.mem == memtype:
    # 					outputs.append(out)
    # 	return sorted(outputs, key=lambda x: x.name)

    # def all_gmem_inouts(self) -> Dict[str, NvOpInput | NvOpOutput]:
    # gmem_inputs = {inp.name: inp for inp in self.all_inputs_with_memtype("gmem")}
    # gmem_outputs = {out.name: out for out in self.all_outputs_with_memtype("gmem")}
    # gmem_inouts = {
    # 	**gmem_inputs,
    # 	**gmem_outputs,
    # 	# TODO: HIGH PRIORITY: add outer op to avoid hard-coding
    # 	"val_soff": NvOpInput(
    # 		name="val_soff",
    # 		mem="gmem",
    # 		shape=Shape(VarlenShape(name="Mo")),
    # 		dtype="int",
    # 		source="val_soff",
    # 		idx=-1,
    # 	),
    # 	"C_val": NvOpOutput(
    # 		name="C_val",
    # 		mem="gmem",
    # 		shape=Shape(VarlenShape(name="N"), VarlenShape(name="M")),
    # 		dtype="float",
    # 		idx=-1,
    # 	),
    # }
    # sorted_keys = sorted(gmem_inouts.keys())
    # return {k: gmem_inouts[k] for k in sorted_keys}

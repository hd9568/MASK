from sparsene.op_gen.nvir.nvop import NvOp, NvOpInput, NvOpOutput, VarlenShape, IoMemType, OpMemType
from typing import List

class NvOpStage:
	ops: List[NvOp]

	def __init__(self, ops: List[NvOp]):
		self.ops = ops

class NvOpPipeline:
	name: str
	stages: List[NvOpStage]
	shifts: List[int]

	def __init__(self, name: str, stages: List[NvOpStage], shifts: List[int]):
		self.name = name
		self.stages = stages
		self.shifts = shifts

	def num_stages(self) -> int:
		return len(self.stages)

	def num_buf(self) -> int:
		return max(self.shifts) + 1
	
	def all_parameters(self) -> List[str]:
		parameters = []
		for stage in self.stages:
			for op in stage.ops:
				parameters.extend(op.parameters)
		return list(dict.fromkeys(parameters)) # fromkeys trick to dedup
	
	def all_varlens(self) -> List[str]:
		varlens = []
		for stage in self.stages:
			for op in stage.ops:
				for inp in op.inputs:
					if isinstance(inp.shape, VarlenShape):
						varlens.append(inp.shape.name)
				for out in op.outputs:
					if isinstance(out.shape, VarlenShape):
						varlens.append(out.shape.name)
		return list(dict.fromkeys(varlens)) # fromkeys trick to dedup
	
	def all_inputs_with_memtype(self, memtype: IoMemType) -> List[NvOpInput]:
		inputs = []
		for stage in self.stages:
			for op in stage.ops:
				for inp in op.inputs:
					if inp.mem == memtype:
						inputs.append(inp)
		return inputs
	
	def all_outputs_with_memtype(self, memtype: IoMemType) -> List[NvOpOutput]:
		outputs = []
		for stage in self.stages:
			for op in stage.ops:
				for out in op.outputs:
					if out.mem == memtype:
						outputs.append(out)
		return outputs
	
	def all_gmem_inouts(self) -> List[str]:
		inp_names = [inp.name for inp in self.all_inputs_with_memtype("gmem")]
		out_names = [out.name for out in self.all_outputs_with_memtype("gmem")]
		return list(dict.fromkeys(inp_names + out_names)) # fromkeys trick to dedup
	
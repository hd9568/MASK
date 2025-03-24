from sparsene.op_gen.nvir.nvop import NvOp
from typing import List

class NvOpStage:
	ops: List[NvOp]

	def __init__(self, ops: List[NvOp]):
		self.ops = ops

class NvOpPipeline:
	stages: List[NvOpStage]
	shifts: List[int]

	def __init__(self, stages: List[NvOpStage], shifts: List[int]):
		self.stages = stages
		self.shifts = shifts

	def num_stages(self) -> int:
		return len(self.stages)

	def num_buf(self) -> int:
		return max(self.shifts) + 1
	
	def gen_fill(self, fill_stage: int) -> List[NvOp]:
		pass

	def gen_empty(self, empty_stage: int) -> List[NvOp]:
		pass

	def gen_small_kernel(self, num_iters: int) -> List[NvOp]:
		pass

	def gen_loop_body(self, induction_var: str) -> List[NvOp]:
		pass

	def gen_pipeline(self) -> List[NvOp]:
		pass

	def gen_kernel(self):
		kernels = {}
		
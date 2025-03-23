from sparsene.op_gen.opir.op_ir import Op
from typing import List

class OpStage:
	ops: List[Op]

	def __init__(self, ops: List[Op]):
		self.ops = ops

class OpPipeline:
	stages: List[OpStage]
	shifts: List[int]

	def __init__(self, stages: List[OpStage], shifts: List[int]):
		self.stages = stages
		self.shifts = shifts

	def num_stages(self) -> int:
		return len(self.stages)

	def num_buf(self) -> int:
		return max(self.shifts) + 1
	
	def gen_fill(self, fill_stage: int) -> List[Op]:
		pass

	def gen_empty(self, empty_stage: int) -> List[Op]:
		pass

	def gen_small_kernel(self, num_iters: int) -> List[Op]:
		pass

	def gen_loop_body(self, induction_var: str) -> List[Op]:
		pass

	def gen_pipeline(self) -> List[Op]:
		pass

	def gen_kernel(self):
		kernels = {}
		
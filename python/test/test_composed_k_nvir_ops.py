import unittest
from sparsene.op_gen.nvir.composed_k_ops import build_composed_k_stage, schedule_composed_k_stage, emit_scheduled_composed_k_stage

class ComposedStageTest(unittest.TestCase):
 def test_linked_dependency_and_padding_guard(self):
  bounds,a,b,sync=build_composed_k_stage()
  self.assertIs(a.inputs[0].tensor.source,bounds.outputs[0]);self.assertIs(b.inputs[0].tensor.source,bounds.outputs[0])
  self.assertIs(a.inputs[1].tensor.source,bounds.outputs[1]);self.assertIs(b.inputs[1].tensor.source,bounds.outputs[1])
  self.assertIs(sync.inputs[0].tensor.source,a.outputs[0]);self.assertIs(sync.inputs[1].tensor.source,b.outputs[0])
  self.assertIn('composed[(tile_begin+k/4)*4+k%4]',b.impl.nv_src());self.assertIn('if(column>=0)',b.impl.nv_src())
 def test_scalar_offsets_preserve_divisor(self):
  bounds,*_=build_composed_k_stage(scalar_offsets=True)
  self.assertIn('bp[q]/4',bounds.impl.nv_src())
  self.assertIn('bp[q+1]-bp[q]',bounds.impl.nv_src())
 def test_existing_scheduler_preserves_shared_a_dependencies(self):
  stages=schedule_composed_k_stage(scalar_offsets=True,share_a=True)
  self.assertEqual([[op.name for op in stage] for stage in stages],[[
   'ComposedKParentBounds','ComposedKStageA','ComposedKStageB','ComposedKStageBarrier']])
  src=emit_scheduled_composed_k_stage(scalar_offsets=True,share_a=True)
  self.assertIn('if(warp==0)',src);self.assertIn('__syncthreads()',src)

if __name__=='__main__':unittest.main()

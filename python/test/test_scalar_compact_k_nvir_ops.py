import json
from pathlib import Path
import unittest
from sparsene.op_gen.nvir.scalar_compact_k_ops import build_scalar_compact_k_stage,emit_scalar_compact_k_stage,lower_scalar_compact_to_nvir

class ScalarCompactNvirTest(unittest.TestCase):
 def test_dependencies_and_indirection(self):
  bounds,a,b,sync=build_scalar_compact_k_stage()
  self.assertIs(a.inputs[0].tensor.source,bounds.outputs[1])
  self.assertIs(b.inputs[0].tensor.source,bounds.outputs[0])
  self.assertIs(b.inputs[1].tensor.source,bounds.outputs[1])
  self.assertIs(sync.inputs[0].tensor.source,a.outputs[0])
  self.assertIs(sync.inputs[1].tensor.source,b.outputs[0])
  src=emit_scalar_compact_k_stage()
  self.assertIn('si[scalar_begin+k]',src)
  self.assertIn('cols[q*8+pos]',src)
  self.assertIn('bp[q]+k/4',src)
  self.assertIn('__syncwarp()',src)
 def test_original_spec_to_nvir_and_rejection(self):
  from dataclasses import replace
  from sparsene.op_gen.scalar_compact_k_lowering import lower_scalar_compact_k_spec
  path=Path(__file__).parent/'fixtures/historical_webbase_four_level.json'
  plan=lower_scalar_compact_k_spec(json.loads(path.read_text()))
  self.assertEqual(len(lower_scalar_compact_to_nvir(plan)),4)
  with self.assertRaises(ValueError):lower_scalar_compact_to_nvir(replace(plan,blk_k_i=2))

if __name__=='__main__':unittest.main()

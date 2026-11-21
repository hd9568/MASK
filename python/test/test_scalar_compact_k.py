import json,unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy import sparse
from sparsene.op_gen.scalar_compact_k_lowering import lower_scalar_compact_k_spec
from sparsene.formats.scalar_compact_k_reference import pack_scalar_compact_k,unpack_scalar_compact_k

class ScalarCompactKTest(unittest.TestCase):
 def test_original_dsl_and_distinct_format_rejected(self):
  root=Path(__file__).parent/'fixtures'
  s=json.loads((root/'historical_webbase_four_level.json').read_text())
  p=lower_scalar_compact_k_spec(s)
  self.assertEqual((p.blk_m,p.blk_k,p.blk_k_i),(16,8,4))
  other=json.loads((root/'nested_active_k4_intent.json').read_text())
  with self.assertRaises(ValueError): lower_scalar_compact_k_spec(other)
 def test_reference_roundtrip_and_boundaries(self):
  rng=np.random.default_rng(99)
  for shape in ((0,0),(11,23),(35,67)):
   a=rng.normal(size=shape);a[rng.random(shape)<.85]=0
   p=pack_scalar_compact_k(sparse.csr_matrix(a),blk_m=16,blk_k=8,blk_k_i=4)
   np.testing.assert_array_equal(unpack_scalar_compact_k(p).toarray(),a)
   sizes=np.diff(p.parent_scalar_ptr)
   np.testing.assert_array_equal(np.diff(p.nested.block_child_ptr),(sizes+3)//4)
   if len(p.scalar_parent_positions):
    bad=p.scalar_parent_positions.copy();bad[0]=8
    with self.assertRaises(ValueError):unpack_scalar_compact_k(replace(p,scalar_parent_positions=bad))
if __name__=='__main__':unittest.main()

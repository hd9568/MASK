import unittest
import json
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
from sparsene.op_gen.composed_sparse_indices import compose_nested_columns

class ComposedSparseIndicesTest(unittest.TestCase):
 def test_nonidentity_holes_and_per_parent_padding(self):
  cols=np.array([[7,11,19,31,45,52,80,91],[3,9,-1,-1,-1,-1,-1,-1],[8,16,30,60,64,72,-1,-1]],np.int64)
  ptr=np.array([0,3,4,9],np.int64);pos=np.array([0,2,7,1,0,1,3,4,5],np.int32)
  x=compose_nested_columns(cols,ptr,pos,inner_width=4,ncols=100)
  np.testing.assert_array_equal(x.padded_scalar_ptr,[0,4,8,16])
  np.testing.assert_array_equal(x.scalar_counts,[3,1,5])
  np.testing.assert_array_equal(x.global_column_tiles,[[7,19,91,-1],[9,-1,-1,-1],[8,16,60,64],[72,-1,-1,-1]])
  np.testing.assert_array_equal(x.local_position_tiles,[[0,2,7,-1],[1,-1,-1,-1],[0,1,3,4],[5,-1,-1,-1]])
 def test_empty_parents_and_bad_metadata(self):
  cols=np.array([[1,2,-1,-1],[4,5,6,-1]],np.int64)
  x=compose_nested_columns(cols,np.array([0,0,2]),np.array([0,2]),inner_width=2,ncols=7)
  np.testing.assert_array_equal(x.padded_scalar_ptr,[0,0,2]);np.testing.assert_array_equal(x.global_column_tiles,[[4,6]])
  for ptr,pos in [([0,1,1],[2]),([0,2,2],[1,0]),([0,2,1],[0]),([0,1,1],[4])]:
   with self.subTest(ptr=ptr,pos=pos):
    with self.assertRaises(ValueError): compose_nested_columns(cols,np.array(ptr),np.array(pos),inner_width=2,ncols=7)
 def test_historical_opir_b_gather_consumes_composed_map(self):
  from sparsene.formats.dsl import build_format_from_spec
  from sparsene.transform.rts import derive_rts
  from sparsene.op_gen.computent.computent import computent_from_rts
  from sparsene.op_gen.opir.generate import generate_from_computent
  s=json.loads((Path(__file__).parent/'fixtures/historical_webbase_four_level.json').read_text())
  with redirect_stdout(StringIO()):
   c=computent_from_rts('SR_BCRS',derive_rts(build_format_from_spec(s)))
   op=generate_from_computent(c)
  ops=[]
  def visit(x):
   ops.append(x)
   if hasattr(x,'block'):
    for y in x.block.ops: visit(y)
  visit(op)
  b_refs=[x for x in ops if x.name=='array_ref' and x.operands[0].source.name_hint=='B_val']
  self.assertEqual(len(b_refs),1)
  column_map=b_refs[0].operands[1].source.defining_op
  self.assertEqual(column_map.name,'array_ref')
  self.assertEqual(column_map.operands[0].source.name_hint,'val_sidx')

if __name__=='__main__': unittest.main()

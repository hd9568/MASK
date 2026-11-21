import unittest
from dataclasses import replace
import numpy as np
from scipy import sparse
from sparsene.formats.scalar_compact_k_reference import pack_scalar_compact_k
from sparsene.op_gen.scalar_compact_k_host_abi import make_scalar_compact_host_abi

class ScalarHostAbiTest(unittest.TestCase):
 def test_offset_units_and_boundaries(self):
  a=sparse.coo_matrix(([2.,3.,5.,7.,11.],([0,2,3,16,17],[1,5,9,2,17])),shape=(18,19))
  payload=pack_scalar_compact_k(a,blk_m=16,blk_k=8,blk_k_i=4)
  abi=make_scalar_compact_host_abi(payload)
  self.assertEqual(abi.padded_rows,32)
  np.testing.assert_array_equal(abi.window_block_offsets,[0,1,2])
  np.testing.assert_array_equal(abi.true_scalar_counts,[3,2])
  np.testing.assert_array_equal(abi.padded_scalar_offsets,[0,4,8])
  np.testing.assert_array_equal(abi.global_column_tiles,[[1,5,9,-1],[2,17,-1,-1]])
  self.assertEqual(abi.global_column_tiles.dtype,np.int32)
  self.assertTrue(np.all(abi.values.transpose(0,2,1)[abi.global_column_tiles<0]==0))
 def test_nonzero_padding_rejected(self):
  p=pack_scalar_compact_k(sparse.eye(2),blk_m=2,blk_k=8,blk_k_i=4)
  bad=p.nested.values.copy();bad[0,0,3]=9
  with self.assertRaises(ValueError):make_scalar_compact_host_abi(replace(p,nested=replace(p.nested,values=bad)))
if __name__=='__main__':unittest.main()

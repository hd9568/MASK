"""Reference-only tests for arbitrary depth K hierarchies."""
import unittest
from dataclasses import replace
import numpy as np
from scipy import sparse
from sparsene.formats.multilevel_k_reference import pack_multilevel_k, unpack_multilevel_k
from sparsene.formats.nested_k_reference import pack_nested_k


class MultiLevelKTest(unittest.TestCase):
    def test_arbitrary_depth_and_leaf_roundtrip(self):
        rng = np.random.default_rng(7731)
        for shape in ((0, 9), (19, 0), (19, 37), (32, 64)):
            a = rng.normal(size=shape).astype(np.float32)
            a[rng.random(shape) < 0.88] = 0
            a = sparse.coo_matrix(a)
            for widths in ((8,), (8, 4), (16, 8, 4, 2), (12, 6, 3, 1)):
                for leaf in ('dense', 'coo'):
                    p = pack_multilevel_k(a, blk_m=8, widths=widths, leaf=leaf)
                    result = unpack_multilevel_k(p)
                    np.testing.assert_array_equal(result.toarray(), a.toarray())
                    b = rng.normal(size=(shape[1], 7)).astype(np.float32)
                    np.testing.assert_allclose(result @ b, a @ b, rtol=1e-6, atol=1e-6)

    def test_existing_two_level_reference_abi(self):
        a = sparse.random(29, 41, density=.15, random_state=5, format='csr')
        for leaf in ('dense', 'coo'):
            x = pack_nested_k(a, blk_m=16, blk_k=8, blk_k_i=4, leaf=leaf)
            y = pack_multilevel_k(a, blk_m=16, widths=(8, 4), leaf=leaf)
            for left, right in ((x.window_block_ptr, y.window_root_ptr), (x.block_child_ptr, y.child_ptrs[0]),
                                (x.block_column_ids, y.root_column_ids), (x.child_index, y.child_indices[0]),
                                (x.values, y.values)):
                np.testing.assert_array_equal(left, right)
            if leaf == 'coo':
                for left, right in ((x.child_value_ptr, y.leaf_value_ptr), (x.local_rows,y.local_rows), (x.local_cols,y.local_cols)):
                    np.testing.assert_array_equal(left, right)

    def test_offsets_and_widths_rejected(self):
        with self.assertRaises(ValueError):
            pack_multilevel_k(sparse.eye(5), blk_m=4, widths=(8, 3))
        p = pack_multilevel_k(sparse.eye(5), blk_m=4, widths=(8, 4, 2))
        bad_ptr = p.child_ptrs[0].copy(); bad_ptr[-1] += 1
        with self.assertRaises(ValueError):
            unpack_multilevel_k(replace(p, child_ptrs=(bad_ptr, *p.child_ptrs[1:])))


if __name__ == '__main__': unittest.main()

"""Numerical and ABI tests for the nested-K reference packer; CPU only."""
import unittest
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy import sparse
from sparsene.formats.nested_k_reference import pack_nested_k, unpack_nested_k
from sparsene.formats.dsl import build_format_from_spec, validate_format_spec
from sparsene.transform.rts import derive_rts


class NestedKReferenceTest(unittest.TestCase):
    def check_matrix(self, a, m, k, inner):
        baseline = sparse.csr_matrix(a); baseline.sum_duplicates(); baseline.eliminate_zeros()
        for leaf in ('dense', 'coo'):
            p = pack_nested_k(a, blk_m=m, blk_k=k, blk_k_i=inner, leaf=leaf)
            result = unpack_nested_k(p)
            np.testing.assert_allclose(result.toarray(), baseline.toarray())
            np.testing.assert_allclose(result @ np.arange(a.shape[1] * 3).reshape(a.shape[1], 3),
                                       baseline @ np.arange(a.shape[1] * 3).reshape(a.shape[1], 3))
            self.assertEqual(len(p.window_block_ptr), (a.shape[0] + m - 1) // m + 1)
            self.assertEqual(len(p.block_child_ptr), len(p.block_column_ids) + 1)
            for b in range(len(p.block_column_ids)):
                child_ids = p.child_index[p.block_child_ptr[b]:p.block_child_ptr[b + 1]]
                self.assertTrue(np.all(np.diff(child_ids) > 0))
        return pack_nested_k(a, blk_m=m, blk_k=k, blk_k_i=inner)

    def test_boundary_empty_rows_and_parent_children(self):
        a = sparse.coo_matrix(([1., 2., 3., 4., 5., 6.],
                               ([0, 1, 2, 3, 4, 4], [0, 5, 9, 12, 1, 15])), shape=(7, 17))
        p = self.check_matrix(a, 4, 8, 2)
        self.assertEqual(p.window_block_ptr.tolist(), [0, 1, 2])
        self.assertEqual(p.block_child_ptr.tolist(), [0, 2, 3])
        self.assertEqual(p.child_index.tolist(), [0, 1, 0])

    def test_empty_shapes_duplicates_and_random(self):
        self.check_matrix(sparse.csr_matrix((0, 4)), 4, 8, 4)
        self.check_matrix(sparse.csr_matrix((9, 4)), 4, 8, 4)
        duplicates = sparse.coo_matrix(([1., 2., -2., 4.], ([0, 1, 1, 3], [0, 2, 2, 5])), shape=(5, 7))
        self.check_matrix(duplicates, 2, 4, 2)
        rng = np.random.default_rng(42)
        for m, k, i in ((16, 8, 4), (4, 12, 3), (2, 4, 1)):
            a = rng.normal(size=(19, 23)); a[rng.random(a.shape) < .85] = 0
            self.check_matrix(sparse.coo_matrix(a), m, k, i)

    def test_explicit_intent_dsl_splits_before_sparsifying_children(self):
        path = Path(__file__).parent / 'fixtures/nested_active_k4_intent.json'
        spec = json.loads(path.read_text())
        self.assertEqual(validate_format_spec(spec)['status'], 'valid')
        ops = derive_rts(build_format_from_spec(spec)).sequence
        inner = next(i for i, op in enumerate(ops)
                     if op.name == 'split' and str(op.split_size) == 'BLK_K_I')
        self.assertEqual(ops[inner + 1].name, 'sparsify')
        self.assertEqual(ops[inner + 1].axes, (0,))

    def test_invalid_metadata_rejected(self):
        p = pack_nested_k(sparse.eye(3), blk_m=2, blk_k=4, blk_k_i=2)
        with self.assertRaises(ValueError):
            unpack_nested_k(replace(p, child_index=np.full_like(p.child_index, 99)))
        with self.assertRaises(ValueError):
            pack_nested_k(sparse.eye(2), blk_m=2, blk_k=7, blk_k_i=4)


if __name__ == '__main__':
    unittest.main()

"""Semantic and NVIR regressions for variable inner K on the strict DSL."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from sparsene.formats.dsl import build_format_from_spec
from sparsene.op_gen.backend_registry import select_backend, materialize_backend_format_spec
from generate_custom_format_operators import lower_format_operator

FIXTURE = Path(__file__).parent / 'fixtures/historical_webbase_four_level.json'

class VariableInnerKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = json.loads(FIXTURE.read_text())

    def spec(self, outer, inner):
        spec = copy.deepcopy(self.original)
        spec['params']['BLK_K'] = outer
        spec['params']['BLK_K_I'] = inner
        return spec

    def test_supported_plans_preserve_dsl_and_physical_k(self):
        for outer in range(8, 65):
            for inner in range(4, 33):
                if inner >= outer or outer % inner:
                    continue
                with self.subTest(outer=outer, inner=inner):
                    spec = self.spec(outer, inner)
                    plan = select_backend(spec, allow_physicalization=False)
                    self.assertEqual(plan.logical_blk_k, outer)
                    self.assertEqual(plan.logical_inner_blk_k, inner)
                    self.assertEqual(plan.physical_blk_k, 1 << (max(8, inner) - 1).bit_length())
                    self.assertEqual(materialize_backend_format_spec(spec, plan), spec)

    def test_reject_invalid_divisibility_and_limits(self):
        for outer,inner in ((8,2),(8,8),(8,5),(21,8),(66,11),(64,33),(7,4)):
            with self.subTest(outer=outer, inner=inner):
                for allow_physicalization in (False, True):
                    with self.assertRaises((ValueError, ZeroDivisionError)):
                        select_backend(self.spec(outer,inner), allow_physicalization=allow_physicalization)

    def test_codegen_specializes_leaf_and_emits_full_k_mma_loop(self):
        for outer,inner in ((8,4),(21,7),(24,12),(32,16),(62,31)):
            with self.subTest(outer=outer, inner=inner), tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                lower_format_operator(
                    format_obj=build_format_from_spec(self.spec(outer,inner)),
                    codegen_format_name='SCALAR_COMPACT_K4',
                    operator_path=d/'kernel.inc',ir_dir=d/'ir',strategy_dir=d/'strategy',
                    specialized_parameters={'BLK_K_I':inner})
                source=(d/'kernel.inc').read_text()
                self.assertIn(f'using BLK_K_I = cute::Int<{inner}>;',source)
                self.assertIn('ki<get<2>(BLK_MMA_MNK{})',source)
                self.assertIn('int k = group * 4 + j;',source)

if __name__ == '__main__':
    unittest.main()

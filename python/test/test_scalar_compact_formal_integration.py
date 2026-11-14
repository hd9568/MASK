"""Contracts at the agent-to-exact-four-level backend boundary."""
import copy
import json
from pathlib import Path
import unittest
from sparsene.op_gen.backend_registry import select_backend, materialize_backend_format_spec
from sparsene.op_gen.format_design_agent import assess_backend_capability
from sparsene.op_gen.strategy_agent import StrategyDecision, normalize_strategy_for_backend

class ScalarCompactFormalIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((Path(__file__).parent / 'fixtures/historical_webbase_four_level.json').read_text())

    def test_exact_format_is_not_rewritten(self):
        plan = select_backend(self.spec, allow_physicalization=False)
        self.assertEqual(plan.backend, 'scalar_compact_k4_tensorcore_v1')
        self.assertEqual(materialize_backend_format_spec(self.spec, plan), self.spec)
        assessment = assess_backend_capability(format_name=self.spec['format_name'], levels=self.spec['levels'], atomic=self.spec['atomic'], params=self.spec['params'], selected_candidate='agent')
        self.assertEqual(assessment['matched_profile'], 'SCALAR_COMPACT_K4_V1')
        self.assertTrue(assessment['execution_ready'])

    def test_different_or_unvalidated_profile_is_rejected_in_exact_mode(self):
        modified = copy.deepcopy(self.spec)
        modified['params']['BLK_K_I'] = 2
        with self.assertRaises(ValueError):
            select_backend(modified, allow_physicalization=False)
        modified = copy.deepcopy(self.spec)
        modified['levels'][2]['axes'][0]['type'] = 'df'
        with self.assertRaises(ValueError):
            select_backend(modified, allow_physicalization=False)

    def test_sync_vector_staging_preserves_alignment(self):
        import tempfile
        from sparsene.formats.dsl import build_format_from_spec
        from sparsene.op_gen.strategy_agent import PipelineStrategy, TensorPlacementStrategy
        from generate_custom_format_operators import lower_format_operator
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            d = StrategyDecision(provider="test", rationale="alignment",
                                 tensor_placement=TensorPlacementStrategy(),
                                 pipeline=PipelineStrategy())
            lower_format_operator(format_obj=build_format_from_spec(self.spec),
                                  codegen_format_name="SCALAR_COMPACT_K4",
                                  operator_path=root / "kernel.inc", ir_dir=root / "ir",
                                  strategy_dir=root / "strategy", strategy_decision=d)
            source = (root / "kernel.inc").read_text()
            self.assertIn("__align__(16) __shared__ float array_ref_1_tensor_o0", source)
            self.assertIn("uint32_t ah[4], al[4], bh[2], bl[2]", source)

    def test_strategy_normalization_limits_unvalidated_layouts(self):
        d = StrategyDecision.from_dict({'provider':'test','rationale':'test','tensor_placement':{'tile_b':32,'materialize_b_array_ref_to_smem':False},'runtime_schedule':{'execution_policy':'multi_bind'}})
        d = normalize_strategy_for_backend(d, packing_mode='scalar_compact_k4_dense')
        self.assertEqual(d.tensor_placement.tile_b,64)
        self.assertTrue(d.tensor_placement.materialize_b_array_ref_to_smem)
        self.assertEqual(d.runtime_schedule.execution_policy,'row')
        self.assertEqual(d.agent_metadata['backend_normalization']['packing_mode'], 'scalar_compact_k4_dense')

if __name__ == '__main__': unittest.main()

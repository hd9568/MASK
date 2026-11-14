"""Regression: nested scalar offsets must not be treated as tile offsets."""
import contextlib,io,json,unittest
from pathlib import Path
from sparsene.formats.dsl import build_format_from_spec
from sparsene.transform.rts import derive_rts
from sparsene.op_gen.computent.computent import computent_from_rts
from sparsene.op_gen.opir.generate import generate_from_computent
from sparsene.op_gen.opir.varlenLoweringPass import VarlenLoweringPass
from sparsene.op_gen.opir.cValFlattenPass import CValFlattenPass
from sparsene.op_gen.nvir.generate import generate_nvir
from sparsene.op_gen.nvir.compiler_driver import _fuse_load_offset_sub_div
from sparsene.op_gen.strategy_agent import StrategyDecision,PipelineStrategy,TensorPlacementStrategy

class NestedOffsetFusionTest(unittest.TestCase):
 def test_preserve_inner_scalar_division(self):
  s=json.loads((Path(__file__).parent/'fixtures/historical_webbase_four_level.json').read_text())
  decision=StrategyDecision(provider='test',rationale='offset units',tensor_placement=TensorPlacementStrategy(),pipeline=PipelineStrategy())
  with contextlib.redirect_stdout(io.StringIO()):
   c=computent_from_rts('SR_BCRS',derive_rts(build_format_from_spec(s)))
   op=generate_from_computent(c)
   op=VarlenLoweringPass(op_builder=None,varlen2LenArrayTable=c.varlen2LenArrayTable).run(op)
   op=CValFlattenPass(op_builder=None).run(op)
   p=generate_nvir(opir=op,format_name='SR_BCRS',strategy_decision_override=decision,strategy_log_dir_override='')
  def allops(seq):
   for x in seq:
    yield x
    if hasattr(x,'body'): yield from allops(x.body.ops)
  _fuse_load_offset_sub_div(p,preserve_offset_names=frozenset({'val_len_1_new_offset'}))
  ops=list(allops(p.ops))
  loops={x.name:x for x in ops if x.name in ('for_loop','for_loop_1')}
  self.assertEqual(len(loops),2)
  self.assertEqual(loops['for_loop'].inputs[1].tensor.source.op.name,'load_offset')
  self.assertEqual(loops['for_loop_1'].inputs[1].tensor.source.op.name,'gen_div_1')
  self.assertIn(' - ',next(x for x in ops if x.name=='load_offset').impl.code_template)
  self.assertNotIn(' - ',next(x for x in ops if x.name=='load_offset_1').impl.code_template)
if __name__=='__main__': unittest.main()

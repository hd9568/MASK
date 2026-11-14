import contextlib,io,json,unittest
from pathlib import Path
from sparsene.formats.dsl import build_format_from_spec
from sparsene.transform.rts import derive_rts
from sparsene.op_gen.computent.computent import computent_from_rts
from sparsene.op_gen.opir.generate import generate_from_computent
from sparsene.op_gen.opir.varlenLoweringPass import VarlenLoweringPass
from sparsene.op_gen.nvir.generate import build_gmem_interfaces

class ExactSymbolLayoutTest(unittest.TestCase):
 def test_separate_nested_offset_layouts_and_legacy_fallback(self):
  spec=json.loads((Path(__file__).parent/'fixtures/historical_webbase_four_level.json').read_text())
  with contextlib.redirect_stdout(io.StringIO()):
   c=computent_from_rts('SR_BCRS',derive_rts(build_format_from_spec(spec)))
   op=generate_from_computent(c)
   op=VarlenLoweringPass(op_builder=None,varlen2LenArrayTable=c.varlen2LenArrayTable).run(op)
   hints={'shape_overrides':{'val_len':['Mo'],'val_len_1_new_offset':['ParentCount']},'dtypes':{'len':'int'},'custom_layouts':{}}
   interfaces=build_gmem_interfaces(op,hints)
  self.assertEqual(str(interfaces['val_len_offset'].shape),'(Varlen(Mo))')
  self.assertEqual(str(interfaces['val_len_1_new_offset'].shape),'(Varlen(ParentCount))')
  self.assertIn('ParentCount',interfaces['val_len_1_new_offset'].tensor_str)
if __name__=='__main__':unittest.main()

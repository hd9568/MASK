import json,unittest
from pathlib import Path
from sparsene.op_gen.multilevel_k_lowering import lower_multilevel_k_spec
from sparsene.op_gen.multilevel_k_scalar_codegen import emit_scalar_cuda

class CodegenTest(unittest.TestCase):
 def test_leaf_specific_abi_and_depth(self):
  p=Path(__file__).parent/'fixtures/nested_active_k4_intent.json'; s=json.loads(p.read_text())
  for leaf in ('dense','coo'):
   s['atomic']['type']=leaf;out=emit_scalar_cuda(lower_multilevel_k_spec(s))
   self.assertIn('sparsene_experimental_multilevel_k',out)
   self.assertIn('ptr0[root]',out)
   self.assertEqual('val_ptr[child0]' in out,leaf=='coo')
if __name__=='__main__': unittest.main()

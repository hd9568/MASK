import json,unittest
from pathlib import Path
from sparsene.op_gen.scalar_compact_k_opir_audit import audit_scalar_compact_opir

class AuditTest(unittest.TestCase):
 def test_historical_opir_contract(self):
  s=json.loads((Path(__file__).parent/'fixtures/historical_webbase_four_level.json').read_text())
  result=audit_scalar_compact_opir(s)
  self.assertEqual(result['b_gather'],'composed_global_columns')
  self.assertEqual(result['preserve_scalar_offset_divisors'],['val_len_1_new_offset'])
  self.assertEqual(result['status'],'audited_not_integrated')
if __name__=='__main__':unittest.main()

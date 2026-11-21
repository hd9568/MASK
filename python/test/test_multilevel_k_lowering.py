import copy
import json
from pathlib import Path
import unittest
import numpy as np
from scipy import sparse
from sparsene.op_gen.multilevel_k_lowering import lower_multilevel_k_spec
from sparsene.formats.multilevel_k_reference import pack_multilevel_k, unpack_multilevel_k


def fixture():
    return json.loads((Path(__file__).parent / 'fixtures/nested_active_k4_intent.json').read_text())


class MultiLevelLoweringTest(unittest.TestCase):
    def test_lowering_and_host_reference_contract(self):
        s=fixture(); original=copy.deepcopy(s)
        for leaf in ('dense', 'coo'):
            s['atomic']['type']=leaf
            plan=lower_multilevel_k_spec(s)
            self.assertEqual(plan.widths, (8, 4))
            self.assertEqual(plan.backend_status, 'host_reference_only')
            a=sparse.csr_matrix(np.array([[1.,0,2.,3.],[0,4.,0,0],[5,0,0,6]],dtype=np.float32))
            p=pack_multilevel_k(a,blk_m=plan.blk_m,widths=plan.widths,leaf=plan.leaf)
            np.testing.assert_array_equal(unpack_multilevel_k(p).toarray(),a.toarray())
            self.assertEqual(plan.to_dict()['metadata_abi']['leaf'],leaf)
        s['atomic']['type']='dense';self.assertEqual(s,original)

    def test_more_levels(self):
        s=fixture();s['params']['K2']=2
        s['levels'].append({'level':4,'name':'smaller_active_children','axes':[
            {'name':'Y_sii_o','type':'sv','direction':'COL','length':'BLK_K_I/K2','reorder':None},
            {'name':'Y_sii_i','type':'df','direction':'COL','length':'K2','reorder':None},
            {'name':'X_i','type':'df','direction':'ROW','length':'BLK_M','reorder':None}]})
        s['atomic']['y_axis']='Y_sii_i'
        plan=lower_multilevel_k_spec(s)
        self.assertEqual(plan.widths,(8,4,2))

    def test_rejects_silently_different_or_unsupported_formats(self):
        cases=[]
        s=fixture();s['atomic']['type']='mco';cases.append(s)
        s=fixture();s['levels'][0]['axes'][0]['reorder']='row_rcm';cases.append(s)
        s=fixture();s['levels'][2]['axes'][0]['type']='df';cases.append(s)
        s=fixture();s['params']['BLK_K_I']=3;cases.append(s)
        s=fixture();s['levels'][2]['axes'][0]['length']='BLK_K';cases.append(s)
        for s in cases:
            with self.subTest(s=s):
                with self.assertRaises(ValueError): lower_multilevel_k_spec(s)

if __name__=='__main__': unittest.main()

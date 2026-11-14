import unittest
from sparsene.op_gen.nvir.codegen import NvIrCodeGenerator
from sparsene.op_gen.nvir.nvop import NvOp,NvOpImpl,NvOpInput,NvOpTensor,Shape

class SpecializedParameterTest(unittest.TestCase):
 def test_runtime_scalar_replaced_without_changing_other_symbols(self):
  gen=NvIrCodeGenerator();gen._specialized_parameters={'BLK_K_I':4}
  op=NvOp('probe',[],[],NvOpImpl(''))
  x=NvOpInput(0,'x',NvOpTensor(Shape(),'rmem','int',source='BLK_K_I'))
  self.assertEqual(gen.dump_f_method_call_arg(op,x,''),'4')
  x.tensor.source='BLK_K';self.assertEqual(gen.dump_f_method_call_arg(op,x,''),'get<2>(BLK_MNK{})')
 def test_rejects_unsafe_or_reserved_parameter(self):
  from sparsene.op_gen.nvir.nvop import NvOpProgram
  for bad in ({'BLK_K':4},{'bad;':4},{'BLK_K_I':True},{'BLK_K_I':0}):
   p=NvOpProgram('invalid',{});p.attrs['specialized_parameters']=bad
   with self.assertRaises(ValueError):NvIrCodeGenerator().dump_nvop_program(p)
 def test_sequential_loop_uses_iteration_for_opt_in(self):
  from sparsene.op_gen.nvir.nvop import ForLoopNvOp,NvOpSequence
  scalar=NvOpTensor(Shape(),'rmem','int',source='zero')
  left=NvOpInput(0,'l',scalar);right=NvOpInput(1,'r',scalar)
  i=NvOpInput(0,'i',NvOpTensor(Shape(),'rmem','int',source='l + {c}'))
  body=NvOp('probe',[i],[],NvOpImpl(''))
  loop=ForLoopNvOp('sequential_probe',left,right,body=NvOpSequence(body))
  g=NvIrCodeGenerator();g._explicit_sequential_iteration=True
  text=g.dump_sequential_for_loop_nvop_class_def(loop,[body])
  self.assertIn('l + _iter',text)

if __name__=='__main__':unittest.main()

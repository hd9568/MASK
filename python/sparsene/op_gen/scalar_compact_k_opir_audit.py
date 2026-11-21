"""Audit original DSL through Computent and OPIR before experimental emission.

This does not lower OPIR into the emitted NVIR stage; it only checks the
structural contract and makes the remaining gap explicit.
"""
from __future__ import annotations
import contextlib
import io
from .scalar_compact_k_lowering import lower_scalar_compact_k_spec
from sparsene.formats.dsl import build_format_from_spec
from sparsene.transform.rts import derive_rts
from sparsene.op_gen.computent.computent import computent_from_rts
from sparsene.op_gen.opir.generate import generate_from_computent
from sparsene.op_gen.opir.varlenLoweringPass import VarlenLoweringPass


def _walk(op):
 yield op
 if hasattr(op,'block'):
  for child in op.block.ops: yield from _walk(child)


def audit_scalar_compact_opir(spec):
 plan=lower_scalar_compact_k_spec(spec)
 with contextlib.redirect_stdout(io.StringIO()):
  comp=computent_from_rts('SR_BCRS',derive_rts(build_format_from_spec(spec)))
  op=generate_from_computent(comp)
  refs=[x for x in _walk(op) if x.name=='array_ref' and x.operands[0].source.name_hint=='B_val']
  if len(refs)!=1: raise ValueError('unexpected B gather structure')
  source=refs[0].operands[1].source.defining_op
  if source.name!='array_ref' or source.operands[0].source.name_hint!='val_sidx':
   raise ValueError('B gather does not use the expected composed column mapping')
  op=VarlenLoweringPass(op_builder=None,varlen2LenArrayTable=comp.varlen2LenArrayTable).run(op)
 symbols={x.attributes.get('symbol') for x in _walk(op) if x.name=='external_symbol'}
 required={'val_len_offset','val_len_1_new_offset','val_sidx_new_1','val_new_1'}
 if not required.issubset(symbols): raise ValueError('unexpected varlen-lowered ABI symbols')
 return {'source_sha256':plan.source_sha256,'b_gather':'composed_global_columns','opir_symbols':sorted(required),'preserve_scalar_offset_divisors':['val_len_1_new_offset'],'status':'audited_not_integrated'}

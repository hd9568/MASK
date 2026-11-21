"""Produce an UNVALIDATED native OPIR/NVIR candidate for diagnostics only."""
import argparse,json
from pathlib import Path
from sparsene.formats.dsl import build_format_from_spec
from sparsene.op_gen.scalar_compact_k_lowering import lower_scalar_compact_k_spec
from sparsene.op_gen.nvir.generate import SPARSE_FORMAT_REGISTRY
from sparsene.op_gen.nvir.scalar_compact_profile import build_scalar_compact_candidate_hints, pad_candidate_k4_for_mma
from sparsene.op_gen.strategy_agent import StrategyDecision,PipelineStrategy,TensorPlacementStrategy
from generate_custom_format_operators import lower_format_operator
p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--tf32x3',action='store_true');p.add_argument('--strategy-json',type=Path);a=p.parse_args()
s=json.loads(a.spec.read_text());plan=lower_scalar_compact_k_spec(s)
if (plan.blk_m,plan.blk_k,plan.blk_k_i)!=(16,8,4):raise ValueError('only 16x8/4 candidate supported')
a.out.mkdir(parents=True,exist_ok=False)
name='SCALAR_COMPACT_K_DIAGNOSTIC';SPARSE_FORMAT_REGISTRY[name]=build_scalar_compact_candidate_hints()
d=(StrategyDecision.from_dict(json.loads(a.strategy_json.read_text())) if a.strategy_json else StrategyDecision(provider='heuristic',rationale='unvalidated native candidate',tensor_placement=TensorPlacementStrategy(),pipeline=PipelineStrategy()))
manifest={'status':'unvalidated_codegen_candidate','source_format_sha256':plan.source_sha256,'experimental_only':True}
try:
 manifest.update(lower_format_operator(format_obj=build_format_from_spec(s),codegen_format_name=name,operator_path=a.out/'kernel.inc',ir_dir=a.out/'ir',strategy_dir=a.out/'strategy',strategy_decision=d,specialized_parameters={'BLK_K_I':4},preserve_scalar_offset_divisors=('val_len_1_new_offset',),explicit_sequential_iteration=True,diagnostic_program_transform=lambda program: pad_candidate_k4_for_mma(program,tf32x3=a.tf32x3)))
finally:
 (a.out/'generation_manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')

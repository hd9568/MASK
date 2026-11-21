"""Emit isolated experimental NVIR stages; does not generate a full kernel."""
import argparse
import hashlib
import json
from pathlib import Path
from sparsene.op_gen.scalar_compact_k_lowering import lower_scalar_compact_k_spec
from sparsene.op_gen.nvir.scalar_compact_k_ops import lower_scalar_compact_to_nvir
p=argparse.ArgumentParser()
p.add_argument('--spec',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--mode',choices=('warp_private','shared_a'),default='warp_private')
a=p.parse_args()
spec=json.loads(a.spec.read_text());plan=lower_scalar_compact_k_spec(spec)
bounds,stage_a,stage_b,barrier=lower_scalar_compact_to_nvir(plan)
if a.mode=='warp_private':
 source='\n'.join(op.impl.nv_src() for op in (bounds,stage_a,stage_b,barrier))+'\n'
else:
 source='\n'.join((bounds.impl.nv_src(),'if(warp==0) {\n'+stage_a.impl.nv_src()+'\n}',stage_b.impl.nv_src(),'__syncthreads();'))+'\n'
a.out.mkdir(parents=True,exist_ok=True)
(a.out/'nvir_stage.inc').write_text(source)
(a.out/'nvir_stage_manifest.json').write_text(json.dumps({'plan':plan.to_dict(),'mode':a.mode,'stage_sha256':hashlib.sha256(source.encode()).hexdigest(),'scope':'experimental staging only; host, outer kernel, MMA, scheduling are not generated'},indent=2)+'\n')

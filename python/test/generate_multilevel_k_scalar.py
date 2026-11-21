"""Generate experimental scalar CUDA; never claims production NVIR support."""
import argparse
import hashlib
import json
from pathlib import Path
from sparsene.op_gen.multilevel_k_lowering import lower_multilevel_k_spec
from sparsene.op_gen.multilevel_k_scalar_codegen import emit_scalar_cuda

p=argparse.ArgumentParser()
p.add_argument('--spec',type=Path,required=True)
p.add_argument('--leaf',choices=('dense','coo'),required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
s=json.loads(a.spec.read_text());s['atomic']['type']=a.leaf
plan=lower_multilevel_k_spec(s);source=emit_scalar_cuda(plan)
a.out.mkdir(parents=True,exist_ok=True)
(a.out/'multilevel_k_generated.inc').write_text(source)
(a.out/'codegen_manifest.json').write_text(json.dumps({'plan':plan.to_dict(),'codegen':'experimental_scalar_not_nvir','source_sha256':hashlib.sha256(source.encode()).hexdigest()},indent=2)+'\n')

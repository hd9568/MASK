"""Emit the experimental complete kernel and provenance from a saved DSL."""
import argparse,hashlib,json
from pathlib import Path
from sparsene.op_gen.scalar_compact_k_lowering import lower_scalar_compact_k_spec
from sparsene.op_gen.scalar_compact_k_kernel_codegen import emit_experimental_full_kernel
p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
spec=json.loads(a.spec.read_text());plan=lower_scalar_compact_k_spec(spec)
source=emit_experimental_full_kernel(plan);a.out.mkdir(parents=True,exist_ok=True)
(a.out/'scalar_compact_generated.inc').write_text(source)
(a.out/'generation_manifest.json').write_text(json.dumps({'schema':'sparsene.experimental_full_kernel.v1','source_format_sha256':plan.source_sha256,'kernel_sha256':hashlib.sha256(source.encode()).hexdigest(),'backend':'experimental_nvir_stage_wmma_shell','production_opir_nvir':False,'production_host_runner':False,'tile':[16,8,4],'threads':128,'data_types':'TF32/FP32'},indent=2)+'\n')

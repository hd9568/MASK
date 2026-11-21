"""Offline experimental DSL -> scheduled stage/WMMA template -> GPU check.

This is a separate research entry point, not production OPIR/NVIR lowering.
Choose a new output directory on Lustre for logs, metadata and build files.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.io import mmread
from scipy import sparse
from sparsene.formats.scalar_compact_k_reference import pack_scalar_compact_k
from sparsene.op_gen.scalar_compact_k_host_abi import make_scalar_compact_host_abi
from sparsene.op_gen.scalar_compact_k_lowering import lower_scalar_compact_k_spec
from sparsene.op_gen.scalar_compact_k_kernel_codegen import emit_experimental_full_kernel,emit_scalar_fp32_diagnostic
from sparsene.op_gen.scalar_compact_k_opir_audit import audit_scalar_compact_opir


def main():
 p=argparse.ArgumentParser();p.add_argument('--matrix',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--arch',choices=('sm_80','sm_90'),required=True);p.add_argument('--nvcc',default='nvcc');p.add_argument('--compute',choices=('tf32','tf32x3','scalar_fp32_diagnostic'),default='tf32');p.add_argument('--n',type=int,default=512);p.add_argument('--repeat',type=int,default=5);p.add_argument('--gpu-lock',type=Path,default=Path('/tmp/sparsekg_sm80_gpu_benchmark.lock'));a=p.parse_args()
 if a.n<=0 or a.repeat<=0: p.error('N and repeat must be positive')
 a.run_dir.mkdir(parents=True,exist_ok=False); root=a.run_dir.resolve();d=root/'dense';d.mkdir()
 spec=json.loads(a.spec.read_text());plan=lower_scalar_compact_k_spec(spec);audit=audit_scalar_compact_opir(spec);source=(emit_scalar_fp32_diagnostic(plan) if a.compute=='scalar_fp32_diagnostic' else emit_experimental_full_kernel(plan,refine=a.compute=='tf32x3'))
 (root/'scalar_compact_generated.inc').write_text(source)
 manifest={'schema':'sparsene.experimental_scalar_compact_e2e.v1','matrix':str(a.matrix.resolve()),'format_sha256':plan.source_sha256,'kernel_sha256':hashlib.sha256(source.encode()).hexdigest(),'backend':('experimental_scalar_fp32_diagnostic' if a.compute=='scalar_fp32_diagnostic' else 'experimental_nvir_stage_wmma_shell'),'compute_mode':a.compute,'opir_audit':audit,'production_opir_nvir':False,'production_host_runner':False,'n':a.n,'repeat':a.repeat,'arch':a.arch,'status':'preparing'}
 def write(): (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 write()
 try:
  start=time.monotonic();raw=mmread(a.matrix)
  if not sparse.issparse(raw): raise ValueError('coordinate sparse matrix input is required')
  if np.iscomplexobj(raw.data): raise ValueError('complex input is unsupported')
  mat=raw.astype(np.float32).tocsr();mat.sum_duplicates();mat.eliminate_zeros();mat.sort_indices()
  if not np.all(np.isfinite(mat.data)): raise ValueError('matrix contains nonfinite values')
  if min(mat.shape)<=0 or mat.nnz<=0: raise ValueError('research runtime requires positive shape and at least one nonzero')
  if max(*mat.shape,mat.nnz)>np.iinfo(np.int32).max: raise OverflowError('runtime requires int32 matrix dimensions and nnz')
  payload=pack_scalar_compact_k(mat,blk_m=plan.blk_m,blk_k=plan.blk_k,blk_k_i=plan.blk_k_i);abi=make_scalar_compact_host_abi(payload)
  arrays={'dense/abi_window_ptr':abi.window_block_offsets,'dense/abi_scalar_ptr':abi.padded_scalar_offsets,'dense/abi_column_tiles':abi.global_column_tiles,'dense/values':abi.values,'csr_ptr':mat.indptr.astype(np.int32),'csr_col':mat.indices.astype(np.int32),'csr_val':mat.data}
  for name,arr in arrays.items(): arr.tofile(root/(name+'.bin'))
  roots=len(payload.nested.block_column_ids);children=len(payload.nested.values)
  manifest.update({'shape':list(mat.shape),'nnz':int(mat.nnz),'roots':roots,'children':children,'preprocess_seconds':time.monotonic()-start,'status':'compiling'});write()
  cpp=Path(__file__).with_name('scalar_compact_generated_runtime.cu').resolve();exe=root/'experimental_e2e'
  cmd=[a.nvcc,'-O3','-std=c++17','-arch='+a.arch,'-I',str(root),str(cpp),'-lcusparse','-o',str(exe)]
  result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(root/'compile.log').write_text(result.stdout)
  if result.returncode: raise RuntimeError('CUDA compilation failed; see compile.log')
  manifest['status']='running';write()
  cmd=[str(exe),str(root),'dense',str(a.n),str(a.repeat),str(mat.shape[0]),str(mat.shape[1]),str(mat.nnz),str(roots),str(children)]
  with a.gpu_lock.open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  (root/'run.log').write_text(result.stdout)
  if result.returncode: raise RuntimeError(f'GPU check failed (exit {result.returncode}); see run.log')
  lines=[x for x in result.stdout.splitlines() if x.startswith('{')];report=json.loads(lines[-1]);(root/'result.json').write_text(json.dumps(report,indent=2)+'\n')
  manifest['status']='validated_experimental';write();print(json.dumps({'run_dir':str(root),'result':report},ensure_ascii=False))
 except Exception as exc:
  manifest.update({'status':'failed','error':str(exc)});write();raise
if __name__=='__main__':main()

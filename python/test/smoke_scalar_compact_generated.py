"""Random/boundary GPU check for experimental full generated kernel."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from scipy import sparse
from sparsene.formats.scalar_compact_k_reference import pack_scalar_compact_k
from sparsene.op_gen.scalar_compact_k_host_abi import make_scalar_compact_host_abi
from sparsene.op_gen.scalar_compact_k_lowering import lower_scalar_compact_k_spec
from sparsene.op_gen.scalar_compact_k_kernel_codegen import emit_experimental_full_kernel

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--arch',required=True);p.add_argument('--nvcc',default='nvcc');a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
s=json.loads((Path(__file__).parent/'fixtures/historical_webbase_four_level.json').read_text())
(a.out/'scalar_compact_generated.inc').write_text(emit_experimental_full_kernel(lower_scalar_compact_k_spec(s)))
rng=np.random.default_rng(731);M,K,N=71,91,33
A=rng.normal(0,2,size=(M,K)).astype('float32');A[rng.random(A.shape)<.85]=0;A[17,:]=0;A[-1,:]=0;A[-1,-1]=1.5
B=rng.normal(size=(K,N)).astype('float32');expected=A@B
abi=make_scalar_compact_host_abi(pack_scalar_compact_k(sparse.csr_matrix(A),blk_m=16,blk_k=8,blk_k_i=4))
def arr(name,typ,x):
 x=np.asarray(x).ravel();body=','.join(f'{float(v):.9e}f' if typ=='float' else str(int(v)) for v in x) or '0'
 return f'static const {typ} {name}[] = {{{body}}};\n'
arrays=''.join((arr('wp','int32_t',abi.window_block_offsets),arr('bp','int32_t',abi.padded_scalar_offsets),arr('cols','int32_t',abi.global_column_tiles),arr('val','float',abi.values),arr('B','float',B),arr('expected','float',expected)))
source=r'''#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "scalar_compact_generated.inc"
#define CK(x) do{auto q=(x);if(q!=cudaSuccess){fprintf(stderr,"%s at %d\n",cudaGetErrorString(q),__LINE__);exit(2);}}while(0)
''' + arrays + r'''
template <class T,size_t L> T* upload(const T (&a)[L]) {T* p;CK(cudaMalloc(&p,sizeof(a)));CK(cudaMemcpy(p,a,sizeof(a),cudaMemcpyHostToDevice));return p;}
int main(){auto dwp=upload(wp);auto dbp=upload(bp);auto dc=upload(cols);auto dv=upload(val);auto db=upload(B);
 float* out;CK(cudaMalloc(&out,71*33*sizeof(float)));compute_tc<<<dim3((71+15)/16,(33+63)/64),128>>>(71,33,dwp,dbp,nullptr,dv,nullptr,nullptr,dc,db,out);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
 std::vector<float> got(71*33);CK(cudaMemcpy(got.data(),out,got.size()*sizeof(float),cudaMemcpyDeviceToHost));float maxerr=0;int bad=0;
 for(size_t i=0;i<got.size();i++){float e=fabsf(got[i]-expected[i]);maxerr=fmaxf(maxerr,e);if(!isfinite(got[i])||e>0.02f+0.005f*fabsf(expected[i]))bad++;}
 printf("{\"M\":71,\"K\":91,\"N\":33,\"max_abs\":%g,\"mismatches\":%d,\"production_opir_nvir\":false}\n",maxerr,bad);return bad?3:0;}
'''
(a.out/'smoke.cu').write_text(source)
subprocess.run([a.nvcc,'-std=c++17','-O3','-arch='+a.arch,str(a.out/'smoke.cu'),'-o',str(a.out/'smoke')],check=True)
subprocess.run([str(a.out/'smoke')],check=True)

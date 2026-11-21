"""Standalone multi-level CUDA ABI probe, not Sparsene-generated code or a benchmark claim."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import numpy as np
from scipy import sparse
from sparsene.formats.multilevel_k_reference import pack_multilevel_k


def literal(a, cast):
    flat = np.asarray(a).ravel()
    if not len(flat): return '0'
    return ','.join(cast(x) for x in flat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--arch', default='sm_80')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(382)
    a = rng.normal(size=(71, 91)).astype('float32')
    a[rng.random(a.shape) < .87] = 0
    a[17, :] = 0
    b = rng.normal(size=(91, 64)).astype('float32')
    expected = a @ b
    dense = pack_multilevel_k(sparse.csr_matrix(a), blk_m=16, widths=(8,4,2), leaf='dense')
    coo = pack_multilevel_k(sparse.csr_matrix(a), blk_m=16, widths=(8,4,2), leaf='coo')
    assert np.array_equal(dense.window_root_ptr, coo.window_root_ptr)
    assert all(np.array_equal(x, y) for x, y in zip(dense.child_ptrs, coo.child_ptrs))
    assert np.array_equal(dense.root_column_ids, coo.root_column_ids)
    def arr(name, typ, values):
        render = (lambda x: f'{float(x):.9e}f') if typ == 'float' else lambda x: str(int(x))
        return f'static const {typ} {name}[] = {{{literal(values, render)}}};\n'
    arrays = ''.join([
        arr('wp', 'long long', dense.window_root_ptr), arr('bp', 'long long', dense.child_ptrs[0]), arr('bp1', 'long long', dense.child_ptrs[1]),
        arr('cols', 'long long', dense.root_column_ids), arr('child', 'int', dense.child_indices[0]), arr('child1', 'int', dense.child_indices[1]),
        arr('dv', 'float', dense.values), arr('vp', 'long long', coo.leaf_value_ptr),
        arr('rr', 'int', coo.local_rows), arr('cc', 'int', coo.local_cols),
        arr('cv', 'float', coo.values), arr('B', 'float', b), arr('expected', 'float', expected)])
    source = r'''#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdlib>
#define CHECK(x) do {auto e=(x); if(e!=cudaSuccess) {fprintf(stderr,"CUDA: %s line %d\n",cudaGetErrorString(e),__LINE__); exit(2);}}while(0)
''' + arrays + r'''
template <class T, size_t S> T* dev(const T (&x)[S]) {T* p; CHECK(cudaMalloc(&p,sizeof(x))); CHECK(cudaMemcpy(p,x,sizeof(x),cudaMemcpyHostToDevice)); return p;}
__global__ void nested(bool coo, const long long* wp,const long long* bp,const long long* cols,const int* child,const long long* bp1,const int* child1,const float* dv,const long long* vp,const int* rr,const int* cc,const float* cv,const float* b,float* out) {
 int tid=blockIdx.x*blockDim.x+threadIdx.x, row=tid/64, n=tid%64; if(row>=71)return;
 int w=row/16, local=row%16; float v=0;
 for(long long q=wp[w];q<wp[w+1];q++) for(long long z0=bp[q];z0<bp[q+1];z0++) for(long long z=bp1[z0];z<bp1[z0+1];z++) {
   int start=4*child[z0]+2*child1[z];
   if(!coo) {for(int j=0;j<2;j++) {long long k=cols[q*8+start+j]; if(k>=0) v+=dv[(z*16+local)*2+j]*b[k*64+n];}}
   else {for(long long t=vp[z];t<vp[z+1];t++) if(rr[t]==local) {long long k=cols[q*8+start+cc[t]]; v+=cv[t]*b[k*64+n];}}
 }
 out[tid]=v;
}
int main(){
 auto dwp=dev(wp); auto dbp=dev(bp); auto dcols=dev(cols); auto dchild=dev(child); auto dbp1=dev(bp1); auto dchild1=dev(child1); auto ddv=dev(dv); auto dvp=dev(vp); auto drr=dev(rr); auto dcc=dev(cc); auto dcv=dev(cv); auto db=dev(B);
 float *out; CHECK(cudaMalloc(&out,71*64*sizeof(float))); std::vector<float> h(71*64);
 for(int mode=0;mode<2;mode++) {
  for(int i=0;i<5;i++) nested<<<36,128>>>(mode,dwp,dbp,dcols,dchild,dbp1,dchild1,ddv,dvp,drr,dcc,dcv,db,out);
  CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize()); CHECK(cudaMemcpy(h.data(),out,h.size()*sizeof(float),cudaMemcpyDeviceToHost));
  float maxerr=0; for(size_t i=0;i<h.size();i++) {float err=fabs(h[i]-expected[i]); maxerr=fmax(maxerr,err); if(err>2e-4f*(1+fabs(expected[i]))) {fprintf(stderr,"mismatch mode=%d i=%zu got=%g expected=%g\n",mode,i,h[i],expected[i]); return 3;}}
  cudaEvent_t x,y; CHECK(cudaEventCreate(&x)); CHECK(cudaEventCreate(&y)); CHECK(cudaEventRecord(x));
  for(int i=0;i<200;i++) nested<<<36,128>>>(mode,dwp,dbp,dcols,dchild,dbp1,dchild1,ddv,dvp,drr,dcc,dcv,db,out);
  CHECK(cudaEventRecord(y)); CHECK(cudaEventSynchronize(y)); float ms; CHECK(cudaEventElapsedTime(&ms,x,y));
  printf("leaf=%s correctness=PASS max_abs=%g prototype_ms=%g\n",mode?"coo":"dense",maxerr,ms/200); CHECK(cudaEventDestroy(x)); CHECK(cudaEventDestroy(y));
 }
}
'''
    path = args.out / 'multilevel_k_abi_probe.cu'; path.write_text(source)
    exe = args.out / 'multilevel_k_abi_probe'
    subprocess.run([args.nvcc, '-O3', '-std=c++17', '-arch='+args.arch, str(path), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)

if __name__ == '__main__': main()

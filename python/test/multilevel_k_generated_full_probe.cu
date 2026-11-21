// Experimental scalar CUDA ABI probe. This is NOT an NVIR-generated Sparsene kernel.
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#define CHECK(x) do {auto cuda_result=(x);if(cuda_result!=cudaSuccess){fprintf(stderr,"CUDA %s line %d\n",cudaGetErrorString(cuda_result),__LINE__);exit(2);}}while(0)
#define SP(x) do {auto e=(x);if(e!=CUSPARSE_STATUS_SUCCESS){fprintf(stderr,"cuSPARSE status %d line %d\n",int(e),__LINE__);exit(3);}}while(0)
template<typename T> T* read_device(const std::string& file,size_t expected) {
 std::ifstream f(file,std::ios::binary|std::ios::ate);if(!f){fprintf(stderr,"missing %s\n",file.c_str());exit(4);} auto size=f.tellg();
 if(size<0 || size_t(size)!=expected*sizeof(T)){fprintf(stderr,"bad bytes %s: %lld expected %zu\n",file.c_str(),(long long)size,expected*sizeof(T));exit(5);}
 std::vector<T> h(expected);f.seekg(0);f.read(reinterpret_cast<char*>(h.data()),size);if(!f){fprintf(stderr,"read error\n");exit(6);} T* p;CHECK(cudaMalloc(&p,size));CHECK(cudaMemcpy(p,h.data(),size,cudaMemcpyHostToDevice));return p;
}
__global__ void initB(float* B,long long count,int n) {for(long long i=blockIdx.x*(long long)blockDim.x+threadIdx.x;i<count;i+=blockDim.x*(long long)gridDim.x) B[i]=float(int((i/n*17+i%n*13)%97)-48)/49.0f;}
#include "multilevel_k_generated.inc"
__global__ void compare(const float* C,const float* R,long long n,unsigned long long* mismatch,unsigned int* maxbits) {
 for(long long i=blockIdx.x*(long long)blockDim.x+threadIdx.x;i<n;i+=blockDim.x*(long long)gridDim.x) {
  float err=fabsf(C[i]-R[i]); if(!isfinite(err) || err>1e-3f*(1+fabsf(R[i]))) atomicAdd(mismatch,1ULL);
  if(isfinite(err)) atomicMax(maxbits,__float_as_uint(err));
 }
}
int main(int argc,char** argv) {
 if(argc!=5){fprintf(stderr,"usage: %s DATA_DIR dense|coo N repeats\n",argv[0]);return 1;}
 std::string base=argv[1],leaf=argv[2],dir=base+"/"+leaf;bool coo=leaf=="coo";if(!coo&&leaf!="dense")return 1;
 if (coo != (GENERATED_IS_COO != 0)) {fprintf(stderr,"leaf/code mismatch\n");return 1;}
 int N=atoi(argv[3]),repeat=atoi(argv[4]);if(N<=0||repeat<=0)return 1;
 constexpr int M=1000005,K=1000005,nnz=3105536,roots=230471,children=411104,windows=(M+15)/16;
 auto wp=read_device<int64_t>(dir+"/window_root_ptr.bin",windows+1);auto cols=read_device<int64_t>(dir+"/root_column_ids.bin",roots*8);
 auto bp=read_device<int64_t>(dir+"/child_ptr0.bin",roots+1);auto child=read_device<int32_t>(dir+"/child_index0.bin",children);
 auto val=read_device<float>(dir+"/values.bin",coo?nnz:children*16*4);int64_t* vp=nullptr;int32_t *rr=nullptr,*cc=nullptr;
 if(coo){vp=read_device<int64_t>(dir+"/value_ptr.bin",children+1);rr=read_device<int32_t>(dir+"/local_rows.bin",nnz);cc=read_device<int32_t>(dir+"/local_cols.bin",nnz);}
 auto row=read_device<int32_t>(base+"/csr_ptr.bin",M+1);auto col=read_device<int32_t>(base+"/csr_col.bin",nnz);auto av=read_device<float>(base+"/csr_val.bin",nnz);
 float *B,*C,*R;long long total=(long long)M*N;CHECK(cudaMalloc(&B,(long long)K*N*sizeof(float)));CHECK(cudaMalloc(&C,total*sizeof(float)));CHECK(cudaMalloc(&R,total*sizeof(float)));
 initB<<<4096,256>>>(B,(long long)K*N,N);CHECK(cudaGetLastError());
 cusparseHandle_t h;cusparseSpMatDescr_t A;cusparseDnMatDescr_t BD,RD;SP(cusparseCreate(&h));
 SP(cusparseCreateCsr(&A,M,K,nnz,row,col,av,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
 SP(cusparseCreateDnMat(&BD,K,N,N,B,CUDA_R_32F,CUSPARSE_ORDER_ROW));SP(cusparseCreateDnMat(&RD,M,N,N,R,CUDA_R_32F,CUSPARSE_ORDER_ROW));
 float alpha=1,beta=0;size_t bytes=0;SP(cusparseSpMM_bufferSize(h,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,A,BD,&beta,RD,CUDA_R_32F,CUSPARSE_SPMM_ALG_DEFAULT,&bytes));
 void* temp=nullptr;if(bytes)CHECK(cudaMalloc(&temp,bytes));
 auto reference=[&](){SP(cusparseSpMM(h,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,A,BD,&beta,RD,CUDA_R_32F,CUSPARSE_SPMM_ALG_DEFAULT,temp));};
 int grid=int((total+255)/256);if(grid>65535)grid=65535;
 auto candidate=[&](){sparsene_experimental_multilevel_k<<<grid,256>>>(M,N,wp,cols,bp,child,val,vp,rr,cc,B,C);};
 reference();candidate();CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 unsigned long long *dm,hm;unsigned int *de,he;CHECK(cudaMalloc(&dm,sizeof(hm)));CHECK(cudaMalloc(&de,sizeof(he)));CHECK(cudaMemset(dm,0,sizeof(hm)));CHECK(cudaMemset(de,0,sizeof(he)));
 compare<<<4096,256>>>(C,R,total,dm,de);CHECK(cudaGetLastError());CHECK(cudaMemcpy(&hm,dm,sizeof(hm),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(&he,de,sizeof(he),cudaMemcpyDeviceToHost));float maxerr;memcpy(&maxerr,&he,sizeof(float));
 cudaEvent_t s,e;CHECK(cudaEventCreate(&s));CHECK(cudaEventCreate(&e));
 CHECK(cudaEventRecord(s));for(int i=0;i<repeat;i++)candidate();CHECK(cudaEventRecord(e));CHECK(cudaEventSynchronize(e));float cms;CHECK(cudaEventElapsedTime(&cms,s,e));
 CHECK(cudaEventRecord(s));for(int i=0;i<repeat;i++)reference();CHECK(cudaEventRecord(e));CHECK(cudaEventSynchronize(e));float rms;CHECK(cudaEventElapsedTime(&rms,s,e));
 printf("{\"dataset\":\"webbase-1M\",\"m\":%d,\"k\":%d,\"n\":%d,\"nnz\":%d,\"leaf\":\"%s\",\"mismatches\":%llu,\"max_abs_error\":%.9g,\"generated_experimental_scalar_ms\":%.9g,\"cusparse_ms\":%.9g,\"repeats\":%d,\"nvir_generated\":false}\n",M,K,N,nnz,leaf.c_str(),hm,maxerr,cms/repeat,rms/repeat,repeat);
 return hm?7:0;
}

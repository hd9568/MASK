// Experimental scalar CUDA ABI probe. This is NOT an NVIR-generated Sparsene kernel.
#include <cuda_runtime.h>
#include <mma.h>
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
// Independent correctness probe; not NVIR-generated or tuned.
__global__ void compute_tc(int M,int N,const int64_t* wp,const int64_t* bp,const int64_t* cols,const float* val,const int64_t* sp,const int32_t* si,const float* B,float* C) {
 namespace wmma=nvcuda::wmma;
 int w=blockIdx.x, warp=threadIdx.x/32, nbase=blockIdx.y*64+warp*16, lane=threadIdx.x%32;
 __shared__ __align__(32) float Aall[16*8]; float* A=Aall;
 __shared__ __align__(32) float BTall[4*8*16]; float* BT=BTall+warp*8*16;
 __shared__ __align__(32) float OUTall[4*16*16]; float* OUT=OUTall+warp*16*16;
 wmma::fragment<wmma::accumulator,16,16,8,float> acc;
 wmma::fill_fragment(acc,0.0f);
 for(int64_t q=wp[w];q<wp[w+1];q++) {
#include "nvir_stage.inc"
  wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> a;
  wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> b;
  wmma::load_matrix_sync(a,A,8); wmma::load_matrix_sync(b,BT,16);
  wmma::mma_sync(acc,a,b,acc); __syncthreads();
 }
 wmma::store_matrix_sync(OUT,acc,16,wmma::mem_row_major); __syncwarp();
 for(int x=lane;x<256;x+=32) {int row=w*16+x/16,n=nbase+x%16; if(row<M&&n<N) C[(int64_t)row*N+n]=OUT[x];}
}
__global__ void compare(const float* C,const float* R,long long n,unsigned long long* mismatch,unsigned int* maxbits,unsigned int* maxref,double* err2,double* ref2) {
 __shared__ double ebuf[256],rbuf[256]; double e=0,r=0;
 for(long long i=blockIdx.x*(long long)blockDim.x+threadIdx.x;i<n;i+=blockDim.x*(long long)gridDim.x) {
  float err=fabsf(C[i]-R[i]),ref=fabsf(R[i]); if(!isfinite(err) || err>1e-3f*(1+ref)) atomicAdd(mismatch,1ULL);
  if(isfinite(err)&&isfinite(ref)) {atomicMax(maxbits,__float_as_uint(err)); atomicMax(maxref,__float_as_uint(ref)); e+=double(err)*err;r+=double(ref)*ref;} else {e=INFINITY;}
 }
 ebuf[threadIdx.x]=e;rbuf[threadIdx.x]=r;__syncthreads();
 for(int step=128;step>0;step/=2) {if(threadIdx.x<step){ebuf[threadIdx.x]+=ebuf[threadIdx.x+step];rbuf[threadIdx.x]+=rbuf[threadIdx.x+step];}__syncthreads();}
 if(threadIdx.x==0){atomicAdd(err2,ebuf[0]);atomicAdd(ref2,rbuf[0]);}
}
int main(int argc,char** argv) {
 if(argc!=5){fprintf(stderr,"usage: %s DATA_DIR dense|coo N repeats\n",argv[0]);return 1;}
 std::string base=argv[1],leaf=argv[2],dir=base+"/"+leaf;bool coo=leaf=="coo";if(!coo&&leaf!="dense")return 1;
 if(coo) {fprintf(stderr,"dense only\n");return 1;}
 int N=atoi(argv[3]),repeat=atoi(argv[4]);if(N<=0||repeat<=0)return 1;
 constexpr int M=1000005,K=1000005,nnz=3105536,roots=230471,children=411104,windows=(M+15)/16;
 auto wp=read_device<int64_t>(dir+"/window_root_ptr.bin",windows+1);auto cols=read_device<int64_t>(dir+"/root_column_ids.bin",roots*8);
 auto bp=read_device<int64_t>(dir+"/child_ptr0.bin",roots+1);auto child=read_device<int32_t>(dir+"/child_index0.bin",children);
 auto sp=read_device<int64_t>(dir+"/parent_scalar_ptr.bin",roots+1); auto si=read_device<int32_t>(dir+"/scalar_parent_positions.bin",1511172);
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
 auto candidate=[&](){compute_tc<<<dim3(windows,(N+63)/64),128>>>(M,N,wp,bp,cols,val,sp,si,B,C);};
 reference();candidate();CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 unsigned long long *dm,hm;unsigned int *de,he,*dr,hr; double *dE,*dR,hE,hR;
 CHECK(cudaMalloc(&dm,sizeof(hm)));CHECK(cudaMalloc(&de,sizeof(he)));CHECK(cudaMalloc(&dr,sizeof(hr)));CHECK(cudaMalloc(&dE,sizeof(hE)));CHECK(cudaMalloc(&dR,sizeof(hR)));
 CHECK(cudaMemset(dm,0,sizeof(hm)));CHECK(cudaMemset(de,0,sizeof(he)));CHECK(cudaMemset(dr,0,sizeof(hr)));CHECK(cudaMemset(dE,0,sizeof(hE)));CHECK(cudaMemset(dR,0,sizeof(hR)));
 compare<<<4096,256>>>(C,R,total,dm,de,dr,dE,dR);CHECK(cudaGetLastError());CHECK(cudaMemcpy(&hm,dm,sizeof(hm),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(&he,de,sizeof(he),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(&hr,dr,sizeof(hr),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(&hE,dE,sizeof(hE),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(&hR,dR,sizeof(hR),cudaMemcpyDeviceToHost));float maxerr,maxref;memcpy(&maxerr,&he,sizeof(float));memcpy(&maxref,&hr,sizeof(float));double relative_l2=sqrt(hE/(hR>1e-30?hR:1e-30)), normalized=maxerr/fmaxf(1e-12f,maxref);
 cudaEvent_t s,e;CHECK(cudaEventCreate(&s));CHECK(cudaEventCreate(&e));
 CHECK(cudaEventRecord(s));for(int i=0;i<repeat;i++)candidate();CHECK(cudaEventRecord(e));CHECK(cudaEventSynchronize(e));float cms;CHECK(cudaEventElapsedTime(&cms,s,e));
 CHECK(cudaEventRecord(s));for(int i=0;i<repeat;i++)reference();CHECK(cudaEventRecord(e));CHECK(cudaEventSynchronize(e));float rms;CHECK(cudaEventElapsedTime(&rms,s,e));
 printf("{\"dataset\":\"webbase-1M\",\"m\":%d,\"k\":%d,\"n\":%d,\"nnz\":%d,\"leaf\":\"%s\",\"strict_1e3_mismatches\":%llu,\"max_abs_error\":%.9g,\"relative_l2\":%.9g,\"max_abs_normalized\":%.9g,\"experimental_tf32_wmma_ms\":%.9g,\"cusparse_ms\":%.9g,\"repeats\":%d,\"nvir_generated\":false}\n",M,K,N,nnz,leaf.c_str(),hm,maxerr,relative_l2,normalized,cms/repeat,rms/repeat,repeat);
 return (relative_l2<5e-4 && normalized<5e-3)?0:7;
}

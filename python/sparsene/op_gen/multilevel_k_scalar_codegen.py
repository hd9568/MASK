"""Experimental scalar code emission for the strict multilevel-K host plan.

This path exists to verify the metadata ABI on GPUs; it is not NVIR lowering,
not the Tensor-Core implementation, and is not selected by production code.
"""
from __future__ import annotations
from .multilevel_k_lowering import MultiLevelKPlan


def emit_scalar_cuda(plan: MultiLevelKPlan) -> str:
    if plan.backend_status != 'host_reference_only' or plan.leaf not in ('dense', 'coo'):
        raise ValueError('invalid experimental scalar plan')
    if len(plan.widths) < 2 or any(a % b for a,b in zip(plan.widths,plan.widths[1:])):
        raise ValueError('plan must have at least one divisible child level')
    args = []
    for d in range(len(plan.widths)-1):
        args.extend([f'const int64_t* ptr{d}',f'const int32_t* idx{d}'])
    args.extend(['const float* val'])
    args.extend(['const int64_t* val_ptr','const int32_t* rows','const int32_t* columns'])
    args.extend(['const float* B','float* C'])
    signature = ', '.join(['int M','int N','const int64_t* windows','const int64_t* root_cols',*args])
    lines=['// Experimental CUDA-core metadata validator; not NVIR or Tensor-Core code.',
           '#include <cstdint>', '#include <cmath>',
           f'extern "C" __global__ void sparsene_experimental_multilevel_k({signature}) {{',
           '  for (int64_t t=blockIdx.x*(int64_t)blockDim.x+threadIdx.x; t<(int64_t)M*N; t+=blockDim.x*(int64_t)gridDim.x) {',
           f'    int row=t/N, n=t%N, w=row/{plan.blk_m}, lr=row%{plan.blk_m};',
           '    float sum=0.0f;',
           '    for(int64_t root=windows[w];root<windows[w+1];++root) {']
    indent='      '; parent='root'
    for d in range(len(plan.widths)-1):
        node=f'child{d}'
        lines.append(f'{indent}for(int64_t {node}=ptr{d}[{parent}];{node}<ptr{d}[{parent}+1];++{node}) {{')
        indent+='  '; parent=node
    offset='+'.join(f'idx{d}[child{d}]*{plan.widths[d+1]}' for d in range(len(plan.widths)-1))
    width=plan.widths[-1]
    if plan.leaf=='dense':
        lines.extend([f'{indent}for(int j=0;j<{width};++j) {{',
                      f'{indent}  int64_t k=root_cols[root*{plan.widths[0]}+{offset}+j];',
                      f'{indent}  if(k>=0) sum=fmaf(val[({parent}*{plan.blk_m}+lr)*{width}+j],B[k*(int64_t)N+n],sum);',
                      f'{indent}}}'])
    else:
        lines.extend([f'{indent}for(int64_t j=val_ptr[{parent}];j<val_ptr[{parent}+1];++j) if(rows[j]==lr) {{',
                      f'{indent}  int64_t k=root_cols[root*{plan.widths[0]}+{offset}+columns[j]];',
                      f'{indent}  sum=fmaf(val[j],B[k*(int64_t)N+n],sum);',
                      f'{indent}}}'])
    for _ in plan.widths[1:]:
        indent=indent[:-2];lines.append(indent+'}')
    lines.extend(['    }','    C[t]=sum;','  }','}'])
    return '\n'.join(lines)+'\n'

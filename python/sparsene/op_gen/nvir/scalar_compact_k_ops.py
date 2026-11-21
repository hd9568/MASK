"""Experimental NVIR nodes for a scalar-compacted K4 host ABI.

This module builds explicit dependency nodes for the handwritten WMMA probe.
It does not integrate with OPIR-to-NVIR lowering or the production scheduler.
"""
from __future__ import annotations
from .nvop import NvOp,NvOpImpl,NvOpInput,NvOpOutput,NvOpTensor,Shape


def _tensor(dtype,source=None,mem='rmem'):
    return NvOpTensor(Shape(),mem,dtype,source=source)


def _input(name,source,dtype='int64'):
    return NvOpInput(0,name,_tensor(dtype,source))


def _inputs(*items):
    result = [_input(*item) for item in items]
    for idx, item in enumerate(result): item.idx = idx
    return result


def build_scalar_compact_k_stage() -> tuple[NvOp,...]:
    """Build and link parent bounds and dependent 16x8 A/B staging nodes.

    The ABI has scalar-position indirection per parent and two compact K4
    leaves per K8 parent. Callers must provide the named live CUDA variables,
    allocate warp-private A and BT, and execute a warp barrier after staging.
    """
    begin=NvOpOutput(0,'scalar_begin',_tensor('int64'))
    count=NvOpOutput(1,'scalar_count',_tensor('int32'))
    bounds=NvOp('ScalarCompactParentBounds',_inputs(('sp','sp'),('q','q')),
        [begin,count],NvOpImpl('''int64_t scalar_begin=sp[q];
int scalar_count=int(sp[q+1]-scalar_begin);'''),mem_type=('g','r'))
    a_ready=NvOpOutput(0,'stage_a_ready',_tensor('float',mem='smem'))
    a=NvOp('ScalarCompactStageA',_inputs(('scalar_count',count,'int32'),('bp','bp'),('val','val','float'),('q','q'),('lane','lane'),('A','A','float')),
       [a_ready],NvOpImpl('''for(int x=lane;x<128;x+=32) {
 int r=x/8,k=x%8; float v=0;
 if(k<scalar_count) {int64_t tile=bp[q]+k/4; v=val[(tile*16+r)*4+k%4];}
 A[x]=wmma::__float_to_tf32(v);
}'''),mem_type=('g','s'))
    b_ready=NvOpOutput(0,'stage_b_ready',_tensor('float',mem='smem'))
    b=NvOp('ScalarCompactStageB',_inputs(('scalar_begin',begin),('scalar_count',count,'int32'),
        ('si','si','int32'),('cols','cols'),('B','B','float'),('q','q'),('lane','lane'),('nbase','nbase'),('N','N'),('BT','BT','float')),
        [b_ready],NvOpImpl('''for(int x=lane;x<128;x+=32) {
 int k=x/16,n=x%16+nbase; float v=0;
 if(k<scalar_count && n<N) {int32_t pos=si[scalar_begin+k]; int64_t column=cols[q*8+pos]; if(column>=0) v=B[column*(int64_t)N+n];}
 BT[x]=wmma::__float_to_tf32(v);
}'''),mem_type=('g','s'))
    sync=NvOp('ScalarCompactStageBarrier',_inputs(('stage_a_ready',a_ready,'float'),('stage_b_ready',b_ready,'float')),
              [],NvOpImpl('__syncwarp();'),mem_type=('s','s'))
    return bounds,a,b,sync


def emit_scalar_compact_k_stage() -> str:
    return '\n'.join(op.impl.nv_src() for op in build_scalar_compact_k_stage())+'\n'


def lower_scalar_compact_to_nvir(plan):
    """Construct the experimental stage only for the exact validated ABI.

    This is not wired to the production compiler driver: the caller remains
    responsible for loop scheduling, allocation, launch, and MMA emission.
    """
    from sparsene.op_gen.scalar_compact_k_lowering import ScalarCompactKPlan
    if not isinstance(plan, ScalarCompactKPlan):
        raise ValueError('requires a validated four-level scalar-compact plan')
    if (plan.blk_m, plan.blk_k, plan.blk_k_i) != (16, 8, 4):
        raise ValueError('experimental NVIR stage requires logical 16x8 with K4 leaves')
    return build_scalar_compact_k_stage()

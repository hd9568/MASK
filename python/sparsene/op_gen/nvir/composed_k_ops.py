"""Opt-in NVIR stage for host-composed nested sparse columns.

Standalone staging graph, not yet connected to OPIR or production runner.
Input `composed` is int64[leaf,4] with -1 sentinels; parent tile offsets
are int64. It is valid only when padded A values are initialized to zero.
"""
from __future__ import annotations
from .nvop import NvOp,NvOpImpl,NvOpInput,NvOpOutput,NvOpTensor,Shape


def _tensor(dtype,source=None,mem='rmem'):
    return NvOpTensor(Shape(),mem,dtype,source=source)

def _inputs(*items):
    return [NvOpInput(i,name,_tensor(dtype,src)) for i,(name,src,dtype) in enumerate(items)]

def build_composed_k_stage(*, scalar_offsets: bool = False, share_a: bool = False):
    left=NvOpOutput(0,'tile_begin',_tensor('int64'))
    count=NvOpOutput(1,'padded_count',_tensor('int32'))
    bounds_code = ('''int64_t tile_begin=bp[q]/4;
int padded_count=int(bp[q+1]-bp[q]);''' if scalar_offsets else '''int64_t tile_begin=bp[q];
int padded_count=int((bp[q+1]-tile_begin)*4);''')
    bounds=NvOp('ComposedKParentBounds',_inputs(('bp','bp','int32' if scalar_offsets else 'int64'),('q','q','int64')),[left,count],NvOpImpl(bounds_code),mem_type=('g','r'))
    aout=NvOpOutput(0,'stage_a_ready',_tensor('float',mem='smem'))
    a=NvOp('ComposedKStageA',_inputs(('tile_begin',left,'int64'),('padded_count',count,'int32'),('val','val','float'),('lane','lane','int32'),('A','A','float')),[aout],NvOpImpl('''for(int x=lane;x<128;x+=32) {
 int r=x/8,k=x%8; float v=0;
 if(k<padded_count) {int64_t tile=tile_begin+k/4; v=val[(tile*16+r)*4+k%4];}
 A[x]=wmma::__float_to_tf32(v);
}'''),mem_type=('g','s'))
    bout=NvOpOutput(0,'stage_b_ready',_tensor('float',mem='smem'))
    b=NvOp('ComposedKStageB',_inputs(('tile_begin',left,'int64'),('padded_count',count,'int32'),('composed','composed','int32' if scalar_offsets else 'int64'),('B','B','float'),('lane','lane','int32'),('nbase','nbase','int32'),('N','N','int32'),('BT','BT','float')),[bout],NvOpImpl('''for(int x=lane;x<128;x+=32) {
 int k=x/16,n=x%16+nbase; float v=0;
 if(k<padded_count && n<N) {int64_t column=composed[(tile_begin+k/4)*4+k%4]; if(column>=0) v=B[column*(int64_t)N+n];}
 BT[x]=wmma::__float_to_tf32(v);
}'''),mem_type=('g','s'))
    if share_a:
        # Four warps share A. CTA synchronization is required both before
        # consumption and after MMA, before the next parent overwrites A.
        a.impl = NvOpImpl('if(warp==0) {\n' + a.impl.nv_src() + '\n}')
        a.add_input(NvOpInput(len(a.inputs), 'warp', _tensor('int32', 'warp')))
    sync=NvOp('ComposedKStageBarrier',_inputs(('stage_a_ready',aout,'float'),('stage_b_ready',bout,'float')),[],NvOpImpl('__syncthreads();' if share_a else '__syncwarp();'),mem_type=('s','s'))
    return bounds,a,b,sync


def schedule_composed_k_stage(*, scalar_offsets: bool = True, share_a: bool = True):
    """Use the existing stage partitioner; caller owns the outer loop/MMA.

    For shared A, a second CTA barrier after MMA is an explicit requirement
    of the surrounding kernel, which this staging-only routine cannot check.
    """
    from .compiler_driver import _partition_pipeline_stages
    from .nvop import ForLoopNvOp, NvOpSequence
    nodes=build_composed_k_stage(scalar_offsets=scalar_offsets,share_a=share_a)
    left=NvOpInput(0,'l',_tensor('int32','zero'))
    right=NvOpInput(1,'r',_tensor('int32','count'))
    loop=ForLoopNvOp('composed_k_parent_loop',left,right,body=NvOpSequence(*nodes))
    stages=_partition_pipeline_stages(loop)
    return stages


def emit_scheduled_composed_k_stage(*, scalar_offsets: bool = True, share_a: bool = True):
    stages=schedule_composed_k_stage(scalar_offsets=scalar_offsets,share_a=share_a)
    if len(stages)!=1:
        raise RuntimeError('staging-only emitter requires a single synchronous NVIR stage')
    return '\n'.join(op.impl.nv_src() for op in stages[0])+'\n'

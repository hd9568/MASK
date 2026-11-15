"""NVIR contracts for the strictly matched four-level scalar compact backend.

The supported four-level dense leaf has logical K in [4,32].
The physical K is padded to 8, 16 or 32 for the swizzled staging layout.
"""
from __future__ import annotations
from copy import deepcopy
from .generate import SPARSE_FORMAT_REGISTRY


def build_scalar_compact_candidate_hints():
    hints=deepcopy(SPARSE_FORMAT_REGISTRY['SR_BCRS'])
    hints['shape_overrides'].update({
        'val_len_offset':['Mo'],
        'val_len_1_new_offset':['ParentCount'],
        'val_sidx_new_1':['BLK_K_I','nnz_dim_BLK_K'],
        'val_sidx_1_new_1':['BLK_K_I','nnz_dim_BLK_K'],
        'val_new_1':['nnz_dim_BLK_K','BLK_K_I','BLK_M'],
    })
    hints['custom_layouts']['val_new_1']={
        'dtype':'float',
        'tensor_str':'make_tensor(make_gmem_ptr(d{name}), make_layout(make_shape(nnz_dim_BLK_K, BLK_K_I{}, get<0>(BLK_MNK{})), make_stride(get<0>(BLK_MNK{}) * BLK_K_I{}, _1{}, BLK_K_I{})))',
        'tensor_type_str':'decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_layout(make_shape(VARLEN, BLK_K_I{}, get<0>(BLK_MNK{})), make_stride(get<0>(BLK_MNK{}) * BLK_K_I{}, _1{}, BLK_K_I{}))))',
    }
    return hints


def pad_candidate_k4_for_mma(program, inner_k=4, tf32x3=False):
    """Stage an inner leaf to K8-aligned fragments without changing its ABI."""
    if not 4 <= inner_k <= 32:
        raise ValueError("unsupported scalar compact inner K")
    physical_k = 1 << (max(8, inner_k) - 1).bit_length()
    from .nvop import ForLoopNvOp, IntShape, Shape, NvOpImpl
    found = {}
    def visit(ops):
        for op in ops:
            if op.name in ('load', 'array_ref_1', 'S2rAValLoadOp', 'S2rBValLoadOp', 'mma'):
                if op.name in found:
                    raise ValueError('ambiguous candidate NVIR operation: ' + op.name)
                found[op.name] = op
            if isinstance(op, ForLoopNvOp):
                visit(op.body.ops)
    visit(program.ops)
    if set(found) != {'load', 'array_ref_1', 'S2rAValLoadOp', 'S2rBValLoadOp', 'mma'}:
        raise ValueError('unexpected candidate NVIR graph: ' + str(sorted(found)))
    a, b = found['load'], found['array_ref_1']
    a.outputs[0].tensor.shape = Shape(a.outputs[0].tensor.shape[0], IntShape(physical_k))
    # ldmatrix.x2 has address operands from 32 lanes; allocate 16 rows even
    # though only the first 8 contain the K8 tile.
    b.outputs[0].tensor.shape = Shape(IntShape(physical_k + 8), b.outputs[0].tensor.shape[1])
    found['S2rAValLoadOp'].inputs[0].tensor.shape = a.outputs[0].tensor.shape
    found['S2rBValLoadOp'].inputs[0].tensor.shape = b.outputs[0].tensor.shape
    a.impl = NvOpImpl('''for (int i = lid; i < get<0>(shape(val_tile)) * (get<2>(BLK_MNK{}) / 4); i += 32) {
    int row = i / (get<2>(BLK_MNK{}) / 4), group = i % (get<2>(BLK_MNK{}) / 4);
    if constexpr (BLK_K_I::value % 4 == 0) {
        float4 v = group * 4 < BLK_K_I{} ? *reinterpret_cast<const float4*>(&operand_0(operand_1, group * 4, row)) : make_float4(0, 0, 0, 0);
        *reinterpret_cast<float4*>(&val_tile(row, group * 4, buf_idx)) = v;
    } else {
        for (int j = 0; j < 4; ++j) {
            int k = group * 4 + j;
            val_tile(row, k, buf_idx) = k < BLK_K_I{} ? operand_0(operand_1, k, row) : 0.0f;
        }
    }
}''')
    b.impl = NvOpImpl('''for (int i = lid; i < get<2>(BLK_MNK{}) * get<1>(shape(result_0)) / 4; i += 32) {
    int k = i / (get<1>(shape(result_0)) / 4), n = (i % (get<1>(shape(result_0)) / 4)) * 4;
    float4 v = k < BLK_K_I{} ? *reinterpret_cast<const float4*>(&operand_0(operand_1(k), n)) : make_float4(0, 0, 0, 0);
    *reinterpret_cast<float4*>(&result_0(k, n, buf_idx)) = v;
}''')
    for op in (a,b):
        op.attrs['cp_async'] = False
        op.outputs[0].attrs['cp_async'] = False
        # The vectorized staging uses float4 even though it is synchronous.
        op.outputs[0].attrs['align_16'] = True
    if tf32x3:
        found['mma'].impl = NvOpImpl('''for (int ki=0; ki<get<2>(BLK_MMA_MNK{}); ++ki) {
    for (int mi=0; mi<get<0>(BLK_MMA_MNK{}); ++mi) {
        for (int ni=0; ni<get<1>(BLK_MMA_MNK{}); ++ni) {
            uint32_t ah[4], al[4], bh[2], bl[2];
            for (int j=0; j<4; ++j) {
                float x=operand_0(j, make_coord(mi,ki));
                asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(ah[j]) : "f"(x));
                float residual=x-__uint_as_float(ah[j]);
                asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(al[j]) : "f"(residual));
            }
            for (int j=0; j<2; ++j) {
                float x=operand_1(j, make_coord(ki,ni));
                asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bh[j]) : "f"(x));
                float residual=x-__uint_as_float(bh[j]);
                asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bl[j]) : "f"(residual));
            }
            auto c=C_val_tile(_, make_coord(mi,ni)).data();
            mma_m16n8k8_fp32_tf32_tf32_fp32(c, ah, bh);
            mma_m16n8k8_fp32_tf32_tf32_fp32(c, al, bh);
            mma_m16n8k8_fp32_tf32_tf32_fp32(c, ah, bl);
        }
    }
}''')
    program.attrs['experimental_k4_padded_staging'] = True

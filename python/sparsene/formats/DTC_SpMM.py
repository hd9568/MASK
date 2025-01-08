from collections import OrderedDict

from sparsene.model.stage import Op, SourceTemplate

OP_S_SRC_BODY = """
    const int smem_st_ptr = i % ({{SFT.0}} + 1);

    float4* src = (float4*)(&{{G.A_TCFValue}}[({{SID.TCStart_offset}} + i) * 16 * 16 + {{HID.lane_id}} * 8]);
    float4* dst = (float4*)(&{{S.tileA_block}}[smem_st_ptr * {{C.tileA_size}} + {{HID.lane_id}} * 8]);
    __pipeline_memcpy_async(dst, src, sizeof(float4));
    if ({{HID.lane_id}} < 4) {
        int4* src_idx = (int4*)(&{{G.A_TCFColIdx}}[({{SID.TCStart_offset}} + i) * 16 + {{HID.lane_id}} * 4]);
        int4* dst_idx = (int4*)(&{{S.sparse_index}}[smem_st_ptr * 16 + {{HID.lane_id}} * 4]);
        __pipeline_memcpy_async(dst_idx, src_idx, sizeof(int4));
    }
    __pipeline_commit();
"""

OP_R_SRC_BODY = """
    const int smem_ld_ptr = i % ({{SFT.0}} + 1);

    __pipeline_wait_prior({{WTN}});
    __syncthreads();

    if ({{HID.lane_id}} < 4) {
        {{R.B}}[{{RID.2}} * 4 + 0] = __ldg(&{{G.X_Val}}[{{S.sparse_index}}[smem_ld_ptr * 16 + {{HID.lane_id}} * 2 + 0]]);
        {{R.B}}[{{RID.2}} * 4 + 1] = __ldg(&{{G.X_Val}}[{{S.sparse_index}}[smem_ld_ptr * 16 + {{HID.lane_id}} * 2 + 1]]);
        {{R.B}}[{{RID.2}} * 4 + 2] = __ldg(&{{G.X_Val}}[{{S.sparse_index}}[smem_ld_ptr * 16 + {{HID.lane_id}} * 2 + 8 + 0]]);
        {{R.B}}[{{RID.2}} * 4 + 3] = __ldg(&{{G.X_Val}}[{{S.sparse_index}}[smem_ld_ptr * 16 + {{HID.lane_id}} * 2 + 8 + 1]]);
    }
    ldmatrix_m8n8k8_x4(&{{R.A}}[{{RID.2}} * 4], &{{S.tileA_block}}[smem_ld_ptr * {{C.tileA_size}} + {{HID.lane_id}} * 8]);
"""

OP_MMA_SRC_BODY = """
    mma_m16n8k16_fp32({{R.C}}, &{{R.A}}[{{RID.2}} * 4], &{{R.B}}[{{RID.2}} * 4]);
"""

OP_WB_SRC_BODY = """
    if ({{HID.lane_id}} % 4 == 0) {
        {{G.Y_Val}}[{{HID.block_idx_x}} * 16 + {{HID.lane_id}} / 4 + 0] = (half){{R.C}}[0];
        {{G.Y_Val}}[{{HID.block_idx_x}} * 16 + {{HID.lane_id}} / 4 + 8] = (half){{R.C}}[2];
    }
"""

OP_S = Op(
    name="s", type="k",
    src_template=SourceTemplate(
        sw_idx_params=OrderedDict([
            ("i", "int"),
            ("TCStart_offset", "int"),
        ]),
        const_params=OrderedDict([
            ("tileA_size", "int"),
        ]),
        gmem_params=OrderedDict([
            ("A_TCFValue", "half*"),
            ("A_TCOffset", "int*"),
            ("A_TCFColIdx", "int*"),
            ("X_Val", "half*"),
            ("Y_Val", "half*"),
        ]),
        smem_params=OrderedDict([
            ("tileA_block", "half*"),
            ("sparse_index", "int*"),
        ]),
        reg_params=OrderedDict([
            ("A", "uint32_t*"),
            ("B", "half*"),
            ("C", "float*"),
        ]),
        body=OP_S_SRC_BODY,
    ),
)

OP_R = Op(
    name="r", type="k",
    src_template=SourceTemplate(
        sw_idx_params=OrderedDict([
            ("i", "int"),
            ("TCStart_offset", "int"),
        ]),
        const_params=OrderedDict([
            ("tileA_size", "int"),
        ]),
        gmem_params=OrderedDict([
            ("A_TCFValue", "half*"),
            ("A_TCOffset", "int*"),
            ("A_TCFColIdx", "int*"),
            ("X_Val", "half*"),
            ("Y_Val", "half*"),
        ]),
        smem_params=OrderedDict([
            ("tileA_block", "half*"),
            ("sparse_index", "int*"),
        ]),
        reg_params=OrderedDict([
            ("A", "uint32_t*"),
            ("B", "half*"),
            ("C", "float*"),
        ]),
        body=OP_R_SRC_BODY,
    ),
)

OP_MMA = Op(
    name="mma", type="k",
    src_template=SourceTemplate(
        sw_idx_params=OrderedDict([
            ("i", "int"),
            ("TCStart_offset", "int"),
        ]),
        const_params=OrderedDict([
            ("tileA_size", "int"),
        ]),
        gmem_params=OrderedDict([
            ("A_TCFValue", "half*"),
            ("A_TCOffset", "int*"),
            ("A_TCFColIdx", "int*"),
            ("X_Val", "half*"),
            ("Y_Val", "half*"),
        ]),
        smem_params=OrderedDict([
            ("tileA_block", "half*"),
            ("sparse_index", "int*"),
        ]),
        reg_params=OrderedDict([
            ("A", "uint32_t*"),
            ("B", "half*"),
            ("C", "float*"),
        ]),
        body=OP_MMA_SRC_BODY,
    ),
)

OP_WB = Op(
    name="wb", type="cta",
    src_template=SourceTemplate(
        sw_idx_params=OrderedDict([
            ("i", "int"),
            ("TCStart_offset", "int"),
        ]),
        const_params=OrderedDict([
            ("tileA_size", "int"),
        ]),
        gmem_params=OrderedDict([
            ("A_TCFValue", "half*"),
            ("A_TCOffset", "int*"),
            ("A_TCFColIdx", "int*"),
            ("X_Val", "half*"),
            ("Y_Val", "half*"),
        ]),
        smem_params=OrderedDict([
            ("tileA_block", "half*"),
            ("sparse_index", "int*"),
        ]),
        reg_params=OrderedDict([
            ("A", "uint32_t*"),
            ("B", "half*"),
            ("C", "float*"),
        ]),
        body=OP_WB_SRC_BODY,
    ),
)

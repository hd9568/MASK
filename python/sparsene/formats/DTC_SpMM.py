from collections import OrderedDict

from sparsene.format.format import (
    Direction,
    Format,
    coo_atomic_format,
    df_axis,
    sv_axis,
    Expr,
    Symbol,
)
from sparsene.op_gen.nvir.op import Op
from sparsene.op_gen.nvir.param import Params, Param

DTC_SPMM_FORMAT = Format(
    name="DTC-SpMM",
    axes=[
        df_axis("X_o", Direction.ROW, length="M"),
        sv_axis("Y_s", Direction.COL, length="N"),
        df_axis("X_i", Direction.ROW, length=16),
    ],
    child=Format(
        axes=[
            df_axis("Y_so", Direction.COL, length=Symbol("N") / 8), # type: ignore
            df_axis("Y_si", Direction.COL, length=8),
            df_axis("X_i", Direction.ROW, length=16),
        ],
        child=coo_atomic_format("X_i", "Y_si"),
    ),
)


OP_S_SRC_BODY = """
    const int smem_st_ptr = {{I.i}} % ({{SFT.0}} + 1);

    float4* src = (float4*)(&{{G.A_TCFValue}}[({{I.TCStart_offset}} + {{I.i}}) * 16 * 16 + {{H.lane_id}} * 8]);
    float4* dst = (float4*)(&{{S.tileA_block}}[smem_st_ptr * {{C.tileA_size}} + {{H.lane_id}} * 8]);
    __pipeline_memcpy_async(dst, src, sizeof(float4));
    if ({{H.lane_id}} < 4) {
        int4* src_idx = (int4*)(&{{G.A_TCFColIdx}}[({{I.TCStart_offset}} + {{I.i}}) * 16 + {{H.lane_id}} * 4]);
        int4* dst_idx = (int4*)(&{{S.sparse_index}}[smem_st_ptr * 16 + {{H.lane_id}} * 4]);
        __pipeline_memcpy_async(dst_idx, src_idx, sizeof(int4));
    }
    __pipeline_commit();
"""

OP_R_SRC_BODY = """
    const int smem_ld_ptr = {{I.i}} % ({{SFT.0}} + 1);

    __pipeline_wait_prior({{WTN}});
    __syncthreads();

    if ({{H.lane_id}} < 4) {
        {{R.B}}[{{RID.2}} * 4 + 0] = __ldg(&{{G.X_Val}}[{{S.sparse_index}}[smem_ld_ptr * 16 + {{H.lane_id}} * 2 + 0]]);
        {{R.B}}[{{RID.2}} * 4 + 1] = __ldg(&{{G.X_Val}}[{{S.sparse_index}}[smem_ld_ptr * 16 + {{H.lane_id}} * 2 + 1]]);
        {{R.B}}[{{RID.2}} * 4 + 2] = __ldg(&{{G.X_Val}}[{{S.sparse_index}}[smem_ld_ptr * 16 + {{H.lane_id}} * 2 + 8 + 0]]);
        {{R.B}}[{{RID.2}} * 4 + 3] = __ldg(&{{G.X_Val}}[{{S.sparse_index}}[smem_ld_ptr * 16 + {{H.lane_id}} * 2 + 8 + 1]]);
    }
    ldmatrix_m8n8k8_x4(&{{R.A}}[{{RID.2}} * 4], &{{S.tileA_block}}[smem_ld_ptr * {{C.tileA_size}} + {{H.lane_id}} * 8]);
"""

OP_MMA_SRC_BODY = """
    mma_m16n8k16_fp32({{R.C}}, &{{R.A}}[{{RID.2}} * 4], &{{R.B}}[{{RID.2}} * 4]);
"""

OP_WB_SRC_BODY = """
    if ({{H.lane_id}} % 4 == 0) {
        {{G.Y_Val}}[{{H.block_idx_x}} * 16 + {{H.lane_id}} / 4 + 0] = (half){{R.C}}[0];
        {{G.Y_Val}}[{{H.block_idx_x}} * 16 + {{H.lane_id}} / 4 + 8] = (half){{R.C}}[2];
    }
"""

OP_S = Op(
    name="s",
    type="k",
    params=Params(
        sw_idx=OrderedDict(
            [
                ("i", Param("i", "int")),
                ("TCStart_offset", Param("TCStart_offset", "int")),
            ]
        ),
        const=OrderedDict(
            [
                ("tileA_size", Param("tileA_size", "int")),
            ]
        ),
        gmem=OrderedDict(
            [
                ("A_TCFValue", Param("A_TCFValue", "half*")),
                ("A_TCOffset", Param("A_TCOffset", "int*")),
                ("A_TCFColIdx", Param("A_TCFColIdx", "int*")),
                ("X_Val", Param("X_Val", "half*")),
                ("Y_Val", Param("Y_Val", "half*")),
            ]
        ),
        smem=OrderedDict(
            [
                ("tileA_block", Param("tileA_block", "half*")),
                ("sparse_index", Param("sparse_index", "int*")),
            ]
        ),
        reg=OrderedDict(
            [
                ("A", Param("A", "uint32_t*")),
                ("B", Param("B", "half*")),
                ("C", Param("C", "float*")),
            ]
        ),
    ),
    func_body=OP_S_SRC_BODY,
)

OP_R = Op(
    name="r",
    type="k",
    params=Params(
        sw_idx=OrderedDict(
            [
                ("i", Param("i", "int")),
                ("TCStart_offset", Param("TCStart_offset", "int")),
            ]
        ),
        const=OrderedDict(
            [
                ("tileA_size", Param("tileA_size", "int")),
            ]
        ),
        gmem=OrderedDict(
            [
                ("A_TCFValue", Param("A_TCFValue", "half*")),
                ("A_TCOffset", Param("A_TCOffset", "int*")),
                ("A_TCFColIdx", Param("A_TCFColIdx", "int*")),
                ("X_Val", Param("X_Val", "half*")),
                ("Y_Val", Param("Y_Val", "half*")),
            ]
        ),
        smem=OrderedDict(
            [
                ("tileA_block", Param("tileA_block", "half*")),
                ("sparse_index", Param("sparse_index", "int*")),
            ]
        ),
        reg=OrderedDict(
            [
                ("A", Param("A", "uint32_t*")),
                ("B", Param("B", "half*")),
                ("C", Param("C", "float*")),
            ]
        ),
    ),
    func_body=OP_R_SRC_BODY,
)

OP_MMA = Op(
    name="mma",
    type="k",
    params=Params(
        sw_idx=OrderedDict(
            [
                ("i", Param("i", "int")),
                ("TCStart_offset", Param("TCStart_offset", "int")),
            ]
        ),
        const=OrderedDict(
            [
                ("tileA_size", Param("tileA_size", "int")),
            ]
        ),
        gmem=OrderedDict(
            [
                ("A_TCFValue", Param("A_TCFValue", "half*")),
                ("A_TCOffset", Param("A_TCOffset", "int*")),
                ("A_TCFColIdx", Param("A_TCFColIdx", "int*")),
                ("X_Val", Param("X_Val", "half*")),
                ("Y_Val", Param("Y_Val", "half*")),
            ]
        ),
        smem=OrderedDict(
            [
                ("tileA_block", Param("tileA_block", "half*")),
                ("sparse_index", Param("sparse_index", "int*")),
            ]
        ),
        reg=OrderedDict(
            [
                ("A", Param("A", "uint32_t*")),
                ("B", Param("B", "half*")),
                ("C", Param("C", "float*")),
            ]
        ),
    ),
    func_body=OP_MMA_SRC_BODY,
)

OP_WB = Op(
    name="wb",
    type="cta",
    params=Params(
        sw_idx=OrderedDict(
            [
                ("i", Param("i", "int")),
                ("TCStart_offset", Param("TCStart_offset", "int")),
            ]
        ),
        const=OrderedDict(
            [
                ("tileA_size", Param("tileA_size", "int")),
            ]
        ),
        gmem=OrderedDict(
            [
                ("A_TCFValue", Param("A_TCFValue", "half*")),
                ("A_TCOffset", Param("A_TCOffset", "int*")),
                ("A_TCFColIdx", Param("A_TCFColIdx", "int*")),
                ("X_Val", Param("X_Val", "half*")),
                ("Y_Val", Param("Y_Val", "half*")),
            ]
        ),
        smem=OrderedDict(
            [
                ("tileA_block", Param("tileA_block", "half*")),
                ("sparse_index", Param("sparse_index", "int*")),
            ]
        ),
        reg=OrderedDict(
            [
                ("A", Param("A", "uint32_t*")),
                ("B", Param("B", "half*")),
                ("C", Param("C", "float*")),
            ]
        ),
    ),
    func_body=OP_WB_SRC_BODY,
)

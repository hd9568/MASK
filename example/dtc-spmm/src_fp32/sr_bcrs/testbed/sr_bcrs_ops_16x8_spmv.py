from pathlib import Path

from sparsene.op_gen.nvir.nvop import (
    NvOp,
    NvOpInput,
    NvOpOutput,
    NvOpImpl,
    IntShape,
    ParamShape,
    Shape,
    VarlenShape,
    MnkShape,
    NvOpSequence,
    GmemInout,
    NvOpTensor,
    ForLoopNvOp,
    ConstantNvOp,
    NvOpProgram,
    SwizzleLayout,
)
from sparsene.op_gen.nvir.plan import apply_pipeline, PipelinePlan

dtypeAB = "float"
dtypeC = "float"

FRAG_A = 4
FRAG_B = 2
FRAG_C = 4


def sr_bcrs():
    sr_bcrs_program = NvOpProgram(
        name="sr_bcrs_spmm_kernel_tf32",
        gmem_inouts={
            "B_val": GmemInout(
                shape=Shape(VarlenShape("K"), VarlenShape("N")),
                name="B_val",
                dtype=dtypeAB,
                tensor_str="logical_divide(make_tensor(make_gmem_ptr(dB_val), make_layout(make_shape(K, N), make_stride(N, _1{}))), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, blockIdx.y))",
                tensor_type_str="decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_layout(make_shape(VARLEN, VARLEN), make_stride(VARLEN, _1{}))), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, VARLEN)))",
            ),
            "C_val": GmemInout(
                shape=Shape(VarlenShape("N"), VarlenShape("M")),
                name="C_val",
                dtype=dtypeC,
                tensor_str="logical_divide(make_tensor(make_gmem_ptr(dC_val), make_shape(N, M)), select<1, 0>(BLK_MNK{}))(make_coord(_, blockIdx.y), make_coord(_, blockIdx.x))",
                tensor_type_str="decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN, VARLEN)), select<1, 0>(BLK_MNK{}))(make_coord(_, VARLEN), make_coord(_, VARLEN)))",
            ),
            "val_sidx": GmemInout(
                shape=Shape(MnkShape("BLK_MNK", "k"), VarlenShape("nnz_block")),
                name="val_sidx",
                dtype="int",
                tensor_str="make_tensor(make_gmem_ptr(dval_sidx), make_shape(get<2>(BLK_MNK{}), nnz_block))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(get<2>(BLK_MNK{}), VARLEN)))",
            ),
            "val_soff": GmemInout(
                shape=Shape(VarlenShape("Mo")),
                name="val_soff",
                dtype="int",
                tensor_str="make_tensor(make_gmem_ptr(dval_soff), make_shape(Mo))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)))",
            ),
            "val_block_val": GmemInout(
                shape=Shape(
                    MnkShape("BLK_MNK", "m"),
                    MnkShape("BLK_MNK", "k"),
                    VarlenShape("nnz_block"),
                ),
                name="val_block_val",
                dtype=dtypeAB,
                tensor_str="make_tensor(make_gmem_ptr(dval_block_val), make_layout(make_shape(get<0>(BLK_MNK{}), get<2>(BLK_MNK{}), VARLEN), make_shape(get<2>(BLK_MNK{}), _1{}, get<0>(BLK_MNK{}) * get<2>(BLK_MNK{}))))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_layout(make_shape(get<0>(BLK_MNK{}), get<2>(BLK_MNK{}), VARLEN), make_stride(get<2>(BLK_MNK{}), _1{}, get<0>(BLK_MNK{}) * get<2>(BLK_MNK{})))))",
            ),
        },
    )

    # dtc program body
    blk_y_loop_zero_op = ConstantNvOp(
        name="BlkYLoopZeroOp",
        shape=Shape(),
        dtype="int",
        value=0,
    )
    sr_bcrs_program.append_op(blk_y_loop_zero_op)

    BlockDimY_op = ConstantNvOp(
        name="BlockDimYOp",
        shape=Shape(),
        dtype="int",
        value="blockDim.y",
    )
    sr_bcrs_program.append_op(BlockDimY_op)

    sr_bcrs_blk_y_loop_op = ForLoopNvOp(
        name="sr_bcrs_blk_y_loop",
        blk_idx_mapping="y",
        loop_l=NvOpInput(
            idx=0,
            name="l",
            tensor=NvOpTensor(
                shape=Shape(),
                mem="rmem",
                dtype="int",
                source=blk_y_loop_zero_op.outputs[0],
            ),
        ),
        loop_r=NvOpInput(
            idx=1,
            name="r",
            tensor=NvOpTensor(
                shape=Shape(),
                mem="rmem",
                dtype="int",
                source=BlockDimY_op.outputs[0],
            ),
        ),
        loop_result=[],
        body=NvOpSequence(),
        iter_args={},
    )
    sr_bcrs_program.append_op(sr_bcrs_blk_y_loop_op)

    #! blk_y_loop body
    blk_x_loop_zero_op = ConstantNvOp(
        name="BlkXLoopZeroOp",
        shape=Shape(),
        dtype="int",
        value=0,
    )
    sr_bcrs_blk_y_loop_op.append(blk_x_loop_zero_op)

    BlockDimX_op = ConstantNvOp(
        name="BlockDimXOp",
        shape=Shape(),
        dtype="int",
        value="blockDim.x",
    )
    sr_bcrs_blk_y_loop_op.append(BlockDimX_op)

    sr_bcrs_blk_x_loop_op = ForLoopNvOp(
        name="sr_bcrs_blk_x_loop",
        blk_idx_mapping="x",
        loop_l=NvOpInput(
            idx=0,
            name="l",
            tensor=NvOpTensor(
                shape=Shape(),
                mem="rmem",
                dtype="int",
                source=blk_x_loop_zero_op.outputs[0],
            ),
        ),
        loop_r=NvOpInput(
            idx=1,
            name="r",
            tensor=NvOpTensor(
                shape=Shape(),
                mem="rmem",
                dtype="int",
                source=BlockDimX_op.outputs[0],
            ),
        ),
        loop_result=[],
        body=NvOpSequence(),
        iter_args={},
    )
    sr_bcrs_blk_y_loop_op.append(sr_bcrs_blk_x_loop_op)

    #! blk_x_loop body
    g2r_sparse_offset_load_op = NvOp(
        name="G2sSparseOffsetLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="val_soff",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("Mo")),
                    mem="gmem",
                    dtype="int",
                    source=sr_bcrs_program.gmem_tensor_ops["val_soff"].outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="i",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                    source="blockIdx.x",
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="l",
                tensor=NvOpTensor(shape=Shape(), mem="rmem", dtype="int"),
            ),
            NvOpOutput(
                idx=1,
                name="r",
                tensor=NvOpTensor(shape=Shape(), mem="rmem", dtype="int"),
            ),
        ],
        impl=NvOpImpl(
            r"""l(0) = val_soff(i);
r(0) = val_soff(i + 1);"""
        ),
        mem_type=("g", "r"),
    )
    sr_bcrs_blk_x_loop_op.append(g2r_sparse_offset_load_op)

    ZerosOp = ConstantNvOp(
        name="ZerosOp",
        shape=Shape(
            IntShape(FRAG_C),
            Shape(
                MnkShape("BLK_MMA_MNK", "m"),
                MnkShape("BLK_MMA_MNK", "n"),
            ),
        ),
        dtype=dtypeC,
        value=0,
    )
    sr_bcrs_blk_x_loop_op.append(ZerosOp)

    sr_bcrs_main_loop_op = ForLoopNvOp(
        name="SRBCRSMainLoopOp",
        loop_l=NvOpInput(
            idx=0,
            name="l",
            tensor=NvOpTensor(
                shape=Shape(),
                mem="rmem",
                dtype="int",
                source=g2r_sparse_offset_load_op.outputs[0],
            ),
        ),
        loop_r=NvOpInput(
            idx=1,
            name="r",
            tensor=NvOpTensor(
                shape=Shape(),
                mem="rmem",
                dtype="int",
                source=g2r_sparse_offset_load_op.outputs[1],
            ),
        ),
        loop_result=[
            NvOpOutput(
                idx=0,
                name="REGC",
                tensor=NvOpTensor(
                    shape=Shape(
                        IntShape(FRAG_C),
                        Shape(
                            MnkShape("BLK_MMA_MNK", "m"), MnkShape("BLK_MMA_MNK", "n")
                        ),
                    ),
                    mem="rmem",
                    dtype=dtypeC,
                ),
            ),
        ],
        body=NvOpSequence(),
        iter_args={},
    )
    sr_bcrs_blk_x_loop_op.append(sr_bcrs_main_loop_op)

    #! main loop body
    G2sSparseIndexLoadOp = NvOp(
        name="G2sSparseIndexLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="val_sidx",
                tensor=NvOpTensor(
                    shape=Shape(MnkShape("BLK_MNK", "k"), VarlenShape("nnz_block")),
                    mem="gmem",
                    dtype="int",
                    source=sr_bcrs_program.gmem_tensor_ops["val_sidx"].outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="i",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                    source="l + {c}",
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="tile_sidx_block",
                tensor=NvOpTensor(
                    shape=Shape(MnkShape("BLK_MNK", "k")),
                    mem="smem",
                    dtype="int",
                ),
            )
        ],
        impl=NvOpImpl(
            r"""
if (lid < get<2>(BLK_MNK{}) / 4) {
    auto thr_tiler = make_shape(Int<4>{});
    auto thr_coord = make_coord(lid);
    auto src = local_tile(val_sidx(_, i), thr_tiler, thr_coord);
    auto dst = local_tile(tile_sidx_block(_, buf_idx), thr_tiler, thr_coord);
    copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, int>{}, src, dst);
}
"""
        ),
        mem_type=("g", "s"),
    )
    sr_bcrs_main_loop_op.append(G2sSparseIndexLoadOp)

    G2sSparseValBlockValLoadOp = NvOp(
        name="G2sSparseValBlockValLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="val_block_val",
                tensor=NvOpTensor(
                    shape=Shape(
                        MnkShape("BLK_MNK", "m"),
                        MnkShape("BLK_MNK", "k"),
                        VarlenShape("nnz_block"),
                    ),
                    mem="gmem",
                    dtype=dtypeAB,
                    source=sr_bcrs_program.gmem_tensor_ops["val_block_val"].outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="i",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                    source="l + {c}",
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="tileA_block",
                tensor=NvOpTensor(
                    shape=Shape(MnkShape("BLK_MNK", "m"), MnkShape("BLK_MNK", "k")),
                    mem="smem",
                    dtype=dtypeAB,
                    row_major=True,
                    swizzle=SwizzleLayout(b=2, m=2, s=3),
                ),
            ),
        ],
        impl=NvOpImpl(
            r"""
auto load_tile_n = _4{};
for (int iter_i = lid; iter_i < get<0>(BLK_MNK{}) * get<2>(BLK_MNK{}) / load_tile_n; iter_i += 32) {
    int row = iter_i / (get<2>(BLK_MNK{}) / load_tile_n);
    int col = iter_i % (get<2>(BLK_MNK{}) / load_tile_n);
    auto thr_tiler_gmem = make_shape(_1{}, _4{});
    auto thr_coord_gmem = make_coord(row, col);
    auto A_val_thr = local_tile(val_block_val(_, _, i), thr_tiler_gmem, thr_coord_gmem);
    //! manual swizzle
    float* dst = tileA_block(_, _, buf_idx).data().get() + iter_i * 4;
    __pipeline_memcpy_async((float4*)dst, (float4*)(A_val_thr(0, _).data().get()), sizeof(float4));
}"""
        ),
    )
    sr_bcrs_main_loop_op.append(G2sSparseValBlockValLoadOp)

    G2sMatrixBLoadOp = NvOp(
        name="G2sMatrixBLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="B_val",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("K"), VarlenShape("N")),
                    mem="gmem",
                    dtype=dtypeAB,
                    source=sr_bcrs_program.gmem_tensor_ops["B_val"].outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="tile_sidx_block",
                tensor=NvOpTensor(
                    shape=Shape(MnkShape("BLK_MNK", "k")),
                    mem="smem",
                    dtype="int",
                    source=G2sSparseIndexLoadOp.outputs[0],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="tileB_block",
                tensor=NvOpTensor(
                    shape=Shape(MnkShape("BLK_MNK", "k"), MnkShape("BLK_MNK", "n")),
                    mem="smem",
                    dtype=dtypeAB,
                    row_major=True,
                    swizzle=SwizzleLayout(b=1, m=2, s=3),
                ),
            ),
        ],
        impl=NvOpImpl(
            r"""
auto load_tile_n = _4{};
for (int iter_i = lid; iter_i < get<2>(BLK_MNK{}) * get<1>(BLK_MNK{}) / load_tile_n; iter_i += 32) {
    int row = iter_i / (get<1>(BLK_MNK{}) / load_tile_n);
    int col = iter_i % (get<1>(BLK_MNK{}) / load_tile_n);
    int sidx = tile_sidx_block(row);
    auto thr_tiler_gmem = make_shape(_1{}, _4{});
    auto thr_coord_gmem = make_coord(sidx, col);
    auto B_val_thr = local_tile(B_val, thr_tiler_gmem, thr_coord_gmem);
    auto thr_tiler_smem = make_shape(_1{}, _4{});
    auto thr_coord_smem = make_coord(row, col);
    auto dst = local_tile(tileB_block(_, _, buf_idx), thr_tiler_smem, thr_coord_smem);
    copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>{}, B_val_thr(0, _), dst(0, _));   
}"""
        ),
        mem_type=("g", "s"),
    )
    sr_bcrs_main_loop_op.append(G2sMatrixBLoadOp)

    S2rAValLoadOp = NvOp(
        name="S2rAValLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="tileA_block",
                tensor=NvOpTensor(
                    shape=Shape(MnkShape("BLK_MNK", "m"), MnkShape("BLK_MNK", "k")),
                    mem="smem",
                    dtype=dtypeAB,
                    source=G2sSparseValBlockValLoadOp.outputs[0],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="REGA",
                tensor=NvOpTensor(
                    shape=Shape(
                        IntShape(FRAG_A),
                        Shape(
                            MnkShape("BLK_MMA_MNK", "m"), MnkShape("BLK_MMA_MNK", "k")
                        ),
                    ),
                    mem="rmem",
                    dtype=dtypeAB,
                ),
            ),
        ],
        impl=NvOpImpl(
            r"""
//> m16n8k8
for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) {
    for (int k_iter = 0; k_iter < get<2>(BLK_MMA_MNK{}); k_iter++) {
        int row = m_iter * get<0>(MMA_MNK{}) + lid % 16;
        int col = lid / 16;
        ldmatrix_m8n8k8_x4(
            (uint32_t*)(REGA(_, make_coord(m_iter, k_iter), buf_idx).data()),
            (void*)(&tileA_block(row, col * 4 + k_iter * get<2>(MMA_MNK{})))
        );
    }
}
"""
        ),
        mem_type=("s", "r"),
    )
    sr_bcrs_main_loop_op.append(S2rAValLoadOp)

    S2rBValLoadOp = NvOp(
        name="S2rBValLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="tileB_block",
                tensor=NvOpTensor(
                    shape=Shape(MnkShape("BLK_MNK", "k"), MnkShape("BLK_MNK", "n")),
                    mem="smem",
                    dtype=dtypeAB,
                    source=G2sMatrixBLoadOp.outputs[0],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="REGB",
                tensor=NvOpTensor(
                    shape=Shape(
                        IntShape(FRAG_B),
                        Shape(
                            MnkShape("BLK_MMA_MNK", "k"), MnkShape("BLK_MMA_MNK", "n")
                        ),
                    ),
                    mem="rmem",
                    dtype=dtypeAB,
                ),
            ),
        ],
        impl=NvOpImpl(
            r"""
for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
    for (int k_iter = 0; k_iter < get<2>(BLK_MMA_MNK{}); k_iter++) {
        int row_b = lid / 2 + k_iter * get<2>(MMA_MNK{});
        int col_b = lid % 2;
        ldmatrix_m8n8k8_x2(
            (uint32_t*)(REGB(_, make_coord(k_iter, n_iter), buf_idx).data()),
            (void*)(&tileB_block(row_b, col_b * 4 + n_iter * 8))
        );
        REGB(_0{}, make_coord(k_iter, n_iter), buf_idx) = __shfl_sync(0xffffffff, REGB(_0{}, make_coord(k_iter, n_iter), buf_idx), lid / 4 + lid % 4 * 8);
        REGB(_1{}, make_coord(k_iter, n_iter), buf_idx) = __shfl_sync(0xffffffff, REGB(_1{}, make_coord(k_iter, n_iter), buf_idx), lid / 4 + lid % 4 * 8);
    }
}
"""
        ),
        mem_type=("s", "r"),
    )
    sr_bcrs_main_loop_op.append(S2rBValLoadOp)

    CalculateOp = NvOp(
        name="CalculateOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="REGA",
                tensor=NvOpTensor(
                    shape=Shape(
                        IntShape(FRAG_A),
                        Shape(
                            MnkShape("BLK_MMA_MNK", "m"),
                            MnkShape("BLK_MMA_MNK", "k"),
                        ),
                    ),
                    mem="rmem",
                    dtype=dtypeAB,
                    source=S2rAValLoadOp.outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="REGB",
                tensor=NvOpTensor(
                    shape=Shape(
                        IntShape(FRAG_B),
                        Shape(
                            MnkShape("BLK_MMA_MNK", "k"),
                            MnkShape("BLK_MMA_MNK", "n"),
                        ),
                    ),
                    mem="rmem",
                    dtype=dtypeAB,
                    source=S2rBValLoadOp.outputs[0],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="REGC",
                tensor=NvOpTensor(
                    shape=Shape(
                        IntShape(FRAG_C),
                        Shape(
                            MnkShape("BLK_MMA_MNK", "m"),
                            MnkShape("BLK_MMA_MNK", "n"),
                        ),
                    ),
                    mem="rmem",
                    dtype=dtypeC,
                ),
                origin=ZerosOp.outputs[0],
            ),
        ],
        impl=NvOpImpl(
            r"""
for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) {
    for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
        for (int k_iter = 0; k_iter < get<2>(BLK_MMA_MNK{}); k_iter++) {
            uint32_t frag_A[4];
            uint32_t frag_B[2];
            asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(REGA(0, make_coord(m_iter, k_iter))));
            asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(REGA(1, make_coord(m_iter, k_iter))));
            asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(REGA(2, make_coord(m_iter, k_iter))));
            asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(REGA(3, make_coord(m_iter, k_iter))));
            asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(REGB(0, make_coord(k_iter, n_iter))));
            asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(REGB(1, make_coord(k_iter, n_iter))));

            mma_m16n8k8_fp32_tf32_tf32_fp32(REGC(_, make_coord(m_iter, n_iter)).data(), frag_A, frag_B);
        }
    }
}
        """
        ),
        mem_type=("r", "r"),
    )
    sr_bcrs_main_loop_op.append(CalculateOp)
    sr_bcrs_main_loop_op.set_loop_result(0, CalculateOp.outputs[0])

    r2g_c_val_store_op = NvOp(
        name="R2gCValStoreOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="REGC",
                tensor=NvOpTensor(
                    shape=Shape(
                        IntShape(FRAG_C),
                        Shape(
                            MnkShape("BLK_MMA_MNK", "m"),
                            MnkShape("BLK_MMA_MNK", "n"),
                        ),
                    ),
                    mem="rmem",
                    dtype=dtypeC,
                    source=sr_bcrs_main_loop_op.outputs[0],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="C_val",
                tensor=NvOpTensor(
                    shape=Shape(
                        MnkShape("BLK_MNK", "n"),
                        MnkShape("BLK_MNK", "m"),
                    ),
                    mem="gmem",
                    dtype=dtypeC,
                ),
                origin=sr_bcrs_program.gmem_tensor_ops["C_val"].outputs[0],
            ),
        ],
        impl=NvOpImpl(
            r"""
//! write back
for (int i_tileM = 0; i_tileM < get<0>(BLK_MMA_MNK{}); i_tileM++) {
    for (int i_tileN = 0; i_tileN < get<1>(BLK_MMA_MNK{}); i_tileN++) {
        int row = lid / 4 + i_tileM * get<0>(MMA_MNK{});
        int col = i_tileN * 8 + lid % 4 * 2;
        C_val(col, row) = REGC(0, make_coord(i_tileM, i_tileN));
        C_val(col + 1, row) = REGC(1, make_coord(i_tileM, i_tileN));
        C_val(col, row + 8) = REGC(2, make_coord(i_tileM, i_tileN));
        C_val(col + 1, row + 8) = REGC(3, make_coord(i_tileM, i_tileN));
    }
}

        """
        ),
        mem_type=("r", "g"),
    )
    sr_bcrs_blk_x_loop_op.append(r2g_c_val_store_op)

    return sr_bcrs_main_loop_op, sr_bcrs_program


if __name__ == "__main__":
    from sparsene.op_gen.nvir.codegen import NvIrCodeGenerator

    sr_bcrs_main_loop_op, sr_bcrs_program = sr_bcrs()

    #! stage 0
    G2sSparseIndexLoadOp = sr_bcrs_program.find_op_by_name("G2sSparseIndexLoadOp")

    #! stage 1
    G2sSparseValBlockValLoadOp = sr_bcrs_program.find_op_by_name(
        "G2sSparseValBlockValLoadOp"
    )
    G2sMatrixBLoadOp = sr_bcrs_program.find_op_by_name("G2sMatrixBLoadOp")

    #! stage 2
    S2rAValLoadOp = sr_bcrs_program.find_op_by_name("S2rAValLoadOp")
    S2rBValLoadOp = sr_bcrs_program.find_op_by_name("S2rBValLoadOp")
    CalculateOp = sr_bcrs_program.find_op_by_name("CalculateOp")

    s0 = NvOpSequence(
        G2sSparseIndexLoadOp,
    )

    s1 = NvOpSequence(
        G2sSparseValBlockValLoadOp,
        G2sMatrixBLoadOp,
    )

    s2 = NvOpSequence(
        S2rAValLoadOp,
        S2rBValLoadOp,
        CalculateOp,
    )

    apply_pipeline(
        sr_bcrs_main_loop_op,
        PipelinePlan([s0, s1, s2], [1, 1]),
    )

    script_dir = Path(__file__).parent
    with open(script_dir / "kernel_16x8_spmv.inc", "w") as f:
        f.write(NvIrCodeGenerator().dump_nvop_program(sr_bcrs_program))

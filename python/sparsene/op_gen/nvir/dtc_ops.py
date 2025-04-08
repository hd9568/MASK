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
    LayoutHint,
    NvOpSequence,
    NvOpPipeline,
    GmemInout,
    NvOpTensor,
    MetaNvOp,
    ForLoopNvOp,
    ConstantNvOp,
    NvOpProgram,
)
from sparsene.op_gen.nvir.plan import apply_pipeline, PipelinePlan
from sparsene.op_gen.nvir.pipeline.pipeline_planner import (
    enumerate_pipeline_plans,
)


def dtc():
    dtc_program = NvOpProgram(
        name="dtc_spmm_kernel_sstage_rstage",
        gmem_inouts={
            "val_sidx": GmemInout(
                shape=Shape(ParamShape("tile_sidx_size"), VarlenShape("nnz_block")),
                name="val_sidx",
                dtype="int",
                tensor_str="make_tensor(make_gmem_ptr(dval_sidx), make_shape(tile_sidx_size{}, nnz_block))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(tile_sidx_size{}, VARLEN)))",
                parameters={"tile_sidx_size": 16},
            ),
            "val_coo_off": GmemInout(
                shape=Shape(IntShape(2), VarlenShape("nnz_block")),
                name="val_coo_off",
                dtype="int",
                tensor_str="make_tensor(make_gmem_ptr(dval_coo_off), make_shape(Int<2>{}, nnz_block))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(Int<2>{}, VARLEN)))",
            ),
            "val_coo_val": GmemInout(
                shape=Shape(VarlenShape("nnz")),
                name="val_coo_val",
                dtype="half",
                tensor_str="make_tensor(make_gmem_ptr(dval_coo_val), make_shape(nnz))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)))",
            ),
            "val_coo_idx": GmemInout(
                shape=Shape(VarlenShape("nnz")),
                name="val_coo_idx",
                dtype="int",
                tensor_str="make_tensor(make_gmem_ptr(dval_coo_idx), make_shape(nnz))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)))",
            ),
            "val_soff": GmemInout(
                shape=Shape(VarlenShape("Mo")),
                name="val_soff",
                dtype="int",
                tensor_str="make_tensor(make_gmem_ptr(dval_soff), make_shape(Mo))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)))",
            ),
            "B_val": GmemInout(
                shape=Shape(VarlenShape("K"), VarlenShape("N")),
                name="B_val",
                dtype="half",
                tensor_str="logical_divide(make_tensor(make_gmem_ptr(dB_val), make_shape(K, N)), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, blockIdx.y))",
                tensor_type_str="decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN, VARLEN)), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, VARLEN)))",
            ),
            "C_val": GmemInout(
                shape=Shape(VarlenShape("N"), VarlenShape("M")),
                name="C_val",
                dtype="float",
                tensor_str="logical_divide(make_tensor(make_gmem_ptr(dC_val), make_shape(N, M)), select<1, 0>(BLK_MNK{}))(make_coord(_, blockIdx.y), make_coord(_, blockIdx.x))",
                tensor_type_str="decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN, VARLEN)), select<1, 0>(BLK_MNK{}))(make_coord(_, VARLEN), make_coord(_, VARLEN)))",
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
    dtc_program.append_op(blk_y_loop_zero_op)
    BlockDimY_op = ConstantNvOp(
        name="BlockDimYOp",
        shape=Shape(),
        dtype="int",
        value="blockDim.y",
    )
    dtc_program.append_op(BlockDimY_op)
    dtc_blk_y_loop_op = ForLoopNvOp(
        name="dtc_spmm_kernel_blk_y_loop",
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
    dtc_program.append_op(dtc_blk_y_loop_op)

    # blk_y_loop body
    blk_x_loop_zero_op = ConstantNvOp(
        name="BlkXLoopZeroOp",
        shape=Shape(),
        dtype="int",
        value=0,
    )
    dtc_blk_y_loop_op.append(blk_x_loop_zero_op)
    BlockDimX_op = ConstantNvOp(
        name="BlockDimXOp",
        shape=Shape(),
        dtype="int",
        value="blockDim.x",
    )
    dtc_blk_y_loop_op.append(BlockDimX_op)
    dtc_blk_x_loop_op = ForLoopNvOp(
        name="dtc_spmm_kernel_blk_x_loop",
        blk_idx_mapping="x",
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
                source=BlockDimX_op.outputs[0],
            ),
        ),
        loop_result=[],
        body=NvOpSequence(),
        iter_args={},
    )
    dtc_blk_y_loop_op.append(dtc_blk_x_loop_op)

    # blk_x_loop body
    g2r_sparse_offset_load_op = NvOp(
        name="G2rSparseOffsetLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="val_soff",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("Mo")),
                    mem="gmem",
                    dtype="int",
                    source=dtc_program.gmem_tensor_ops["val_soff"].outputs[0],
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
            code_template="""l(0) = val_soff(i);
r(0) = val_soff(i + 1);"""
        ),
        mem_type=("g", "r"),
        parameters={},
    )
    dtc_blk_x_loop_op.append(g2r_sparse_offset_load_op)

    zeros_op = ConstantNvOp(
        name="ZerosOp",
        shape=Shape(
            IntShape(4),
            Shape(
                MnkShape("BLK_MMA_MNK", "m"),
                MnkShape("BLK_MMA_MNK", "n"),
            ),
        ),
        dtype="float",
        value=0,
    )

    dtc_blk_x_loop_op.append(zeros_op)

    dtc_main_loop_op = ForLoopNvOp(
        name="DtcMainLoopOp",
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
                        IntShape(4),
                        Shape(
                            MnkShape("BLK_MMA_MNK", "m"),
                            MnkShape("BLK_MMA_MNK", "n"),
                        ),
                    ),
                    mem="rmem",
                    dtype="float",
                ),
            ),
        ],
        body=NvOpSequence(),
        iter_args={},
    )
    dtc_blk_x_loop_op.append(dtc_main_loop_op)

    # main_loop body
    g2s_val_sidx_load_op = NvOp(
        name="G2sValSidxLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="val_sidx",
                tensor=NvOpTensor(
                    shape=Shape(ParamShape("tile_sidx_size"), VarlenShape("nnz_block")),
                    mem="gmem",
                    dtype="int",
                    source=dtc_program.gmem_tensor_ops["val_sidx"].outputs[0],
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
                    shape=Shape(ParamShape("tile_sidx_size")),
                    mem="smem",
                    dtype="int",
                ),
            )
        ],
        impl=NvOpImpl(
            code_template="""if (lid < 4) {
    auto thr_tiler = make_shape(Int<4>{});
    auto thr_coord = make_coord(lid);
    auto src = local_tile(val_sidx(_, i), thr_tiler, thr_coord);
    auto dst = local_tile(tile_sidx_block(_, buf_idx), thr_tiler, thr_coord);
    copy(Copy_Atom<UniversalCopy<uint128_t>, int>{}, src, dst);
}"""
        ),
        mem_type=("g", "s"),
        parameters={"tile_sidx_size": 16},
    )
    dtc_main_loop_op.append(g2s_val_sidx_load_op)

    g2r_coo_atomic_format_load_off_op = NvOp(
        name="G2rCooAtomicFormatLoadOffOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="val_coo_off",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(2), VarlenShape("nnz_block")),
                    mem="gmem",
                    dtype="int",
                    source=dtc_program.gmem_tensor_ops["val_coo_off"].outputs[0],
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
            code_template="""l(buf_idx) = val_coo_off(0, i);
r(buf_idx) = val_coo_off(1, i);"""
        ),
        mem_type=("g", "r"),
    )
    dtc_main_loop_op.append(g2r_coo_atomic_format_load_off_op)

    s2r_val_sidx_load_op = NvOp(
        name="S2rValSidxLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="tile_sidx_block",
                tensor=NvOpTensor(
                    shape=Shape(ParamShape("tile_sidx_size")),
                    mem="smem",
                    dtype="int",
                    source=g2s_val_sidx_load_op.outputs[0],
                ),
            )
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="val_sidx_reg",
                tensor=NvOpTensor(
                    shape=Shape(ParamShape("tile_sidx_size")),
                    mem="rmem",
                    dtype="int",
                ),
            )
        ],
        impl=NvOpImpl(
            code_template="""auto src = tile_sidx_block;
auto dst = val_sidx_reg(_, buf_idx);
copy(src, dst);"""
        ),
        mem_type=("s", "r"),
        parameters={"tile_sidx_size": 16},
    )
    dtc_main_loop_op.append(s2r_val_sidx_load_op)

    g2s_coo_atomic_format_load_val_op = NvOp(
        name="G2sCooAtomicFormatLoadValOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="val_coo_val",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("nnz")),
                    mem="gmem",
                    dtype="half",
                    source=dtc_program.gmem_tensor_ops["val_coo_val"].outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="ll",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                    source=g2r_coo_atomic_format_load_off_op.outputs[0],
                ),
            ),
            NvOpInput(
                idx=2,
                name="rr",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                    source=g2r_coo_atomic_format_load_off_op.outputs[1],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="tile_coo_val_block",
                tensor=NvOpTensor(
                    shape=Shape(ParamShape("tile_coo_val_size")),
                    mem="smem",
                    dtype="half",
                ),
            ),
        ],
        impl=NvOpImpl(
            code_template="""auto thr_tiler = make_shape(_8{});
auto thr_coord = make_coord(lid);
auto input = make_tensor(val_coo_val.data() + ll, make_shape(tile_coo_val_size{}));
auto output = tile_coo_val_block(_, buf_idx);

if (lid * 8 < rr - ll) {
    auto src = local_tile(input, thr_tiler, thr_coord);
    auto dst = local_tile(output, thr_tiler, thr_coord);
    copy(Copy_Atom<UniversalCopy<uint128_t>, half>{}, src, dst);
}"""
        ),
        mem_type=("g", "s"),
        parameters={"tile_coo_val_size": 256},
    )
    dtc_main_loop_op.append(g2s_coo_atomic_format_load_val_op)

    g2r_coo_atomic_format_load_idx_op = NvOp(
        name="G2rCooAtomicFormatLoadIdxOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="val_coo_idx",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("nnz")),
                    mem="gmem",
                    dtype="int",
                    source=dtc_program.gmem_tensor_ops["val_coo_idx"].outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="ll",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                    source=g2r_coo_atomic_format_load_off_op.outputs[0],
                ),
            ),
            NvOpInput(
                idx=2,
                name="rr",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                    source=g2r_coo_atomic_format_load_off_op.outputs[1],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="coo_idx_reg",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(8)),
                    mem="rmem",
                    dtype="int",
                ),
            ),
            NvOpOutput(
                idx=1,
                name="coo_range_reg",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                ),
            ),
        ],
        impl=NvOpImpl(
            code_template="""auto thr_tiler = make_shape(_8{});
auto thr_coord = make_coord(lid);
auto input = make_tensor(val_coo_idx.data() + ll, make_shape(tile_coo_val_size{}));
coo_range_reg(buf_idx) = rr - ll;
if (lid * 8 < rr - ll) {
    auto src = local_tile(input, thr_tiler, thr_coord);
    auto dst = coo_idx_reg(_, buf_idx);
    copy(src, dst);
}"""
        ),
        mem_type=("g", "r"),
        parameters={"tile_coo_val_size": 256},
    )
    dtc_main_loop_op.append(g2r_coo_atomic_format_load_idx_op)

    g2r_b_val_load_op = NvOp(
        name="G2rBValLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="B_val",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("K"), MnkShape("BLK_MNK", "n")),
                    mem="gmem",
                    dtype="half",
                    source=dtc_program.gmem_tensor_ops["B_val"].outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="val_sidx_reg",
                tensor=NvOpTensor(
                    shape=Shape(MnkShape("BLK_MNK", "k")),
                    mem="rmem",
                    dtype="int",
                    source=s2r_val_sidx_load_op.outputs[0],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="REGB",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(4), MnkShape("BLK_MMA_MNK", "n")),
                    mem="rmem",
                    dtype="half",
                ),
            ),
        ],
        impl=NvOpImpl(
            code_template="""for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
    int row = lid % 4 * 2;
    int col = n_iter * 8 + lid / 4;
    REGB(0, n_iter, buf_idx) = B_val(val_sidx_reg(row + 0), col);
    REGB(1, n_iter, buf_idx) = B_val(val_sidx_reg(row + 1), col);
    REGB(2, n_iter, buf_idx) = B_val(val_sidx_reg(row + 8), col);
    REGB(3, n_iter, buf_idx) = B_val(val_sidx_reg(row + 9), col);
}"""
        ),
        mem_type=("g", "r"),
    )
    dtc_main_loop_op.append(g2r_b_val_load_op)

    s2s_coo_atomic_val_restore_op = NvOp(
        name="S2sCooAtomicValRestoreOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="tile_coo_val_block",
                tensor=NvOpTensor(
                    shape=Shape(ParamShape("tile_coo_val_size")),
                    mem="smem",
                    dtype="half",
                    source=g2s_coo_atomic_format_load_val_op.outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="coo_idx_reg",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(8)),
                    mem="rmem",
                    dtype="int",
                    source=g2r_coo_atomic_format_load_idx_op.outputs[0],
                ),
            ),
            NvOpInput(
                idx=2,
                name="nnz_num",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                    source=g2r_coo_atomic_format_load_idx_op.outputs[1],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="tile_coo_restore_val_block",
                tensor=NvOpTensor(
                    shape=Shape(
                        MnkShape("BLK_MNK", "k"),
                        MnkShape("BLK_MNK", "m"),
                    ),
                    mem="smem",
                    dtype="half",
                ),
            ),
        ],
        impl=NvOpImpl(
            code_template="""// TODO: use tiler to distribute the tensor and use clear(Tensor) to set zero
for (int i_o2s = 0; i_o2s < 4; i_o2s++) {
    *((int*)(tile_coo_restore_val_block(_, _, buf_idx).data().get() + 32 * 2 * i_o2s + lid * 2)) = 0;
}
__syncthreads();
for (int i_restore = 0; i_restore < 8; i_restore++) {
    int idx = coo_idx_reg(i_restore);
    if (lid * 8 + i_restore < nnz_num) {
        tile_coo_restore_val_block(_, _, buf_idx)(idx) = tile_coo_val_block(lid * 8 + i_restore);
    }
}"""
        ),
        mem_type=("s", "s"),
        parameters={"tile_coo_val_size": 256},
    )
    dtc_main_loop_op.append(s2s_coo_atomic_val_restore_op)

    s2r_a_val_load_op = NvOp(
        name="S2rAValLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="tile_coo_restore_val_block",
                tensor=NvOpTensor(
                    shape=Shape(MnkShape("BLK_MNK", "k"), MnkShape("BLK_MNK", "m")),
                    mem="smem",
                    dtype="half",
                    source=s2s_coo_atomic_val_restore_op.outputs[0],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="REGA",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(8), MnkShape("BLK_MMA_MNK", "m")),
                    mem="rmem",
                    dtype="half",
                ),
            ),
        ],
        impl=NvOpImpl(
            code_template="""
__syncthreads();
int row = lid % 16;
int col = lid / 16;
// TODO: bank conflict
for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) { // now the m_iter would always be zero, which means the for-loop body would run only once
    ldmatrix_m8n8k8_x4(
        (uint32_t*)(REGA(_, _0{}, buf_idx).data()),
        (void*)&tile_coo_restore_val_block(col * 8, row)
    );
}"""
        ),
        mem_type=("s", "r"),
    )
    dtc_main_loop_op.append(s2r_a_val_load_op)

    mma_op = NvOp(
        name="MmaOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="REGA",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(8), MnkShape("BLK_MMA_MNK", "m")),
                    mem="rmem",
                    dtype="half",
                    source=s2r_a_val_load_op.outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="REGB",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(4), MnkShape("BLK_MMA_MNK", "n")),
                    mem="rmem",
                    dtype="half",
                    source=g2r_b_val_load_op.outputs[0],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="REGC",
                tensor=NvOpTensor(
                    shape=Shape(
                        IntShape(4),
                        Shape(
                            MnkShape("BLK_MMA_MNK", "m"),
                            MnkShape("BLK_MMA_MNK", "n"),
                        ),
                    ),
                    mem="rmem",
                    dtype="float",
                ),
                origin=zeros_op.outputs[0],
            ),
        ],
        impl=NvOpImpl(
            code_template="""for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) {
    for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
        mma_m16n8k16_fp32(REGC(_, make_coord(m_iter, n_iter)).data(), (uint32_t*)(REGA(_, m_iter).data()), REGB(_, n_iter).data());
    }
}"""
        ),
        mem_type=("r", "r"),
    )
    dtc_main_loop_op.append(mma_op)

    dtc_main_loop_op.set_loop_result(0, mma_op.outputs[0])

    r2g_c_val_store_op = NvOp(
        name="R2gCValStoreOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="REGC",
                tensor=NvOpTensor(
                    shape=Shape(
                        IntShape(4),
                        Shape(
                            MnkShape("BLK_MMA_MNK", "m"),
                            MnkShape("BLK_MMA_MNK", "n"),
                        ),
                    ),
                    mem="rmem",
                    dtype="float",
                    source=dtc_main_loop_op.outputs[0],
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
                    dtype="float",
                ),
                origin=dtc_program.gmem_tensor_ops["C_val"].outputs[0],
            ),
        ],
        impl=NvOpImpl(
            code_template="""//! write back
for (int i_tileN = 0; i_tileN < get<1>(BLK_MNK{}) / 8; i_tileN++) {
    int row = lid / 4;
    int col = i_tileN * 8 + lid % 4 * 2;
    C_val(col, row) = REGC(0, i_tileN);
    C_val(col + 1, row) = REGC(1, i_tileN);
    C_val(col, row + 8) = REGC(2, i_tileN);
    C_val(col + 1, row + 8) = REGC(3, i_tileN);
}"""
        ),
        mem_type=("r", "g"),
    )
    dtc_blk_x_loop_op.append(r2g_c_val_store_op)

    return dtc_main_loop_op, dtc_program


if __name__ == "__main__":
    from sparsene.op_gen.nvir.codegen import NvIrCodeGenerator

    from sparsene.op_gen.nvir.opgraph.graphviz_visualizer import visualize_op_graph
    from sparsene.op_gen.nvir.opgraph.graph import construct_graph_from_op_sequence

    dtc_main_loop_op, dtc_program = dtc()

    # op_graph = construct_graph_from_op_sequence(dtc_main_loop_op.body)
    # visualize_op_graph(op_graph, "dtc.dot")

    g2s_val_sidx_load_op = dtc_program.find_op_by_name("G2sValSidxLoadOp")
    g2r_coo_atomic_format_load_off_op = dtc_program.find_op_by_name(
        "G2rCooAtomicFormatLoadOffOp"
    )
    s2r_val_sidx_load_op = dtc_program.find_op_by_name("S2rValSidxLoadOp")
    g2s_coo_atomic_format_load_val_op = dtc_program.find_op_by_name(
        "G2sCooAtomicFormatLoadValOp"
    )
    g2r_coo_atomic_format_load_idx_op = dtc_program.find_op_by_name(
        "G2rCooAtomicFormatLoadIdxOp"
    )
    g2r_b_val_load_op = dtc_program.find_op_by_name("G2rBValLoadOp")
    s2s_coo_atomic_val_restore_op = dtc_program.find_op_by_name(
        "S2sCooAtomicValRestoreOp"
    )
    s2r_a_val_load_op = dtc_program.find_op_by_name("S2rAValLoadOp")
    mma_op = dtc_program.find_op_by_name("MmaOp")

    s0 = NvOpSequence(
        g2s_val_sidx_load_op,
        g2r_coo_atomic_format_load_off_op,
    )
    s1 = NvOpSequence(
        s2r_val_sidx_load_op,
        g2s_coo_atomic_format_load_val_op,
        g2r_coo_atomic_format_load_idx_op,
    )
    s2 = NvOpSequence(
        g2r_b_val_load_op,
        s2s_coo_atomic_val_restore_op,
        s2r_a_val_load_op,
    )
    s3 = NvOpSequence(
        mma_op,
    )

    plans = enumerate_pipeline_plans(dtc_main_loop_op)
    print("Possible plans:", len(plans))
    for plan in plans:
        new_loop, new_dtc_program = dtc()
        apply_pipeline(new_loop, plan)
        print(NvIrCodeGenerator().dump_nvop_program(new_dtc_program))

    # bit-bsr
    # flash-llm
    #### acc-spmm (dtc-spmm + 修改restore A val)

    apply_pipeline(
        dtc_main_loop_op,
        PipelinePlan([s0, s1, s2, s3], [1, 1, 1]),
    )

    print(NvIrCodeGenerator().dump_nvop_program(dtc_program))

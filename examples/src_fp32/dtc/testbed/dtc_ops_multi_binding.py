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
from sparsene.op_gen.nvir.pipeline.pipeline_planner import (
    enumerate_pipeline_plans,
    NeighborDependencyValidator,
)
from sparsene.op_gen.nvir.opgraph.graph import construct_graph_from_op_sequence

def dtc():
    dtc_program = NvOpProgram(
        name="dtc_spmm_kernel_fp32_val_idx_bind_stage2",
        gmem_inouts={
            "B_val": GmemInout(
                shape=Shape(VarlenShape("K"), VarlenShape("N")),
                name="B_val",
                dtype="float",
                tensor_str="logical_divide(make_tensor(make_gmem_ptr(dB_val), make_layout(make_shape(K, N), make_stride(N, _1{}))), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, blockIdx.y))",
                tensor_type_str="decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_layout(make_shape(VARLEN, VARLEN), make_stride(VARLEN, _1{}))), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, VARLEN)))",
            ),
            #! split the C_val according to the BLK_M and BLK_N
            "C_val": GmemInout(
                shape=Shape(VarlenShape("N"), VarlenShape("M")),
                name="C_val",
                dtype="float",
                # tensor_str="logical_divide(make_tensor(make_gmem_ptr(dC_val), make_shape(N, M)), select<1, 0>(BLK_MNK{}))(make_coord(_, blockIdx.y), make_coord(_, blockIdx.x))",
                # tensor_type_str="decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN, VARLEN)), select<1, 0>(BLK_MNK{}))(make_coord(_, VARLEN), make_coord(_, VARLEN)))",
                tensor_str="logical_divide(make_tensor(make_gmem_ptr(dC_val), make_shape(N, M)), select<1, 0>(BLK_MNK{}))(make_coord(_, blockIdx.y), _)",
                tensor_type_str="decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN, VARLEN)), select<1, 0>(BLK_MNK{}))(make_coord(_, VARLEN), _))"
            ),
            "val_sidx": GmemInout(
                shape=Shape(ParamShape("tile_sidx_size"), VarlenShape("nnz_block")),
                name="val_sidx",
                dtype="int",
                tensor_str="make_tensor(make_gmem_ptr(dval_sidx), make_shape(tile_sidx_size{}, nnz_block))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(tile_sidx_size{}, VARLEN)))",
                parameters={"tile_sidx_size": 8},
            ),
            "val_soff": GmemInout(
                shape=Shape(VarlenShape("Mo")),
                name="val_soff",
                dtype="int",
                tensor_str="make_tensor(make_gmem_ptr(dval_soff), make_shape(Mo))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)))",
            ),
            "val_coo_idx": GmemInout(
                shape=Shape(VarlenShape("nnz")),
                name="val_coo_idx",
                dtype="int",
                tensor_str="make_tensor(make_gmem_ptr(dval_coo_idx), make_shape(nnz))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)))",
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
                dtype="float",
                tensor_str="make_tensor(make_gmem_ptr(dval_coo_val), make_shape(nnz))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)))",
            ),
            "row_window_binding": GmemInout(
                shape=Shape(VarlenShape("row_window_blockDim_size")),
                name="row_window_binding",
                dtype="int",
                tensor_str="make_tensor(make_gmem_ptr(drow_window_binding), make_shape(row_window_blockDim_size))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)))",
            ),
            "row_window_split": GmemInout(
                shape=Shape(VarlenShape("Mo")),
                name="row_window_split",
                dtype="int",
                tensor_str="make_tensor(make_gmem_ptr(drow_window_split), make_shape(Mo))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)))",
            )
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

    g2r_row_window_binding_op = NvOp(
        name="G2rRowWindowBindingOp",
        inputs=[
            NvOpInput(
                idx=0, 
                name="row_window_binding",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("row_window_blockDim_size")),
                    mem="gmem",
                    dtype="int",
                    source=dtc_program.gmem_tensor_ops["row_window_binding"].outputs[0],
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
                name="bind_i",
                tensor=NvOpTensor(shape=Shape(), mem="rmem", dtype="int"),
            ),
        ],
        impl=NvOpImpl(
            r"""
bind_i(0) = row_window_binding(i);
"""
        ),
        mem_type=("g", "r"),
    )
    dtc_blk_x_loop_op.append(g2r_row_window_binding_op)

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
            r"""l(0) = val_soff(i);
r(0) = val_soff(i + 1);"""
        ),
        mem_type=("g", "r"),
    )
    dtc_blk_x_loop_op.append(g2r_sparse_offset_load_op)
    ZerosOp = ConstantNvOp(
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
    dtc_blk_x_loop_op.append(ZerosOp)
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
    G2sSparseIndexLoadOp = NvOp(
        name="G2sSparseIndexLoadOp",
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
            ),
        ],
        impl=NvOpImpl(
            r"""if (lid < 2) {
    auto thr_tiler = make_shape(Int<4>{});
    auto thr_coord = make_coord(lid);
    auto src = local_tile(val_sidx(_, i), thr_tiler, thr_coord);
    auto dst = local_tile(tile_sidx_block(_, buf_idx), thr_tiler, thr_coord);
    copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, int>{}, src, dst);
}"""
        ),
        mem_type=("g", "s"),
        parameters={
            "tile_sidx_size": 8,
        },
    )
    dtc_main_loop_op.append(G2sSparseIndexLoadOp)
    G2sSparseCooIdxLoadOp = NvOp(
        name="G2sSparseCooIdxLoadOp",
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
                name="val_coo_idx",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("nnz")),
                    mem="gmem",
                    dtype="int",
                    source=dtc_program.gmem_tensor_ops["val_coo_idx"].outputs[0],
                ),
            ),
            NvOpInput(
                idx=2,
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
                name="coo_range_reg",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                ),
            ),
            NvOpOutput(
                idx=1,
                name="tileA_coo_idx",
                tensor=NvOpTensor(
                    shape=Shape(ParamShape("tile_coo_val_size_no_pad")),
                    mem="smem",
                    dtype="int",
                ),
            ),
        ],
        impl=NvOpImpl(
            r"""int ll = val_coo_off(0, i);
int rr = val_coo_off(1, i);
coo_range_reg(buf_idx) = rr - ll;

auto thr_tiler = make_shape(_4{});
auto input = make_tensor(val_coo_idx.data() + ll, tile_coo_val_size_no_pad{});
for (int i_load = lid; i_load * 4 + ll < rr; i_load += 32) {
    auto thr_coord = make_coord(i_load);
    auto src = local_tile(input, thr_tiler, thr_coord);
    auto dst = local_tile(tileA_coo_idx(_, buf_idx), thr_tiler, thr_coord);
    copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, int>{}, src, dst);
}"""
        ),
        mem_type=("g", "s"),
        parameters={
            "tile_coo_val_size_no_pad": 16 * 8,
        },
    )
    dtc_main_loop_op.append(G2sSparseCooIdxLoadOp)
    G2sSparseCooValLoadOp = NvOp(
        name="G2sSparseCooValLoadOp",
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
                name="val_coo_val",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("nnz")),
                    mem="gmem",
                    dtype="float",
                    source=dtc_program.gmem_tensor_ops["val_coo_val"].outputs[0],
                ),
            ),
            NvOpInput(
                idx=2,
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
                name="tileA_coo_val",
                tensor=NvOpTensor(
                    shape=Shape(ParamShape("tile_coo_val_size_no_pad")),
                    mem="smem",
                    dtype="float",
                ),
            ),
        ],
        impl=NvOpImpl(
            r"""int ll = val_coo_off(0, i);
int rr = val_coo_off(1, i);

auto thr_tiler = make_shape(_4{});
auto input = make_tensor(val_coo_val.data() + ll, tile_coo_val_size_no_pad{});
for (int i_load = lid; i_load * 4 + ll < rr; i_load += 32) {
    auto thr_coord = make_coord(i_load);
    auto src = local_tile(input, thr_tiler, thr_coord);
    auto dst = local_tile(tileA_coo_val(_, buf_idx), thr_tiler, thr_coord);
    copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>{}, src, dst);
}"""
        ),
        mem_type=("g", "s"),
        parameters={
            "tile_coo_val_size_no_pad": 16 * 8,
        },
    )
    dtc_main_loop_op.append(G2sSparseCooValLoadOp)
    G2sMatrixBLoadOp = NvOp(
        name="G2sMatrixBLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="B_val",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("K"), VarlenShape("N")),
                    mem="gmem",
                    dtype="float",
                    source=dtc_program.gmem_tensor_ops["B_val"].outputs[0],
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
                    dtype="float",
                    row_major=True,
                    swizzle=SwizzleLayout(b=3, m=2, s=3),
                ),
            ),
        ],
        impl=NvOpImpl(
            r"""// auto thr_tiler = make_shape(_4{});
// auto thr_coord = make_coord(lid);
__syncthreads();
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
    dtc_main_loop_op.append(G2sMatrixBLoadOp)
    S2sRestoreMatrixAOp = NvOp(
        name="S2sRestoreMatrixAOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="tileA_coo_idx",
                tensor=NvOpTensor(
                    shape=Shape(ParamShape("tile_coo_val_size_no_pad")),
                    mem="smem",
                    dtype="int",
                    source=G2sSparseCooIdxLoadOp.outputs[1],
                ),
            ),
            NvOpInput(
                idx=1,
                name="tileA_coo_val",
                tensor=NvOpTensor(
                    shape=Shape(ParamShape("tile_coo_val_size_no_pad")),
                    mem="smem",
                    dtype="float",
                    source=G2sSparseCooValLoadOp.outputs[0],
                ),
            ),
            NvOpInput(
                idx=2,
                name="nnz_num",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                    source=G2sSparseCooIdxLoadOp.outputs[0],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="tile_coo_restore_val_block",
                tensor=NvOpTensor(
                    shape=Shape(MnkShape("BLK_MNK", "m"), MnkShape("BLK_MNK", "k")),
                    mem="smem",
                    dtype="float",
                    row_major=True,
                    swizzle=SwizzleLayout(b=1, m=2, s=3),
                ),
            ),
        ],
        impl=NvOpImpl(
            r"""for (int i_o2s = lid; i_o2s < tile_coo_val_size{}; i_o2s += 32) {   
    *((float*)(tile_coo_restore_val_block(_, _, buf_idx).data().get() + i_o2s)) = 0;
}
__syncthreads();
for (int i_restore = lid; i_restore < nnz_num; i_restore += 32) {
    float value = tileA_coo_val(i_restore);
    int idx = tileA_coo_idx(i_restore);
    //! without manual swizzle
    // int row = idx / 8;
    // int col = idx % 8;
    // tile_coo_restore_val_block(row, col, buf_idx) = value;
    //! manual swizzle
    *((float*)tile_coo_restore_val_block(_, _, buf_idx).data().get() + idx) = value;
}"""
        ),
        mem_type=("s", "s"),
        parameters={
            "tile_coo_val_size_no_pad": 16 * 8,
            "tile_coo_val_size": 16 * 8,
        },
    )
    dtc_main_loop_op.append(S2sRestoreMatrixAOp)
    S2rAValLoadOp = NvOp(
        name="S2rAValLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="tile_coo_restore_val_block",
                tensor=NvOpTensor(
                    shape=Shape(
                        MnkShape("BLK_MNK", "m"),
                        MnkShape("BLK_MNK", "k"),
                    ),
                    mem="smem",
                    dtype="float",
                    source=S2sRestoreMatrixAOp.outputs[0],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="REGA",
                tensor=NvOpTensor(
                    shape=Shape(
                        IntShape(4),
                        MnkShape("BLK_MMA_MNK", "m"),
                    ),
                    mem="rmem",
                    dtype="float",
                ),
            ),
        ],
        impl=NvOpImpl(
            r"""__syncthreads();
//> m16n8k8
int row = lid % 16;
int col = lid / 16;
ldmatrix_m8n8k8_x4(
    (uint32_t*)(REGA(_, _0{}, buf_idx).data()),
    (void*)(&tile_coo_restore_val_block(row, col * 4))
);"""
        ),
        mem_type=("s", "r"),
    )
    dtc_main_loop_op.append(S2rAValLoadOp)
    S2rBValLoadOp = NvOp(
        name="S2rBValLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="tileB_block",
                tensor=NvOpTensor(
                    shape=Shape(
                        MnkShape("BLK_MNK", "k"),
                        MnkShape("BLK_MNK", "n"),
                    ),
                    mem="smem",
                    dtype="float",
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
                        IntShape(2),
                        MnkShape("BLK_MMA_MNK", "n"),
                    ),
                    mem="rmem",
                    dtype="float",
                ),
            ),
        ],
        impl=NvOpImpl(
            r"""__syncthreads();
//> row major B float - cannot use ldmatrix.trans
// way1: direct load
// for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
//     int row_b = lid % 4;
//     int col_b = lid / 4;
//     REGB(_0{}, n_iter, buf_idx) = tileB_block(row_b, col_b + n_iter * 8);
//     REGB(_1{}, n_iter, buf_idx) = tileB_block(row_b + 4, col_b + n_iter * 8);
// }

// way2: ldmatrix + shuffle
for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
    int row_b = lid / 2;
    int col_b = lid % 2;
    ldmatrix_m8n8k8_x2(
        (uint32_t*)(REGB(_, n_iter, buf_idx).data()),
        (void*)(&tileB_block(row_b, col_b * 4 + n_iter * 8))
    );
    REGB(_0{}, n_iter, buf_idx) = __shfl_sync(0xffffffff, REGB(_0{}, n_iter, buf_idx), lid / 4 + lid % 4 * 8);
    REGB(_1{}, n_iter, buf_idx) = __shfl_sync(0xffffffff, REGB(_1{}, n_iter, buf_idx), lid / 4 + lid % 4 * 8);
}"""
        ),
        mem_type=("s", "r"),
    )
    dtc_main_loop_op.append(S2rBValLoadOp)
    CalculateOp = NvOp(
        name="CalculateOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="REGA",
                tensor=NvOpTensor(
                    shape=Shape(
                        IntShape(4),
                        MnkShape("BLK_MMA_MNK", "m"),
                    ),
                    mem="rmem",
                    dtype="float",
                    source=S2rAValLoadOp.outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="REGB",
                tensor=NvOpTensor(
                    shape=Shape(
                        IntShape(2),
                        MnkShape("BLK_MMA_MNK", "n"),
                    ),
                    mem="rmem",
                    dtype="float",
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
                        IntShape(4),
                        Shape(
                            MnkShape("BLK_MMA_MNK", "m"),
                            MnkShape("BLK_MMA_MNK", "n"),
                        ),
                    ),
                    mem="rmem",
                    dtype="float",
                ),
                origin=ZerosOp.outputs[0],
            ),
        ],
        impl=NvOpImpl(
            r"""for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) {
    for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
        uint32_t frag_A[4];
        uint32_t frag_B[2];
        asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(REGA(0, m_iter)));
        asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(REGA(1, m_iter)));
        asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(REGA(2, m_iter)));
        asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(REGA(3, m_iter)));
        asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(REGB(0, n_iter)));
        asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(REGB(1, n_iter)));

        mma_m16n8k8_fp32_tf32_tf32_fp32(REGC(_, make_coord(m_iter, n_iter)).data(), frag_A, frag_B);
    }
}"""
        ),
        mem_type=("r", "r"),
    )
    dtc_main_loop_op.append(CalculateOp)
    dtc_main_loop_op.set_loop_result(0, CalculateOp.outputs[0])

    g2r_row_window_is_split_op = NvOp(
        name="G2rRowWindowIsSplitOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="row_window_split",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("Mo")),
                    mem="gmem",
                    dtype="int",
                    source=dtc_program.gmem_tensor_ops["row_window_split"].outputs[0],
                )
            ),
            NvOpInput(
                idx=1,
                name="bind_i",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                    source=g2r_row_window_binding_op.outputs[0],
                )
            )
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="is_split",
                tensor=NvOpTensor(shape=Shape(), mem="rmem", dtype="int"),
            )
        ],
        impl=NvOpImpl(
            r"""
is_split(0) = row_window_split(bind_i);
"""
        ),
        mem_type=("g", "r"),
    )
    dtc_blk_x_loop_op.append(g2r_row_window_is_split_op)

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
            NvOpInput(
                idx=1,
                name="is_split",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                    source=g2r_row_window_is_split_op.outputs[0],
                )
            ),
            NvOpInput(
                idx=2,
                name="bind_id",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                    source=g2r_row_window_binding_op.outputs[0],
                )
            )
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
            r"""//! write back
for (int i_tileN = 0; i_tileN < get<1>(BLK_MNK{}) / 8; i_tileN++) {
    int row = lid / 4;
    int col = i_tileN * 8 + lid % 4 * 2;
    if (is_split) {
        atomicAdd(&C_val(col, make_coord(row, bind_id)), REGC(0, i_tileN));
        atomicAdd(&C_val(col + 1, make_coord(row, bind_id)), REGC(1, i_tileN));
        atomicAdd(&C_val(col, make_coord(row + 8, bind_id)), REGC(2, i_tileN));
        atomicAdd(&C_val(col + 1, make_coord(row + 8, bind_id)), REGC(3, i_tileN));
    } else {
        C_val(col, make_coord(row, bind_id)) = REGC(0, i_tileN);
        C_val(col + 1, make_coord(row, bind_id)) = REGC(1, i_tileN);
        C_val(col, make_coord(row + 8, bind_id)) = REGC(2, i_tileN);
        C_val(col + 1, make_coord(row + 8, bind_id)) = REGC(3, i_tileN);
    }
    
}"""
        ),
        mem_type=("r", "g"),
    )
    dtc_blk_x_loop_op.append(r2g_c_val_store_op)

    return dtc_main_loop_op, dtc_program


if __name__ == "__main__":
    from sparsene.op_gen.nvir.codegen import NvIrCodeGenerator

    dtc_main_loop_op, dtc_program = dtc()

    # Stage 0
    # 1. G2sSparseIndexLoadOp
    G2sSparseIndexLoadOp = dtc_program.find_op_by_name("G2sSparseIndexLoadOp")
    # 2. G2sSparseCooIdxLoadOp
    G2sSparseCooIdxLoadOp = dtc_program.find_op_by_name("G2sSparseCooIdxLoadOp")
    # 3. G2sSparseCooValLoadOp
    G2sSparseCooValLoadOp = dtc_program.find_op_by_name("G2sSparseCooValLoadOp")

    # Stage 1
    # 4. G2sMatrixBLoadOp
    G2sMatrixBLoadOp = dtc_program.find_op_by_name("G2sMatrixBLoadOp")
    # 5. S2sRestoreMatrixAOp
    S2sRestoreMatrixAOp = dtc_program.find_op_by_name("S2sRestoreMatrixAOp")

    # Stage 2
    # 6. S2rAValLoadOp
    S2rAValLoadOp = dtc_program.find_op_by_name("S2rAValLoadOp")
    # 7. S2rBValLoadOp
    S2rBValLoadOp = dtc_program.find_op_by_name("S2rBValLoadOp")
    # 8. CalculateOp
    CalculateOp = dtc_program.find_op_by_name("CalculateOp")

    s0 = NvOpSequence(
        G2sSparseIndexLoadOp,
        G2sSparseCooIdxLoadOp,
        G2sSparseCooValLoadOp,
    )
    s1 = NvOpSequence(
        G2sMatrixBLoadOp,
        S2sRestoreMatrixAOp,
    )
    s2 = NvOpSequence(
        S2rAValLoadOp,
        S2rBValLoadOp,
        CalculateOp,
    )

    apply_pipeline(
        dtc_main_loop_op,
        PipelinePlan([s0, s1, s2], [1, 1]),
    )

    script_dir = Path(__file__).parent
    with open(script_dir / "kernel_multi_binding.inc", "w") as f:
        f.write(NvIrCodeGenerator().dump_nvop_program(dtc_program))

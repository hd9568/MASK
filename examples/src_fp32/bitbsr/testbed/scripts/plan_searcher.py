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
from sparsene.op_gen.nvir.opgraph.graphviz_visualizer import visualize_op_graph

DTYPE_A = "float"
DTYPE_B = "float"
DTYPE_C = "float"
DTYPE_MASK = "uint64_t"
NFRAG_A = 4
NFRAG_B = 2
NFRAG_C = 4

BLK_MNK = (16, 64, 8)
MMA_MNK = (16, 8, 8)
BLK_MMA_MNK = (
    BLK_MNK[0] // MMA_MNK[0],
    BLK_MNK[1] // MMA_MNK[1],
    BLK_MNK[2] // MMA_MNK[2],
)

NBITS_PER_MASK = 64
TILE_MCO_VAL_SIZE = BLK_MNK[0] * BLK_MNK[2]
NMASK_PER_TILE = TILE_MCO_VAL_SIZE // NBITS_PER_MASK


def bitbsr():
    bitbsr_program = NvOpProgram(
        name="bitbsr_spmm_kernel_tf32tf32fp32",
        gmem_inouts={
            "B_val": GmemInout(
                shape=Shape(VarlenShape("K"), VarlenShape("N")),
                name="B_val",
                dtype=DTYPE_B,
                tensor_str="logical_divide(make_tensor(make_gmem_ptr(dB_val), make_layout(make_shape(K, N), make_stride(N, _1{}))), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, blockIdx.y))",
                tensor_type_str="decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_layout(make_shape(VARLEN, VARLEN), make_stride(VARLEN, _1{}))), make_shape(_1{}, get<1>(BLK_MNK{})))(_, make_coord(_, VARLEN)))",
            ),
            "C_val": GmemInout(
                shape=Shape(VarlenShape("M"), VarlenShape("N")),
                name="C_val",
                dtype=DTYPE_C,
                tensor_str="logical_divide(make_tensor(make_gmem_ptr(dC_val), make_layout(make_shape(M, N), make_stride(N, _1{}))), select<0, 1>(BLK_MNK{}))(make_coord(_, blockIdx.x), make_coord(_, blockIdx.y))",
                tensor_type_str="decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_layout(make_shape(VARLEN, VARLEN), make_stride(VARLEN, _1{}))), select<0, 1>(BLK_MNK{}))(make_coord(_, VARLEN), make_coord(_, VARLEN)))",
            ),
            "val_sidx": GmemInout(
                shape=Shape(VarlenShape("nnz_block")),
                name="val_sidx",
                dtype="int",
                tensor_str="make_tensor(make_gmem_ptr(dval_sidx), make_shape(nnz_block))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)))",
            ),
            "val_soff": GmemInout(
                shape=Shape(VarlenShape("Mo")),
                name="val_soff",
                dtype="int",
                tensor_str="make_tensor(make_gmem_ptr(dval_soff), make_shape(Mo))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)))",
            ),
            "val_mco_mask": GmemInout(
                shape=Shape(ParamShape("nmask_per_tile"), VarlenShape("nnz_block")),
                name="val_mco_mask",
                dtype=DTYPE_MASK,
                tensor_str="make_tensor(make_gmem_ptr(dval_mco_mask), make_shape(nmask_per_tile{}, nnz_block))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(nmask_per_tile{}, VARLEN)))",
                parameters={
                    "nmask_per_tile": NMASK_PER_TILE,
                },
            ),
            "val_mco_off": GmemInout(
                shape=Shape(ParamShape("nmask_per_tile"), VarlenShape("nnz_block")),
                name="val_mco_off",
                dtype="int",
                tensor_str="make_tensor(make_gmem_ptr(dval_mco_off), make_shape(nnz_block))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)))",
                parameters={
                    "nmask_per_tile": NMASK_PER_TILE,
                },
            ),
            "val_mco_val": GmemInout(
                shape=Shape(VarlenShape("nnz")),
                name="val_mco_val",
                dtype=DTYPE_A,
                tensor_str="make_tensor(make_gmem_ptr(dval_mco_val), make_shape(nnz))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)))",
            ),
        },
    )

    # bitbsr program body
    Zero_op = ConstantNvOp(
        name="ZeroOp",
        shape=Shape(),
        dtype="int",
        value=0,
    )
    bitbsr_program.append_op(Zero_op)
    BlockDimY_op = ConstantNvOp(
        name="BlockDimYOp",
        shape=Shape(),
        dtype="int",
        value="blockDim.y",
    )
    bitbsr_program.append_op(BlockDimY_op)
    BlockDimX_op = ConstantNvOp(
        name="BlockDimXOp",
        shape=Shape(),
        dtype="int",
        value="blockDim.x",
    )
    bitbsr_program.append_op(BlockDimX_op)

    bitbsr_blk_y_loop_op = ForLoopNvOp(
        name="bitbsr_spmm_kernel_blk_y_loop",
        blk_idx_mapping="y",
        loop_l=NvOpInput(
            idx=0,
            name="l",
            tensor=NvOpTensor(
                shape=Shape(),
                mem="rmem",
                dtype="int",
                source=Zero_op.outputs[0],
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
    )
    bitbsr_program.append_op(bitbsr_blk_y_loop_op)

    # blk_y_loop body
    bitbsr_blk_x_loop_op = ForLoopNvOp(
        name="bitbsr_spmm_kernel_blk_x_loop",
        blk_idx_mapping="x",
        loop_l=NvOpInput(
            idx=0,
            name="l",
            tensor=NvOpTensor(
                shape=Shape(),
                mem="rmem",
                dtype="int",
                source=Zero_op.outputs[0],
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
    )
    bitbsr_blk_y_loop_op.append(bitbsr_blk_x_loop_op)

    # blk_x_loop body
    G2rSparseOffsetLoadOp = NvOp(
        name="G2rSparseOffsetLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="val_soff",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("Mo")),
                    mem="gmem",
                    dtype="int",
                    source=bitbsr_program.gmem_tensor_ops["val_soff"].outputs[0],
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
    bitbsr_blk_x_loop_op.append(G2rSparseOffsetLoadOp)

    ZerosOp = ConstantNvOp(
        name="ZerosOp",
        shape=Shape(
            IntShape(NFRAG_C),
            Shape(MnkShape("BLK_MMA_MNK", "m"), MnkShape("BLK_MMA_MNK", "n")),
        ),
        dtype=DTYPE_C,
        value=0,
    )
    bitbsr_blk_x_loop_op.append(ZerosOp)

    bitbsr_main_loop_op = ForLoopNvOp(
        name="BitBsrMainLoopOp",
        loop_l=NvOpInput(
            idx=0,
            name="l",
            tensor=NvOpTensor(
                shape=Shape(),
                mem="rmem",
                dtype="int",
                source=G2rSparseOffsetLoadOp.outputs[0],
            ),
        ),
        loop_r=NvOpInput(
            idx=1,
            name="r",
            tensor=NvOpTensor(
                shape=Shape(),
                mem="rmem",
                dtype="int",
                source=G2rSparseOffsetLoadOp.outputs[1],
            ),
        ),
        loop_result=[
            NvOpOutput(
                idx=0,
                name="REGC",
                tensor=NvOpTensor(
                    shape=Shape(
                        IntShape(NFRAG_C),
                        Shape(
                            MnkShape("BLK_MMA_MNK", "m"), MnkShape("BLK_MMA_MNK", "n")
                        ),
                    ),
                    mem="rmem",
                    dtype=DTYPE_C,
                ),
            ),
        ],
    )
    bitbsr_blk_x_loop_op.append(bitbsr_main_loop_op)

    # main_loop body
    G2rSparseIndexLoadOp = NvOp(
        name="G2rSparseIndexLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="val_sidx",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("nnz_block")),
                    mem="gmem",
                    dtype="int",
                    source=bitbsr_program.gmem_tensor_ops["val_sidx"].outputs[0],
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
                name="tile_sidx",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                ),
            ),
        ],
        impl=NvOpImpl(r"""tile_sidx(0, buf_idx) = val_sidx(i);"""),
        mem_type=("g", "r"),
    )
    bitbsr_main_loop_op.append(G2rSparseIndexLoadOp)

    G2rSparseMcoOffLoadOp = NvOp(
        name="G2rSparseMcoOffLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="val_mco_off",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("nnz_block")),
                    mem="gmem",
                    dtype="int",
                    source=bitbsr_program.gmem_tensor_ops["val_mco_off"].outputs[0],
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
                name="mco_off_range",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(2)),
                    mem="rmem",
                    dtype="int",
                ),
            ),
        ],
        impl=NvOpImpl(
            r"""mco_off_range(0, buf_idx) = val_mco_off(i);
mco_off_range(1, buf_idx) = val_mco_off(i + 1);"""
        ),
        mem_type=("g", "r"),
        parameters={
            "nmask_per_tile": NMASK_PER_TILE,
        },
    )
    bitbsr_main_loop_op.append(G2rSparseMcoOffLoadOp)

    G2rSparseMcoMaskLoadOp = NvOp(
        name="G2rSparseMcoMaskLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="val_mco_mask",
                tensor=NvOpTensor(
                    shape=Shape(ParamShape("nmask_per_tile"), VarlenShape("nnz_block")),
                    mem="gmem",
                    dtype="uint64_t",
                    source=bitbsr_program.gmem_tensor_ops["val_mco_mask"].outputs[0],
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
                name="tile_mco_mask",
                tensor=NvOpTensor(
                    shape=Shape(ParamShape("nmask_per_tile")),
                    mem="rmem",
                    dtype=DTYPE_MASK,
                ),
            ),
        ],
        impl=NvOpImpl(
            r"""CUTE_STATIC_ASSERT_V((get<0>(shape_i0) == size<0>(shape(val_mco_mask))));
auto tiler = make_shape(Int<sizeof(uint128_t) / sizeof(uint64_t)>{});
auto src = flat_divide(val_mco_mask(_, i), tiler);
auto dst = flat_divide(tile_mco_mask(_, buf_idx), tiler);
copy(Copy_Atom<UniversalCopy<uint128_t>, uint64_t>{}, src, dst);"""
        ),
        mem_type=("g", "r"),
        parameters={
            "nmask_per_tile": NMASK_PER_TILE,
        },
    )
    bitbsr_main_loop_op.append(G2rSparseMcoMaskLoadOp)

    G2sSparseMcoValLoadOp = NvOp(
        name="G2sSparseMcoValLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="val_mco_val",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("nnz")),
                    mem="gmem",
                    dtype=DTYPE_A,
                    source=bitbsr_program.gmem_tensor_ops["val_mco_val"].outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="mco_off_range",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(2)),
                    mem="rmem",
                    dtype="int",
                    source=G2rSparseMcoOffLoadOp.outputs[0],
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
                name="tile_mco_val",
                tensor=NvOpTensor(
                    shape=Shape(ParamShape("tile_mco_val_size")),
                    mem="smem",
                    dtype=DTYPE_A,
                ),
            ),
        ],
        impl=NvOpImpl(
            r"""int ll = mco_off_range(0);
int rr = mco_off_range(1);

auto thr_tiler = make_shape(_4{});
auto input = make_tensor(val_mco_val.data() + ll, tile_mco_val_size{});
for (int i_load = lid; i_load * 4 + ll < rr; i_load += 32) {
    auto thr_coord = make_coord(i_load);
    auto src = local_tile(input, thr_tiler, thr_coord);
    auto dst = local_tile(tile_mco_val(_, buf_idx), thr_tiler, thr_coord);
    copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>{}, src, dst);
}"""
        ),
        mem_type=("g", "s"),
        parameters={
            "tile_mco_val_size": TILE_MCO_VAL_SIZE,
        },
    )
    bitbsr_main_loop_op.append(G2sSparseMcoValLoadOp)

    G2sMatrixBLoadOp = NvOp(
        name="G2sMatrixBLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="B_val",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("K"), VarlenShape("N")),
                    mem="gmem",
                    dtype=DTYPE_B,
                    source=bitbsr_program.gmem_tensor_ops["B_val"].outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="tile_sidx",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                    source=G2rSparseIndexLoadOp.outputs[0],
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
                    dtype=DTYPE_B,
                    row_major=True,
                    swizzle=SwizzleLayout(b=3, m=3, s=3),
                ),
            ),
        ],
        impl=NvOpImpl(
            r"""auto load_tile_n = _4{};
for (int iter_i = lid; iter_i < get<2>(BLK_MNK{}) * get<1>(BLK_MNK{}) / load_tile_n; iter_i += 32) {
    int row = iter_i / (get<1>(BLK_MNK{}) / load_tile_n);
    int col = iter_i % (get<1>(BLK_MNK{}) / load_tile_n);
    int gmem_row = tile_sidx * get<2>(BLK_MNK{}) + row;
    auto thr_tiler_gmem = make_shape(_1{}, _4{});
    auto thr_coord_gmem = make_coord(gmem_row, col);
    auto B_val_thr = local_tile(B_val, thr_tiler_gmem, thr_coord_gmem);
    auto thr_tiler_smem = make_shape(_1{}, _4{});
    auto thr_coord_smem = make_coord(row, col);
    auto dst = local_tile(tileB_block(_, _, buf_idx), thr_tiler_smem, thr_coord_smem);
    copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>{}, B_val_thr(0, _), dst(0, _));   
}"""
        ),
        mem_type=("g", "s"),
    )
    bitbsr_main_loop_op.append(G2sMatrixBLoadOp)

    S2sRestoreMatrixAOp = NvOp(
        name="S2sRestoreMatrixAOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="tile_mco_val",
                tensor=NvOpTensor(
                    shape=Shape(ParamShape("tile_mco_val_size")),
                    mem="smem",
                    dtype=DTYPE_A,
                    source=G2sSparseMcoValLoadOp.outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="tile_mco_mask",
                tensor=NvOpTensor(
                    shape=Shape(ParamShape("nmask_per_tile")),
                    mem="rmem",
                    dtype=DTYPE_MASK,
                    source=G2rSparseMcoMaskLoadOp.outputs[0],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="tile_mco_restore_val_block",
                tensor=NvOpTensor(
                    shape=Shape(MnkShape("BLK_MNK", "m"), MnkShape("BLK_MNK", "k")),
                    mem="smem",
                    dtype=DTYPE_A,
                    row_major=True,
                    swizzle=SwizzleLayout(b=1, m=3, s=3),
                ),
            ),
        ],
        impl=NvOpImpl(
            r"""int off = 0;
for (int i_mask = 0; i_mask < nmask_per_tile{}; i_mask++) {
    uint64_t mask = tile_mco_mask(i_mask);
    for (int local_vid = tid; local_vid < 64; local_vid += 32 /* num threads */) {
        int local_idx = 0;
        if (mask & (1ULL << local_vid)) {
            local_idx = __popcll(mask << (63 - local_vid));
        }
        int vid = i_mask * 64 + local_vid;
        if (local_idx == 0) {
            *((float*)tile_mco_restore_val_block(_, _, buf_idx).data().get() + vid) = 0;
        } else {
            *((float*)tile_mco_restore_val_block(_, _, buf_idx).data().get() + vid) = tile_mco_val(off + local_idx - 1);
        }
    }
    off += __popcll(mask);
}"""
        ),
        mem_type=("s", "s"),
        parameters={
            "tile_mco_val_size": TILE_MCO_VAL_SIZE,
            "nmask_per_tile": NMASK_PER_TILE,
        },
        cp_async=False,
    )
    bitbsr_main_loop_op.append(S2sRestoreMatrixAOp)

    S2rAValLoadOp = NvOp(
        name="S2rAValLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="tile_mco_restore_val_block",
                tensor=NvOpTensor(
                    shape=Shape(
                        MnkShape("BLK_MNK", "m"),
                        MnkShape("BLK_MNK", "k"),
                    ),
                    mem="smem",
                    dtype=DTYPE_A,
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
                        IntShape(NFRAG_A),
                        MnkShape("BLK_MMA_MNK", "m"),
                    ),
                    mem="rmem",
                    dtype=DTYPE_A,
                ),
            ),
        ],
        impl=NvOpImpl(
            r"""int row = lid % 16;
int col = lid / 16;
ldmatrix_m8n8k8_x4(
    (uint32_t*)(REGA(_, _0{}, buf_idx).data()),
    (void*)(&tile_mco_restore_val_block(row, col * 4))
);"""
        ),
        mem_type=("s", "r"),
    )
    bitbsr_main_loop_op.append(S2rAValLoadOp)

    S2rBValLoadOp = NvOp(
        name="S2rBValLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="tileB_block",
                tensor=NvOpTensor(
                    shape=Shape(MnkShape("BLK_MNK", "k"), MnkShape("BLK_MNK", "n")),
                    mem="smem",
                    dtype=DTYPE_B,
                    source=G2sMatrixBLoadOp.outputs[0],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="REGB",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(NFRAG_B), MnkShape("BLK_MMA_MNK", "n")),
                    mem="rmem",
                    dtype=DTYPE_B,
                ),
            ),
        ],
        impl=NvOpImpl(
            r"""for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
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
    bitbsr_main_loop_op.append(S2rBValLoadOp)

    CalculateOp = NvOp(
        name="CalculateOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="REGA",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(NFRAG_A), MnkShape("BLK_MMA_MNK", "m")),
                    mem="rmem",
                    dtype=DTYPE_A,
                    source=S2rAValLoadOp.outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="REGB",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(NFRAG_B), MnkShape("BLK_MMA_MNK", "n")),
                    mem="rmem",
                    dtype=DTYPE_B,
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
                        IntShape(NFRAG_C),
                        Shape(
                            MnkShape("BLK_MMA_MNK", "m"), MnkShape("BLK_MMA_MNK", "n")
                        ),
                    ),
                    mem="rmem",
                    dtype=DTYPE_C,
                ),
                origin=ZerosOp.outputs[0],
            )
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
    bitbsr_main_loop_op.append(CalculateOp)
    bitbsr_main_loop_op.set_loop_result(0, CalculateOp.outputs[0])

    R2gCValStoreOp = NvOp(
        name="R2gCValStoreOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="REGC",
                tensor=NvOpTensor(
                    shape=Shape(
                        IntShape(NFRAG_C),
                        Shape(
                            MnkShape("BLK_MMA_MNK", "m"),
                            MnkShape("BLK_MMA_MNK", "n"),
                        ),
                    ),
                    mem="rmem",
                    dtype=DTYPE_C,
                    source=bitbsr_main_loop_op.outputs[0],
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
                    dtype=DTYPE_C,
                ),
                origin=bitbsr_program.gmem_tensor_ops["C_val"].outputs[0],
            ),
        ],
        impl=NvOpImpl(
            r"""//! write back
for (int i_tileN = 0; i_tileN < get<1>(BLK_MNK{}) / 8; i_tileN++) {
    int row = lid / 4;
    int col = i_tileN * 8 + lid % 4 * 2;
    C_val(row, col) = REGC(0, i_tileN);
    C_val(row, col + 1) = REGC(1, i_tileN);
    C_val(row + 8, col) = REGC(2, i_tileN);
    C_val(row + 8, col + 1) = REGC(3, i_tileN);
}"""
        ),
        mem_type=("r", "g"),
    )
    bitbsr_blk_x_loop_op.append(R2gCValStoreOp)

    return bitbsr_main_loop_op, bitbsr_program


if __name__ == "__main__":
    from sparsene.op_gen.nvir.codegen import NvIrCodeGenerator

    bitbsr_main_loop_op, bitbsr_program = bitbsr()

    op_graph = construct_graph_from_op_sequence(bitbsr_main_loop_op.body)
    visualize_op_graph(op_graph, str(Path(__file__).parent / "bitbsr.dot"))

    validator = NeighborDependencyValidator(op_graph)
    plans = enumerate_pipeline_plans(
        bitbsr_main_loop_op,
        validator,
        min_nstages=2,
        max_nstages=3,
        min_ops_per_stage=2,
        max_ops_per_stage=4,
        min_shift=1,
        max_shift=3,
    )
    plans_dir = Path(__file__).parent.parent / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    with open(plans_dir.parent / "plans.txt", "w") as f_plans:
        for idx, plan in enumerate(plans):
            print(f"Plan {idx}: {plan}")
            new_loop, new_bitbsr_program = bitbsr()
            apply_pipeline(new_loop, plan)
            with open(plans_dir / f"plan_{idx}.inc", "w") as f_kernel:
                f_kernel.write(
                    NvIrCodeGenerator().dump_nvop_program(new_bitbsr_program)
                )
            f_plans.write(f"{idx},{plan}\n")

    print("Possible plans:", len(plans))

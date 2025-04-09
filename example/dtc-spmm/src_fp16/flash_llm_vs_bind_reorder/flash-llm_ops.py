import os

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

#> flash llm code template
# int l = 0;
# int r = (K + get<2>(BLK_MNK{}) - 1) / get<2>(BLK_MNK{});
# int k = r - l;
# int i = 0;
# for (; i < k; i++) {
#     // g2r load tile offsets 

#     // g2r load tile A
    
#     // g2s load tile B

#     // r2s restore tile A

#     // s2r load tile A 
    
#     // s2r load tile B

#     // compute
# }

def flash_llm():

    flash_llm_program = NvOpProgram(
        name="flash_llm_spmm_kernel_sstage_rstage",
        gmem_inouts={
            "tile_offsets": GmemInout( #! how to represent divide in this case
                shape=Shape(VarlenShape("K / get<2>(BLK_MNK{})"), VarlenShape("M / get<0>(BLK_MNK{})")),
                name="tile_offsets",
                dtype="int",
                tensor_str="logical_divide(make_tensor(make_gmem_ptr(dtile_offsets), make_shape(K / get<2>(BLK_MNK{}), M / get<0>(BLK_MNK{}))), make_shape(K / get<2>(BLK_MNK{}), _1{}))(_, blockIdx.x)",
                tensor_type_str="decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN, _1{})))(_, VARLEN))"
            ),
            "compressed_A": GmemInout( #! load directly from GMEM to reg (no need to pad)
                shape=Shape(VarlenShape("nnz")),
                name="compressed_A",
                dtype="uint32_t",
                tensor_str="make_tensor(make_gmem_ptr(dcompressed_A), make_shape(nnz))",
                tensor_type_str="decltype(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN)))",
            ),
            "B_val": GmemInout(
                shape=Shape(VarlenShape("K"), VarlenShape("N")),
                name="B_val",
                dtype="half",
                tensor_str="logical_divide(make_tensor(make_gmem_ptr(dB_val), make_shape(K, N)), make_shape(K, get<1>(BLK_MNK{})))(_, blockIdx.y)",
                tensor_type_str="decltype(logical_divide(make_tensor(make_gmem_ptr((TO0*)nullptr), make_shape(VARLEN, VARLEN)), make_shape(VARLEN, get<1>(BLK_MNKP{})))(_, VARLEN))",
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

    # flash-llm program body
    blk_y_loop_zero_op = ConstantNvOp(
        name="BlkYLoopZeroOp",
        shape=Shape(),
        dtype="int",
        value=0
    )
    flash_llm_program.append_op(blk_y_loop_zero_op)
    BlockDimY_op = ConstantNvOp(
        name="BlockDimYOp",
        shape=Shape(),
        dtype="int",
        value="blockDim.y",
    )
    flash_llm_program.append_op(BlockDimY_op)
    flash_llm_blk_y_loop_op = ForLoopNvOp(
        name="flash_spmm_kernel_blk_y_loop",
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
                source=BlockDimY_op.outputs[0]
            ),
        ),
        loop_result=[],
        body=NvOpSequence(),
        iter_args={},
    )
    flash_llm_program.append_op(flash_llm_blk_y_loop_op)
    
    # blk_y_loop_op body
    blk_x_loop_zero_op = ConstantNvOp(
        name="BlkXLoopZeroOp",
        shape=Shape(),
        dtype="int",
        value=0
    )
    # TODO: check is 'append' or 'append_op'
    flash_llm_blk_y_loop_op.append(blk_x_loop_zero_op)
    BlockDimX_op = ConstantNvOp(
        name="BlockDimXOp",
        shape=Shape(),
        dtype="int",
        value="blockDim.x",
    )
    flash_llm_blk_y_loop_op.append(BlockDimX_op)
    flash_llm_blk_x_loop_op = ForLoopNvOp(
        name="flash_llm_spmm_kernel_blk_x_loop",
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
    flash_llm_blk_y_loop_op.append(flash_llm_blk_x_loop_op)

    # blk_x_loop_op body
    # TODO: loop的左右两边范围有没有轻量级的写法？
    flash_llm_loop_bound_zero_op = ConstantNvOp(
        name="FlashLLMLoopBoundZeroOp",
        shape=Shape(),
        dtype="int",
        value=0
    )

    flash_llm_loop_bound_up_op = ConstantNvOp(
        name="FlashLLMLoopBoundUpOp",
        shape=Shape(),
        dtype="int",
        value="((K + get<2>(BLK_MNK{}) - 1) / get<2>(BLK_MNK{}))"
    )

    flash_llm_blk_x_loop_op.append(flash_llm_loop_bound_zero_op)
    flash_llm_blk_x_loop_op.append(flash_llm_loop_bound_up_op)

    #> 这两个参数怎么写比较好？
    # in flash-llm compressed_A load op, every thread load uint4 a time
    flash_llm_load_compressed_A_vec_size_op = ConstantNvOp(
        name="FlashLLMLoadCompressedAVecSizeOp",
        shape=Shape(),
        dtype="int",
        value=4
    )
    flash_llm_blk_x_loop_op.append(flash_llm_load_compressed_A_vec_size_op)

    flash_llm_load_compressed_A_num_reg_op = ConstantNvOp(
        name="FlashLLMLoadCompressedANumRegOp",
        shape=Shape(),
        dtype="int",
        value="(get<0>(BLK_MNK{}) * get<2>(BLK_MNK{}) + 32 - 1) / 32"
    )
    flash_llm_blk_x_loop_op.append(flash_llm_load_compressed_A_num_reg_op)

    #> loop result, outputs
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
    flash_llm_blk_x_loop_op.append(zeros_op)

    flash_llm_main_loop_op = ForLoopNvOp(
        name="FlashLLMMainLoopOp",
        loop_l=NvOpInput(
            idx=0,
            name="l",
            tensor=NvOpTensor(
                shape=Shape(),
                mem="rmem",
                dtype="int",
                source=flash_llm_loop_bound_zero_op.outputs[0],
            ),
        ),
        loop_r=NvOpInput(
            idx=1,
            name="r",
            tensor=NvOpTensor(
                shape=Shape(),
                mem="rmem",
                dtype="int",
                source=flash_llm_loop_bound_up_op.outputs[0],
            ),
        ),
        loop_result=[
            NvOpOutput(
                idx=0,
                name="REGC",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(4), Shape(MnkShape("BLK_MMA_MNK", "m"), MnkShape("BLK_MMA_MNK", "n"))),
                    mem="rmem",
                    dtype="float",
                ),
            ),
        ],
        body=NvOpSequence(),
        iter_args={},
    )
    flash_llm_blk_x_loop_op.append(flash_llm_main_loop_op)

    # main_loop body
    g2r_tile_offsets_load_op = NvOp(
        name="G2rTileOffsetsLoadOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="tile_offsets",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("(K + get<2>(BLK_MNK{}) - 1) / get<2>(BLK_MNK{})"), IntShape(1)),
                    mem="gmem",
                    dtype="int",
                    source=flash_llm_program.gmem_tensor_ops["tile_offsets"].outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="i",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                    source="{c}",
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="tile_offsets_reg",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(2)),
                    mem="rmem",
                    dtype="int",
                ),
            ),
        ],
        impl=NvOpImpl(
            code_template="""
tile_offsets_reg(0) = tile_offsets(i);
tile_offsets_reg(1) = tile_offsets(i + 1);
            """
        ),
        mem_type=("g", "r"),
    )
    flash_llm_main_loop_op.append(g2r_tile_offsets_load_op)

    #> g2r load tile A, input(compressed_A), input(tile_offsets: l & r), 
    #> behavior: load a tile, start from l to r in compressed_A, 
    #> outputs: compressed_A_reg
    g2r_load_compressed_tile_A_op = NvOp(
        name="G2rLoadCompressedTileAOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="compressed_A",
                tensor=NvOpTensor(
                    shape=Shape(VarlenShape("nnz")),
                    mem="gmem",
                    dtype="uint32_t",
                    source=flash_llm_program.gmem_tensor_ops["compressed_A"].outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="tile_offsets_reg",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(2)),
                    mem="rmem",
                    dtype="int",
                    source=g2r_tile_offsets_load_op.outputs[0],
                ),
            ),
        ],
        # TODO: reg number per thread: 1 warp per thread block, thread block work on BLK_M * BLK_K, so (BLK_M * BLK_K / 32) is the reg number per thread
        outputs=[
            NvOpOutput(
                idx=0,
                name="compressed_A_reg",
                tensor=NvOpTensor(
                    # reg number = vec len (4) * xxx
                    shape=Shape(IntShape(4), ParamShape("compressed_A_reg_size / 4")),
                    mem="rmem",
                    dtype="uint32_t",
                ),
            ),
        ],
        impl=NvOpImpl(
            code_template="""
int max_nnz_vector_this_tile = tile_offsets_reg(1) - tile_offsets_reg(0);
for (int i = 0; i < compressed_A_reg_size{} / 4; i++) {
    int index = lid + i * (32);
    if (index >= max_nnz_vector_this_tile) {
        break;
    }
    auto dst = compressed_A_reg(_, i);
    auto src = compressed_A(_, tile_offsets_reg(0) + index);
    copy(Copy_Atom<UniversalCopy<uint128_t>, uint32_t>{}, src, dst);
}
            """
        ),
        mem_type=("g", "r"),
        # 这个parameters是[str, int]类型的？能否表示 BLK_M * BLK_K / 32
        parameters={
            "compressed_A_reg_size": "(get<0>(BLK_MNK{}) * get<2>(BLK_MNK{}) + 32 - 1) / 32",
            "tile_val_idx_bind_size": "get<0>(BLK_MNK{}) * get<2>(BLK_MNK{})",
        },
    )
    flash_llm_main_loop_op.append(g2r_load_compressed_tile_A_op)

    # TODO: input(i) is the iter_arg of main_loop_op, can we get the iter_arg
    #> g2s_load_tile_B_op: input(B_val), input(i)
    #> behavior: load a tileB, 
    #> outputs: tileB
    g2s_load_tile_B_op = NvOp(
        name="c",
        inputs=[
            NvOpInput(
                idx=0,
                name="B_val",
                tensor=NvOpTensor(
                    Shape=Shape(VarlenShape("K"), MnkShape("BLK_MNK", "n")),
                    mem="smem",
                    dtype="half",
                    source=flash_llm_program.gmem_tensor_ops["B_val"].outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="tile_idx",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                    source=None, # TODO : which source
                ),
            ),
        ],
        # TODO: nbuf 自动添加吗？
        outputs=[
            NvOpOutput(
                idx=0,
                name="tileB_block",
                tensor=NvOpTensor(
                    shape=Shape(MnkShape("BLK_MNK", "k"), MnkShape("BLK_MNK", "n"), ParamShape("NBUF")),
                    mem="smem",
                    dtype="half"
                ),
            ),
        ],
        impl=NvOpImpl(
            code_template="""
__syncthreads();
auto load_tile_n = _8{};
for (int iter_i = 0; iter_i < get<2>(BLK_MNK{}) * get<1>(BLK_MNK{}) / load_tile_n; tile_i += 32) {
    int row = tile_i / (get<1>(BLK_MNK{}) / load_tile_n);
    int col = tile_i % (get<1>(BLK_MNK{}) / load_tile_n);
    int gmem_row = tile_idx * get<1>(BLK_MNK{}) + row;
    auto thr_tiler_gmem = make_shape(_1{}, _8{});
    auto thr_coord_gmem = make_coord(gmem_row, col);
    auto B_val_thr = local_tile(B_val, thr_tiler_gmem, thr_coord_gmem);
    auto thr_tiler_smem = make_shape(_1{}, _8{});
    auto thr_coord_smem = make_coord(row, col);
    auto dst = local_tile(tileB_block(_, _, buf_idx), thr_tiler_smem, thr_coord_smem);
    copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{}, B_val_thr(0, _), dst(0, _));
}
            """
        ),
        mem_type=("g", "s"),
    )
    flash_llm_main_loop_op.append(g2s_load_tile_B_op)

    #> r2s_restore_tile_A_op: restore tile A from reg to SMEM
    #> inputs: input(compressedA_reg)
    #> outputs: outputs(tileA_block)
    r2s_restore_tile_A_op = NvOp(
        name="R2sRestoreTileAOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="compressed_A_reg",
                tensor=NvOpTensor(
                    shape=(IntShape(4), ParamShape("compressed_A_reg_size / 4")),
                    mem="rmem",
                    dtype="uint32_t",
                    source=g2r_load_compressed_tile_A_op.outputs[0],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="tileA_block",
                tensor=NvOpTensor(
                    shape=(MnkShape("BLK_MNK", "m"), MnkShape("BLK_MNK", "k")),
                    mem="smem",
                    dtype="half",
                ),
            ),
        ],
        impl=NvOpImpl(
            code_template="""
for (int i_o2s = lid; i_o2s < () / 2; i_o2s += 32) {
    *((float*)(tileA_block(_, _, buf_idx).data().get() + i_o2s)) = 0;
}
__syncthreads();
for (int i = 0; i < compressed_A_reg_size{} / 4; i++) {
    int index = lid + i * (32);
    if (index >= max_nnz_vector_this_tile) {
        break;
    }
    for (int reg_ptr = 0; reg_ptr < 4; reg_ptr++) {
        uint32_t val_idx_bind = compressed_A_reg(reg_ptr, i);
        half value;
        short idx;
        unpack_half_short(val_idx_bind, value, idx);
        int row = idx / get<2>(BLK_MNK{});
        int col = idx % get<2>(BLK_MNK{});
        tile_coo_restore_val_block(row, col, buf_idx) = value;
    }
}
            """
        ),
        mem_type=("r", "s"),
    )
    flash_llm_main_loop_op.append(r2s_restore_tile_A)

    #> s2r_load_tile_A_op
    # inputs: tileA_block
    # outputs: REGA
    s2r_load_tile_A_op = NvOp(
        name="S2rLoadTileAOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="tileA_block",
                tensor=NvOpTensor(
                    shape=Shape(MnkShape("BLK_MNK", "k"), MnkShape("BLK_MNK", "m")),
                    mem="smem",
                    dtype="half",
                    source=r2s_restore_tile_A_op.outputs[0],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0, 
                name="REGA",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(8), Shape(MnkShape("BLK_MMA_MNK", "m"), MnkShape("BLK_MMA_MNK", "k"))),
                    mem="rmem",
                    dtype="half",
                ),
            ),
        ],
        impl=NvOpImpl(
            code_template="""
__syncthreads();
for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) {
    for (int k_iter = 0; k_iter < get<2>(BLK_MMA_MNK{}); k_iter++) {
        int row = lid % 16 + m_iter * 16;
        int col = lid / 16 + k_iter * 2;

        ldmatrix_m8n8k8_x4(
            (uint32_t*)(REGA(_, make_coord(m_iter, n_iter), buf_idx).data()),
            (void*)&tileA_block(row, col * 8)
        )
    }
}
            """
        ),
        mem_type=("s", "r"),
    )
    flash_llm_main_loop_op.append(s2r_load_tile_A_op)

    #> s2r_load_tile_B_op:
    # inputs: tileB_block
    # outputs REGB
    s2r_load_tile_B_op = NvOp(
        name="S2rLoadTileBOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="tileB_block",
                tensor=NvOpTensor(
                    shape=Shape(MnkShape("BLK_MNK", "k"), MnkShape("BLK_MNK", "n")),
                    mem="smem",
                    dtype="half",
                    source=g2s_load_tile_B_op.outputs[0],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="REGB",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(4), Shape(MnkShape("BLK_MMA_MNK", "k"), MnkShape("BLK_MMA_MNK", "n"))),
                    mem="rmem",
                    dtype="half",
                ),
            ),
        ],
        impl=NvOpImpl(
            code_template="""
__syncthreads();
for (int k_iter = 0; k_iter < get<2>(BLK_MMA_MNK{}); k_iter++) {
    for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
        int row_b = lid % 16 + k_iter * 16;
        int col_b = n_iter;
        ldmatrix_m8n8k8_x2_trans(
            (uint32_t*)(REGB(_, make_coord(k_iter, n_iter), buf_idx).data()),
            (void*)(&tileB_block(row_b, col_b * 8))
        );
    }
}
            """
        ),
        mem_type=("s", "r"),
    )
    flash_llm_main_loop_op.append(s2r_load_tile_B_op)

    #> mma_op
    # inputs: REGA, REGB
    # outputs: 
    mma_op = NvOp(
        name="MmaOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="REGA",
                tensor=NvOpTensor(
                    shape=(IntShape(8), Shape(MnkShape("BLK_MMA_MNK", "m"), MnkShape("BLK_MMA_MNK", "k"))),
                    mem="rmem",
                    dtype="half",
                    source=s2r_load_tile_A_op.outputs[0],
                ),
            ),
            NvOpInput(
                idx=1,
                name="REGB",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(4), Shape(MnkShape("BLK_MMA_MNK", "k"), MnkShape("BLK_MMA_MNK", "n"))),
                    mem="rmem",
                    dtype="half",
                    source=s2r_load_tile_B_op.outputs[0],
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="REGC",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(4), Shape(MnkShape("BLK_MMA_MNK", "m"), MnkShape("BLK_MMA_MNK", "n"))),
                    mem="rmem",
                    dtype="float",
                ),
                origin=zeros_op.outputs[0],
            ),
        ],
        impl=NvOpImpl(
            code_template="""
for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) {
    for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
        for (int k_iter = 0; k_iter < get<2>(BLK_MMA_MNK{}); k_iter++) {
            mma_m16n8k16_fp32(REGC(_, make_coord(m_iter, n_iter)).data(), (uint32_t*)(REGA(_, make_coord(m_iter, k_iter)).data()), REGB(_, make_coord(k_iter, n_iter)).data());
        }
    }
}
            """
        ),
        mem_type=("r", "r"),
    )
    flash_llm_main_loop_op.append(mma_op)

    flash_llm_main_loop_op.set_loop_result(0, mma_op.outputs[0])

    #> r2g_val_store_op
    # inputs: 
    r2g_c_val_store_op = NvOp(
        name="R2gCValStoreOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="REGC",
                tensor=NvOpTensor(
                    shape=Shape(IntShape(4), Shape(MnkShape("BLK_MMA_MNK", "m"), MnkShape("BLK_MMA_MNK", "n"))),
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
                    shape=Shape(MnkShape("BLK_MNK", "n"), MnkShape("BLK_MNK", "m")),
                    mem="gmem",
                    dtype="float",
                ),
                origin=flash_llm_program.gmem_tensor_ops["C_val"].outputs[0],
            )
        ],
        impl=NvOpImpl(
            code_template="""
//! write back
for (int i_tileM = 0; i_tileM < get<0>(BLK_MMA_MNK{}); i_tileM++) {
    for (int i_tileN = 0; i_tileN < get<1>(BLK_MMA_MNK{}); i_tileN++) {
        int row = lid / 4 + i_tileM * 16;
        int col = lid % 4 * 2 + i_tileN * 8;
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
    flash_llm_blk_x_loop_op.append(r2g_c_val_store_op)

    return flash_llm_main_loop_op, flash_llm_program

if __name__ == "__main__":
    from sparsene.op_gen.nvir.codegen import NvIrCodeGenerator

    from sparsene.op_gen.nvir.opgraph.graphviz_visualizer import visualize_op_graph
    from sparsene.op_gen.nvir.opgraph.graph import construct_graph_from_op_sequence

    flash_llm_main_loop_op, flash_llm_program = flash_llm()

    # TODO: next part
    g2r_tile_offsets_load_op = flash_llm_program.find_op_by_name("G2rTileOffsetsLoadOp")
    g2r_load_compressed_tile_A_op = flash_llm_program.find_op_by_name("G2rLoadCompressedTileAOp")
    g2s_load_tile_B_op = flash_llm_program.find_op_by_name("G2rLoadTileBOp")
    r2s_restore_tile_A_op = flash_llm_program.find_op_by_name("R2sRestoreTileAOp")
    s2r_load_tile_A_op = flash_llm_program.find_op_by_name("S2rLoadTileAOp")
    s2r_load_tile_B_op = flash_llm_program.find_op_by_name("S2rLoadTileBOp")
    mma_op = flash_llm_program.find_op_by_name("mma_op")

    s0 = NvOpSequence(
        g2r_tile_offsets_load_op,
        g2r_load_compressed_tile_A_op,
        g2s_load_tile_B_op,
        r2s_restore_tile_A_op,
        s2r_load_tile_A_op,
        s2r_load_tile_B_op,
        mma_op
    )

    apply_pipeline(
        flash_llm_main_loop_op,
        PipelinePlan([s0], [0]),
    )

    print(NvIrCodeGenerator().dump_nvop_program(flash_llm_program))
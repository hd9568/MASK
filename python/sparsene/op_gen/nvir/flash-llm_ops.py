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
from sparsene.op_gen.nvir.pipeliner import PipelinePlan, apply_pipeline, enumerate_pipeline_plans

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
        gmem_inouts=(
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
        ),
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
    flash_llm_loop_bound_op = NvOp(
        name="FlashLLMLoopBoundOp",
        inputs=[
            NvOpInput(
                idx=0,
                name="K",
                tensor=NvOpTensor(
                    shape=Shape(),
                    mem="rmem",
                    dtype="int",
                    source="blockIdx.x", #TODO 这里的blockIdx.x 是什么意思？
                ),
            ),
        ],
        outputs=[
            NvOpOutput(
                idx=0,
                name="start",
                tensor=NvOpTensor(shape=Shape(), mem="rmem", dtype="int"),
            ),
            NvOpOutput(
                idx=1,
                name="end",
                tensor=NvOpTensor(shape=Shape(), mem="rmem", dtype="int"),
            ),
        ],
        impl=NvOpImpl(
            code_template="""
start = 0;
end = (K + get<2>(BLK_MNK{}) - 1) / get<2>(BLK_MNK{});
            """
        ),
        mem_type=("g", "r"), # TODO: 如何简化这里的mem type
        parameters={},
    )
    flash_llm_blk_x_loop_op.append(flash_llm_loop_bound_op)

    #> loop result, output
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
                source=flash_llm_loop_bound_op.outputs[0],
            ),
        ),
        loop_r=NvOpInput(
            idx=1,
            name="r",
            tensor=NvOpTensor(
                shape=Shape(),
                mem="rmem",
                dtype="int",
                source=flash_llm_loop_bound_op.outputs[1],
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

    g2r_load_compressed_tile_A_op = NvOp(
        name="G2rLoadCompressedTileAOp",
        inputs=[

        ],
        outputs=[

        ],
        impl=NvOpImpl(

        ),
        mem_type=("g", "r"),
    )
    flash_llm_main_loop_op.append(g2r_load_compressed_tile_A_op)

    g2s_load_tile_B_op = NvOp(
        name="G2rLoadTileBOp",
        inputs=[

        ],
        outputs=[

        ],
        impl=NvOpImpl(

        ),
        mem_type=("g", "s"),
    )
    flash_llm_main_loop_op.append(g2s_load_tile_B_op)

    r2s_restore_tile_A_op = NvOp(
        name="R2sRestoreTileAOp",
        inputs=[

        ],
        outputs=[

        ],
        impl=NvOpImpl(

        ),
        mem_type=("r", "s"),
    )
    flash_llm_main_loop_op.append(r2s_restore_tile_A)

    s2r_load_tile_A_op = NvOp(
        name="S2rLoadTileAOp",
        inputs=[],
        outputs=[],
        impl=NvOpImpl(),
        mem_type=("s", "r"),
    )
    flash_llm_main_loop_op.append(s2r_load_tile_A_op)

    s2r_load_tile_B_op = NvOp(
        name="S2rLoadTileBOp",
        inputs=[],
        outputs=[],
        impl=NvOpImpl(),
        mem_type=("s", "r"),
    )
    flash_llm_main_loop_op.append(s2r_load_tile_B_op)

    mma_op = NvOp(
        name="MmaOp",
        inputs=[],
        outputs=[],
        impl=NvOpImpl(),
        mem_type=("r", "r"),
    )
    flash_llm_main_loop_op.append(mma_op)

    flash_llm_main_loop_op.set_loop_result(0, mma_op.outputs[0])

    r2g_c_val_store_op = NvOp(
        name="R2gCValStoreOp",
        inputs=[],
        outputs=[],
        impl=NvOpImpl(),
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
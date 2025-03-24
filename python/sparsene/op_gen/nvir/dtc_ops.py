from sparsene.op_gen.nvir.code_generator import (
    NvOp,
    NvOpInput,
    NvOpOutput,
    NvOpImpl,
    IntShape,
    ParamShape,
    Shape,
    VarlenShape,
    MnkShape,
)

g2s_val_sidx_load_op = NvOp(
    name="G2sValSidxLoadOp",
    inputs=[
        NvOpInput(
            idx=0,
            shape=Shape(ParamShape("tile_sidx_size"), VarlenShape("nnz_block")),
            mem="gmem",
            name="dval_sidx",
            dtype="int",
        ),
        NvOpInput(idx=1, shape=Shape(), mem="rmem", name="i", dtype="int"),
    ],
    outputs=[
        NvOpOutput(
            idx=0,
            shape=Shape(ParamShape("tile_sidx_size")),
            mem="smem",
            name="tile_sidx_block",
            dtype="int",
        )
    ],
    impl=NvOpImpl(
        code_template="""if (lid < 4) {
auto thr_tiler = make_shape(Int<4>{});
auto thr_coord = make_coord(lid);
auto src = local_tile(dval_sidx(_, i), thr_tiler, thr_coord);
auto dst = local_tile(tile_sidx_block(_, buf_idx), thr_tiler, thr_coord);
copy(Copy_Atom<UniversalCopy<uint128_t>, int>{}, src, dst);
}"""
    ),
    mem_type=("g", "s"),
    parameters={"tile_sidx_size": 16},
)

g2r_coo_atomic_format_load_off_op = NvOp(
    name="G2rCooAtomicFormatLoadOffOp",
    inputs=[
        NvOpInput(
            idx=0,
            shape=Shape(IntShape(2), VarlenShape("nnz_block")),
            mem="gmem",
            name="dval_coo_off",
            dtype="int",
        ),
        NvOpInput(idx=1, shape=Shape(), mem="rmem", name="i", dtype="int"),
    ],
    outputs=[
        NvOpOutput(idx=0, shape=Shape(), mem="rmem", name="l", dtype="int"),
        NvOpOutput(idx=1, shape=Shape(), mem="rmem", name="r", dtype="int"),
    ],
    impl=NvOpImpl(
        code_template="""l(buf_idx) = dval_coo_off(0, i);
r(buf_idx) = dval_coo_off(1, i);"""
    ),
    mem_type=("g", "r"),
    parameters={},
)

s2r_val_sidx_load_op = NvOp(
    name="S2rValSidxLoadOp",
    inputs=[
        NvOpInput(
            idx=0,
            shape=Shape(ParamShape("tile_sidx_size")),
            mem="smem",
            name="tile_sidx_block",
            dtype="int",
        )
    ],
    outputs=[
        NvOpOutput(
            idx=0,
            shape=Shape(ParamShape("tile_sidx_size")),
            mem="rmem",
            name="val_sidx_reg",
            dtype="int",
        )
    ],
    impl=NvOpImpl(
        code_template="""__pipeline_wait_prior(wait_num);
__syncthreads();
auto src = tile_sidx_block;
auto dst = val_sidx_reg(_, buf_idx);
copy(src, dst);"""
    ),
    mem_type=("s", "r"),
    parameters={"tile_sidx_size": 16},
)

g2s_coo_atomic_format_load_val_op = NvOp(
    name="G2sCooAtomicFormatLoadValOp",
    inputs=[
        NvOpInput(
            idx=0,
            shape=Shape(VarlenShape("nnz")),
            mem="gmem",
            name="dval_coo_val",
            dtype="half",
        ),
        NvOpInput(idx=1, shape=Shape(), mem="rmem", name="ll", dtype="int"),
        NvOpInput(idx=2, shape=Shape(), mem="rmem", name="rr", dtype="int"),
    ],
    outputs=[
        NvOpOutput(
            idx=0,
            shape=Shape(ParamShape("tile_coo_val_size")),
            mem="smem",
            name="tile_coo_val_block",
            dtype="half",
        ),
    ],
    impl=NvOpImpl(
        code_template="""auto thr_tiler = make_shape(_8{});
auto thr_coord = make_coord(lid);
auto input = make_tensor(dval_coo_val.data() + ll, make_shape(tile_coo_val_size{}));
auto output = tile_coo_val_block(_, buf_idx);

if (lid * 8 < rr - ll) {
    auto src = local_tile(input, thr_tiler, thr_coord);
    auto dst = local_tile(output, thr_tiler, thr_coord);
    copy(Copy_Atom<UniversalCopy<uint128_t>, half>{}, src, dst);
}"""
    ),
    mem_type=("g", "s"),
    parameters={"tile_coo_val_size": 16},
)

g2r_coo_atomic_format_load_idx_op = NvOp(
    name="G2rCooAtomicFormatLoadIdxOp",
    inputs=[
        NvOpInput(
            idx=0,
            shape=Shape(VarlenShape("nnz")),
            mem="gmem",
            name="dval_coo_idx",
            dtype="int",
        ),
        NvOpInput(idx=1, shape=Shape(), mem="rmem", name="ll", dtype="int"),
        NvOpInput(idx=2, shape=Shape(), mem="rmem", name="rr", dtype="int"),
    ],
    outputs=[
        NvOpOutput(
            idx=0, shape=Shape(IntShape(8)), mem="rmem", name="coo_idx_reg", dtype="int"
        ),
        NvOpOutput(idx=1, shape=Shape(), mem="rmem", name="coo_range_reg", dtype="int"),
    ],
    impl=NvOpImpl(
        code_template="""auto thr_tiler = make_shape(_8{});
auto thr_coord = make_coord(lid);
auto input = make_tensor(dval_coo_idx.data() + ll, make_shape(tile_coo_val_size{}));
coo_range_reg(buf_idx) = rr - ll;
if (lid * 8 < rr - ll) {
    auto src = local_tile(input, thr_tiler, thr_coord);
    auto dst = coo_idx_reg(_, buf_idx);
    copy(src, dst);
}"""
    ),
    mem_type=("g", "r"),
    parameters={"tile_coo_val_size": 16},
)

g2r_b_val_load_op = NvOp(
    name="G2rBValLoadOp",
    inputs=[
        NvOpInput(
            idx=0,
            shape=Shape(VarlenShape("K"), VarlenShape("N")),
            mem="gmem",
            name="B_val",
            dtype="half",
        ),
        NvOpInput(
            idx=1,
            shape=Shape(MnkShape("BLK_MNK", "k")),
            mem="rmem",
            name="val_sidx_reg",
            dtype="int",
        ),
    ],
    outputs=[
        NvOpOutput(
            idx=0,
            shape=Shape(IntShape(4), MnkShape("BLK_MMA_MNK", "n")),
            mem="rmem",
            name="REGB",
            dtype="half",
        ),
    ],
    impl=NvOpImpl(
        code_template="""for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
    int row = lid % 4 * 2;
    int col = blockIdx.y * get<1>(BLK_MNK{}) + n_iter * 8 + lid / 4;
    REGB(0, n_iter, buf_idx) = B_val(val_sidx_reg(row + 0), col);
    REGB(1, n_iter, buf_idx) = B_val(val_sidx_reg(row + 1), col);
    REGB(2, n_iter, buf_idx) = B_val(val_sidx_reg(row + 8), col);
    REGB(3, n_iter, buf_idx) = B_val(val_sidx_reg(row + 9), col);
}"""
    ),
    mem_type=("g", "r"),
    parameters={},
)

s2s_coo_atomic_val_restore_op = NvOp(
    name="S2sCooAtomicValRestoreOp",
    inputs=[
        NvOpInput(
            idx=0,
            shape=Shape(ParamShape("tile_coo_val_size")),
            mem="smem",
            name="tile_coo_val_block",
            dtype="half",
        ),
        NvOpInput(
            idx=1, shape=Shape(IntShape(8)), mem="rmem", name="coo_idx_reg", dtype="int"
        ),
        NvOpInput(idx=2, shape=Shape(), mem="rmem", name="nnz_num", dtype="int"),
    ],
    outputs=[
        NvOpOutput(
            idx=0,
            shape=Shape(
                MnkShape("BLK_MNK", "m"),
                MnkShape("BLK_MNK", "k"),
            ),
            mem="smem",
            name="tile_coo_restore_val_block",
            dtype="half",
        ),
    ],
    impl=NvOpImpl(
        code_template="""__pipeline_wait_prior(wait_num);
__syncthreads();
// TODO: use tiler to distribute the tensor and use clear(Tensor) to set zero
for (int i_o2s = 0; i_o2s < 4; i_o2s++) {
    *((int*)(tile_coo_restore_val_block(_, _, buf_idx).data().get() + 32 * 2 * i_o2s + lid * 2)) = 0;
}
__syncthreads();
for (int i_restore = 0; i_restore < 8; i_restore++) {
    int idx = coo_idx_reg(i_restore);
    if (lid * 8 + i_restore < nnz_num) {
        tile_coo_restore_val_block(_, _, buf_idx).data()[(idx)] = tile_coo_val_block(lid * 8 + i_restore);
    }
}"""
    ),
    mem_type=("s", "s"),
    parameters={"tile_coo_val_size": 16},
)

s2r_a_val_load_op = NvOp(
    name="S2rAValLoadOp",
    inputs=[
        NvOpInput(
            idx=0,
            shape=Shape(MnkShape("BLK_MNK", "m"), MnkShape("BLK_MNK", "k")),
            mem="smem",
            name="tile_coo_restore_val_block",
            dtype="half",
        ),
    ],
    outputs=[
        NvOpOutput(
            idx=0,
            shape=Shape(IntShape(8), MnkShape("BLK_MMA_MNK", "m")),
            mem="rmem",
            name="REGA",
            dtype="half",
        ),
    ],
    impl=NvOpImpl(
        code_template="""// no need to __pipeline_wait_prior here because the prior op s2s_coo_atomic_val_restore does not use cp.async
__syncthreads();
int row = lid % 16;
int col = lid / 16;
// TODO: bank conflict
for (int m_iter = 0; m_iter < get<0>(BLK_MMA_MNK{}); m_iter++) { // now the m_iter would always be zero, which means the for-loop body would run only once
    ldmatrix_m8n8k8_x4(
        (uint32_t*)(REGA(_, _0{}, buf_idx).data()),
        (void*)&tile_coo_restore_val_block((row) * 16 + (col * 8))
    );
}"""
    ),
    mem_type=("s", "r"),
    parameters={},
)

mma_op = NvOp(
    name="MmaOp",
    inputs=[
        NvOpInput(
            idx=0,
            shape=Shape(IntShape(8), MnkShape("BLK_MMA_MNK", "m")),
            mem="rmem",
            name="REGA",
            dtype="half",
        ),
        NvOpInput(
            idx=1,
            shape=Shape(IntShape(4), MnkShape("BLK_MMA_MNK", "n")),
            mem="rmem",
            name="REGB",
            dtype="half",
        ),
    ],
    outputs=[
        NvOpOutput(
            idx=0,
            shape=Shape(
                IntShape(4),
                Shape(MnkShape("BLK_MMA_MNK", "m"), MnkShape("BLK_MMA_MNK", "n")),
            ),
            mem="rmem",
            name="REGC",
            dtype="float",
        ),
    ],
    impl=NvOpImpl(
        code_template="""for (int m_iter = 0; m_iter < 1; m_iter++) {
    for (int n_iter = 0; n_iter < get<1>(BLK_MMA_MNK{}); n_iter++) {
        mma_m16n8k16_fp32(REGC(_, (m_iter, n_iter), buf_idx).data(), (uint32_t*)(REGA(_, m_iter).data()), REGB(_, n_iter).data());
    }
}"""
    ),
    mem_type=("r", "r"),
    parameters={},
    reduction=True,
)


if __name__ == "__main__":
    from sparsene.op_gen.nvir.code_generator import NvIrCodeGenerator

    print("// shift 1")
    print(NvIrCodeGenerator().dump_nvop_class_def(g2s_val_sidx_load_op))
    print()
    print(NvIrCodeGenerator().dump_nvop_class_def(g2r_coo_atomic_format_load_off_op))
    print()
    print("// shift 2")
    print(NvIrCodeGenerator().dump_nvop_class_def(s2r_val_sidx_load_op))
    print()
    print(NvIrCodeGenerator().dump_nvop_class_def(g2s_coo_atomic_format_load_val_op))
    print()
    print(NvIrCodeGenerator().dump_nvop_class_def(g2r_coo_atomic_format_load_idx_op))
    print()
    print("// shift 3")
    print(NvIrCodeGenerator().dump_nvop_class_def(g2r_b_val_load_op))
    print()
    print(NvIrCodeGenerator().dump_nvop_class_def(s2s_coo_atomic_val_restore_op))
    print()
    print(NvIrCodeGenerator().dump_nvop_class_def(s2r_a_val_load_op))
    print()
    print(NvIrCodeGenerator().dump_nvop_class_def(mma_op))
    print()
    print()
    print("    // shift 1")
    print(NvIrCodeGenerator().dump_nvop_create(g2s_val_sidx_load_op, 1))
    print()
    print(NvIrCodeGenerator().dump_nvop_create(g2r_coo_atomic_format_load_off_op, 1))
    print()
    print("    // shift 2")
    print(NvIrCodeGenerator().dump_nvop_create(s2r_val_sidx_load_op, 1))
    print()
    print(NvIrCodeGenerator().dump_nvop_create(g2s_coo_atomic_format_load_val_op, 1))
    print()
    print(NvIrCodeGenerator().dump_nvop_create(g2r_coo_atomic_format_load_idx_op, 1))
    print()
    print("    // shift 3")
    print(NvIrCodeGenerator().dump_nvop_create(g2r_b_val_load_op, 1))
    print()
    print(NvIrCodeGenerator().dump_nvop_create(s2s_coo_atomic_val_restore_op, 1))
    print()
    print(NvIrCodeGenerator().dump_nvop_create(s2r_a_val_load_op, 1))
    print()
    print(NvIrCodeGenerator().dump_nvop_create(mma_op, 1))
    print()
    

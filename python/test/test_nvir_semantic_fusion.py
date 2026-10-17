import contextlib
import copy
import io

from sparsene.formats.Acc_SpMM import BIT_TCF_FORMAT
from sparsene.formats.DTC_SpMM import ME_TCF_FORMAT
from sparsene.formats.SR_BCRS import SR_BCRS_FORMAT
from sparsene.formats.Spaden import BIT_BSR_FORMAT
from sparsene.op_gen.computent.computent import computent_from_rts
from sparsene.op_gen.nvir.compiler_driver import (
    _apply_nvir_semantic_fusion,
    _computational_leaf_ops,
    _find_leaf_op_by_name,
    _fuse_load_offset_sub_div,
    _partition_pipeline_stages,
    _select_main_loop,
)
from sparsene.op_gen.nvir.generate import generate_nvir
from sparsene.op_gen.opir.cValFlattenPass import CValFlattenPass
from sparsene.op_gen.opir.generate import generate_from_computent
from sparsene.op_gen.opir.varlenLoweringPass import VarlenLoweringPass
from sparsene.op_gen.strategy_agent import (
    PipelineStrategy,
    StrategyDecision,
    TensorPlacementStrategy,
)
from sparsene.transform.rts import derive_rts


FORMAT_CASES = [
    ("ME_TCF", ME_TCF_FORMAT, 8, 10, [3, 2, 3]),
    ("BIT_TCF", BIT_TCF_FORMAT, 9, 11, [4, 2, 3]),
    ("BIT_BSR", BIT_BSR_FORMAT, 9, 11, [4, 2, 3]),
    ("SR_BCRS", SR_BCRS_FORMAT, 6, 8, [1, 2, 3]),
]


def _heuristic_strategy() -> StrategyDecision:
    return StrategyDecision(
        provider="test",
        rationale="semantic fusion count test",
        tensor_placement=TensorPlacementStrategy(),
        pipeline=PipelineStrategy(),
    )


def _build_nvir(format_name, format_obj):
    with contextlib.redirect_stdout(io.StringIO()):
        computent = computent_from_rts(format_name, derive_rts(copy.deepcopy(format_obj)))
        ops = generate_from_computent(computent)
        lowered = VarlenLoweringPass(
            op_builder=None,
            varlen2LenArrayTable=computent.varlen2LenArrayTable,
        ).run(ops)
        flattened = CValFlattenPass(op_builder=None).run(lowered)
        return generate_nvir(
            opir=flattened,
            format_name=format_name,
            strategy_decision_override=_heuristic_strategy(),
            strategy_log_dir_override="",
        )


def test_format_aware_nvir_semantic_fusion_counts():
    for format_name, format_obj, main_target, leaf_target, stage_target in FORMAT_CASES:
        program = _build_nvir(format_name, format_obj)
        _fuse_load_offset_sub_div(program)
        main_loop = _select_main_loop(program)
        assert main_loop is not None

        assert _apply_nvir_semantic_fusion(program, main_loop)

        stage_ops = _partition_pipeline_stages(main_loop)
        assert len(main_loop.body.ops) == main_target
        assert len(_computational_leaf_ops(program)) == leaf_target
        assert [len(stage) for stage in stage_ops] == stage_target


def _main_loop_op(program, name):
    main_loop = _select_main_loop(program)
    assert main_loop is not None
    for op in main_loop.body.ops:
        if op.name == name:
            return op
    raise AssertionError(f"{name} not found in DTC main loop")


def _fused_dtc_program():
    return _fused_program("ME_TCF", ME_TCF_FORMAT)


def _fused_program(format_name, format_obj):
    program = _build_nvir(format_name, format_obj)
    _fuse_load_offset_sub_div(program)
    main_loop = _select_main_loop(program)
    assert main_loop is not None
    assert _apply_nvir_semantic_fusion(program, main_loop)
    return program


def test_dtc_hand_impl_specialization_shapes():
    program = _fused_dtc_program()

    sparse_index = _main_loop_op(program, "G2sSparseIndexLoadOp").impl.code_template
    assert "if (lid < 2)" in sparse_index
    assert "Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, int>" in sparse_index
    assert "sparsene_copy_g2s_128" not in sparse_index

    b_load = _main_loop_op(program, "G2sMatrixBLoadOp").impl.code_template
    assert "__syncthreads();" in b_load
    assert "get<2>(BLK_MNK{}) * get<1>(BLK_MNK{}) / load_tile_n" in b_load
    assert "uintptr_t" not in b_load

    s2r_a = _find_leaf_op_by_name(program, "S2rAValLoadOp")
    assert s2r_a is not None
    s2r_a_impl = s2r_a.impl.code_template
    assert "for (int k_iter" not in s2r_a_impl
    assert "ldmatrix_m8n8k8_x4" in s2r_a_impl
    assert "make_coord(_0{}, _0{})" in s2r_a_impl

    s2r_b = _find_leaf_op_by_name(program, "S2rBValLoadOp")
    assert s2r_b is not None
    s2r_b_impl = s2r_b.impl.code_template
    assert "for (int k_iter" not in s2r_b_impl
    assert "ldmatrix_m8n8k8_x2" in s2r_b_impl
    assert "__shfl_sync" in s2r_b_impl

    calculate = _main_loop_op(program, "CalculateOp").impl.code_template
    assert "for (int k_iter" not in calculate
    assert "make_coord(m_iter, _0{})" in calculate
    assert "make_coord(_0{}, n_iter)" in calculate

    store = _find_leaf_op_by_name(program, "R2gCValStoreOp")
    assert store is not None
    store_impl = store.impl.code_template
    assert "for (int i_tileM" not in store_impl
    assert "for (int i_tileN" in store_impl
    assert "(col, row)" in store_impl
    assert "(col + 1, row + 8)" in store_impl


def _assert_mco_single_k_hand_impl(program):
    mco_val = _main_loop_op(program, "G2sSparseMcoValLoadOp").impl.code_template
    mco_val_op = _main_loop_op(program, "G2sSparseMcoValLoadOp")
    assert mco_val_op.attrs.get("cp_async") is True
    assert mco_val_op.outputs[0].attrs.get("cp_async") is True
    assert "Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>" in mco_val
    assert "int ll = mco_len_l" in mco_val
    assert "int rr = mco_len_r" in mco_val

    mask = _main_loop_op(program, "G2rSparseMcoMaskLoadOp").impl.code_template
    assert "UniversalCopy<uint128_t>" in mask
    assert "flat_divide" in mask

    restore = _main_loop_op(program, "S2sRestoreMatrixAOp").impl.code_template
    assert "if (lid == 0)" not in restore
    assert "__popcll(mask << (63 - local_vid))" in restore
    assert "for (int local_vid = threadIdx.x; local_vid < 64" in restore

    b_load = _main_loop_op(program, "G2sMatrixBLoadOp").impl.code_template
    assert "uintptr_t" not in b_load
    assert "Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>" in b_load

    s2r_a = _find_leaf_op_by_name(program, "S2rAValLoadOp")
    assert s2r_a is not None
    assert "for (int k_iter" not in s2r_a.impl.code_template

    calculate = _main_loop_op(program, "CalculateOp").impl.code_template
    assert "for (int k_iter" not in calculate
    assert "make_coord(m_iter, _0{})" in calculate

    store = _find_leaf_op_by_name(program, "R2gCValStoreOp")
    assert store is not None
    store_impl = store.impl.code_template
    assert "for (int i_tileM" not in store_impl
    assert "for (int i_tileN" in store_impl


def test_bit_tcf_flat_acc_compute_wrapper_shapes():
    program = _fused_program("BIT_TCF", BIT_TCF_FORMAT)

    s2r_a = _find_leaf_op_by_name(program, "S2rAValLoadOp")
    assert s2r_a is not None
    assert str(s2r_a.outputs[0].tensor.shape) == "(Int(4), Mnk(BLK_MMA_MNK, m))"
    assert "__syncthreads();" not in s2r_a.impl.code_template
    assert "REGA(_, _0{}, buf_idx)" in s2r_a.impl.code_template
    assert "make_coord(_0{}, _0{})" not in s2r_a.impl.code_template

    s2r_b = _find_leaf_op_by_name(program, "S2rBValLoadOp")
    assert s2r_b is not None
    assert str(s2r_b.outputs[0].tensor.shape) == "(Int(2), Mnk(BLK_MMA_MNK, n))"
    assert "__syncthreads();" not in s2r_b.impl.code_template
    assert "REGB(_, n_iter, buf_idx)" in s2r_b.impl.code_template
    assert "make_coord(_0{}, n_iter)" not in s2r_b.impl.code_template

    calculate = _main_loop_op(program, "CalculateOp")
    assert len(calculate.inputs) == 2
    assert str(calculate.inputs[0].tensor.shape) == "(Int(4), Mnk(BLK_MMA_MNK, m))"
    assert str(calculate.inputs[1].tensor.shape) == "(Int(2), Mnk(BLK_MMA_MNK, n))"
    calculate_impl = calculate.impl.code_template
    assert "for (int k_iter" not in calculate_impl
    assert "REGA(0, m_iter)" in calculate_impl
    assert "REGB(0, n_iter)" in calculate_impl
    assert "make_coord(m_iter, _0{})" not in calculate_impl
    assert "make_coord(_0{}, n_iter)" not in calculate_impl


def test_mco_format_hand_impl_specialization_shapes():
    _assert_mco_single_k_hand_impl(_fused_program("BIT_TCF", BIT_TCF_FORMAT))
    _assert_mco_single_k_hand_impl(_fused_program("BIT_BSR", BIT_BSR_FORMAT))


def test_sr_bcrs_hand_impl_specialization_shapes():
    program = _fused_program("SR_BCRS", SR_BCRS_FORMAT)

    sparse_index = _main_loop_op(program, "G2sSparseIndexLoadOp").impl.code_template
    assert "lid < get<2>(BLK_MNK{}) / 4" in sparse_index
    assert "sparsene_copy_g2s_128" not in sparse_index

    block_val = _main_loop_op(program, "G2sSparseValBlockValLoadOp").impl.code_template
    assert "__pipeline_memcpy_async" in block_val
    assert "local_tile" in block_val
    assert "val_tile" in block_val

    s2r_a = _find_leaf_op_by_name(program, "S2rAValLoadOp")
    assert s2r_a is not None
    assert "for (int k_iter" in s2r_a.impl.code_template
    assert "m_iter * get<0>(MMA_MNK{})" in s2r_a.impl.code_template

    calculate = _main_loop_op(program, "CalculateOp").impl.code_template
    assert "for (int k_iter" in calculate
    assert "for (int m_iter" in calculate
    assert "for (int n_iter" in calculate

    store = _find_leaf_op_by_name(program, "R2gCValStoreOp")
    assert store is not None
    store_impl = store.impl.code_template
    assert "for (int i_tileM" in store_impl
    assert "for (int i_tileN" in store_impl

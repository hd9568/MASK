from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from sparsene.formats.DTC_SpMM import ME_TCF_FORMAT
from sparsene.logging import set_logging_level_for_all
from sparsene.op_gen.computent.computent import computent_from_rts
from sparsene.op_gen.nvir.codegen import NvIrCodeGenerator
from sparsene.op_gen.nvir.compiler_driver import (
    _fuse_inline_gen_add,
    _fuse_load_offset_sub_div,
    _mark_codegen_inline_load_offset,
    _select_main_loop,
)
from sparsene.op_gen.nvir.generate import generate_nvir
from sparsene.op_gen.nvir.nvop import ForLoopNvOp, NvOpProgram, NvOpSequence
from sparsene.op_gen.nvir.plan import PipelinePlan, apply_pipeline
from sparsene.op_gen.opir.cValFlattenPass import CValFlattenPass
from sparsene.op_gen.opir.generate import generate_from_computent
from sparsene.op_gen.opir.varlenLoweringPass import VarlenLoweringPass
from sparsene.op_gen.strategy_agent import StrategyConfig
from sparsene.transform.rts import derive_rts


DEFAULT_OUTPUT = REPO_ROOT / "python" / "test" / "results" / "dtc.inc"


def dtc(strategy_config: StrategyConfig | None = None) -> tuple[ForLoopNvOp, NvOpProgram]:
    """Build the generated ME_TCF/DTC NVIR program before software pipelining.

    This is the generated counterpart of ``dtc_ops.py::dtc``.  It keeps the
    automatic format-to-NVIR lowering path, then exposes the produced
    ``NvOpProgram`` so we can manually tune the generated NVOPs.
    """

    if strategy_config is None:
        strategy_config = StrategyConfig(provider="heuristic")

    rts = derive_rts(ME_TCF_FORMAT)
    computent = computent_from_rts("ME_TCF", rts)
    ops = generate_from_computent(computent)
    lowered_ops = VarlenLoweringPass(
        op_builder=None,
        varlen2LenArrayTable=computent.varlen2LenArrayTable,
    ).run(ops)
    flattened_ops = CValFlattenPass(op_builder=None).run(lowered_ops)
    program = generate_nvir(
        opir=flattened_ops,
        format_name="ME_TCF",
        strategy_config=strategy_config,
    )

    main_loop = _select_main_loop(program)
    if main_loop is None:
        raise RuntimeError("generated DTC program has no main ForLoopNvOp")
    return main_loop, program


def optimize_generated_dtc(main_loop: ForLoopNvOp, program: NvOpProgram) -> ForLoopNvOp:
    """Manual optimization hook.

    Add generated-DTC-specific NVOP rewrites here.  This hook runs after the
    current compiler-driver scalar fusions and before the explicit pipeline plan
    is applied, so op names match the stage list below.
    """

    return main_loop


def apply_generated_dtc_pipeline(
    main_loop: ForLoopNvOp,
    program: NvOpProgram,
) -> ForLoopNvOp:
    """Apply the current generated DTC fusion and 3-stage pipeline explicitly."""

    _fuse_load_offset_sub_div(program)
    selected_main_loop = _select_main_loop(program)
    if selected_main_loop is None:
        raise RuntimeError("generated DTC program has no main ForLoopNvOp")
    main_loop = selected_main_loop

    if _fuse_inline_gen_add(main_loop):
        _mark_codegen_inline_load_offset(main_loop)

    main_loop = optimize_generated_dtc(main_loop, program)

    strategy = program.attrs.get("strategy_decision")
    pipeline_strategy = getattr(strategy, "pipeline", None)
    if pipeline_strategy is not None:
        main_loop.attrs["pipeline_strategy"] = pipeline_strategy

    stage0 = NvOpSequence(
        program.find_op_by_name("array_ref"),
        program.find_op_by_name("load_offset_1"),
        program.find_op_by_name("coo_atomic_format_load_idx"),
        program.find_op_by_name("coo_atomic_format_load_val"),
    )
    stage1 = NvOpSequence(
        program.find_op_by_name("array_ref_1"),
        program.find_op_by_name("coo_atomic_val_restore"),
    )
    stage2 = NvOpSequence(
        program.find_op_by_name("S2rAValLoadOp"),
        program.find_op_by_name("S2rBValLoadOp"),
        program.find_op_by_name("mma"),
    )

    shifts = [1, 1]
    if pipeline_strategy is not None and pipeline_strategy.stage_shifts:
        if len(pipeline_strategy.stage_shifts) == 2:
            shifts = list(pipeline_strategy.stage_shifts)

    apply_pipeline(main_loop, PipelinePlan([stage0, stage1, stage2], shifts))
    main_loop.set_loop_result(0, program.find_op_by_name("mma").outputs[0])
    return main_loop


def dump_tree(path: Path, program: NvOpProgram, *, show_impl: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        program.dump_tree(
            indent_size=2,
            show_io=True,
            show_impl=show_impl,
            impl_max_lines=0,
        )
    )


def write_inc(path: Path, program: NvOpProgram) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(NvIrCodeGenerator().dump_nvop_program(program))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the generated DTC NVIR program and emit a CUDA .inc file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .inc path.",
    )
    parser.add_argument(
        "--dump-before-pipeline",
        type=Path,
        default=None,
        help="Optional NVIR tree dump before software pipelining.",
    )
    parser.add_argument(
        "--dump-after-pipeline",
        type=Path,
        default=None,
        help="Optional NVIR tree dump after software pipelining.",
    )
    parser.add_argument(
        "--dump-impl",
        action="store_true",
        help="Include NVOP impl bodies in tree dumps.",
    )
    parser.add_argument(
        "--strategy-provider",
        default="heuristic",
        choices=[
            "heuristic",
            "manual_json",
            "openai_http",
            "anthropic_http",
            "subprocess_json",
        ],
        help="Strategy provider passed to generated NVIR lowering.",
    )
    parser.add_argument(
        "--strategy-manual-json",
        default=None,
        help="Manual strategy JSON path when --strategy-provider=manual_json.",
    )
    parser.add_argument(
        "--strategy-log-dir",
        default="docs/agent_runs",
        help="Strategy-agent log directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce Sparsene logging noise.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_logging_level_for_all("ERROR" if args.quiet else "INFO")

    strategy_config = StrategyConfig(
        provider=args.strategy_provider,
        manual_json_path=args.strategy_manual_json,
        log_dir=args.strategy_log_dir,
    )
    main_loop, program = dtc(strategy_config=strategy_config)

    if args.dump_before_pipeline is not None:
        dump_tree(args.dump_before_pipeline, program, show_impl=args.dump_impl)

    apply_generated_dtc_pipeline(main_loop, program)

    if args.dump_after_pipeline is not None:
        dump_tree(args.dump_after_pipeline, program, show_impl=args.dump_impl)

    write_inc(args.output, program)
    print(f"[ok] generated DTC inc -> {args.output}")


if __name__ == "__main__":
    main()

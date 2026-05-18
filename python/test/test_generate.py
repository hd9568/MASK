import argparse
from pathlib import Path

from sparsene.op_gen.opir import printer
from sparsene.formats.DTC_SpMM import ME_TCF_FORMAT
from sparsene.formats.Acc_SpMM import BIT_TCF_FORMAT
from sparsene.formats.HME_TCF import HME_TCF_FORMAT
from sparsene.formats.ROW_REORDER_SR_BCRS import ROW_REORDER_SR_BCRS_FORMAT
from sparsene.formats.Spaden import BIT_BSR_FORMAT
from sparsene.formats.SR_BCRS import SR_BCRS_FORMAT
from sparsene.transform.rts import derive_rts
from sparsene.op_gen.computent.computent import computent_from_rts
from sparsene.op_gen.opir.generate import generate_from_computent
from sparsene.op_gen.opir.varlenLoweringPass import VarlenLoweringPass
from sparsene.op_gen.opir.cValFlattenPass import CValFlattenPass
from sparsene.op_gen.nvir.generate import generate_nvir
from sparsene.op_gen.nvir.printer import NvProgramPrinter
from sparsene.logging import get_logger, set_logging_level_for_all
from sparsene.op_gen.strategy_agent import StrategyConfig

from sparsene.formats.ROW_REORDER_SR_BCRS import ROW_REORDER_SR_BCRS_FORMAT

from sparsene.op_gen.nvir.compiler_driver import apply_software_pipeline_and_codegen


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--debug", "-d", action="store_true")
    args.add_argument(
        "--dump-impl",
        action="store_true",
        help="在 NVIR 树形 dump 中包含 op.impl 代码片段",
    )
    args.add_argument(
        "--dump-no-io",
        action="store_true",
        help="在 NVIR 树形 dump 中隐藏 inputs/outputs",
    )
    args.add_argument(
        "--dump-file",
        type=str,
        default="./me_tcf_nvir_tree.txt",
        help="ME-TCF 的 NVIR 树形 dump 输出文件",
    )
    args.add_argument(
        "--strategy-provider",
        type=str,
        default="heuristic",
        choices=[
            "heuristic",
            "manual_json",
            "openai_http",
            "anthropic_http",
            "subprocess_json",
        ],
        help="张量放置/流水线策略智能体提供者",
    )
    args.add_argument(
        "--strategy-manual-json",
        type=str,
        default=None,
        help="manual_json provider 使用的策略 JSON 文件",
    )
    args.add_argument(
        "--strategy-agent-command",
        type=str,
        default=None,
        help="subprocess_json provider 使用的外部命令，stdin 输入 request/prompt，stdout 输出 JSON response",
    )
    args.add_argument(
        "--strategy-log-dir",
        type=str,
        default="docs/agent_runs",
        help="智能体 request/response 日志目录",
    )
    args.add_argument(
        "--strategy-api-key",
        type=str,
        default=None,
        help="在线策略智能体使用的 API key，默认使用 strategy_agent.py 中的预设值",
    )
    args.add_argument(
        "--strategy-api-base-url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL，默认使用预设值",
    )
    args.add_argument(
        "--strategy-anthropic-base-url",
        type=str,
        default=None,
        help="Anthropic-compatible API base URL，默认使用预设的 token-plan anthropic 入口",
    )
    args.add_argument(
        "--strategy-model",
        type=str,
        default=None,
        help="在线策略智能体使用的模型名",
    )
    args.add_argument(
        "--strategy-max-tokens",
        type=int,
        default=None,
        help="在线策略智能体允许返回的最大 token 数",
    )
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_logging_level_for_all("DEBUG" if args.debug else "INFO")

    logger = get_logger("test")

    print("=" * 100)
    print("ME_TCF")
    rts = derive_rts(ME_TCF_FORMAT)
    computent = computent_from_rts("ME_TCF", rts)
    ops = generate_from_computent(computent)
    print("[BEFORE VarlenLoweringPass]")
    print(printer.Printer().dump(ops))

    varlen_lowering_pass = VarlenLoweringPass(op_builder=None, varlen2LenArrayTable=computent.varlen2LenArrayTable)
    lowered_ops = varlen_lowering_pass.run(ops)

    print("[AFTER VarlenLoweringPass]")
    print(printer.Printer().dump(lowered_ops))
    
    c_val_flatten_pass = CValFlattenPass(op_builder=None)
    flattened_ops = c_val_flatten_pass.run(lowered_ops)
    print("[AFTER cValFlattenPass]")
    print(printer.Printer().dump(flattened_ops))

    default_strategy_config = StrategyConfig()
    strategy_config = StrategyConfig(
        provider=args.strategy_provider,
        manual_json_path=args.strategy_manual_json,
        agent_command=args.strategy_agent_command,
        log_dir=args.strategy_log_dir,
        api_key=args.strategy_api_key or default_strategy_config.api_key,
        api_base_url=(
            args.strategy_api_base_url
            or default_strategy_config.api_base_url
        ),
        anthropic_base_url=(
            args.strategy_anthropic_base_url
            or default_strategy_config.anthropic_base_url
        ),
        model=args.strategy_model or default_strategy_config.model,
        max_tokens=args.strategy_max_tokens or default_strategy_config.max_tokens,
    )
    nvir_program = generate_nvir(
        opir=flattened_ops,
        format_name="ME_TCF",
        strategy_config=strategy_config,
    )
    print("[AFTER NVIR Lowering]")
    print(f"program={nvir_program.name}, num_ops={len(nvir_program.ops)}")
    print(
        "top_ops=",
        [op.name for op in nvir_program.ops[: len(nvir_program.ops)]],
    )

    show_io = not args.dump_no_io

    tree_from_method = nvir_program.dump_tree(
        indent_size=2,
        show_io=show_io,
        show_impl=args.dump_impl,
    )
    print("[AFTER NVIR Lowering - dump_tree]")
    print(tree_from_method)

    # tree_from_printer = NvProgramPrinter(
    #     indent_size=2,
    #     show_io=show_io,
    #     show_impl=args.dump_impl,
    # ).dump_program(nvir_program)
    # if tree_from_method != tree_from_printer:
    #     logger.warning("dump_tree output differs from NvProgramPrinter output")

    # dump_path = Path(args.dump_file)
    # dump_path.write_text(tree_from_method)
    # print(f"[NVIR Tree Dumped] {dump_path.resolve()}")

    
    output_file = "./me_tcf_kernel3.inc"
    apply_software_pipeline_and_codegen(nvir_program, output_file)

    exit(0)
    # logger.info(rts)
    # logger.info(computent)


    # print("=" * 100)
    # print("BIT_TCF")
    # rts = derive_rts(BIT_TCF_FORMAT)
    # computent = computent_from_rts("BIT_TCF", rts)
    # ops = generate_from_computent(computent)
    # print("[BEFORE VarlenLoweringPass]")
    # print(printer.default_printer.dump(ops))
    # varlen_lowering_pass = VarlenLoweringPass(op_builder=None, varlen2LenArrayTable=computent.varlen2LenArrayTable)
    # lowered_ops = varlen_lowering_pass.run(ops)
    # print("[AFTER VarlenLoweringPass]")
    # print(printer.default_printer.dump(lowered_ops))
    # c_val_flatten_pass = CValFlattenPass(op_builder=None)
    # flattened_ops = c_val_flatten_pass.run(lowered_ops)
    # print("[AFTER cValFlattenPass]")
    # print(printer.default_printer.dump(flattened_ops))
    # exit(0)
    # logger.info(rts)
    # logger.info(computent)


    # print("=" * 100)
    # print("HME_TCF")
    # rts = derive_rts(HME_TCF_FORMAT)
    # computent = computent_from_rts("HME_TCF", rts)
    # ops = generate_from_computent(computent)
    # print("[BEFORE VarlenLoweringPass]")
    # print(printer.default_printer.dump(ops))
    # varlen_lowering_pass = VarlenLoweringPass(op_builder=None, varlen2LenArrayTable=computent.varlen2LenArrayTable)
    # lowered_ops = varlen_lowering_pass.run(ops)
    # print("[AFTER VarlenLoweringPass]")
    # print(printer.default_printer.dump(lowered_ops))
    # exit(0)
    # logger.info(rts)
    # logger.info(computent)


    # print("=" * 100)
    # print("ROW_REORDER_SR_BCRS")
    # rts = derive_rts(ROW_REORDER_SR_BCRS_FORMAT)
    # computent = computent_from_rts("ROW_REORDER_SR_BCRS", rts)
    # ops = generate_from_computent(computent)
    # print("[BEFORE VarlenLoweringPass]")
    # print(printer.default_printer.dump(ops))
    # varlen_lowering_pass = VarlenLoweringPass(op_builder=None, varlen2LenArrayTable=computent.varlen2LenArrayTable)
    # lowered_ops = varlen_lowering_pass.run(ops)
    # print("[AFTER VarlenLoweringPass]")
    # print(printer.default_printer.dump(lowered_ops))
    # c_val_flatten_pass = CValFlattenPass(op_builder=None, varlen2IdxArrayTable=None)
    # flattened_ops = c_val_flatten_pass.run(lowered_ops)
    # print("(debug)", c_val_flatten_pass.varlen2IdxArrayTable)
    # print("[AFTER cValFlattenPass]")
    # print(printer.default_printer.dump(flattened_ops))
    # exit(0)
    # logger.info(rts)
    # logger.info(computent)


    # print("=" * 100)
    # print("BIT_BSR")
    # rts = derive_rts(BIT_BSR_FORMAT)
    # computent = computent_from_rts("BIT_BSR", rts)
    # ops = generate_from_computent(computent)
    # print("[BEFORE VarlenLoweringPass]")
    # print(printer.default_printer.dump(ops))
    # varlen_lowering_pass = VarlenLoweringPass(op_builder=None, varlen2LenArrayTable=computent.varlen2LenArrayTable)
    # lowered_ops = varlen_lowering_pass.run(ops)
    # print("[AFTER VarlenLoweringPass]")
    # print(printer.default_printer.dump(lowered_ops))
    # c_val_flatten_pass = CValFlattenPass(op_builder=None, varlen2IdxArrayTable=None)
    # flattened_ops = c_val_flatten_pass.run(lowered_ops)
    # print("[AFTER CValFlattenPass]")
    # print(printer.default_printer.dump(flattened_ops))
    # exit(0)
    # logger.info(rts)
    # logger.info(computent)
    # exit(0)


    # print("=" * 100)
    # print("SR_BCRS")
    # rts = derive_rts(SR_BCRS_FORMAT)
    # computent = computent_from_rts("SR_BCRS", rts)
    # ops = generate_from_computent(computent)
    # print("[BEFORE VarlenLoweringPass]")
    # print(printer.default_printer.dump(ops))
    # varlen_lowering_pass = VarlenLoweringPass(op_builder=None, varlen2LenArrayTable=computent.varlen2LenArrayTable)
    # lowered_ops = varlen_lowering_pass.run(ops)
    # print("[AFTER VarlenLoweringPass]")
    # print(printer.default_printer.dump(lowered_ops))
    # c_val_flatten_pass = CValFlattenPass(op_builder=None, varlen2IdxArrayTable=None)
    # flattened_ops = c_val_flatten_pass.run(lowered_ops)
    # print("[AFTER CValFlattenPass]")
    # print(printer.default_printer.dump(flattened_ops))
    # exit(0)
    # logger.info(rts)
    # logger.info(computent)


    print("=" * 100)
    # rts = derive_rts(ROW_REORDER_SR_BCRS_FORMAT)
    # logger.info(rts)
    # computent = computent_from_rts("row_reorder_sr_bcrs", rts)
    
    # logger.info(rts)
    # logger.info(computent)

    # ops = generate_from_computent(computent)
    # print(ops)
    # logger.info(printer.default_printer.dump(ops))
    # print(1)
    # print(printer.default_printer.dump(ops))
    # print(2)

    # nvop_program = generate_nvir(ops)
    

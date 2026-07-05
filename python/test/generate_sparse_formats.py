from __future__ import annotations

import argparse
from pathlib import Path

from sparsene.formats.Acc_SpMM import BIT_TCF_FORMAT
from sparsene.formats.DTC_SpMM import ME_TCF_FORMAT
from sparsene.formats.SR_BCRS import SR_BCRS_FORMAT
from sparsene.formats.Spaden import BIT_BSR_FORMAT
from sparsene.op_gen.computent.computent import computent_from_rts
from sparsene.op_gen.nvir.compiler_driver import apply_software_pipeline_and_codegen
from sparsene.op_gen.nvir.generate import generate_nvir
from sparsene.op_gen.opir.cValFlattenPass import CValFlattenPass
from sparsene.op_gen.opir.generate import generate_from_computent
from sparsene.op_gen.opir.varlenLoweringPass import VarlenLoweringPass
from sparsene.op_gen.strategy_agent import StrategyConfig
from sparsene.transform.rts import derive_rts


FORMAT_SPECS = [
    ("ME_TCF", "dtc", ME_TCF_FORMAT),
    ("BIT_TCF", "acc", BIT_TCF_FORMAT),
    ("BIT_BSR", "bitbsr", BIT_BSR_FORMAT),
    ("SR_BCRS", "sr_bcrs", SR_BCRS_FORMAT),
]


def _generate_one(format_name: str, format_obj, output_path: Path) -> None:
    rts = derive_rts(format_obj)
    computent = computent_from_rts(format_name, rts)
    ops = generate_from_computent(computent)
    lowered = VarlenLoweringPass(
        op_builder=None,
        varlen2LenArrayTable=computent.varlen2LenArrayTable,
    ).run(ops)
    flattened = CValFlattenPass(op_builder=None).run(lowered)
    nvir_program = generate_nvir(
        opir=flattened,
        format_name=format_name,
        strategy_config=StrategyConfig(provider="heuristic"),
    )
    apply_software_pipeline_and_codegen(nvir_program, str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DTC/ACC/BITBSR/SR_BCRS kernels into results/*.inc",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/home/huoda/sparsene/python/test/results"),
        help="Directory used to store generated .inc files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    for format_name, file_stem, format_obj in FORMAT_SPECS:
        output_path = args.results_dir / f"{file_stem}.inc"
        _generate_one(format_name, format_obj, output_path)
        print(f"[ok] {format_name} -> {output_path}")


if __name__ == "__main__":
    main()

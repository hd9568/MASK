import argparse

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
from sparsene.logging import get_logger, set_logging_level_for_all

from sparsene.formats.ROW_REORDER_SR_BCRS import ROW_REORDER_SR_BCRS_FORMAT




def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--debug", "-d", action="store_true")
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
    # exit(0)
    # logger.info(rts)
    # logger.info(computent)


    print("=" * 100)
    print("BIT_TCF")
    rts = derive_rts(BIT_TCF_FORMAT)
    computent = computent_from_rts("BIT_TCF", rts)
    ops = generate_from_computent(computent)
    print("[BEFORE VarlenLoweringPass]")
    print(printer.default_printer.dump(ops))
    varlen_lowering_pass = VarlenLoweringPass(op_builder=None, varlen2LenArrayTable=computent.varlen2LenArrayTable)
    lowered_ops = varlen_lowering_pass.run(ops)
    print("[AFTER VarlenLoweringPass]")
    print(printer.default_printer.dump(lowered_ops))
    c_val_flatten_pass = CValFlattenPass(op_builder=None)
    flattened_ops = c_val_flatten_pass.run(lowered_ops)
    print("[AFTER cValFlattenPass]")
    print(printer.default_printer.dump(flattened_ops))
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


    print("=" * 100)
    print("ROW_REORDER_SR_BCRS")
    rts = derive_rts(ROW_REORDER_SR_BCRS_FORMAT)
    computent = computent_from_rts("ROW_REORDER_SR_BCRS", rts)
    ops = generate_from_computent(computent)
    print("[BEFORE VarlenLoweringPass]")
    print(printer.default_printer.dump(ops))
    varlen_lowering_pass = VarlenLoweringPass(op_builder=None, varlen2LenArrayTable=computent.varlen2LenArrayTable)
    lowered_ops = varlen_lowering_pass.run(ops)
    print("[AFTER VarlenLoweringPass]")
    print(printer.default_printer.dump(lowered_ops))
    c_val_flatten_pass = CValFlattenPass(op_builder=None, varlen2IdxArrayTable=None)
    flattened_ops = c_val_flatten_pass.run(lowered_ops)
    print("(debug)", c_val_flatten_pass.varlen2IdxArrayTable)
    print("[AFTER cValFlattenPass]")
    print(printer.default_printer.dump(flattened_ops))
    # exit(0)
    # logger.info(rts)
    # logger.info(computent)


    print("=" * 100)
    print("BIT_BSR")
    rts = derive_rts(BIT_BSR_FORMAT)
    computent = computent_from_rts("BIT_BSR", rts)
    ops = generate_from_computent(computent)
    print("[BEFORE VarlenLoweringPass]")
    print(printer.default_printer.dump(ops))
    varlen_lowering_pass = VarlenLoweringPass(op_builder=None, varlen2LenArrayTable=computent.varlen2LenArrayTable)
    lowered_ops = varlen_lowering_pass.run(ops)
    print("[AFTER VarlenLoweringPass]")
    print(printer.default_printer.dump(lowered_ops))
    c_val_flatten_pass = CValFlattenPass(op_builder=None, varlen2IdxArrayTable=None)
    flattened_ops = c_val_flatten_pass.run(lowered_ops)
    print("[AFTER CValFlattenPass]")
    print(printer.default_printer.dump(flattened_ops))
    # exit(0)
    # logger.info(rts)
    # logger.info(computent)
    # exit(0)


    print("=" * 100)
    print("SR_BCRS")
    rts = derive_rts(SR_BCRS_FORMAT)
    computent = computent_from_rts("SR_BCRS", rts)
    ops = generate_from_computent(computent)
    print("[BEFORE VarlenLoweringPass]")
    print(printer.default_printer.dump(ops))
    varlen_lowering_pass = VarlenLoweringPass(op_builder=None, varlen2LenArrayTable=computent.varlen2LenArrayTable)
    lowered_ops = varlen_lowering_pass.run(ops)
    print("[AFTER VarlenLoweringPass]")
    print(printer.default_printer.dump(lowered_ops))
    c_val_flatten_pass = CValFlattenPass(op_builder=None, varlen2IdxArrayTable=None)
    flattened_ops = c_val_flatten_pass.run(lowered_ops)
    print("[AFTER CValFlattenPass]")
    print(printer.default_printer.dump(flattened_ops))
    exit(0)
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
    
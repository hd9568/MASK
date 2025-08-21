import argparse

from sparsene.op_gen.opir import printer
from sparsene.formats.DTC_SpMM import ME_TCF_FORMAT
from sparsene.transform.rts import derive_rts
from sparsene.op_gen.computent.computent import computent_from_rts
from sparsene.op_gen.opir.generate import generate_from_computent
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

    rts = derive_rts(ME_TCF_FORMAT)
    computent = computent_from_rts("me_tcf", rts)
    
    # rts = derive_rts(ROW_REORDER_SR_BCRS_FORMAT)
    # logger.info(rts)
    # computent = computent_from_rts("row_reorder_sr_bcrs", rts)
    
    logger.info(rts)
    logger.info(computent)

    ops = generate_from_computent(computent)
    print(ops)
    # logger.info(printer.default_printer.dump(ops))
    print(1)
    print(printer.default_printer.dump(ops))
    print(2)
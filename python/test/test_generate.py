import argparse

from sparsene.op_gen.opir import printer
from sparsene.formats.DTC_SpMM import DTC_SPMM_FORMAT
from sparsene.transform.rts import derive_rts
from sparsene.op_gen.computent.computent import computent_from_rts
from sparsene.op_gen.opir.generate import generate_from_computent
from sparsene.op_gen.nvir.generate import generate_nvir
from sparsene.logging import get_logger, set_logging_level_for_all


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--debug", "-d", action="store_true")
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_logging_level_for_all("DEBUG" if args.debug else "INFO")

    logger = get_logger("test")

    rts = derive_rts(DTC_SPMM_FORMAT)
    computent = computent_from_rts("dtc_spmm", rts)
    logger.info(rts)
    logger.info(computent)

    ops = generate_from_computent(computent)
    logger.info(printer.default_printer.dump(ops))

    nvir = generate_nvir(ops)
    for gmem in nvir.gmem_inouts.values():
        logger.info(gmem)

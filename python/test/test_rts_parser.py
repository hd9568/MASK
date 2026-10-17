import argparse

from sparsene.formats.DTC_SpMM import ME_TCF_FORMAT
from sparsene.formats.HME_TCF import HME_TCF_FORMAT
from sparsene.formats.Acc_SpMM import BIT_TCF_FORMAT
from sparsene.formats.Spaden import BIT_BSR_FORMAT
from sparsene.formats.SR_BCRS import SR_BCRS_FORMAT
from sparsene.transform.rts import derive_rts
from sparsene.op_gen.computent.computent import computent_from_rts

from sparsene.logging import get_logger, set_logging_level_for_all


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--debug", "-d", action="store_true")
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_logging_level_for_all("DEBUG" if args.debug else "INFO")

    logger = get_logger("test")

    # rts = derive_rts(BIT_BSR_FORMAT)
    # computent = computent_from_rts("bit_bsr", rts)
    # logger.info(rts)
    # logger.info(computent)

    # rts = derive_rts(BIT_TCF_FORMAT)
    # computent = computent_from_rts("bit_tcf", rts)
    # logger.info(rts)
    # logger.info(computent)

    rts = derive_rts(ME_TCF_FORMAT)
    computent = computent_from_rts("dtc_spmm", rts)
    logger.info(rts)
    logger.info(computent)

    # rts = derive_rts(BIT_TCF_FORMAT)
    # computent = generate_from_rts("bit_tcf", rts)

    # logger.info("BIT-TCF:")
    # logger.info(rts)
    # logger.info(computent)

    # breakpoint()

    # rts = derive_rts(BIT_BSR_FORMAT)
    # computent = generate_from_rts("bit_bsr", rts)

    # logger.info("BIT-BSR:")
    # logger.info(rts)
    # logger.info(computent)

    # breakpoint()

    # rts = derive_rts(SR_BCRS_FORMAT)
    # computent = generate_from_rts("sr_bcrs", rts)

    # logger.info("SR-BCRS:")
    # logger.info(rts)
    # logger.info(computent)

    # breakpoint()

    # rts = derive_rts(HME_TCF_FORMAT)
    # computent = generate_from_rts("hmec_tcf", rts)

    # logger.info("HME-TCF:")
    # logger.info(rts)
    # logger.info(computent)

    # breakpoint()

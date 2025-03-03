from sparsene.formats.DTC_SpMM import DTC_SPMM_FORMAT
from sparsene.formats.HME_TCF import HME_TCF_FORMAT
from sparsene.formats.Acc_SpMM import BIT_TCF_FORMAT
from sparsene.formats.Spaden import BIT_BSR_FORMAT
from sparsene.formats.SR_BCRS import SR_BCRS_FORMAT
from sparsene.transform.rts import derive_rts
from sparsene.op_gen.rts_parser import generate_from_rts

from sparsene.logging import get_logger

if __name__ == "__main__":
    logger = get_logger("test")
    rts = derive_rts(DTC_SPMM_FORMAT)
    computent = generate_from_rts("dtc_spmm", rts)

    logger.info("DTC-SPMM:")
    logger.info(rts)
    logger.info(computent)

    # breakpoint()

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
    # computent = generate_from_rts("sr_bcr", rts)

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

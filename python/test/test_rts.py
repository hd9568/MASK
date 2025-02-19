from sparsene.formats.DTC_SpMM import DTC_SPMM_FORMAT

from sparsene.transformation.rts import derive_rts

from sparsene.logging import get_logger

if __name__ == "__main__":
	logger = get_logger("test")
	rts = derive_rts(DTC_SPMM_FORMAT)
	logger.info("Done")
	logger.info(rts)
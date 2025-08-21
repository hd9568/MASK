import argparse

from sparsene.formats.DTC_SpMM import ME_TCF_FORMAT

from sparsene.transform.rts import derive_rts

from sparsene.logging import get_logger, set_logging_level_for_all

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--debug", "-d", action="store_true")
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_logging_level_for_all("DEBUG" if args.debug else "INFO")

    logger = get_logger("test")
    rts = derive_rts(ME_TCF_FORMAT)
    logger.info("Done")
    logger.info(rts)

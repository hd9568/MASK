import json
import logging.config
import logging.handlers
import pathlib
from typing import Literal


def setup_logging():
    config_file = pathlib.Path(__file__).parent / "logger.json"
    with open(config_file) as f:
        config = json.load(f)
    logging.config.dictConfig(config)


setup_logging()


def get_logger(name):
    return logging.getLogger(name)


root_logger = logging.getLogger("root")


def set_logging_level_for_all(
    level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ],
):
    for _, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            logger.setLevel(level)

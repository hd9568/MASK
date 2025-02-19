import json
import logging.config
import logging.handlers
import pathlib


def setup_logging():
    config_file = pathlib.Path(__file__).parent / "logger.json"
    with open(config_file) as f:
        config = json.load(f)
    logging.config.dictConfig(config)


setup_logging()


def get_logger(name):
    return logging.getLogger(name)


root_logger = logging.getLogger("root")

import logging

from .configuration import get_config
from .utils import initialize_logger, track_execution_time

@track_execution_time
def main():

    initialize_logger()

    config = get_config()

    logging.info("Config: %s", config.data)
    
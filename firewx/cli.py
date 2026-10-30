import logging

from .configuration import get_general_config, get_data_config
from .utils import initialize_logger, track_execution_time

@track_execution_time
def main():

    initialize_logger()

    general_config = get_general_config()
    data_config = get_data_config(general_config)

    logging.info("General Config: %s", general_config.data)
    logging.info("Data Config: %s", data_config.data)
    
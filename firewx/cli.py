import logging

from .configuration import get_config
from .utils import initialize_logger, track_execution_time, import_cffdrs
from .data_classes import DatasetModel
from .fire_season import compute_fire_season_mask

@track_execution_time
def main():

    initialize_logger()

    # Load config and import relevant CFFDRS module
    config = get_config()
    cffdrs = import_cffdrs(config)
    logging.info("CFFDRS module imported successfully.")
    import time
    time.sleep(1299)

    logging.info("Opening and initializing dataset...")
    dataset = DatasetModel(config)

    # Compute fire season mask for the entire dataset and set as attribute
    logging.info("Computing fire season mask for the dataset...")
    dataset.fire_season_mask = compute_fire_season_mask(config, dataset)


    # Main loop; FWI calculations must be done iteratively.
    # for year in range(config.start_year, config.end_year + 1):
    #     logging.info("Processing year: %d", year)

    
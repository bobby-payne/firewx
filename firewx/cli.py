import logging

from .configuration import get_config
from .utils import initialize_logger, track_execution_time
from .data_classes import DatasetModel
from .fire_season import compute_fire_season_mask


@track_execution_time
def main():

    initialize_logger()

    # Load config
    config = get_config()

    logging.info("Opening and initializing dataset...")
    dataset = DatasetModel(config)
    logging.info(f"Success! Detected {dataset.number_of_locations} unique locations.")

    # Iterate over each location using the dataset's iterator method and process its data in-place
    for i, _ in enumerate(dataset.iter_location(), start=1):

        logging.info(f"Starting on location {i} / {dataset.number_of_locations}...")


        logging.info(f"⛈️ Processing weather data...")
        dataset.process_data_for_location()

        logging.info("🔥 Computing fire season mask...")
        dataset.fire_season_mask = compute_fire_season_mask(config, dataset)

        logging.info(f"Finished processing location {i}.")




    # Compute fire season mask for the entire dataset and set as attribute
    # logging.info("Computing fire season mask for the dataset...")
    # dataset.fire_season_mask = compute_fire_season_mask(config, dataset)


    # Main loop; FWI calculations must be done iteratively.
    # for year in range(config.start_year, config.end_year + 1):
    #     logging.info("Processing year: %d", year)

    
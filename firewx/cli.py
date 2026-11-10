import os
import logging

from .configuration import get_config
from .utils import initialize_logger, track_execution_time
from .data_classes import DatasetModel
from .fire_season import compute_fire_season_mask
from .fwi_daily import compute_daily_fwi


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

        logging.info("📊 Computing fire weather indices...")
        if dataset.cffwis_version.upper() == 'FWI1987':
            dataset.fire_weather_indices = compute_daily_fwi(config, dataset)
        elif dataset.cffwis_version.upper() == 'FWI2025':
            raise NotImplementedError("FWI2025 computation is not yet implemented.")
        else:
            raise ValueError(f"Unrecognized CFFWIS version: {dataset.cffwis_version}")

        logging.info(f"✏️ Writing to file at {config.output_directory}...")
        dataset.save_fwi_data(f"{config.output_directory}")

        logging.info(f"Success! Finished processing location {i}.\n")
    
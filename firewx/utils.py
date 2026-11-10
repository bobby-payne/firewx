import time
import logging
import pytz
import os
import sys
from types import SimpleNamespace
from timezonefinder import TimezoneFinder
from datetime import datetime
from functools import wraps
from pathlib import Path


def initialize_logger():

    # Configure logging to write to both console or a file
    base_dir = Path(__file__).resolve().parent 
    log_path = (base_dir / 'logs').resolve()
    if not log_path.exists():
        log_path.mkdir(parents=True)
    log_file_location = log_path / 'firewx.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file_location,  # Remove this line if you want to log to the console instead
        filemode='w'
    )


def track_execution_time(func):
    
    @wraps(func)
    def inner_function(*args, **kwargs):

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        if elapsed_time < 60:
            logging.info(f"Finished in {elapsed_time:.2f} seconds.")
        else:
            logging.info(f"Finished in {elapsed_time/60:.2f} minutes.")

        return result
        
    return inner_function


def get_timezone_UTC_offset(lat: float, long: float) -> float:
    """
    Get the timezone offset from UTC at a given latitude and longitude.

    Parameters
    ----------
    lat : float
        Latitude of the location in (-90.0, 90.0).
    long : float
        Longitude of the location in (-180.0, 180.0).

    Returns
    -------
    float
        The UTC offset in hours for the timezone at the given coordinates.
        e.g., -3.5 (for Newfoundland, Canada)
    """

    tz_str = TimezoneFinder().timezone_at(lng=long, lat=lat)
    if tz_str is None:
        raise ValueError(f"Could not determine timezone for coordinates: lat={lat}, long={long}")
    tz = pytz.timezone(tz_str)
    tz_offset = tz.localize(datetime(2024, 1, 1)).utcoffset()
    return tz_offset.total_seconds() / 3600

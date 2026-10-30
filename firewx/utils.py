import time
import logging

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
        level=logging.DEBUG,
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

        logging.info(f"Finished in {elapsed_time/60:.2f} minutes.")

        return result
        
    return inner_function

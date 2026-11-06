import xarray as xr
import pandas as pd
from glob import glob


def open_location_parameters(config):
    """Open location parameters .csv based on the provided configuration.

    Args:
        config: Configuration object containing paths and settings for location parameters.

    Returns:
        pandas.DataFrame: DataFrame containing location parameters.
    """

    location_parameters = pd.read_csv(
        config.path_to_location_parameters,
        )

    # Raise exception if there isn't a latitude, longitude, and year column
    required_columns = ['latitude', 'longitude', 'year']
    for col in required_columns:
        if col not in location_parameters.columns:
            raise ValueError(f"Missing required column '{col}' in location parameters file.")

    # Raise exception if there are duplicates of (latitude, longitude, year)
    if location_parameters.duplicated(subset=['latitude', 'longitude', 'year']).any():
        raise ValueError("Duplicate year(s) detected for the same latitude and longitude.")

    # Convert start_date and end_date columns to datetime if they exist
    if "start_date" in location_parameters.columns:
        location_parameters['start_date'] = pd.to_datetime(
            location_parameters['start_date'],
            format="%Y-%m-%d"
            )
    if "end_date" in location_parameters.columns:
        location_parameters['end_date'] = pd.to_datetime(
            location_parameters['end_date'],
            format="%Y-%m-%d"
            )

    return location_parameters.sort_values(["latitude", "longitude", "year"])


def open_mfdataset_eager(path):
    """Open one or more NetCDF files WITHOUT Dask lazy-loading and load into memory.

    Args:
        path: Path to the NetCDF file(s). Can include wildcards.

    Returns:
        xr.Dataset: Combined dataset from all matching files.
    """

    files = sorted(glob(path))

    if not files:
        raise FileNotFoundError(f"No files found matching: {path}")

    datasets = []

    for file in files:
        ds = xr.open_dataset(file)
        ds.load()  # Load data into memory, without Dask
        datasets.append(ds)
        ds.close()

    if len(datasets) == 1:
        return datasets[0]

    return xr.combine_by_coords(datasets)


def get_wx_data(config):
    """Fetch weather data based on the provided configuration.

    Args:
        config: Configuration object containing paths and settings for weather data.

    Returns:
        dict[str, xr.DataArray]: Dictionary containing weather data.
    """
    if config.lazy_load_data:

        chunks = config.chunks
        return {
            "wind_speed_10m": xr.open_mfdataset(config.wind_speed_10m['path'], chunks={}).chunk(chunks),
            "air_temperature_2m": xr.open_mfdataset(config.air_temperature_2m['path'], chunks={}).chunk(chunks),
            "relative_humidity_2m": xr.open_mfdataset(config.relative_humidity_2m['path'], chunks={}).chunk(chunks),
            "precip_accumulation_1hr": xr.open_mfdataset(config.precip_accumulation_1hr['path'], chunks={}).chunk(chunks),
        }
    
    else:
        
        return {
            "wind_speed_10m": open_mfdataset_eager(config.wind_speed_10m['path']),
            "air_temperature_2m": open_mfdataset_eager(config.air_temperature_2m['path']),
            "relative_humidity_2m": open_mfdataset_eager(config.relative_humidity_2m['path']),
            "precip_accumulation_1hr": open_mfdataset_eager(config.precip_accumulation_1hr['path']),
        }
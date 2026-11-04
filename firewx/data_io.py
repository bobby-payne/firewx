import xarray as xr
from glob import glob


def open_mfdataset_eager(path: str) -> xr.Dataset:
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


def get_wx_data(config) -> dict:
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
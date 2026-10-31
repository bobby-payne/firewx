import xarray as xr


def get_wx_data(config) -> dict:
    """Fetch weather data based on the provided configuration.

    Args:
        config: Configuration object containing paths and settings for weather data.

    Returns:
        dict[str, xr.DataArray]: Dictionary containing weather data.
    """
    if config.lazy_load_data:
        return {
            "wind_speed_10m": xr.open_mfdataset(config.wind_speed_10m['path'], chunks={}),
            "air_temperature_2m": xr.open_mfdataset(config.air_temperature_2m['path'], chunks={}),
            "relative_humidity_2m": xr.open_mfdataset(config.relative_humidity_2m['path'], chunks={}),
            "precip_accumulation_1hr": xr.open_mfdataset(config.precip_accumulation_1hr['path'], chunks={}),
        }
    else:
        return {
            "wind_speed_10m": xr.open_mfdataset(config.wind_speed_10m['path'], chunks=None),
            "air_temperature_2m": xr.open_mfdataset(config.air_temperature_2m['path'], chunks=None),
            "relative_humidity_2m": xr.open_mfdataset(config.relative_humidity_2m['path'], chunks=None),
            "precip_accumulation_1hr": xr.open_mfdataset(config.precip_accumulation_1hr['path'], chunks=None),
        }
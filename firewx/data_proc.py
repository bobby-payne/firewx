
import logging
import numpy as np
from haversine import haversine


def convert_dataset_UTC_to_LST(config, dataset, utc_offset):
    """
    Convert the dataset's time dimension from UTC to Local Standard Time (LST) based on the UTC offset specified.
    Assumes the dataset's time dimension is in UTC, and all spatial coordinates in the dataset share the same timezone
    given by the UTC offset specified.
    Also sets the order of coordinates as ([other coordinates], time).

    Parameters
    ----------
    config : dict
        The configuration dictionary containing the time coordinate name.
    dataset : xarray.Dataset
        The dataset containing the time dimension.
    utc_offset : float
        The UTC offset of the location for LST conversion.

    Returns
    -------
    xarray.Dataset
        The dataset with the time dimension converted to LST.
    """
    time_coordinate_name = config.time_coordinate_name
    dataset = dataset.assign_coords(
        {time_coordinate_name: dataset[time_coordinate_name] + np.timedelta64(int(utc_offset * 3600), 's')}
        )
    return dataset


def select_time_range(config, dataset, start_year, final_year):
    """
    Select the time range from the dataset based on the start and final year specified.

    Parameters
    ----------
    config : dict
        The configuration dictionary containing the time coordinate name.
    dataset : xarray.Dataset
        The dataset containing the time dimension.
    start_year : int
        The start year for the time range selection.
    final_year : int
        The final year for the time range selection.

    Returns
    -------
    xarray.Dataset
        The dataset restricted to the specified time range.
    """
    
    if start_year is not None:
        start_datetime = np.datetime64(f"{start_year}-01-01 00:00")
    else:
        start_datetime = None
    if final_year is not None:
        final_datetime = np.datetime64(f"{final_year}-12-31 23:59")
    else:
        final_datetime = None

    time_coordinate_name = config.time_coordinate_name

    return dataset.sel({time_coordinate_name: slice(start_datetime, final_datetime)})


def select_nearest_gridpoint(config, dataset, latitude, longitude):
    """
    Select the nearest grid point in the dataset to the latitude and longitude specified in config.

    Parameters
    ----------
    config : dict
        The configuration dictionary containing grid specifications.
    dataset : xarray.Dataset
        The dataset containing the grid points.
    latitude : float
        The target latitude for selecting the nearest grid point.
    longitude : float
        The target longitude for selecting the nearest grid point.

    Returns
    -------
    xarray.Dataset
        The dataset corresponding to the nearest grid point.
    """
    lat_name = config.latitude_coordinate_name
    lon_name = config.longitude_coordinate_name
    target_lat = latitude
    target_lon = longitude

    nearest_lat = dataset[lat_name].sel({lat_name: target_lat}, method='nearest').item()
    nearest_lon = dataset[lon_name].sel({lon_name: target_lon}, method='nearest').item()

    distance = haversine((target_lat, target_lon), (nearest_lat, nearest_lon), unit='deg')
    if abs(distance) > 1:
        logging.warning(
            "Nearest grid point to specified latitude and longitude is more than 1° away. Is this intended?"
            )

    return dataset.sel({lat_name: nearest_lat, lon_name: nearest_lon})


def convert_longitude_range(longitude, to_centered: bool = True):
    """
    Converts longitude float between ranges [0,360) and [-180,180).
    
    If "to_centered=True", converts the longitude from the [0,360) convention
    to the standard [-180,180) convention, required for timezone retrieval
    and by the cffdrs FWI calculation function.

    If "to_centered=False", converts the longitude from the [-180,180) convention
    to the [0,360) convention.

    Parameters
    ----------
    longitude : float
        The longitude value to convert.
    to_centered : bool, optional
        If True, convert from [0,360) to [-180,180).
        If False, convert from [-180,180) to [0,360).
        Default is True

    Returns
    -------
    float
        The converted longitude value.
    """

    if to_centered:
        longitude = ((longitude + 180.) % 360.) - 180.
    else:
        longitude = longitude % 360.
    return longitude


def convert_dataset_longitude_range(config, dataset, to_centered: bool = True):
    """
    Converts longitude range of an xarray.Dataset between [0,360) and [-180,180).
    
    If "to_centered=True", converts the longitude coordinate from the [0,360) convention
    to the standard [-180,180) convention, required for timezone retrieval
    and by the cffdrs FWI calculation function.

    If "to_centered=False", converts the longitude coordinate from the [-180,180) convention
    to the [0,360) convention.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset containing the longitude coordinate.
    to_centered : bool, optional
        If True, convert from [0,360) to [-180,180).
        If False, convert from [-180,180) to [0,360).
        Default is True

    Returns
    -------
    xarray.Dataset
        The dataset containing the longitude coordinate.
    """

    longitude_name = config.longitude_coordinate_name
    dataset = dataset.assign_coords({
        longitude_name: convert_longitude_range(dataset[longitude_name], to_centered=to_centered)
    })

    return dataset


def apply_transform(config, variable, dataset):
    """
    Transform the specified variable in the dataset by applying a mathematical operation.
    Transforms are specified in the config file and evaluated using Python's
    in-built eval function.

    Parameters
    ----------
    config : dict
        The configuration dictionary containing transformation specifications.
    variable : str
        The name of the variable to transform.
    dataset : xarray.Dataset
        The dataset containing the weather variable to be transformed.

    Returns
    -------
    xarray.Dataset
        The transformed dataset with the transformations applied.
    """
    # read in the user-specified transform
    transform = config[variable]["transform"]
    variable_name = config[variable]["variable_name"]

    # Apply each transformation to the corresponding variable
    if transform is not None:
        print(f"Applying transformation to {variable}: {transform}")
        transform_func = eval("lambda x:" + transform)  # TODO: replace eval with a secure alternative (e.g numexpr)
        dataset[variable_name] = transform_func(dataset[variable_name])

    return dataset


def apply_unit_conversion(config, variable, dataset):
    """
    Apply unit conversions to the specified variable in the dataset based on the configuration.

    Parameters
    ----------
    config : dict
        The configuration dictionary containing unit specifications.
    variable : str
        The name of the variable to convert.
    dataset : xarray.Dataset
        The dataset containing the weather variable to be converted.

    Returns
    -------
    xarray.Dataset
        The dataset with the variable converted to the desired units.
    """

    if variable == "wind_speed_10m":
        variable_name = config.wind_speed_10m["variable_name"]
        dataset[variable_name] = convert_windspeed_units_to_kmh(
            dataset[variable_name], config.wind_speed_10m["units"]
            )
    elif variable == "air_temperature_2m":
        variable_name = config.air_temperature_2m["variable_name"]
        dataset[variable_name] = convert_temperature_units_to_celsius(
            dataset[variable_name], config.air_temperature_2m["units"]
            )
    elif variable == "relative_humidity_2m":
        variable_name = config.relative_humidity_2m["variable_name"]
        dataset[variable_name] = convert_relative_humidity_units_to_percentage(
            dataset[variable_name], config.relative_humidity_2m["units"]
            )
    elif variable == "precip_accumulation_1hr":
        variable_name = config.precip_accumulation_1hr["variable_name"]
        dataset[variable_name] = convert_precipitation_units_to_mm(
            dataset[variable_name], config.precip_accumulation_1hr["units"]
            )
    else:
        raise ValueError(f"Unsupported variable name: {variable}.")

    return dataset


def convert_windspeed_units_to_kmh(wind_speed, from_unit):

    if from_unit in ["km/h", "kph"]:
        return wind_speed
    elif from_unit in ["m/s"]:
        return wind_speed * 3.6
    elif from_unit in ["mph", "mi/h"]:
        return wind_speed * 1.60934
    elif from_unit in ["kts", "knots"]:
        return wind_speed * 1.852
    else:
        raise ValueError(
            f"Unsupported wind speed unit: {from_unit}. Choose from ['km/h', 'm/s', 'mph', 'kts']."
            )


def convert_temperature_units_to_celsius(temperature, from_unit):

    if from_unit in ["C", "degC"]:
        return temperature
    elif from_unit in ["F", "degF"]:
        return (temperature - 32) * 5.0/9.0
    elif from_unit in ["K"]:
        return temperature - 273.15
    else:
        raise ValueError(
            f"Unsupported temperature unit: {from_unit}. Choose from ['C', 'F', 'K']."
            )


def convert_relative_humidity_units_to_percentage(relative_humidity, from_unit=None):

    if from_unit in ["%", "percent"]:
        return relative_humidity
    elif from_unit in [None, "fraction", "frac", "decimal"]:
        return relative_humidity * 100
    else:
        raise ValueError(
            f"Unsupported relative humidity unit: {from_unit}. Choose from ['%', null]."
        )


def convert_precipitation_units_to_mm(precipitation, from_unit):

    if from_unit in ["mm", "millimeter", "millimeters"]:
        return precipitation
    elif from_unit in ["cm", "centimeter", "centimeters"]:
        return precipitation * 10
    elif from_unit in ["m", "meter", "meters"]:
        return precipitation * 1000
    elif from_unit in ["in", "inch", "inches"]:
        return precipitation * 25.4
    else:
        raise ValueError(
            f"Unsupported precipitation unit: {from_unit}. Choose from ['mm', 'cm', 'm', 'in']."
        )

def select_noon_values(config, dataset):
    """
    Select only the values corresponding to noon (12:00) from the dataset.

    Parameters:
    - config: Configuration object containing the time coordinate name.
    - dataset: xarray DataArray or Dataset with a time coordinate.

    Returns:
    - dataset: xarray DataArray or Dataset containing only the noon values.
    """

    time_coordinate_name = config.time_coordinate_name
    dataset = dataset.sel({time_coordinate_name: dataset[time_coordinate_name].dt.hour == 12})
    return dataset


def convert_precipitation_hourly_to_daily(config, precipitation_hourly):
    """
    Convert hourly precipitation data to daily precipitation data.
    Daily precipitation accumulation is from NOON to NOON of the next day.

    Parameters:
    - config: Configuration object containing the time coordinate name.
    - precipitation_hourly: Hourly precipitation data (xarray DataArray or Dataset).

    Returns:
    - precipitation_daily: Daily precipitation data (xarray DataArray or Dataset).
    """

    time_coordinate_name = config.time_coordinate_name

    precipitation_daily = precipitation_hourly.resample(
        **{time_coordinate_name: "24h"},
        offset="12h",
        closed="right",
        label="right"
    ).sum(min_count=24)

    precipitation_daily = precipitation_daily.isel(time=slice(None, -1))

    return precipitation_daily

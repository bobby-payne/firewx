import logging
import numpy as np
import pandas as pd

from cffdrs.fwi import (
    fine_fuel_moisture_code,
    duff_moisture_code,
    drought_code,
    initial_spread_index,
    buildup_index,
    fire_weather_index,
)


def compute_daily_fwi(config, dataset_object):
    """
    Calculate the daily Fire Weather Index (FWI) for all years in the dataset,
    combined into a single pandas dataframe.

    Parameters:
    config: Configuration object containing relevant settings.
    dataset_object: Dataset object containing the necessary meteorological data.

    Returns:
    pd.DataFrame: DataFrame containing the daily FWI and its components for all years, indexed by date.
    """

    fwi_df_dict = {}
    for year in dataset_object.startup_data['year'].unique():
        fwi_df_dict[year] = compute_daily_fwi_for_year(config, dataset_object, year)

    # combine all yearly dataframes into a single dataframe
    combined_fwi_df = pd.concat(fwi_df_dict.values())
    return combined_fwi_df


def compute_daily_fwi_for_year(config, dataset_object, year):
    """
    Calculate the daily Fire Weather Index (FWI) for a given year.

    Parameters:
    config: Configuration object containing relevant settings.
    dataset_object: Dataset object containing the necessary meteorological data.
    year (int): Year for which to calculate the daily FWI.

    Returns:
    pd.DataFrame: DataFrame containing the daily FWI and its components for the given year.
    """

    # Retrieve relevant data from the dataset object and config for the given year
    time_coordinate_name = config.time_coordinate_name
    temp_name = config.air_temperature_2m["variable_name"]
    rh_name = config.relative_humidity_2m["variable_name"]
    ws_name = config.wind_speed_10m["variable_name"]
    prec_name = config.precip_accumulation_1hr["variable_name"]

    datetimes = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")

    fire_season_mask = dataset_object.fire_season_mask['fire_season_mask'].sel(
        {time_coordinate_name: slice(f"{year}-01-01", f"{year}-12-31")}).values
    air_temperature_2m = dataset_object.air_temperature_2m_noon[temp_name].sel(
        {time_coordinate_name: slice(f"{year}-01-01", f"{year}-12-31")}).values
    relative_humidity_2m = dataset_object.relative_humidity_2m_noon[rh_name].sel(
        {time_coordinate_name: slice(f"{year}-01-01", f"{year}-12-31")}).values
    wind_speed_10m = dataset_object.wind_speed_10m_noon[ws_name].sel(
        {time_coordinate_name: slice(f"{year}-01-01", f"{year}-12-31")}).values
    precipitation_24hr = dataset_object.precip_accumulation_24hr[prec_name].sel(
        {time_coordinate_name: slice(f"{year}-01-01", f"{year}-12-31")}).values
    latitude = dataset_object.latitude
    longitude = dataset_object.longitude
    first_active_day = datetimes[np.argmax(fire_season_mask)]

    # Loop over all datetimes; if fire season mask is true, then calculate FWIs.
    ffmc_arr, dmc_arr, dc_arr, isi_arr, bui_arr, fwi_arr = [], [], [], [], [], []
    for i, dt in enumerate(datetimes):
        if not fire_season_mask[i]:
            ffmc_arr.append(None)
            dmc_arr.append(None)
            dc_arr.append(None)
            isi_arr.append(None)
            bui_arr.append(None)
            fwi_arr.append(None)
            continue

        temperature = air_temperature_2m[i]
        relative_humidity = relative_humidity_2m[i]
        wind_speed = wind_speed_10m[i]
        precipitation = precipitation_24hr[i]

        if dt == first_active_day:
            ffmc, dmc, dc = get_moisture_code_starting_values(config, dataset_object, year)
        else:
            ffmc = fine_fuel_moisture_code(ffmc, temperature, relative_humidity, wind_speed, precipitation)
            dmc = duff_moisture_code(dmc, temperature, relative_humidity, precipitation, latitude, dt.month)
            dc = drought_code(dc, temperature, relative_humidity, precipitation, latitude, dt.month)

        isi = initial_spread_index(ffmc, wind_speed, fbp_mod=False)
        bui = buildup_index(dmc, dc)
        fwi = fire_weather_index(isi, bui)

        ffmc_arr.append(ffmc)
        dmc_arr.append(dmc)
        dc_arr.append(dc)
        isi_arr.append(isi)
        bui_arr.append(bui)
        fwi_arr.append(fwi)

    # put arrays into a dataframe
    fwi_df = pd.DataFrame(
        {
            "latitude": latitude,
            "longitude": longitude,
            "fire_season_mask": fire_season_mask,
            "ws": wind_speed_10m,
            "temp": air_temperature_2m,
            "rh": relative_humidity_2m,
            "prec": precipitation_24hr,
            "ffmc": ffmc_arr,
            "dmc": dmc_arr,
            "dc": dc_arr,
            "isi": isi_arr,
            "bui": bui_arr,
            "fwi": fwi_arr,
        },
        index=datetimes
    )

    return fwi_df


def get_moisture_code_starting_values(config, dataset_object, year):
    """
    Get the starting values for the moisture codes (FFMC, DMC, DC) for a given fire season.

    Parameters:
    config: Configuration object containing default values for the moisture codes.
    dataset_object: Dataset object containing startup data.
    year (int): Year for which to retrieve the starting values.

    Returns:
    tuple: A tuple containing the starting values for FFMC, DMC, and DC.
    """

    startup_data = dataset_object.startup_data[dataset_object.startup_data['year'] == year]

    # check if ffmc_startup column exists in the startup data
    if 'ffmc_startup' in startup_data.columns:
        ffmc_startup = startup_data['ffmc_startup'].values[0]
    else:
        logging.info(f"No startup values provided for FFMC for year {year}; using default.")
        ffmc_startup = config.FFMC_default

    if 'dmc_startup' in startup_data.columns:
        dmc_startup = startup_data['dmc_startup'].values[0]
    else:
        logging.info(f"No startup values provided for DMC for year {year}; using default.")
        dmc_startup = config.DMC_default

    if 'dc_startup' in startup_data.columns:
        dc_startup = startup_data['dc_startup'].values[0]
    else:
        logging.info(f"No startup values provided for DC for year {year}; using default.")
        dc_startup = config.DC_default

    return ffmc_startup, dmc_startup, dc_startup



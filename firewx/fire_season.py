import numpy as np
import xarray as xr
import pandas as pd
import logging


def compute_fire_season_mask(
                        config,
                        dataset_object,
                        ):
    """
    High-level function to compute the fire season from the dataset object.

    Parameters
    ----------
    config : data_classes.Config
        The configuration object containing fire season parameters.
    dataset_object : data_classes.DatasetModel
        The dataset object containing hourly air temperature data in deg C.

    Returns
    -------
    data_classes.DatasetModel
        A dataset indicating the fire season on an hourly basis. 
        The returned dataset will have the same structure as the input dataset, 
        but with a fire season mask attribute named 'fire_season_mask'.
    """

    # assume time dim has the same name as the time coordinate given in config
    time_coordinate_name = config.time_coordinate_name

    # compute the daily maximum temperature
    air_temperature_daymax = dataset_object.air_temperature_2m.resample(time="1D").max()

    # empty list to store fire season masks for each year
    fire_season_masks = []

    # calculate fire season mask for each year
    for row in dataset_object.startup_data.itertuples(index=False):

        year = row.year
        if not pd.isna(config.crop["start_year"]):
            if year < config.crop["start_year"]:
                continue
        if not pd.isna(config.crop["final_year"]):
            if year > config.crop["final_year"]:
                continue

        try:
            start_date = row.start_date
        except:
            start_date = None
        try:
            end_date = row.end_date
        except:
            end_date = None

        print(f"Processing year {year}...")

        # select the daily max temperature for the current year
        air_temperature_daymax_for_year = air_temperature_daymax.sel(time=str(year))

        # get the masks for when the start and stop thresholds are being met
        masks_for_thresholds_met = compute_masks_for_thresholds_met(
            config,
            air_temperature_daymax_for_year,
            )

        # compute the fire season mask for the current year
        fire_season_mask_daily = compute_daily_fire_season_mask_for_year(masks_for_thresholds_met)

        # update the fire season mask to only include the longest season
        fire_season_mask_daily = handle_shoulder_seasons(fire_season_mask_daily)

        # impose user-specified start and end dates for the fire season if provided
        fire_season_mask_daily = modify_fire_season_mask_with_user_dates(
            fire_season_mask_daily,
            user_start_date=start_date,
            user_end_date=end_date,
        )

        # make hourly by repeating each entry 24 times
        fire_season_mask_hourly = make_fire_season_mask_hourly(config, fire_season_mask_daily)

        # attach the fire season mask to the dataset object for the current year
        fire_season_masks.append(fire_season_mask_hourly)

    # combine all yearly fire season masks into a single DataArray
    fire_season_mask_all_years = xr.concat(fire_season_masks, dim=time_coordinate_name)

    return fire_season_mask_all_years


def compute_masks_for_thresholds_met(config, air_temperature_2m_daymax):
    """
    Compute masks where the daily maximum temperature exceeds the start threshold
    or falls below the stop threshold for a consecutive number of days.

    The first mask in the tuple with be true at times when the daily max temp > start threshold
    for N consecutive days, and false otherwise.
    The second mask in the tuple will be true at times where the daily max temp < stop threshold 
    for N consecutive days, and false otherwise.

    Parameters
    ----------
    config : data_classes.Config
        The configuration object containing fire season parameters.
    air_temperature_2m_daymax : xarray.DataArray
        The daily maximum air temperature for a SINGLE year.

    Returns
    -------
    xarray.Dataset
        A dataset containing two boolean arrays indicating where the start and stop temperature thresholds
        are met for the required consecutive days. The variables are named "above_start_temp" and "below_stop_temp".
    """
    var_name = config.air_temperature_2m["variable_name"]
    season_start_temperature = config.season_start_temp
    season_stop_temperature = config.season_stop_temp
    season_consecutive_days = config.season_consecutive_days

    # Create boolean masks for when the temperature exceeds the start threshold
    # and falls below the stop threshold.
    mask_above_start_temp = air_temperature_2m_daymax[var_name] > season_start_temperature
    mask_below_stop_temp = air_temperature_2m_daymax[var_name] < season_stop_temperature

    # Apply rolling window to check for when the threshold is met for the required consecutive days.
    mask_above_start_temp_consecutive = (
        mask_above_start_temp.rolling(time=season_consecutive_days, min_periods=season_consecutive_days)
        .sum().shift(time=1, fill_value=False) == season_consecutive_days
    )

    mask_below_stop_temp_consecutive = (
        mask_below_stop_temp.rolling(time=season_consecutive_days, min_periods=season_consecutive_days)
        .sum().shift(time=1, fill_value=False) == season_consecutive_days
    )

    masks_for_thresholds_met = xr.Dataset({
        "above_start_temp": mask_above_start_temp_consecutive,
        "below_stop_temp": mask_below_stop_temp_consecutive
    })

    return masks_for_thresholds_met


def compute_daily_fire_season_mask_for_year(masks_for_thresholds_met):
    """
    Compute the daily fire season mask for a single year based on the provided temperature threshold masks.

    Parameters
    ----------
    masks_for_thresholds_met : xarray.Dataset
        A dataset containing boolean arrays "above_start_temp" and "below_stop_temp" indicating where
        the start and stop temperature thresholds are met for the required consecutive days.
        See `compute_temperature_threshold_masks_for_year` for details on how these masks are computed.

    Returns
    -------
    xarray.DataArray
        A boolean array indicating the fire season mask for the year.
    """

    above_start_temp = masks_for_thresholds_met["above_start_temp"].values
    below_stop_temp = masks_for_thresholds_met["below_stop_temp"].values

    fire_season = np.zeros(len(above_start_temp), dtype=bool)

    active = False

    for i in range(len(above_start_temp)):

        if not active and above_start_temp[i]:
            active = True

        elif active and below_stop_temp[i]:
            active = False

        fire_season[i] = active

    # raise error if no fire season was detected based on temperature thresholds
    if not fire_season.any():
        raise ValueError("No fire season detected at location.")

    # put in xarray Dataset
    fire_season = xr.Dataset(
        {
            "fire_season_mask": xr.DataArray(
                fire_season,
                coords=masks_for_thresholds_met["above_start_temp"].coords,
                dims=masks_for_thresholds_met["above_start_temp"].dims,
            )
        }
    )

    return fire_season


def handle_shoulder_seasons(fire_season_mask_daily: xr.Dataset) -> xr.Dataset:
    """
    Retain only the longest active fire season in a single-year dataset,
    as per McElhinny et al. (2020).

    All other active fire seasons are set to inactive (False).

    Args:
        fire_season_mask_daily: Dataset containing a Boolean variable called 'fire_season_mask'.

    Returns:
        Dataset with only the longest fire season retained in 'fire_season_mask'.
    """
    fire_season_mask_daily_updated = fire_season_mask_daily.copy()

    # Extract fire-season mask
    fire_season_mask_values = fire_season_mask_daily["fire_season_mask"].fillna(False).values.astype(bool)

    # Find indices of active days
    active_indices = np.where(fire_season_mask_values)[0]

    # No active fire seasons
    if len(active_indices) == 0:
        fire_season_mask_daily_updated["fire_season_mask"] = fire_season_mask_values
        return fire_season_mask_daily_updated

    # Identify separate consecutive active periods
    seasons = np.split(
        active_indices,
        np.where(np.diff(active_indices) > 1)[0] + 1
    )

    # Find longest season (first retained in case of a tie)
    longest_season = max(seasons, key=len)

    # Retain only the longest season
    fire_season_mask_values_updated = np.zeros_like(fire_season_mask_values)
    fire_season_mask_values_updated[longest_season] = True

    fire_season_mask_daily_updated["fire_season_mask"].data = (
        fire_season_mask_values_updated
    )

    return fire_season_mask_daily_updated


def modify_fire_season_mask_with_user_dates(
        fire_season_mask,
        user_start_date=None,
        user_end_date=None,
):
    """
    Override the detected fire-season start and/or end dates
    according to user-specified start_date and end_date.
    If a boundary is not specified, the function will use the start or end date
    derived from the temperature thresholds. If neither is available, an error is raised.
    Assumes input fire_season_mask is not empty, and only contains a single fire season.

    Parameters
    ----------
    fire_season_mask : xarray.Dataset
        Dataset containing the fire season mask.
    user_start_date : str or pd.Timestamp, optional
        User-specified start date of the fire season.
    user_end_date : str or pd.Timestamp, optional
        User-specified end date of the fire season.

    Returns
    -------
    xarray.Dataset
        Dataset with the fire season mask modified according to the user-specified start and end dates.
    """

    # relevant data
    datetimes = pd.DatetimeIndex(fire_season_mask.time.values)
    mask_values = fire_season_mask["fire_season_mask"].values.astype(bool)
    
    # get the time index of start and end of the fire season, according to the given mask
    active_indices = np.flatnonzero(mask_values)
    tempbased_start_idx = active_indices[0]
    tempbased_end_idx = active_indices[-1] + 1

    # get the time index of start and/or end of the fire season as provided by the user
    if not pd.isna(user_start_date):

        if pd.Timestamp(user_start_date).year != datetimes[0].year:
            raise ValueError(f"User-provided season start date ({user_start_date.date()}) for year {datetimes[0].year} is not compatible.")

        user_start_date = pd.Timestamp(user_start_date).normalize()
        user_start_idx = datetimes.get_loc(user_start_date)

    if not pd.isna(user_end_date):

        if pd.Timestamp(user_end_date).year != datetimes[0].year:
            raise ValueError(f"User-provided season end date ({user_end_date.date()}) for year {datetimes[0].year} is not compatible.")

        user_end_date = pd.Timestamp(user_end_date).normalize()
        user_end_idx = datetimes.get_loc(user_end_date)

    # modify the mask according to user-specified start and end dates
    if not pd.isna(user_start_date):

        # case: user-specified start date is later than temperature-derived start date
        if tempbased_start_idx < user_start_idx:
            mask_values[tempbased_start_idx:user_start_idx] = False

        # case: user-specified start date is earlier than temperature-derived start date
        elif tempbased_start_idx > user_start_idx:
            mask_values[user_start_idx:tempbased_start_idx] = True

        # case: equal
        else:
            pass

    if not pd.isna(user_end_date):

        # case: user-specified end date is earlier than temperature-derived end date
        if tempbased_end_idx > user_end_idx:
            mask_values[user_end_idx:tempbased_end_idx] = False

        # case: user-specified end date is later than temperature-derived end date
        elif tempbased_end_idx < user_end_idx:
            mask_values[tempbased_end_idx:user_end_idx] = True

        # case: equal
        else:
            pass


    # Put into an xarray dataset
    updated_mask = xr.Dataset(
        {
            "fire_season_mask": xr.DataArray(
                mask_values,
                coords=fire_season_mask.coords,
                dims=fire_season_mask.dims,
            )
        }
    )

    return updated_mask


def make_fire_season_mask_hourly(config, fire_season_mask_daily):
    """
    Convert a daily fire season mask to an hourly fire season mask by forward filling.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the time coordinate name.
    fire_season_mask_daily : xarray.Dataset
        Daily fire season mask.

    Returns
    -------
    xarray.Dataset
        Hourly fire season mask.
    """
    time_coordinate_name = config.time_coordinate_name
    datetimes_daily = fire_season_mask_daily[time_coordinate_name]

    datetimes_hourly = pd.date_range(
        start=datetimes_daily.values[0],
        end=datetimes_daily.values[-1] + pd.Timedelta(hours=23),
        freq="1h"
    )

    fire_season_mask_hourly = (
        fire_season_mask_daily
        .reindex({time_coordinate_name: datetimes_hourly}, method="ffill")
    )

    return fire_season_mask_hourly
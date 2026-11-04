import numpy as np
import xarray as xr


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

    # compute the daily maximum temperature
    air_temperature_daymax = dataset_object.air_temperature_2m.resample(time="1D").max()

    # empty list to store fire season masks for each year
    fire_season_masks = []

    # calculate fire season mask for each year
    # for year in range(config.start_year, config.final_year + 1):
    for year in [2000]:

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

        # make hourly by repeating each entry 24 times
        fire_season_mask_hourly = fire_season_mask_daily.resample(time="1h").ffill()

        # attach the fire season mask to the dataset object for the current year
        fire_season_masks.append(fire_season_mask_hourly)

    # combine all yearly fire season masks into a single DataArray
    fire_season_mask_all_years = xr.concat(fire_season_masks, dim="time")

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
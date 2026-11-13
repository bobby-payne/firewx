
import os

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


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PATH_TO_INPUT_CSV = "/sapho/rpayne/tankervision/tv_stns.csv"
OUTPUT_DIRECTORY = "/sapho/rpayne/tankervision/"

TEMP_COL = "temp"   # Temperature in Celsius
RH_COL = "rh"       # Relative humidity in percent
WS_COL = "ws"       # Wind speed in km/h
PREC_COL = "prec"   # Precipitation in mm

DAILY_FWI_COLS = [
    "ffmc_daily",
    "dmc_daily",
    "dc_daily",
    "isi_daily",
    "bui_daily",
    "fwi_daily",
]


# ---------------------------------------------------------------------
# FWI calculations
# ---------------------------------------------------------------------

def calculate_daily_fwi(
    dataframe_station_fr,
    station,
    fire_row,
    lat,
):
    """
    Calculate daily FWI components for one station and fire row.

    Daily calculations use noon weather observations. The 1600 values
    on the first day are used as startup moisture codes, so behaviour
    indices are not calculated for that day.

    Returns
    -------
    pandas.DataFrame
        Daily FWI components mapped onto all rows in the input
        station/fire-row DataFrame.
    """

    datetimes = dataframe_station_fr["datetime"]

    if not datetimes.is_unique:
        raise ValueError(
            f"Datetimes are not unique for station {station} "
            f"and fire row {fire_row}"
        )

    # Acquire start day and startup moisture codes
    start_date = datetimes.min().date()

    # Lists for storing daily FWI components
    ffmc_list = []
    dmc_list = []
    dc_list = []
    isi_list = []
    bui_list = []
    fwi_list = []

    # Loop over each datetime and calculate FWI at noon
    for j, dt in enumerate(datetimes):

        if j == 0 and dt.hour > 16:
            raise ValueError(
                "Very first observation is after 1600; "
                "your output will be wrong"
            )

        # Daily FWI only considers noon values
        if dt.hour != 12:
            continue

        # Use 1600 moisture codes as startup values on the first day
        if dt.date() == start_date:

            ffmc_0 = dataframe_station_fr[
                datetimes.dt.hour == 16
            ]["ffmc"].iloc[0]

            dmc_0 = dataframe_station_fr[
                datetimes.dt.hour == 16
            ]["dmc"].iloc[0]

            dc_0 = dataframe_station_fr[
                datetimes.dt.hour == 16
            ]["dc"].iloc[0]

            ffmc_list.append(ffmc_0)
            dmc_list.append(dmc_0)
            dc_list.append(dc_0)

            isi_list.append(np.nan)
            bui_list.append(np.nan)
            fwi_list.append(np.nan)

            continue

        # Get weather variables for the current datetime
        month = dt.month
        temp = dataframe_station_fr[TEMP_COL].iloc[j]
        rh = dataframe_station_fr[RH_COL].iloc[j]
        ws = dataframe_station_fr[WS_COL].iloc[j]

        # Sum hourly precipitation over the previous 24 hours
        if j < 24:
            prec_24h = dataframe_station_fr[PREC_COL].iloc[0:j].sum()
        else:
            prec_24h = dataframe_station_fr[PREC_COL].iloc[j - 24:j].sum()

        # Get previous day's moisture codes
        ffmc_prev = ffmc_list[-1]
        dmc_prev = dmc_list[-1]
        dc_prev = dc_list[-1]

        # Calculate new moisture codes
        ffmc = float(
            fine_fuel_moisture_code(
                ffmc_prev, temp, rh, ws, prec_24h
            )
        )

        dmc = float(
            duff_moisture_code(
                dmc_prev, temp, rh, prec_24h, lat, month,
                lat_adjust=True
            )
        )

        dc = float(
            drought_code(
                dc_prev, temp, rh, prec_24h, lat, month,
                lat_adjust=True
            )
        )

        # Calculate corresponding daily FWIs
        isi = float(initial_spread_index(ffmc, ws, fbp_mod=False))
        bui = float(buildup_index(dmc, dc))
        fwi = float(fire_weather_index(isi, bui))

        # Store results
        ffmc_list.append(ffmc)
        dmc_list.append(dmc)
        dc_list.append(dc)
        isi_list.append(isi)
        bui_list.append(bui)
        fwi_list.append(fwi)

    # Map daily results onto every hourly row using the date
    dates = datetimes.dt.date.unique()

    daily_dicts = {
        "ffmc_daily": dict(zip(dates, ffmc_list)),
        "dmc_daily": dict(zip(dates, dmc_list)),
        "dc_daily": dict(zip(dates, dc_list)),
        "isi_daily": dict(zip(dates, isi_list)),
        "bui_daily": dict(zip(dates, bui_list)),
        "fwi_daily": dict(zip(dates, fwi_list)),
    }

    daily_values = pd.DataFrame(index=dataframe_station_fr.index)

    for column, daily_dict in daily_dicts.items():
        daily_values[column] = datetimes.dt.date.map(daily_dict)

    return daily_values


# ---------------------------------------------------------------------
# DataFrame processing
# ---------------------------------------------------------------------

def process_dataframe(dataframe):
    """
    Calculate daily FWI components for every centroid, station,
    and fire row in the input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Original DataFrame with additional daily FWI columns.
    """

    # Convert timestamp column to pandas datetime
    datetime_original = dataframe["datetime"].copy()
    dataframe["datetime"] = pd.to_datetime(dataframe["datetime"])

    # Initialize daily FWI columns
    for column in DAILY_FWI_COLS:
        dataframe[column] = None

    # Get list of centroids
    centroid_list = dataframe["centroid"].unique()
    Nc = len(centroid_list)

    # Loop over each centroid
    for i, centroid in enumerate(centroid_list, start=1):

        print(f"Working on centroid {i} / {Nc}")

        dataframe_subset = dataframe[
            dataframe["centroid"] == centroid
        ]

        nearest_stations = dataframe_subset[
            "nearest_stations_name"
        ].unique()

        # Loop over nearest stations
        for station in nearest_stations:

            print(f"Processing nearest station: {station}")

            dataframe_station = dataframe_subset[
                dataframe_subset["nearest_stations_name"] == station
            ]

            lat = dataframe_station["lat"].iloc[0]
            fire_rows = dataframe_station["fire_row"].unique()

            # Loop over fire rows
            for fire_row in fire_rows:

                dataframe_station_fr = dataframe_station[
                    dataframe_station["fire_row"] == fire_row
                ].copy()

                # Calculate daily FWI components
                daily_values = calculate_daily_fwi(
                    dataframe_station_fr=dataframe_station_fr,
                    station=station,
                    fire_row=fire_row,
                    lat=lat,
                )

                # Assign results to the original DataFrame
                dataframe.loc[
                    dataframe_station_fr.index, DAILY_FWI_COLS
                ] = daily_values[DAILY_FWI_COLS]

    # Restore original datetime format
    dataframe["datetime"] = datetime_original

    return dataframe


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    # Read input data
    dataframe = pd.read_csv(PATH_TO_INPUT_CSV)

    # Calculate daily FWI components
    dataframe = process_dataframe(dataframe)

    # Save output
    output_path = os.path.join(
        OUTPUT_DIRECTORY,
        "tv_stns_daily_fwis.csv"
    )

    print(f"Saving modified dataframe to: {output_path}")

    dataframe.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

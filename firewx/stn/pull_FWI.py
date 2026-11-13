from pathlib import Path
import geopandas as gpd
import pandas as pd
from ast import literal_eval

def join_stations_to_FWI(FIRES_CENTROIDS_STATIONS_PATH = Path(r"C:\Users\jbuskas\OneDrive - NRCan RNCan\TankerVision\Nearest Weather Station Analysis\_centroids_nearestStations_v3.csv"),
                         FWI_DATA_PATH = Path(r"C:\Users\jbuskas\OneDrive - NRCan RNCan\Fire Weather Stations Project\QC'd Station Data\BC_fwi2025_hourly\BC_fwi2025_hourly.csv")):
    """
    Load the BC FWI 2025 hourly dataset and join observations to fires list.
    Join based on the 3 closest weather stations.
    Returns the final dataframe of hourly FWI observations for the 3 closest weather stations for each fire, and exports it as a csv in the same path as the fire centroids dataframe provided.
    """
    fwi_data = pd.read_csv(FWI_DATA_PATH)
    fires = pd.read_csv(FIRES_CENTROIDS_STATIONS_PATH)
    # clean fires dataset - remove old irrelevant nearest weather station column
    fires = fires.drop(columns="NEAR_WX_ST")

    # narrow down FWI observations to 2025 only
    fwi_data_2025 = fwi_data[fwi_data['yr']==2025]

    # convert datettime column to pandas datetime format
    # The Z at the end of the datetime value indicates Zulu time which is UTC 0
    fwi_data_2025['datetime'] = pd.to_datetime(fwi_data_2025["datetime"], utc=True)

    # drop irrelevant fwi_data columns to reduce load time and keep all other observations
    fwi_data_2025 = fwi_data_2025.drop(columns=['id', 'yr', 'mon', 'day', 'hr', 'timezone', 'province', 'reason', 'obs_type', 'technique'])

    # create new dataframe with the columns to keep
    fire_stations = fires[['fr_nmbr', 'YEAR', 'centroid', 'nearest_stations']].copy()

    # convert nearest_station column back to tuple from string
    fire_stations['nearest_stations'] = fire_stations['nearest_stations'].apply(lambda value: literal_eval(value) if isinstance(value, str) else value)

    # extract station names as a separate column and add rank column
    fire_stations['nearest_stations_name'] = fire_stations['nearest_stations'].apply(lambda x: tuple(station for station, dist in x))
    fire_stations = (
        fire_stations.reset_index(drop=True)
        .rename_axis("fire_row")
        .reset_index()
        .explode("nearest_stations_name")
    )
    fire_stations["station_rank"] = (fire_stations.groupby("fire_row").cumcount() + 1)

    # join fire stations to the fwi_data_2025 dataset
    fires_final = fire_stations.merge(
        fwi_data_2025,
        left_on="nearest_stations_name",
        right_on="name",
        how="left",
        indicator=True,
    ).drop(columns="name")

    fires_final = fires_final.drop(columns="_merge")

    fires_final.to_csv(f"{FIRES_CENTROIDS_STATIONS_PATH.parent}\\TankerVision_Fires_Hourly_Weather_2025_v2.csv")
    return fires_final
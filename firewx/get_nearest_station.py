from pathlib import Path
import geopandas as gpd

def nearest_stations(centroid_geom, stations_gdf, station_col="STTN_NM"):
    """
    find the 3 nearest stations to the centroid and save station name and distance to column in provided dataframe.
    """
    distances = stations_gdf.distance(centroid_geom)

    nearest_idx = distances.nsmallest(3).index

    return tuple(
        (
            stations_gdf.loc[i, station_col],
            float(distances.loc[i])
        )
        for i in nearest_idx
    )

def create_nearest_stations_csv(WEATHER_STATIONS_PATH = Path(r"C:\Users\jbuskas\OneDrive - NRCan RNCan\TankerVision\BCGW_02001F02_1792435697276_10220\PROT_WEATHER_STATIONS_SP\PRT_WTHRST_point.shp"),
                                RESULTS_PATH = Path(r"C:\Users\jbuskas\OneDrive - NRCan RNCan\TankerVision\Nearest Weather Station Analysis"),
                                FIRES_PATH = Path(r"C:\Users\jbuskas\OneDrive - NRCan RNCan\TankerVision\2025 Fire Locations\TV_poly_2025_EPSG3005_LidarAvail.shp")):
    """
    Find the 3 closest weather stations to each fire polygon (based on the centroid) and save the results as a csv in RESULTS_PATH.
    Returns the updated fire dataframe with the centroids and 3 closest weather stations added.
    """
    fires = gpd.read_file(FIRES_PATH)
    weather_stations = gpd.read_file(WEATHER_STATIONS_PATH)
    # calculate centroids
    fires['centroid'] = fires.centroid
    
    # drop stations which have no observations in fwi_2025
    problematic_sttn = [
        'BAKER CREEK',
        'QD8 SIWASH PB',
        'QD1 GALLAGHERS',
        'QD 2 DEER PARK',
        'CAMPBELL RIVER AP (EC)'
    ]
    weather_stations = weather_stations[~weather_stations['STTN_NM'].isin(problematic_sttn)]
    
    # find nearest stations to each fire and add column to fires dataframe
    fires["nearest_stations"] = fires['centroid'].geometry.apply(
        lambda geom: nearest_stations(
            geom,
            weather_stations,
            station_col="STTN_NM" # STTN_NM for BCWS
        )
    )
    
    fires.to_csv(f"{RESULTS_PATH}\\_centroids_nearestStations.csv")
    return fires
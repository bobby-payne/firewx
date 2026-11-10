import logging
from os import path

from .data_io import (
    open_location_parameters,
    get_wx_data,
    save_fwi_data,
)
from .data_proc import (
    convert_dataset_UTC_to_LST,
    select_time_range,
    select_nearest_gridpoint,
    convert_dataset_longitude_range,
    apply_transform,
    apply_unit_conversion,
    select_noon_values,
    convert_precipitation_hourly_to_daily,
)
from .utils import (
    get_timezone_UTC_offset,
)


class DatasetModel:
    
    def __init__(self, config):

        # Each config parameter becomes an attribute of the instance
        self.config = config
        for key, value in self.config.data.items():
            setattr(self, key, value)

        # Read in location parameters
        self.location_parameters = open_location_parameters(self.config)
        self.number_of_locations = len(self.location_parameters.groupby(["latitude", "longitude"]))

        # Open weather data, do pre-processing, load into memory, then set as attributes of the instance
        self.wx_data_dict = get_wx_data(self.config)


    def iter_location(self):
        """
        Iterate over each unique location (latitude, longitude) in the location parameters.
        For each location, set the corresponding attributes and yield the instance.
        """

        # Iterate over unique latitude and longitude combinations in the location parameters
        for (latitude, longitude), startup_data in (
                    self.location_parameters.groupby(["latitude", "longitude"])
                ):
            self.latitude = latitude
            self.longitude = longitude
            self.startup_data = startup_data
            self.utc_offset = get_timezone_UTC_offset(latitude, longitude)
            yield self


    def process_data_for_location(self):
        """
        Process weather data for the current location (latitude, longitude) set in the instance.
        (see iterate_location for how the location is set.)
        Applies preprocessing, transformations, and unit conversions to each dataset.
        Sets the processed datasets as attributes of the instance.
        """

        # check if latitude and longitude are set
        if not hasattr(self, "latitude") or not hasattr(self, "longitude"):
            raise ValueError("Latitude and longitude must be set before processing data for the location.")

        for var_name, dataset in self.wx_data_dict.items():

            dataset = dataset.copy()

            # Preprocessing for longitude and grid point selection
            if self.longitude_convention == [0, 360]:
                dataset = convert_dataset_longitude_range(self.config, dataset, to_centered=True)
            dataset = select_nearest_gridpoint(self.config, dataset, self.latitude, self.longitude)

            # Preprocessing for time conversion and time range selection
            # NOTE: Converting to LST assumes select_nearest_gridpoint has been applied
            if self.timezone == 'UTC':
                dataset = convert_dataset_UTC_to_LST(self.config, dataset, self.utc_offset)  
            dataset = select_time_range(self.config, dataset, self.crop["start_year"], self.crop["final_year"])

            # Apply transformations and unit conversions
            dataset = apply_transform(self.config, var_name, dataset)
            dataset = apply_unit_conversion(self.config, var_name, dataset)

            # Compute and store daily values if using CFFWIS 1987 daily system
            if self.cffwis_version == "FWI1987":
                if var_name == "precip_accumulation_1hr":
                    setattr(
                        self,
                        "precip_accumulation_24hr",
                        convert_precipitation_hourly_to_daily(self.config, dataset)
                        )
                else:
                    setattr(
                        self,
                        var_name + "_noon",
                        select_noon_values(self.config, dataset)
                    )

            # Daily maximum temperature is required for fire season computation
            if var_name == "air_temperature_2m":
                setattr(
                    self,
                    "air_temperature_2m_daymax",
                    dataset.resample(time="1D").max()
                )

            # Dataset is loaded into memory at this point
            if self.lazy_load_data:
                logging.info(f"Loading {var_name} data into memory... this may take a while!")
                dataset = dataset.compute()

            # Set as attribute
            setattr(self, var_name, dataset)


    def save_fwi_data(self, output_directory, file_name="fwi_data.csv", **kwargs):
        """Save FWI data to a CSV file by calling `save_fwi_data` from the data_io module.

        Args:
            fwi_data (pd.DataFrame): DataFrame containing FWI data.
            output_directory (str): Directory to save the output CSV file.
            file_name (str, optional): Name of the output CSV file. Default is "fwi_data.csv".
            **kwargs: Additional keyword arguments forwarded to
                pd.DataFrame.to_csv().
        """
        if hasattr(self, "fire_weather_indices"):
            save_fwi_data(self.fire_weather_indices, output_directory, file_name=file_name, **kwargs)
        else:
            raise RuntimeError("No fire weather index data available to save to file.")
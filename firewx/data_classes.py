import logging

from .data_io import get_wx_data
from .data_proc import (
    convert_dataset_to_LST,
    select_time_range,
    select_nearest_gridpoint,
    convert_longitude_range,
    convert_dataset_longitude_range,
    apply_transform,
    apply_unit_conversion
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

        # Get UTC offset for the user-specified location and set as attribute
        if self.longitude_convention == [0, 360]:
            self.longitude = convert_longitude_range(self.longitude, to_centered=True)
        self.utc_offset = get_timezone_UTC_offset(self.latitude, self.longitude)

        # Open weather data, do pre-processing, load into memory, then set as attributes of the instance
        wx_data_dict = get_wx_data(self.config)
        for var_name, dataset in wx_data_dict.items():

            # Preprocessing for longitude and grid point selection
            if self.longitude_convention == [0, 360]:
                dataset = convert_dataset_longitude_range(self.config, dataset, to_centered=True)
            dataset = select_nearest_gridpoint(self.config, dataset)

            # Preprocessing for time conversion and time range selection
            # NOTE: Converting to LST assumes select_nearest_gridpoint has been applied
            if self.timezone == 'UTC':
                dataset = convert_dataset_to_LST(self.config, dataset)  
            dataset = select_time_range(self.config, dataset)

            # Apply transformations and unit conversions
            dataset = apply_transform(self.config, var_name, dataset)
            dataset = apply_unit_conversion(self.config, var_name, dataset)

            # Dataset is loaded into memory at this point
            if self.lazy_load_data:
                logging.info(f"Loading {var_name} data into memory... this may take a while!")
                dataset = dataset.compute()

            # Set as attribute
            setattr(self, var_name, dataset)

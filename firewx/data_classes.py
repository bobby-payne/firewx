
from .data_io import get_wx_data
from .data_proc import (
    select_time_range,
    select_nearest_gridpoint,
    convert_longitude_range,
    convert_dataset_longitude_range,
    apply_transform,
    apply_unit_conversion
)
from .fire_season import (
    get_timezone_UTC_offset,
)


class DatasetModel:
    
    def __init__(self, config):

        # Each config parameter becomes an attribute of the instance
        self.config = config
        for key, value in self.config.data.items():
            setattr(self, key, value)

        # Get UTC offset for the user-specified location and set as attribute
        if self.config.longitude_convention == [0, 360]:
            self.config.longitude = convert_longitude_range(self.config.longitude, to_centered=True)
        self.utc_offset = get_timezone_UTC_offset(self.config.latitude, self.config.longitude)

        # Open weather data, do pre-processing, then set as attributes of the instance
        wx_data_dict = get_wx_data(self.config)
        for var_name, dataset in wx_data_dict.items():

            # Preprocessing/unit standardization
            dataset = select_time_range(self.config, dataset)
            if self.config.longitude_convention == [0, 360]:
                dataset = convert_dataset_longitude_range(self.config, dataset, to_centered=True)
            dataset = select_nearest_gridpoint(self.config, dataset)
            dataset = apply_transform(self.config, var_name, dataset)
            dataset = apply_unit_conversion(self.config, var_name, dataset)

            # Set as attribute
            setattr(self, var_name, dataset)


class DatasetStation:

    def __init__(self, config):

        raise NotImplementedError("DatasetStation class is not implemented yet.")

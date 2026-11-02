
from torch import mm

from .data_io import get_wx_data
from .data_proc import (
    select_nearest_gridpoint,
    convert_longitude_range,
    apply_transform,
    apply_unit_conversion
)


class DatasetModel:
    
    def __init__(self, config):

        # Each config parameter for the dataset becomes an attribute of the instance
        self.config = config
        for key, value in self.config.data.items():
            setattr(self, key, value)

        # Open data, do processing, then set as attributes of the instance
        wx_data_dict = get_wx_data(self.config)
        for var_name, dataset in wx_data_dict.items():

            # Select the nearest grid point to the specified latitude and longitude
            dataset = select_nearest_gridpoint(self.config, dataset, self.config.latitude, self.config.longitude)

            # cffdrs_ng requires longitude to be in the [-180,180) range
            if self.config.longitude_convention == [0, 360]:
                dataset = convert_longitude_range(self.config, dataset, to_centered=True)

            # Apply any user-specified transformations to the dataset
            dataset = apply_transform(self.config, var_name, dataset)

            # Apply unit conversions
            dataset = apply_unit_conversion(self.config, var_name, dataset)

            # Set as attribute
            setattr(self, var_name, dataset)


class DatasetStation:

    def __init__(self, config):

        raise NotImplementedError("DatasetStation class is not implemented yet.")

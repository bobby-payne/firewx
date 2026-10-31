
def apply_transform(config, variable, dataset):
    """
    Transform the specified variable in the dataset by applying a mathematical operation.
    Transforms are specified in the config file and evaluated using Python's
    in-built eval function.

    Parameters
    ----------
    config : dict
        The configuration dictionary containing transformation specifications.
    variable : str
        The name of the variable to transform.
    dataset : xarray.Dataset
        The dataset containing the weather variable to be transformed.

    Returns
    -------
    xarray.Dataset
        The transformed dataset with the transformations applied.
    """
    # read in the user-specified transform
    transform = config[variable]["transform"]
    variable_name = config[variable]["variable_name"]

    # Apply each transformation to the corresponding variable
    if transform is not None:
        print(f"Applying transformation to {variable}: {transform}")
        transform_func = eval("lambda x:" + transform)  # TODO: replace eval with a secure alternative (e.g numexpr)
        dataset[variable_name] = transform_func(dataset[variable_name])

    return dataset


def apply_unit_conversion(config, variable, dataset):

    if variable == "wind_speed_10m":
        variable_name = config.wind_speed_10m["variable_name"]
        dataset[variable_name] = convert_windspeed_units_to_kmh(
            dataset[variable_name], config.wind_speed_10m["units"]
            )
    elif variable == "air_temperature_2m":
        variable_name = config.air_temperature_2m["variable_name"]
        dataset[variable_name] = convert_temperature_units_to_celsius(
            dataset[variable_name], config.air_temperature_2m["units"]
            )
    elif variable == "relative_humidity_2m":
        variable_name = config.relative_humidity_2m["variable_name"]
        dataset[variable_name] = convert_relative_humidity_units_to_percentage(
            dataset[variable_name], config.relative_humidity_2m["units"]
            )
    elif variable == "precip_accumulation_1hr":
        variable_name = config.precip_accumulation_1hr["variable_name"]
        dataset[variable_name] = convert_precipitation_units_to_mm(
            dataset[variable_name], config.precip_accumulation_1hr["units"]
            )
    else:
        raise ValueError(f"Unsupported variable name: {variable}.")

    return dataset


def convert_windspeed_units_to_kmh(wind_speed, from_unit):

    if from_unit in ["km/h", "kph"]:
        return wind_speed
    elif from_unit in ["m/s"]:
        return wind_speed * 3.6
    elif from_unit in ["mph", "mi/h"]:
        return wind_speed * 1.60934
    elif from_unit in ["kts", "knots"]:
        return wind_speed * 1.852
    else:
        raise ValueError(
            f"Unsupported wind speed unit: {from_unit}. Choose from ['km/h', 'm/s', 'mph', 'kts']."
            )


def convert_temperature_units_to_celsius(temperature, from_unit):

    if from_unit in ["C", "degC"]:
        return temperature
    elif from_unit in ["F", "degF"]:
        return (temperature - 32) * 5.0/9.0
    elif from_unit in ["K"]:
        return temperature - 273.15
    else:
        raise ValueError(
            f"Unsupported temperature unit: {from_unit}. Choose from ['C', 'F', 'K']."
            )


def convert_relative_humidity_units_to_percentage(relative_humidity, from_unit=None):

    if from_unit in ["%", "percent"]:
        return relative_humidity
    elif from_unit in [None, "fraction", "frac", "decimal"]:
        return relative_humidity * 100
    else:
        raise ValueError(
            f"Unsupported relative humidity unit: {from_unit}. Choose from ['%', null]."
        )


def convert_precipitation_units_to_mm(precipitation, from_unit):

    if from_unit in ["mm", "millimeter", "millimeters"]:
        return precipitation
    elif from_unit in ["cm", "centimeter", "centimeters"]:
        return precipitation * 10
    elif from_unit in ["m", "meter", "meters"]:
        return precipitation * 1000
    elif from_unit in ["in", "inch", "inches"]:
        return precipitation * 25.4
    else:
        raise ValueError(
            f"Unsupported precipitation unit: {from_unit}. Choose from ['mm', 'cm', 'm', 'in']."
        )
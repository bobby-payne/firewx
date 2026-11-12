# This package is currently under active development!

This is an unofficial Python package and is not endorsed by NRCAN/RNCAN.

---

## Installation

Create and then activate a virtual environment with Python. For example:

```
python -m venv .venv
source .venv/bin/activate
```

Clone this repository:

```
git clone https://github.com/bobby-payne/firewx
```

Install the package in editable mode, along with its dependencies:

```
pip install -e firewx
```

That's it!

---

## How to Use

1. You will need to create an `.csv` file containing the years and locations (and optionally, fire season start dates and moisture code startups) for which you want to compute fire weather indices. See `example.csv` (coming soon!) for what this should look like; at minimum, there should be a `latitude`, `longitude`, and `year`, where longitude values use the West-Negative / East-Positive system, (i.e., in [-180, 180]).

2. In `conf/models/` create a `[example_model].yaml`, replacing [example_model] with the name of the gridded data product. This config contains all the necessary information and configuration specific to the dataset. See `example.yaml` (coming soon!) for how this should look like.

3. Edit `conf/config.yaml` with the necessary information. This config includes general information that is not specific to the gridded data product, in addition to pointing to the model that you wish to use.

4. You should now be ready to run `firewx`. If the package is installed correctly in your Python environment, you should be able to run the package simply by entering the package name.

```
firewx
```

---

## Description of Modules

`cli.py` - Command line interface; where main() is found.

`configuration.py` - Class and methods related to opening and reading the config files.

`data_classes.py` - The gridded dataset is stored as an instance of the `DatasetModel` class; this is where the class is defined.

`data_io.py` - Methods related to opening, reading, and writing `netcdf` and `csv` files.

`data_proc.py` - Data processing; methods related to processing the gridded weather data themselves, such as unit conversion and user-defined transforms.

`fire_season.py` - Methods for computing the fire season mask.

`fwi_daily.py` - Methods for computing daily FWIs according to the FWI1987 system.

`utils.py` - Contains helper functions that don't fit elsewhere.
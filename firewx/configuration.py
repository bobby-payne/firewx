import yaml
from pathlib import Path


class Config:
    def __init__(self, relative_path="conf/config.yaml"):
        # Resolve path relativto this file, not the working directory
        base_dir = Path(__file__).resolve().parent
        config_path = (base_dir / relative_path).resolve()

        with open(config_path, "r") as f:
            self.data = yaml.safe_load(f)

    def __getitem__(self, key):
        return self.data[key]

    def __setattr__(self, key, value):
        if key == "data":
            super().__setattr__(key, value)
        else:
            self.data[key] = value

    def __getattr__(self, key):
        if key == "data":
            return super().__getattribute__(key)
        return self.data.get(key)


# Singleton pattern to avoid rereading the config every time
_general_config_instance = None
_data_config_instance = None
_config_instance = None


def get_general_config(relative_path="conf/config.yaml"):
    global _general_config_instance
    if _general_config_instance is None:
        _general_config_instance = Config(relative_path)
    return _general_config_instance

def get_data_config(general_config):
    global _data_config_instance
    if _data_config_instance is None:
        relative_path=f"conf/models/{general_config.dataset_name}.yaml"
        _data_config_instance = Config(relative_path)
    return _data_config_instance

def get_config(relative_path="conf/config.yaml"):
    global _config_instance
    if _config_instance is None:
        # combine and return the general and data-specific configurations into a single config file
        general_config = get_general_config(relative_path)
        data_config = get_data_config(general_config)
        combined_config = Config()
        combined_config.data = {**general_config.data, **data_config.data}
        _config_instance = combined_config
    return _config_instance

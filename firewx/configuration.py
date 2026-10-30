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


# Singleton pattern to avoid rereading the config every time
_general_config_instance = None
_data_config_instance = None


def get_general_config(relative_path="conf/config.yaml"):
    global _general_config_instance
    if _general_config_instance is None:
        _general_config_instance = Config(relative_path)
    return _general_config_instance

def get_data_config(general_config):
    global _data_config_instance
    if _data_config_instance is None:
        dataset_name = general_config["dataset_name"]
        _data_config_instance = Config(relative_path=f"conf/models/{dataset_name}.yaml")
    return _data_config_instance

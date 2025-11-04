from funkybob import RandomNameGenerator
import yaml
from datetime import datetime
import mergedeep


def generate_run_name() -> str:
    """Generate a timestamped random run name once on rank 0 and share it across ranks."""
    return next(iter(RandomNameGenerator()))


def read_yaml_config(path) -> dict:
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def read_configs(default_config_path, custom_config_path=None):
    config = read_yaml_config(default_config_path)
    if custom_config_path is not None:
        update_config = read_yaml_config(custom_config_path)
        mergedeep.merge(config, update_config, strategy=mergedeep.Strategy.REPLACE)
    config['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return config


def save_config(config, path):
    with open(path, 'w') as f:
        yaml.safe_dump(config, f)

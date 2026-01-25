from funkybob import RandomNameGenerator
import yaml


def generate_run_name() -> str:
    """Generate a timestamped random run name once on rank 0 and share it across ranks."""
    return next(iter(RandomNameGenerator()))


def read_yaml_config(path) -> dict:
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, path):
    with open(path, 'w') as f:
        yaml.safe_dump(config, f)

def replace_at(s, i, char):
    return s[:i] + char + s[i + 1:]

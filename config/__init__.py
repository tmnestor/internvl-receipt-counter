"""
Configuration module for InternVL2 receipt counter.
"""
from pathlib import Path

import yaml


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
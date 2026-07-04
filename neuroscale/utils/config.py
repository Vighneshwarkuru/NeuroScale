"""Configuration loading and management."""

import yaml
from pathlib import Path
from typing import Any


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary containing configuration parameters.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_config(dataset_name: str) -> dict[str, Any]:
    """Load config by dataset name.

    Args:
        dataset_name: One of 'cifar10', 'cifar100', 'imagenet'.

    Returns:
        Configuration dictionary.
    """
    config_dir = Path(__file__).parent.parent.parent / "configs"
    config_path = config_dir / f"{dataset_name}.yaml"
    return load_config(str(config_path))

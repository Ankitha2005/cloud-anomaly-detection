"""
Configuration loader utility for the Cloud Node Anomaly Detection project.
"""

import os
import yaml
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    # Navigate up from src/utils/config_loader.py to project root
    return current.parent.parent.parent


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
        
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        config_path = get_project_root() / "configs" / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_data_path(relative_path: str) -> Path:
    """
    Get absolute path for a data file.
    
    Args:
        relative_path: Path relative to project root
        
    Returns:
        Absolute Path object
    """
    return get_project_root() / relative_path


def ensure_directories():
    """Create all necessary directories if they don't exist."""
    root = get_project_root()
    directories = [
        "data/raw",
        "data/processed",
        "data/sequences",
        "outputs/models",
        "outputs/results",
        "outputs/figures",
    ]
    
    for dir_path in directories:
        (root / dir_path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test the config loader
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"Project name: {config['project']['name']}")
    print(f"Window size: {config['data']['window_size_seconds']} seconds")


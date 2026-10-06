"""MEWM-Agent package root."""

__version__ = "1.0.0"

from .config import MEWMConfig, load_config

__all__ = ["MEWMConfig", "load_config", "__version__"]

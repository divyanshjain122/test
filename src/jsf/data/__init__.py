"""Data loading and preprocessing module."""

from .base import DataLoader, PriceData, DataLoadError
from .loaders import (
    CSVLoader,
    ParquetLoader,
    InMemoryLoader,
    SyntheticDataLoader,
    load_data,
)
from .universe import Universe, UniverseFilter, create_universe, UNIVERSE_CONSTITUENTS
from .preprocessing import (
    handle_missing_data,
    resample_data,
    calculate_returns,
    normalize_prices,
    calculate_rolling_stats,
    align_data,
    remove_outliers,
    winsorize,
    forward_fill_gaps,
    ensure_business_days,
    calculate_volatility,
)

__all__ = [
    # Base classes
    "DataLoader",
    "PriceData",
    "DataLoadError",
    # Loaders
    "CSVLoader",
    "ParquetLoader",
    "InMemoryLoader",
    "SyntheticDataLoader",
    "load_data",
    # Universe
    "Universe",
    "UniverseFilter",
    "create_universe",
    "UNIVERSE_CONSTITUENTS",
    # Preprocessing
    "handle_missing_data",
    "resample_data",
    "calculate_returns",
    "normalize_prices",
    "calculate_rolling_stats",
    "align_data",
    "remove_outliers",
    "winsorize",
    "forward_fill_gaps",
    "ensure_business_days",
    "calculate_volatility",
]

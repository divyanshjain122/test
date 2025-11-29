"""Test configuration and fixtures."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Generator


@pytest.fixture
def sample_dates() -> pd.DatetimeIndex:
    """Generate sample trading dates."""
    return pd.date_range(start="2020-01-01", end="2023-12-31", freq="B")


@pytest.fixture
def sample_prices(sample_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate sample price data for testing."""
    np.random.seed(42)
    n_assets = 5
    n_days = len(sample_dates)
    
    # Generate realistic price paths
    returns = np.random.normal(0.0005, 0.015, size=(n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    return pd.DataFrame(
        prices,
        index=sample_dates,
        columns=[f"ASSET_{i}" for i in range(n_assets)]
    )


@pytest.fixture
def sample_returns(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """Generate sample returns from prices."""
    return sample_prices.pct_change().fillna(0.0)


@pytest.fixture
def temp_data_dir(tmp_path: str) -> Generator[str, None, None]:
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    yield str(data_dir)


@pytest.fixture
def random_seed() -> int:
    """Provide a fixed random seed for reproducibility."""
    return 42

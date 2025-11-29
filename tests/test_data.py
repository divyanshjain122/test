"""Unit tests for the data module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from jsf.data import (
    PriceData,
    DataLoadError,
    CSVLoader,
    ParquetLoader,
    InMemoryLoader,
    SyntheticDataLoader,
    load_data,
    Universe,
    UniverseFilter,
    create_universe,
    handle_missing_data,
    resample_data,
    calculate_returns,
    normalize_prices,
    remove_outliers,
    calculate_volatility,
)
from jsf.config import UniverseType


# ============================================================================
# Test Synthetic Data Loader
# ============================================================================

class TestSyntheticDataLoader:
    """Test synthetic data generation."""
    
    def test_basic_generation(self) -> None:
        """Test basic synthetic data generation."""
        loader = SyntheticDataLoader(
            symbols=["AAPL", "GOOGL"],
            start_date="2020-01-01",
            end_date="2020-01-31",
            seed=42,
        )
        
        df = loader.load()
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.MultiIndex)
        assert len(df.index.get_level_values(1).unique()) == 2
        assert "close" in df.columns
        assert "volume" in df.columns
    
    def test_reproducibility(self) -> None:
        """Test that same seed produces same data."""
        loader1 = SyntheticDataLoader(
            symbols=["AAPL"],
            start_date="2020-01-01",
            end_date="2020-01-10",
            seed=42,
        )
        
        loader2 = SyntheticDataLoader(
            symbols=["AAPL"],
            start_date="2020-01-01",
            end_date="2020-01-10",
            seed=42,
        )
        
        df1 = loader1.load()
        df2 = loader2.load()
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_price_ranges(self) -> None:
        """Test that generated prices are reasonable."""
        loader = SyntheticDataLoader(
            symbols=["TEST"],
            start_date="2020-01-01",
            end_date="2020-12-31",
            initial_price=100.0,
            seed=42,
        )
        
        df = loader.load()
        
        # Check OHLC relationships
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["close"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["low"] <= df["open"]).all()
        
        # Check volumes are positive
        assert (df["volume"] > 0).all()


# ============================================================================
# Test PriceData Container
# ============================================================================

class TestPriceData:
    """Test PriceData container class."""
    
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample price data."""
        dates = pd.bdate_range("2020-01-01", periods=10)
        symbols = ["AAPL", "GOOGL"]
        
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        
        np.random.seed(42)
        data = pd.DataFrame({
            "open": np.random.uniform(90, 110, len(index)),
            "high": np.random.uniform(95, 115, len(index)),
            "low": np.random.uniform(85, 105, len(index)),
            "close": np.random.uniform(90, 110, len(index)),
            "volume": np.random.randint(1000000, 10000000, len(index)),
        }, index=index)
        
        return data
    
    def test_initialization(self, sample_data: pd.DataFrame) -> None:
        """Test PriceData initialization."""
        price_data = PriceData(sample_data)
        
        assert price_data.data is not None
        assert len(price_data.symbols) == 2
        assert "AAPL" in price_data.symbols
        assert "GOOGL" in price_data.symbols
    
    def test_get_symbol_data(self, sample_data: pd.DataFrame) -> None:
        """Test extracting single symbol data."""
        price_data = PriceData(sample_data)
        
        aapl_data = price_data.get_symbol_data("AAPL")
        
        assert isinstance(aapl_data, pd.DataFrame)
        assert "close" in aapl_data.columns
        assert len(aapl_data) == 10  # 10 business days
    
    def test_get_close_prices(self, sample_data: pd.DataFrame) -> None:
        """Test getting close prices."""
        price_data = PriceData(sample_data)
        
        close = price_data.get_close_prices()
        
        assert isinstance(close, pd.DataFrame)
        assert close.shape[1] == 2  # 2 symbols
        assert list(close.columns) == ["AAPL", "GOOGL"]
    
    def test_get_returns(self, sample_data: pd.DataFrame) -> None:
        """Test returns calculation."""
        price_data = PriceData(sample_data)
        
        returns = price_data.get_returns(periods=1)
        
        assert isinstance(returns, pd.DataFrame)
        assert returns.shape[1] == 2
        # First row should be NaN
        assert returns.iloc[0].isna().all()
    
    def test_summary(self, sample_data: pd.DataFrame) -> None:
        """Test summary statistics."""
        price_data = PriceData(sample_data)
        
        summary = price_data.summary()
        
        assert summary["n_symbols"] == 2
        assert summary["n_days"] == 10
        assert "AAPL" in summary["symbols"]
        assert "memory_usage_mb" in summary


# ============================================================================
# Test Universe
# ============================================================================

class TestUniverse:
    """Test Universe class."""
    
    def test_from_symbols(self) -> None:
        """Test creating universe from symbol list."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        universe = Universe.from_symbols(symbols, name="tech")
        
        assert universe.name == "tech"
        assert len(universe) == 3
        assert "AAPL" in universe
        assert "TSLA" not in universe
    
    def test_from_predefined(self) -> None:
        """Test creating universe from predefined type."""
        universe = Universe.from_predefined(UniverseType.DOW_30)
        
        assert universe.name == "DOW_30"
        assert len(universe) == 30
    
    def test_create_universe_convenience(self) -> None:
        """Test convenience function."""
        # From string
        universe1 = create_universe("SP500")
        assert len(universe1) > 0
        
        # From list
        universe2 = create_universe(["AAPL", "GOOGL"])
        assert len(universe2) == 2
        
        # From enum
        universe3 = create_universe(UniverseType.DOW_30)
        assert len(universe3) == 30
    
    def test_intersection(self) -> None:
        """Test universe intersection."""
        u1 = Universe.from_symbols(["AAPL", "GOOGL", "MSFT"])
        u2 = Universe.from_symbols(["GOOGL", "MSFT", "TSLA"])
        
        u_inter = u1.intersection(u2)
        
        assert len(u_inter) == 2
        assert "GOOGL" in u_inter
        assert "MSFT" in u_inter
        assert "AAPL" not in u_inter
        assert "TSLA" not in u_inter
    
    def test_union(self) -> None:
        """Test universe union."""
        u1 = Universe.from_symbols(["AAPL", "GOOGL"])
        u2 = Universe.from_symbols(["MSFT", "TSLA"])
        
        u_union = u1.union(u2)
        
        assert len(u_union) == 4
        assert all(s in u_union for s in ["AAPL", "GOOGL", "MSFT", "TSLA"])
    
    def test_sample(self) -> None:
        """Test sampling from universe."""
        universe = Universe.from_symbols(["A", "B", "C", "D", "E"])
        
        sampled = universe.sample(3, seed=42)
        
        assert len(sampled) == 3
        assert all(s in universe for s in sampled.symbols)


# ============================================================================
# Test Preprocessing
# ============================================================================

class TestPreprocessing:
    """Test preprocessing functions."""
    
    @pytest.fixture
    def sample_prices(self) -> pd.DataFrame:
        """Create sample price DataFrame."""
        dates = pd.bdate_range("2020-01-01", periods=20)
        return pd.DataFrame({
            "AAPL": np.random.uniform(90, 110, 20),
            "GOOGL": np.random.uniform(1000, 1200, 20),
        }, index=dates)
    
    def test_handle_missing_data_ffill(self, sample_prices: pd.DataFrame) -> None:
        """Test forward fill missing data."""
        # Introduce missing values
        data = sample_prices.copy()
        data.iloc[5, 0] = np.nan
        data.iloc[10, 1] = np.nan
        
        filled = handle_missing_data(data, method="ffill")
        
        assert not filled.isna().any().any()
        assert filled.iloc[5, 0] == data.iloc[4, 0]
    
    def test_handle_missing_data_drop(self, sample_prices: pd.DataFrame) -> None:
        """Test dropping missing data."""
        data = sample_prices.copy()
        data.iloc[5, 0] = np.nan
        
        cleaned = handle_missing_data(data, method="drop")
        
        assert not cleaned.isna().any().any()
        assert len(cleaned) == len(data) - 1
    
    def test_calculate_returns(self, sample_prices: pd.DataFrame) -> None:
        """Test returns calculation."""
        returns = calculate_returns(sample_prices, method="simple", periods=1)
        
        assert returns.shape == sample_prices.shape
        assert returns.iloc[0].isna().all()  # First row is NaN
        
        # Check returns are calculated (non-NaN after first row)
        assert not returns.iloc[1:].isna().all().all()
    
    def test_normalize_prices(self, sample_prices: pd.DataFrame) -> None:
        """Test price normalization."""
        normalized = normalize_prices(sample_prices, base_value=100.0)
        
        # First row should be 100.0 for all symbols
        assert np.allclose(normalized.iloc[0], 100.0)
    
    def test_remove_outliers_zscore(self, sample_prices: pd.DataFrame) -> None:
        """Test outlier removal."""
        data = sample_prices.copy()
        # Add an extreme outlier
        data.iloc[10, 0] = 10000.0
        
        cleaned = remove_outliers(data, method="zscore", threshold=3.0)
        
        assert len(cleaned) < len(data)
        assert cleaned["AAPL"].max() < 10000.0
    
    def test_calculate_volatility(self, sample_prices: pd.DataFrame) -> None:
        """Test volatility calculation."""
        returns = calculate_returns(sample_prices)
        vol = calculate_volatility(returns, window=5, annualization_factor=252)
        
        assert vol.shape == returns.shape
        # First few rows will be NaN due to rolling window
        assert vol.iloc[:4].isna().all().all()
        # Later values should be positive
        assert (vol.iloc[5:] > 0).all().all()


# ============================================================================
# Test Load Data Convenience Function
# ============================================================================

class TestLoadData:
    """Test load_data convenience function."""
    
    def test_load_synthetic(self) -> None:
        """Test loading synthetic data."""
        price_data = load_data(
            source="synthetic",
            symbols=["AAPL", "GOOGL"],
            start_date="2020-01-01",
            end_date="2020-01-31",
            seed=42,
        )
        
        assert isinstance(price_data, PriceData)
        assert len(price_data.symbols) == 2
        assert price_data.start_date == pd.Timestamp("2020-01-01")
    
    def test_missing_parameters_synthetic(self) -> None:
        """Test that synthetic requires parameters."""
        with pytest.raises(ValueError, match="symbols required"):
            load_data(source="synthetic", start_date="2020-01-01", end_date="2020-12-31")
        
        with pytest.raises(ValueError, match="start_date and end_date required"):
            load_data(source="synthetic", symbols=["AAPL"])


# ============================================================================
# Test CSV Loader
# ============================================================================

class TestCSVLoader:
    """Test CSV data loader."""
    
    def test_load_from_dataframe(self, tmp_path: Path) -> None:
        """Test loading CSV from saved DataFrame."""
        # Create test data
        dates = pd.bdate_range("2020-01-01", periods=5)
        symbols = ["AAPL", "GOOGL"]
        
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        data = pd.DataFrame({
            "open": np.random.uniform(90, 110, len(index)),
            "high": np.random.uniform(95, 115, len(index)),
            "low": np.random.uniform(85, 105, len(index)),
            "close": np.random.uniform(90, 110, len(index)),
            "volume": np.random.randint(1000000, 10000000, len(index)),
        }, index=index)
        
        # Save to CSV
        csv_path = tmp_path / "test_data.csv"
        data.reset_index().to_csv(csv_path, index=False)
        
        # Load using CSV loader
        loader = CSVLoader(file_path=csv_path, symbols=["AAPL", "GOOGL"])
        loaded_data = loader.load()
        
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == 10  # 5 dates × 2 symbols


# ============================================================================
# Test In-Memory Loader
# ============================================================================

class TestInMemoryLoader:
    """Test in-memory data loader."""
    
    def test_load_dataframe(self) -> None:
        """Test loading from DataFrame."""
        dates = pd.bdate_range("2020-01-01", periods=10)
        symbols = ["AAPL", "GOOGL"]
        
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        data = pd.DataFrame({
            "open": np.random.uniform(90, 110, len(index)),
            "high": np.random.uniform(95, 115, len(index)),
            "low": np.random.uniform(85, 105, len(index)),
            "close": np.random.uniform(90, 110, len(index)),
            "volume": np.random.randint(1000000, 10000000, len(index)),
        }, index=index)
        
        loader = InMemoryLoader(data)
        loaded = loader.load()
        
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == 20  # 10 dates × 2 symbols

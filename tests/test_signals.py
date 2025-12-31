"""Unit tests for the signals module."""

import pytest
import pandas as pd
import numpy as np

from jsf.signals import (
    Signal,
    SignalType,
    SignalMetadata,
    SignalError,
    CompositeSignal,
    MomentumSignal,
    MovingAverageCrossSignal,
    RSISignal,
    BollingerBandsSignal,
    MACDSignal,
    VolumeWeightedSignal,
    MeanReversionSignal,
    TrendStrengthSignal,
    VolatilitySignal,
)
from jsf.data import load_data


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    return load_data(
        source="synthetic",
        symbols=["AAPL", "GOOGL", "MSFT"],
        start_date="2020-01-01",
        end_date="2023-12-31",
        initial_price=100.0,
        annual_return=0.10,
        annual_volatility=0.20,
        seed=42,
    )


# ============================================================================
# Test Technical Signals
# ============================================================================

class TestMomentumSignal:
    """Test momentum signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test basic momentum signal generation."""
        signal = MomentumSignal(lookback=20)
        result = signal(sample_price_data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 3  # 3 symbols
        assert not result.isnull().all().any()
    
    def test_normalization(self, sample_price_data):
        """Test signal normalization."""
        signal = MomentumSignal(lookback=20, normalize=True)
        result = signal(sample_price_data)
        
        # Should be roughly in [-1, 1] range
        assert result.abs().max().max() <= 1.5
    
    def test_caching(self, sample_price_data):
        """Test signal caching works."""
        signal = MomentumSignal(lookback=20)
        
        # Generate twice
        result1 = signal(sample_price_data)
        result2 = signal(sample_price_data)
        
        # Should be identical (from cache)
        pd.testing.assert_frame_equal(result1, result2)


class TestMovingAverageCrossSignal:
    """Test MA crossover signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test MA cross signal."""
        signal = MovingAverageCrossSignal(fast_period=20, slow_period=50)
        result = signal(sample_price_data)
        
        assert isinstance(result, pd.DataFrame)
        assert set(result.dropna().values.flatten()).issubset({-1.0, 1.0})
    
    def test_invalid_periods(self):
        """Test that fast must be less than slow."""
        with pytest.raises(ValueError, match="fast_period must be less than slow_period"):
            MovingAverageCrossSignal(fast_period=50, slow_period=20)
    
    def test_metadata(self):
        """Test metadata is correct."""
        signal = MovingAverageCrossSignal(fast_period=20, slow_period=50)
        metadata = signal.get_metadata()
        
        assert metadata.lookback_period == 50
        assert metadata.signal_type == SignalType.TECHNICAL
        assert not metadata.requires_volume


class TestRSISignal:
    """Test RSI signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test RSI signal generation."""
        signal = RSISignal(period=14)
        result = signal(sample_price_data)
        
        assert isinstance(result, pd.DataFrame)
        # Should only have -1, 0, or 1
        unique_vals = set(result.dropna().values.flatten())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})


class TestBollingerBandsSignal:
    """Test Bollinger Bands signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test Bollinger Bands signal."""
        signal = BollingerBandsSignal(period=20, num_std=2.0)
        result = signal(sample_price_data)
        
        assert isinstance(result, pd.DataFrame)
        # Should be in [-1, 1] range
        assert result.abs().max().max() <= 1.0


class TestMACDSignal:
    """Test MACD signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test MACD signal generation."""
        signal = MACDSignal(fast_period=12, slow_period=26, signal_period=9)
        result = signal(sample_price_data)
        
        assert isinstance(result, pd.DataFrame)
        assert set(result.dropna().values.flatten()).issubset({-1.0, 1.0})
    
    def test_metadata(self):
        """Test MACD metadata."""
        signal = MACDSignal(fast_period=12, slow_period=26, signal_period=9)
        metadata = signal.get_metadata()
        
        assert metadata.lookback_period == 35  # slow + signal


class TestVolumeWeightedSignal:
    """Test volume-weighted signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test volume-weighted signal."""
        signal = VolumeWeightedSignal(lookback=20)
        result = signal(sample_price_data)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_requires_volume(self):
        """Test metadata indicates volume requirement."""
        signal = VolumeWeightedSignal(lookback=20)
        metadata = signal.get_metadata()
        
        assert metadata.requires_volume


# ============================================================================
# Test Statistical Signals
# ============================================================================

class TestMeanReversionSignal:
    """Test mean reversion signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test mean reversion signal."""
        signal = MeanReversionSignal(lookback=20, entry_threshold=1.5)
        result = signal(sample_price_data)
        
        assert isinstance(result, pd.DataFrame)
        # Should be roughly in [-1, 1]
        assert result.abs().max().max() <= 1.5
    
    def test_signal_type(self):
        """Test signal type is statistical."""
        signal = MeanReversionSignal(lookback=20)
        assert signal.signal_type == SignalType.STATISTICAL


class TestTrendStrengthSignal:
    """Test trend strength signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test trend strength signal."""
        signal = TrendStrengthSignal(lookback=20, min_r_squared=0.7)
        result = signal(sample_price_data)
        
        assert isinstance(result, pd.DataFrame)


class TestVolatilitySignal:
    """Test volatility signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test volatility signal."""
        signal = VolatilitySignal(lookback=20, vol_lookback=60)
        result = signal(sample_price_data)
        
        assert isinstance(result, pd.DataFrame)
        # Should be in [-1, 1]
        assert result.abs().max().max() <= 1.5
    
    def test_invert_option(self, sample_price_data):
        """Test invert parameter."""
        signal_normal = VolatilitySignal(lookback=20, invert=False)
        signal_inverted = VolatilitySignal(lookback=20, invert=True)
        
        result_normal = signal_normal(sample_price_data)
        result_inverted = signal_inverted(sample_price_data)
        
        # Signals should be negatives (approximately)
        corr = np.corrcoef(
            result_normal.dropna().values.flatten()[:100],
            result_inverted.dropna().values.flatten()[:100]
        )[0, 1]
        assert corr < -0.9


# ============================================================================
# Test Composite Signal
# ============================================================================

class TestCompositeSignal:
    """Test composite signal."""
    
    def test_average_combination(self, sample_price_data):
        """Test averaging multiple signals."""
        signals = [
            MomentumSignal(lookback=20),
            MeanReversionSignal(lookback=20),
        ]
        
        composite = CompositeSignal(
            name="test_composite",
            signals=signals,
            method="average",
        )
        
        result = composite(sample_price_data)
        assert isinstance(result, pd.DataFrame)
    
    def test_weighted_average(self, sample_price_data):
        """Test weighted average combination."""
        signals = [
            MomentumSignal(lookback=20),
            MeanReversionSignal(lookback=20),
        ]
        
        composite = CompositeSignal(
            name="weighted_composite",
            signals=signals,
            method="weighted_average",
            weights=[0.7, 0.3],
        )
        
        result = composite(sample_price_data)
        assert isinstance(result, pd.DataFrame)
    
    def test_vote_combination(self, sample_price_data):
        """Test voting combination."""
        signals = [
            MomentumSignal(lookback=10),
            MomentumSignal(lookback=20),
            MomentumSignal(lookback=30),
        ]
        
        composite = CompositeSignal(
            name="vote_composite",
            signals=signals,
            method="vote",
        )
        
        result = composite(sample_price_data)
        # Vote should produce -1, 0, or 1
        unique_vals = set(result.dropna().values.flatten())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})
    
    def test_invalid_weights(self):
        """Test that invalid weights raise error."""
        signals = [MomentumSignal(), MeanReversionSignal()]
        
        with pytest.raises(ValueError, match="weights"):
            CompositeSignal(
                name="bad_composite",
                signals=signals,
                method="weighted_average",
                weights=[0.5],  # Wrong length
            )
    
    def test_metadata_aggregation(self):
        """Test metadata aggregates from components."""
        signals = [
            MomentumSignal(lookback=20),
            VolumeWeightedSignal(lookback=30),
        ]
        
        composite = CompositeSignal(
            name="test_composite",
            signals=signals,
            method="average",
        )
        
        metadata = composite.get_metadata()
        assert metadata.lookback_period == 30  # Max of components
        assert metadata.requires_volume  # From VolumeWeightedSignal


# ============================================================================
# Test Signal Base Class
# ============================================================================

class TestSignalBase:
    """Test signal base class functionality."""
    
    def test_cache_enable_disable(self, sample_price_data):
        """Test cache can be enabled/disabled."""
        signal = MomentumSignal(lookback=20)
        
        # Generate with cache
        result1 = signal(sample_price_data)
        
        # Disable cache
        signal.enable_cache(False)
        result2 = signal(sample_price_data)
        
        # Should still be equal but not from cache
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_clear_cache(self, sample_price_data):
        """Test cache clearing."""
        signal = MomentumSignal(lookback=20)
        
        # Generate and cache
        _ = signal(sample_price_data)
        assert len(signal._cache) > 0
        
        # Clear cache
        signal.clear_cache()
        assert len(signal._cache) == 0
    
    def test_validate_data_insufficient(self):
        """Test validation fails with insufficient data."""
        # Create data with only 10 periods
        short_data = load_data(
            source="synthetic",
            symbols=["AAPL"],
            start_date="2020-01-01",
            end_date="2020-01-14",  # Only ~10 business days
            seed=42,
        )
        
        signal = MomentumSignal(lookback=20)
        
        with pytest.raises(SignalError, match="Insufficient data"):
            signal(short_data)
    
    def test_repr(self):
        """Test string representation."""
        signal = MomentumSignal(lookback=20)
        repr_str = repr(signal)
        
        assert "MomentumSignal" in repr_str
        assert "momentum" in repr_str

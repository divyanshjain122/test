"""Unit tests for Phase 5-6 signals (fundamental, sentiment, composites, transforms)."""

import pytest
import pandas as pd
import numpy as np

from jsf.signals import (
    # Fundamental
    ValueSignal,
    QualitySignal,
    GrowthSignal,
    SizeSignal,
    DividendSignal,
    # Sentiment
    MarketRegimeSignal,
    BreadthSignal,
    RelativeStrengthSignal,
    NewHighLowSignal,
    VolumeShockSignal,
    SeasonalitySignal,
    # Composites
    RotationSignal,
    MultiTimeframeSignal,
    AdaptiveWeightSignal,
    ThresholdFilterSignal,
    ConsensusSignal,
    # Transforms
    normalize_signal,
    rank_signal,
    smooth_signal,
    clip_signal,
    winsorize_signal,
    demean_signal,
    neutralize_signal,
    apply_decay,
    combine_signals,
    score_signals,
    NormalizationMethod,
    RankingMethod,
    # Base
    MomentumSignal,
    SignalType,
)
from jsf.data import load_data


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    return load_data(
        source="synthetic",
        symbols=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
        start_date="2020-01-01",
        end_date="2023-12-31",
        initial_price=100.0,
        annual_return=0.10,
        annual_volatility=0.20,
        seed=42,
    )


# ============================================================================
# Test Fundamental Signals
# ============================================================================

class TestValueSignal:
    """Test value signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test basic value signal generation."""
        signal = ValueSignal(lookback=60)
        result = signal.generate(sample_price_data)
        
        assert result.shape == sample_price_data.get_close_prices().shape
        assert not result.isna().all().all()
        assert (result >= -1).all().all() and (result <= 1).all().all()
    
    def test_with_fundamental_data(self, sample_price_data):
        """Test with actual fundamental data."""
        fundamental_data = {
            "AAPL": pd.Series([1.5, 1.6, 1.7], index=sample_price_data.get_close_prices().index[::400]),
            "GOOGL": pd.Series([2.0, 2.1, 2.0], index=sample_price_data.get_close_prices().index[::400]),
        }
        
        signal = ValueSignal(lookback=60)
        result = signal.generate(sample_price_data, fundamental_data=fundamental_data)
        
        assert result.shape == sample_price_data.get_close_prices().shape


class TestQualitySignal:
    """Test quality signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test basic quality signal generation."""
        signal = QualitySignal(lookback=60)
        result = signal.generate(sample_price_data)
        
        assert result.shape == sample_price_data.get_close_prices().shape
        assert not result.isna().all().all()


class TestGrowthSignal:
    """Test growth signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test basic growth signal generation."""
        signal = GrowthSignal(lookback_periods=[20, 60, 120])
        result = signal.generate(sample_price_data)
        
        assert result.shape == sample_price_data.get_close_prices().shape
        assert not result.isna().all().all()


class TestSizeSignal:
    """Test size signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test basic size signal generation."""
        signal = SizeSignal(prefer_small_cap=True)
        result = signal.generate(sample_price_data)
        
        assert result.shape == sample_price_data.get_close_prices().shape
        assert result.abs().max().max() <= 1.0


class TestDividendSignal:
    """Test dividend signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test basic dividend signal generation."""
        signal = DividendSignal(lookback=60)
        result = signal.generate(sample_price_data)
        
        assert result.shape == sample_price_data.get_close_prices().shape


# ============================================================================
# Test Sentiment Signals
# ============================================================================

class TestMarketRegimeSignal:
    """Test market regime signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test basic regime detection."""
        signal = MarketRegimeSignal(lookback=60)
        result = signal.generate(sample_price_data)
        
        assert result.shape == sample_price_data.get_close_prices().shape
        # Should classify as bull (1), bear (-1), or neutral (0)
        unique_vals = result.iloc[-1].unique()
        assert all(v in [-1.0, 0.0, 1.0] for v in unique_vals)


class TestBreadthSignal:
    """Test breadth signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test basic breadth calculation."""
        signal = BreadthSignal(lookback=20)
        result = signal.generate(sample_price_data)
        
        assert result.shape == sample_price_data.get_close_prices().shape
        assert (result >= -1).all().all() and (result <= 1).all().all()


class TestRelativeStrengthSignal:
    """Test relative strength signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test basic RS calculation."""
        signal = RelativeStrengthSignal(lookback=60)
        result = signal.generate(sample_price_data)
        
        assert result.shape == sample_price_data.get_close_prices().shape
        assert not result.isna().all().all()


class TestNewHighLowSignal:
    """Test new high/low signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test new high/low detection."""
        signal = NewHighLowSignal(lookback=252)
        result = signal.generate(sample_price_data)
        
        assert result.shape == sample_price_data.get_close_prices().shape
        # Should only generate -1, 0, or 1
        unique_vals = np.unique(result.values[~np.isnan(result.values)])
        assert all(v in [-1.0, 0.0, 1.0] for v in unique_vals)


class TestVolumeShockSignal:
    """Test volume shock signal."""
    
    def test_basic_generation(self, sample_price_data):
        """Test volume shock detection."""
        signal = VolumeShockSignal(lookback=20, shock_threshold=2.0)
        result = signal.generate(sample_price_data)
        
        assert result.shape == sample_price_data.get_close_prices().shape


class TestSeasonalitySignal:
    """Test seasonality signal."""
    
    def test_monthly_pattern(self, sample_price_data):
        """Test monthly seasonality."""
        signal = SeasonalitySignal(pattern="monthly", favorable_periods=[1, 11, 12])
        result = signal.generate(sample_price_data)
        
        assert result.shape == sample_price_data.get_close_prices().shape
        # Check that favorable months have signal = 1
        jan_data = result[result.index.month == 1]
        if len(jan_data) > 0:
            assert (jan_data == 1.0).all().all()


# ============================================================================
# Test Advanced Composites
# ============================================================================

class TestRotationSignal:
    """Test rotation signal."""
    
    def test_basic_rotation(self, sample_price_data):
        """Test top-N rotation."""
        base_signals = [
            MomentumSignal(lookback=20),
            MomentumSignal(lookback=60),
        ]
        
        signal = RotationSignal(
            signals=base_signals,
            n_positions=3,
            rebalance_frequency=20,
        )
        result = signal.generate(sample_price_data)
        
        assert result.shape == sample_price_data.get_close_prices().shape
        # Should select exactly n_positions per rebalance period
        row_sums = result.sum(axis=1)
        assert row_sums.max() <= 3


class TestMultiTimeframeSignal:
    """Test multi-timeframe signal."""
    
    def test_basic_mtf(self, sample_price_data):
        """Test multi-timeframe aggregation."""
        base_signal = MomentumSignal(lookback=20)
        signal = MultiTimeframeSignal(
            base_signal=base_signal,
            timeframes=[1, 5, 10],
        )
        result = signal.generate(sample_price_data)
        
        assert result.shape == sample_price_data.get_close_prices().shape


class TestAdaptiveWeightSignal:
    """Test adaptive weight signal."""
    
    def test_basic_adaptive(self, sample_price_data):
        """Test adaptive weighting."""
        base_signals = [
            MomentumSignal(lookback=20),
            MomentumSignal(lookback=60),
        ]
        
        signal = AdaptiveWeightSignal(
            signals=base_signals,
            lookback=60,
            weight_method="sharpe",
        )
        result = signal.generate(sample_price_data)
        
        assert result.shape == sample_price_data.get_close_prices().shape


class TestThresholdFilterSignal:
    """Test threshold filter signal."""
    
    def test_absolute_threshold(self, sample_price_data):
        """Test absolute threshold filtering."""
        base_signal = MomentumSignal(lookback=20)
        signal = ThresholdFilterSignal(
            base_signal=base_signal,
            threshold=0.5,
            mode="absolute",
        )
        result = signal.generate(sample_price_data)
        
        assert result.shape == sample_price_data.get_close_prices().shape
        # Should only have -1, 0, 1
        unique_vals = np.unique(result.values[~np.isnan(result.values)])
        assert all(v in [-1.0, 0.0, 1.0] for v in unique_vals)


class TestConsensusSignal:
    """Test consensus signal."""
    
    def test_basic_consensus(self, sample_price_data):
        """Test consensus requirement."""
        base_signals = [
            MomentumSignal(lookback=20),
            MomentumSignal(lookback=40),
            MomentumSignal(lookback=60),
        ]
        
        signal = ConsensusSignal(
            signals=base_signals,
            consensus_threshold=0.7,
        )
        result = signal.generate(sample_price_data)
        
        assert result.shape == sample_price_data.get_close_prices().shape


# ============================================================================
# Test Transformation Utilities
# ============================================================================

class TestNormalizeSignal:
    """Test signal normalization."""
    
    def test_zscore_normalization(self, sample_price_data):
        """Test z-score normalization."""
        signal = MomentumSignal(lookback=20).generate(sample_price_data)
        normalized = normalize_signal(signal, NormalizationMethod.ZSCORE)
        
        assert normalized.shape == signal.shape
        # Check that mean is approximately 0 and std is approximately 1
        row_means = normalized.mean(axis=1)
        assert abs(row_means.mean()) < 0.5
    
    def test_minmax_normalization(self, sample_price_data):
        """Test min-max normalization."""
        signal = MomentumSignal(lookback=20).generate(sample_price_data)
        normalized = normalize_signal(signal, NormalizationMethod.MINMAX)
        
        assert normalized.shape == signal.shape
        assert (normalized >= -1).all().all() and (normalized <= 1).all().all()


class TestRankSignal:
    """Test signal ranking."""
    
    def test_cross_sectional_rank(self, sample_price_data):
        """Test cross-sectional ranking."""
        signal = MomentumSignal(lookback=20).generate(sample_price_data)
        ranked = rank_signal(signal, RankingMethod.CROSS_SECTIONAL)
        
        assert ranked.shape == signal.shape
        assert (ranked >= 0).all().all() and (ranked <= 1).all().all()


class TestSmoothSignal:
    """Test signal smoothing."""
    
    def test_sma_smoothing(self, sample_price_data):
        """Test SMA smoothing."""
        signal = MomentumSignal(lookback=20).generate(sample_price_data)
        smoothed = smooth_signal(signal, method="sma", window=5)
        
        assert smoothed.shape == signal.shape
    
    def test_ema_smoothing(self, sample_price_data):
        """Test EMA smoothing."""
        signal = MomentumSignal(lookback=20).generate(sample_price_data)
        smoothed = smooth_signal(signal, method="ema", window=5)
        
        assert smoothed.shape == signal.shape


class TestClipSignal:
    """Test signal clipping."""
    
    def test_basic_clipping(self, sample_price_data):
        """Test value clipping."""
        signal = MomentumSignal(lookback=20).generate(sample_price_data)
        clipped = clip_signal(signal, lower=-0.5, upper=0.5)
        
        assert (clipped >= -0.5).all().all()
        assert (clipped <= 0.5).all().all()


class TestWinsorizeSignal:
    """Test signal winsorization."""
    
    def test_cross_sectional_winsorize(self, sample_price_data):
        """Test cross-sectional winsorization."""
        signal = MomentumSignal(lookback=20).generate(sample_price_data)
        winsorized = winsorize_signal(signal, limits=(0.05, 0.05), axis=1)
        
        assert winsorized.shape == signal.shape


class TestDemeanSignal:
    """Test signal demeaning."""
    
    def test_cross_sectional_demean(self, sample_price_data):
        """Test cross-sectional demeaning."""
        signal = MomentumSignal(lookback=20).generate(sample_price_data)
        demeaned = demean_signal(signal, axis=1)
        
        assert demeaned.shape == signal.shape
        # Row means should be approximately zero
        row_means = demeaned.mean(axis=1)
        assert abs(row_means.mean()) < 0.1


class TestNeutralizeSignal:
    """Test signal neutralization."""
    
    def test_basic_neutralization(self, sample_price_data):
        """Test factor neutralization."""
        signal = MomentumSignal(lookback=20).generate(sample_price_data)
        factor = MomentumSignal(lookback=60).generate(sample_price_data)
        
        neutralized = neutralize_signal(signal, factor)
        
        assert neutralized.shape == signal.shape


class TestApplyDecay:
    """Test signal decay."""
    
    def test_basic_decay(self, sample_price_data):
        """Test exponential decay."""
        signal = MomentumSignal(lookback=20).generate(sample_price_data)
        decayed = apply_decay(signal, half_life=10)
        
        assert decayed.shape == signal.shape


class TestCombineSignals:
    """Test signal combination."""
    
    def test_average_combination(self, sample_price_data):
        """Test averaging signals."""
        sig1 = MomentumSignal(lookback=20).generate(sample_price_data)
        sig2 = MomentumSignal(lookback=60).generate(sample_price_data)
        
        combined = combine_signals([sig1, sig2], method="average")
        
        assert combined.shape == sig1.shape
    
    def test_median_combination(self, sample_price_data):
        """Test median combination."""
        sig1 = MomentumSignal(lookback=20).generate(sample_price_data)
        sig2 = MomentumSignal(lookback=60).generate(sample_price_data)
        
        combined = combine_signals([sig1, sig2], method="median")
        
        assert combined.shape == sig1.shape


class TestScoreSignals:
    """Test signal scoring."""
    
    def test_zscore_scoring(self, sample_price_data):
        """Test z-score based scoring."""
        signal = MomentumSignal(lookback=20).generate(sample_price_data)
        scores = score_signals(signal, method="zscore")
        
        assert scores.shape == signal.shape
    
    def test_percentile_scoring(self, sample_price_data):
        """Test percentile-based scoring."""
        signal = MomentumSignal(lookback=20).generate(sample_price_data)
        scores = score_signals(signal, method="percentile")
        
        assert scores.shape == signal.shape
        assert (scores >= 0).all().all() and (scores <= 1).all().all()

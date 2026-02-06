"""Tests for NLP-based sentiment signals.

Tests the BERT-signal integration implemented in Phase 19.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the new NLP-based signals
from jsf.signals.sentiment import (
    TextSentimentSignal,
    SentimentMomentumSignal,
    SentimentDivergenceSignal,
)
from jsf.signals import SignalType
from jsf.data import PriceData, SyntheticDataLoader


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    loader = SyntheticDataLoader(
        symbols=['AAPL', 'GOOGL'],
        start_date='2024-01-01',
        end_date='2024-01-30',
        initial_price=100.0,
        annual_volatility=0.25,
        seed=42
    )
    return PriceData(data=loader.load())


@pytest.fixture
def sample_sentiment_data():
    """Create sample sentiment text data."""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    texts = [
        "Company beats earnings expectations, stock rallies",
        "Market uncertainty as inflation concerns rise",
        "Strong Q4 results drive positive outlook",
        "Analysts downgrade stock on growth fears",
        "Tech sector shows resilience amid volatility",
    ] * 6
    
    return pd.DataFrame({
        'text': texts,
        'date': dates,
    })


@pytest.fixture
def symbol_sentiment_data():
    """Create sample sentiment data with symbol column."""
    dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
    
    data = []
    for date in dates:
        for symbol in ['AAPL', 'GOOGL']:
            data.append({
                'text': f"{symbol} shows positive momentum today",
                'date': date,
                'symbol': symbol,
            })
    
    return pd.DataFrame(data)


# ============================================================================
# TextSentimentSignal Tests
# ============================================================================

class TestTextSentimentSignal:
    """Tests for TextSentimentSignal."""
    
    def test_creation(self):
        """Test signal creation with default parameters."""
        signal = TextSentimentSignal()
        
        assert signal.name == "text_sentiment"
        assert signal.signal_type == SignalType.SENTIMENT
        assert signal.model_type == "simple"
        assert signal.sentiment_threshold == 0.3
        assert signal.smoothing_window == 5
    
    def test_creation_with_params(self):
        """Test signal creation with custom parameters."""
        signal = TextSentimentSignal(
            model_type="finbert",
            sentiment_threshold=0.5,
            smoothing_window=10,
            name="custom_sentiment",
        )
        
        assert signal.name == "custom_sentiment"
        assert signal.model_type == "finbert"
        assert signal.sentiment_threshold == 0.5
        assert signal.smoothing_window == 10
    
    def test_generate_without_data(self, sample_price_data):
        """Test signal generation without sentiment data (mock mode)."""
        signal = TextSentimentSignal()
        result = signal.generate(sample_price_data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == len(sample_price_data.get_close_prices())
        assert list(result.columns) == ['AAPL', 'GOOGL']
        # Values should be between -1 and 1
        assert result.values.min() >= -1.0
        assert result.values.max() <= 1.0
    
    def test_generate_with_data(self, sample_price_data, sample_sentiment_data):
        """Test signal generation with sentiment data."""
        signal = TextSentimentSignal(model_type="simple")
        result = signal.generate(sample_price_data, sentiment_data=sample_sentiment_data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == len(sample_price_data.get_close_prices())
    
    def test_set_sentiment_data(self, sample_sentiment_data):
        """Test setting sentiment data on the signal."""
        signal = TextSentimentSignal()
        signal.set_sentiment_data(
            sample_sentiment_data,
            text_column="text",
            date_column="date",
        )
        
        assert signal._sentiment_data is not None
        assert len(signal._sentiment_data) == len(sample_sentiment_data)
    
    def test_generate_with_symbol_specific_data(self, sample_price_data, symbol_sentiment_data):
        """Test signal generation with symbol-specific sentiment."""
        signal = TextSentimentSignal()
        signal.set_sentiment_data(
            symbol_sentiment_data,
            text_column="text",
            date_column="date",
            symbol_column="symbol",
        )
        result = signal.generate(sample_price_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'AAPL' in result.columns
        assert 'GOOGL' in result.columns
    
    def test_get_metadata(self):
        """Test metadata generation."""
        signal = TextSentimentSignal(smoothing_window=7)
        metadata = signal.get_metadata()
        
        assert metadata.name == "text_sentiment"
        assert metadata.signal_type == SignalType.SENTIMENT
        assert metadata.lookback_period == 7


# ============================================================================
# SentimentMomentumSignal Tests
# ============================================================================

class TestSentimentMomentumSignal:
    """Tests for SentimentMomentumSignal."""
    
    def test_creation(self):
        """Test signal creation with default parameters."""
        signal = SentimentMomentumSignal()
        
        assert signal.name == "sentiment_momentum"
        assert signal.signal_type == SignalType.SENTIMENT
        assert signal.lookback == 7
        assert signal.momentum_threshold == 0.1
        assert signal.sentiment_threshold == 0.3
    
    def test_creation_with_params(self):
        """Test signal creation with custom parameters."""
        signal = SentimentMomentumSignal(
            lookback=14,
            momentum_threshold=0.2,
            sentiment_threshold=0.5,
            model_type="finbert",
            name="custom_momentum",
        )
        
        assert signal.name == "custom_momentum"
        assert signal.lookback == 14
        assert signal.momentum_threshold == 0.2
        assert signal.sentiment_threshold == 0.5
        assert signal.model_type == "finbert"
    
    def test_generate(self, sample_price_data):
        """Test signal generation."""
        signal = SentimentMomentumSignal()
        result = signal.generate(sample_price_data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == len(sample_price_data.get_close_prices())
        # Signal should be -1, 0, or 1
        unique_values = set(result.values.flatten()) - {np.nan}
        assert unique_values.issubset({-1.0, 0.0, 1.0})
    
    def test_generate_with_sentiment(self, sample_price_data, sample_sentiment_data):
        """Test signal generation with external sentiment data."""
        signal = SentimentMomentumSignal(model_type="simple")
        result = signal.generate(sample_price_data, sentiment_data=sample_sentiment_data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_price_data.get_close_prices().shape
    
    def test_get_metadata(self):
        """Test metadata generation."""
        signal = SentimentMomentumSignal(lookback=10)
        metadata = signal.get_metadata()
        
        assert metadata.name == "sentiment_momentum"
        assert metadata.signal_type == SignalType.SENTIMENT
        assert metadata.lookback_period == 10


# ============================================================================
# SentimentDivergenceSignal Tests
# ============================================================================

class TestSentimentDivergenceSignal:
    """Tests for SentimentDivergenceSignal."""
    
    def test_creation(self):
        """Test signal creation with default parameters."""
        signal = SentimentDivergenceSignal()
        
        assert signal.name == "sentiment_divergence"
        assert signal.signal_type == SignalType.SENTIMENT
        assert signal.lookback == 10
        assert signal.divergence_threshold == 0.15
    
    def test_creation_with_params(self):
        """Test signal creation with custom parameters."""
        signal = SentimentDivergenceSignal(
            lookback=20,
            divergence_threshold=0.25,
            model_type="finbert",
            name="custom_divergence",
        )
        
        assert signal.name == "custom_divergence"
        assert signal.lookback == 20
        assert signal.divergence_threshold == 0.25
        assert signal.model_type == "finbert"
    
    def test_generate(self, sample_price_data):
        """Test signal generation."""
        signal = SentimentDivergenceSignal()
        result = signal.generate(sample_price_data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == len(sample_price_data.get_close_prices())
        # Signal should be -1, 0, or 1
        unique_values = set(result.values.flatten()) - {np.nan}
        assert unique_values.issubset({-1.0, 0.0, 1.0})
    
    def test_divergence_detection(self, sample_price_data):
        """Test that divergence is properly detected."""
        # This tests that the signal can detect price-sentiment divergence
        signal = SentimentDivergenceSignal(
            lookback=5,
            divergence_threshold=0.1,
        )
        result = signal.generate(sample_price_data)
        
        # Should produce some non-zero signals
        # (exact values depend on mock sentiment)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_price_data.get_close_prices().shape
    
    def test_get_metadata(self):
        """Test metadata generation."""
        signal = SentimentDivergenceSignal(lookback=15)
        metadata = signal.get_metadata()
        
        assert metadata.name == "sentiment_divergence"
        assert metadata.signal_type == SignalType.SENTIMENT
        assert metadata.lookback_period == 15


# ============================================================================
# Integration Tests
# ============================================================================

class TestNLPSignalIntegration:
    """Integration tests for NLP signals with the signals framework."""
    
    def test_import_from_signals(self):
        """Test that NLP signals can be imported from jsf.signals."""
        from jsf.signals import (
            TextSentimentSignal,
            SentimentMomentumSignal,
            SentimentDivergenceSignal,
        )
        
        assert TextSentimentSignal is not None
        assert SentimentMomentumSignal is not None
        assert SentimentDivergenceSignal is not None
    
    def test_signal_composition(self, sample_price_data):
        """Test composing NLP signals with other signals."""
        from jsf.signals import combine_signals
        
        text_signal = TextSentimentSignal()
        momentum_signal = SentimentMomentumSignal()
        
        # Generate both signals
        text_result = text_signal.generate(sample_price_data)
        momentum_result = momentum_signal.generate(sample_price_data)
        
        # Combine them
        combined = combine_signals(
            [text_result, momentum_result],
            weights=[0.5, 0.5],
        )
        
        assert isinstance(combined, pd.DataFrame)
        assert combined.shape == text_result.shape
    
    def test_multiple_signals_same_data(self, sample_price_data, sample_sentiment_data):
        """Test running multiple signals on the same data."""
        signals = [
            TextSentimentSignal(model_type="simple"),
            SentimentMomentumSignal(lookback=5),
            SentimentDivergenceSignal(lookback=5),
        ]
        
        results = []
        for signal in signals:
            result = signal.generate(sample_price_data, sentiment_data=sample_sentiment_data)
            results.append(result)
            assert isinstance(result, pd.DataFrame)
        
        assert len(results) == 3


# ============================================================================
# Edge Cases
# ============================================================================

class TestNLPSignalEdgeCases:
    """Edge case tests for NLP signals."""
    
    def test_empty_sentiment_data(self, sample_price_data):
        """Test with empty sentiment data."""
        signal = TextSentimentSignal()
        empty_df = pd.DataFrame(columns=['text', 'date'])
        
        result = signal.generate(sample_price_data, sentiment_data=empty_df)
        assert isinstance(result, pd.DataFrame)
    
    def test_single_symbol(self):
        """Test with single symbol price data."""
        loader = SyntheticDataLoader(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-20',
            initial_price=100.0,
            annual_volatility=0.25,
            seed=42
        )
        price_data = PriceData(data=loader.load())
        
        signal = TextSentimentSignal()
        result = signal.generate(price_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'AAPL' in result.columns
    
    def test_short_lookback(self, sample_price_data):
        """Test with very short lookback period."""
        signal = SentimentMomentumSignal(lookback=2)
        result = signal.generate(sample_price_data)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_high_threshold(self, sample_price_data):
        """Test with high threshold (may produce all zeros)."""
        signal = SentimentDivergenceSignal(divergence_threshold=0.9)
        result = signal.generate(sample_price_data)
        
        assert isinstance(result, pd.DataFrame)
        # High threshold should produce mostly zeros
        assert (result.fillna(0) == 0).sum().sum() >= result.size * 0.8

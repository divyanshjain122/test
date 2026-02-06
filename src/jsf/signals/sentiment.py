"""Sentiment and market regime signals.

This module implements signals based on market sentiment, regime detection,
and behavioral patterns.
"""

from typing import Optional, Any, List, Union, Dict

import pandas as pd
import numpy as np

from jsf.signals.base import Signal, SignalType, SignalMetadata
from jsf.data import PriceData
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


# Lazy import for NLP models to avoid import errors when not installed
def _get_sentiment_analyzer(model_type: str = "simple"):
    """Get sentiment analyzer model.
    
    Args:
        model_type: 'simple' for rule-based, 'finbert' for BERT-based
        
    Returns:
        Sentiment analyzer instance
    """
    try:
        if model_type == "simple":
            from jsf.ml.transformers.sentiment import SimpleSentiment
            return SimpleSentiment()
        elif model_type == "finbert":
            from jsf.ml.transformers.bert import FinBERT
            return FinBERT(use_mock=True)  # Default to mock mode
        else:
            from jsf.ml.transformers.sentiment import SimpleSentiment
            return SimpleSentiment()
    except ImportError:
        logger.warning("NLP models not available, using mock sentiment")
        return None


class MarketRegimeSignal(Signal):
    """
    Market regime detection signal.
    
    Identifies bull, bear, and neutral market conditions.
    """
    
    def __init__(
        self,
        lookback: int = 60,
        regime_threshold: float = 0.15,
        name: str = "market_regime",
    ):
        """
        Initialize market regime signal.
        
        Args:
            lookback: Lookback period for regime detection
            regime_threshold: Threshold for regime classification
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.SENTIMENT,
            description=f"Market regime detection ({lookback}-period)",
            lookback=lookback,
            regime_threshold=regime_threshold,
        )
        self.lookback = lookback
        self.regime_threshold = regime_threshold
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate market regime signal."""
        close_prices = price_data.get_close_prices()
        
        # Calculate market proxy (average of all symbols)
        market_proxy = close_prices.mean(axis=1)
        
        # Calculate trend strength
        sma = market_proxy.rolling(window=self.lookback).mean()
        trend = (market_proxy - sma) / sma
        
        # Classify regime
        # Bull: trend > threshold, Bear: trend < -threshold
        regime = pd.Series(0.0, index=market_proxy.index)
        regime[trend > self.regime_threshold] = 1.0  # Bull
        regime[trend < -self.regime_threshold] = -1.0  # Bear
        
        # Broadcast regime to all symbols
        signal = pd.DataFrame(
            np.tile(regime.values.reshape(-1, 1), (1, len(close_prices.columns))),
            index=close_prices.index,
            columns=close_prices.columns,
        )
        
        return signal
    
    def get_metadata(self) -> SignalMetadata:
        """Get signal metadata."""
        return SignalMetadata(
            name=self.name,
            signal_type=self.signal_type,
            description=self.description,
            parameters=self.parameters,
            lookback_period=self.lookback,
            requires_volume=False,
        )


class BreadthSignal(Signal):
    """
    Market breadth signal.
    
    Measures the percentage of stocks advancing vs declining.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        name: str = "breadth",
    ):
        """
        Initialize breadth signal.
        
        Args:
            lookback: Lookback period for breadth calculation
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.SENTIMENT,
            description=f"Market breadth ({lookback}-period)",
            lookback=lookback,
        )
        self.lookback = lookback
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate breadth signal."""
        returns = price_data.get_returns(periods=1)
        
        # Calculate rolling breadth (% of stocks with positive returns)
        positive_returns = (returns > 0).astype(float)
        breadth = positive_returns.rolling(window=self.lookback).mean()
        
        # Convert to signal [-1, 1]
        # breadth > 0.5 = bullish, < 0.5 = bearish
        signal = 2 * breadth - 1
        
        return signal
    
    def get_metadata(self) -> SignalMetadata:
        """Get signal metadata."""
        return SignalMetadata(
            name=self.name,
            signal_type=self.signal_type,
            description=self.description,
            parameters=self.parameters,
            lookback_period=self.lookback,
            requires_volume=False,
        )


class RelativeStrengthSignal(Signal):
    """
    Relative strength vs market signal.
    
    Compares individual stock performance to market benchmark.
    """
    
    def __init__(
        self,
        lookback: int = 60,
        benchmark_symbol: Optional[str] = None,
        name: str = "relative_strength",
    ):
        """
        Initialize relative strength signal.
        
        Args:
            lookback: Lookback period for RS calculation
            benchmark_symbol: Symbol to use as benchmark (None = market average)
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.SENTIMENT,
            description=f"Relative strength vs benchmark ({lookback}-period)",
            lookback=lookback,
            benchmark_symbol=benchmark_symbol,
        )
        self.lookback = lookback
        self.benchmark_symbol = benchmark_symbol
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate relative strength signal."""
        close_prices = price_data.get_close_prices()
        
        # Calculate returns
        returns = close_prices.pct_change(periods=self.lookback)
        
        # Get benchmark returns
        if self.benchmark_symbol and self.benchmark_symbol in close_prices.columns:
            benchmark_returns = returns[self.benchmark_symbol]
        else:
            # Use market average
            benchmark_returns = returns.mean(axis=1)
        
        # Calculate relative strength (excess return vs benchmark)
        signal = pd.DataFrame(
            index=close_prices.index,
            columns=close_prices.columns,
        )
        
        for symbol in close_prices.columns:
            rs = returns[symbol] - benchmark_returns
            signal[symbol] = np.tanh(rs * 10)  # Normalize
        
        return signal
    
    def get_metadata(self) -> SignalMetadata:
        """Get signal metadata."""
        return SignalMetadata(
            name=self.name,
            signal_type=self.signal_type,
            description=self.description,
            parameters=self.parameters,
            lookback_period=self.lookback,
            requires_volume=False,
        )


class NewHighLowSignal(Signal):
    """
    New highs/lows signal.
    
    Generates signals based on prices making new highs or lows.
    """
    
    def __init__(
        self,
        lookback: int = 252,
        threshold: float = 0.98,
        name: str = "new_high_low",
    ):
        """
        Initialize new high/low signal.
        
        Args:
            lookback: Lookback period for high/low detection
            threshold: Proximity threshold for "new" high/low (0.98 = within 2%)
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.SENTIMENT,
            description=f"New highs/lows ({lookback}-period)",
            lookback=lookback,
            threshold=threshold,
        )
        self.lookback = lookback
        self.threshold = threshold
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate new high/low signal."""
        close_prices = price_data.get_close_prices()
        
        # Calculate rolling highs and lows
        rolling_high = close_prices.rolling(window=self.lookback).max()
        rolling_low = close_prices.rolling(window=self.lookback).min()
        
        # Check if near new high or low
        near_high = close_prices >= (rolling_high * self.threshold)
        near_low = close_prices <= (rolling_low / self.threshold)
        
        # Generate signal
        signal = pd.DataFrame(
            0.0,
            index=close_prices.index,
            columns=close_prices.columns,
        )
        signal[near_high] = 1.0  # Bullish
        signal[near_low] = -1.0  # Bearish
        
        return signal
    
    def get_metadata(self) -> SignalMetadata:
        """Get signal metadata."""
        return SignalMetadata(
            name=self.name,
            signal_type=self.signal_type,
            description=self.description,
            parameters=self.parameters,
            lookback_period=self.lookback,
            requires_volume=False,
        )


class VolumeShockSignal(Signal):
    """
    Volume shock/surge detection signal.
    
    Identifies unusual volume spikes as sentiment indicators.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        shock_threshold: float = 2.0,
        name: str = "volume_shock",
    ):
        """
        Initialize volume shock signal.
        
        Args:
            lookback: Lookback for average volume
            shock_threshold: Multiple of average volume to qualify as shock
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.SENTIMENT,
            description=f"Volume shock detection (threshold={shock_threshold}x)",
            lookback=lookback,
            shock_threshold=shock_threshold,
        )
        self.lookback = lookback
        self.shock_threshold = shock_threshold
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate volume shock signal."""
        volume = price_data.get_field("volume")
        returns = price_data.get_returns(periods=1)
        
        # Calculate average volume
        avg_volume = volume.rolling(window=self.lookback).mean()
        
        # Detect volume shocks
        volume_ratio = volume / (avg_volume + 1e-10)
        is_shock = volume_ratio > self.shock_threshold
        
        # Direction based on price movement during shock
        signal = pd.DataFrame(
            0.0,
            index=volume.index,
            columns=volume.columns,
        )
        
        # Positive shock with positive return = bullish
        signal[is_shock & (returns > 0)] = 1.0
        # Positive shock with negative return = bearish
        signal[is_shock & (returns < 0)] = -1.0
        
        # Decay signal over next few periods
        signal = signal.rolling(window=5).mean().fillna(0)
        
        return signal
    
    def get_metadata(self) -> SignalMetadata:
        """Get signal metadata."""
        return SignalMetadata(
            name=self.name,
            signal_type=self.signal_type,
            description=self.description,
            parameters=self.parameters,
            lookback_period=self.lookback,
            requires_volume=True,
        )


class SeasonalitySignal(Signal):
    """
    Seasonality and calendar effect signal.
    
    Generates signals based on time of year, month, day of week patterns.
    """
    
    def __init__(
        self,
        pattern: str = "monthly",
        favorable_periods: Optional[list] = None,
        name: str = "seasonality",
    ):
        """
        Initialize seasonality signal.
        
        Args:
            pattern: Type of pattern ('monthly', 'quarterly', 'day_of_week')
            favorable_periods: List of favorable periods (e.g., [1, 4, 11, 12] for months)
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.SENTIMENT,
            description=f"Seasonality signal ({pattern} pattern)",
            pattern=pattern,
            favorable_periods=favorable_periods or [],
        )
        self.pattern = pattern
        self.favorable_periods = favorable_periods or []
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate seasonality signal."""
        close_prices = price_data.get_close_prices()
        
        signal = pd.DataFrame(
            0.0,
            index=close_prices.index,
            columns=close_prices.columns,
        )
        
        if self.pattern == "monthly":
            # Month of year effect
            months = close_prices.index.month
            for month in self.favorable_periods:
                signal[months == month] = 1.0
        
        elif self.pattern == "quarterly":
            # Quarter effect
            quarters = close_prices.index.quarter
            for quarter in self.favorable_periods:
                signal[quarters == quarter] = 1.0
        
        elif self.pattern == "day_of_week":
            # Day of week effect (0=Monday, 4=Friday)
            days = close_prices.index.dayofweek
            for day in self.favorable_periods:
                signal[days == day] = 1.0
        
        # If no favorable periods specified, use neutral
        if not self.favorable_periods:
            signal[:] = 0.0
        
        return signal
    
    def get_metadata(self) -> SignalMetadata:
        """Get signal metadata."""
        return SignalMetadata(
            name=self.name,
            signal_type=self.signal_type,
            description=self.description,
            parameters=self.parameters,
            lookback_period=0,
            requires_volume=False,
        )


# =============================================================================
# NLP-BASED SENTIMENT SIGNALS (Phase 19 BERT Integration)
# =============================================================================

class TextSentimentSignal(Signal):
    """
    NLP-based sentiment signal using BERT or rule-based models.
    
    Analyzes text data (news, social media, filings) to generate
    trading signals based on sentiment scores.
    
    This signal requires external text data aligned with price data timestamps.
    
    Example:
        >>> signal = TextSentimentSignal(model_type="simple")
        >>> sentiment_data = pd.DataFrame({
        ...     "text": ["Stock beats earnings", "Market crashes"],
        ...     "date": ["2024-01-01", "2024-01-02"]
        ... })
        >>> signal.set_sentiment_data(sentiment_data)
        >>> signals = signal.generate(price_data)
    """
    
    def __init__(
        self,
        model_type: str = "simple",
        sentiment_threshold: float = 0.3,
        smoothing_window: int = 5,
        name: str = "text_sentiment",
    ):
        """
        Initialize text sentiment signal.
        
        Args:
            model_type: 'simple' for rule-based, 'finbert' for BERT-based
            sentiment_threshold: Threshold for signal generation
            smoothing_window: Window for smoothing sentiment scores
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.SENTIMENT,
            description=f"NLP sentiment signal ({model_type})",
            model_type=model_type,
            sentiment_threshold=sentiment_threshold,
            smoothing_window=smoothing_window,
        )
        self.model_type = model_type
        self.sentiment_threshold = sentiment_threshold
        self.smoothing_window = smoothing_window
        self._sentiment_data: Optional[pd.DataFrame] = None
        self._analyzer = None
    
    def set_sentiment_data(
        self,
        data: pd.DataFrame,
        text_column: str = "text",
        date_column: str = "date",
        symbol_column: Optional[str] = None,
    ) -> None:
        """
        Set external sentiment data.
        
        Args:
            data: DataFrame with text and date columns
            text_column: Column containing text to analyze
            date_column: Column containing dates
            symbol_column: Optional column for symbol-specific sentiment
        """
        self._sentiment_data = data.copy()
        self._text_column = text_column
        self._date_column = date_column
        self._symbol_column = symbol_column
    
    def _get_analyzer(self):
        """Get or create the sentiment analyzer."""
        if self._analyzer is None:
            self._analyzer = _get_sentiment_analyzer(self.model_type)
        return self._analyzer
    
    def _analyze_texts(self, texts: List[str]) -> List[float]:
        """Analyze text sentiment scores."""
        analyzer = self._get_analyzer()
        
        if analyzer is None:
            # Fallback to random mock sentiment
            return [np.random.uniform(-1, 1) for _ in texts]
        
        try:
            if self.model_type == "finbert":
                # FinBERT returns BERTSentimentResult objects
                results = analyzer.predict(texts)
                # Convert to float: positive=1, neutral=0, negative=-1
                scores = []
                for r in results:
                    if r.label.value == "positive":
                        scores.append(r.score)
                    elif r.label.value == "negative":
                        scores.append(-r.score)
                    else:
                        scores.append(0.0)
                return scores
            else:
                # SimpleSentiment returns SentimentResult objects
                results = analyzer.analyze(texts)
                if not isinstance(results, list):
                    results = [results]
                return [r.score for r in results]
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return [0.0] * len(texts)
    
    def generate(
        self,
        price_data: PriceData,
        sentiment_data: Optional[pd.DataFrame] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Generate NLP sentiment signal.
        
        If sentiment_data is not provided, will use mock sentiment.
        
        Args:
            price_data: Price data for alignment
            sentiment_data: Optional DataFrame with text and dates
            **kwargs: Additional arguments
            
        Returns:
            Signal DataFrame (date x symbol) with values in [-1, 1]
        """
        close_prices = price_data.get_close_prices()
        
        # Initialize signal DataFrame
        signal = pd.DataFrame(
            0.0,
            index=close_prices.index,
            columns=close_prices.columns,
        )
        
        # Use provided data or stored data
        data = sentiment_data if sentiment_data is not None else self._sentiment_data
        
        if data is not None and len(data) > 0:
            # Analyze sentiment from text data
            text_col = getattr(self, '_text_column', 'text')
            date_col = getattr(self, '_date_column', 'date')
            symbol_col = getattr(self, '_symbol_column', None)
            
            texts = data[text_col].tolist()
            scores = self._analyze_texts(texts)
            
            # Create sentiment series
            sentiment_df = data.copy()
            sentiment_df['_sentiment_score'] = scores
            sentiment_df[date_col] = pd.to_datetime(sentiment_df[date_col])
            
            # Aggregate by date (and symbol if available)
            if symbol_col and symbol_col in sentiment_df.columns:
                # Symbol-specific sentiment
                grouped = sentiment_df.groupby([date_col, symbol_col])['_sentiment_score'].mean()
                for (date, symbol), score in grouped.items():
                    if date in signal.index and symbol in signal.columns:
                        signal.loc[date, symbol] = score
            else:
                # Broadcast to all symbols
                date_sentiment = sentiment_df.groupby(date_col)['_sentiment_score'].mean()
                for date, score in date_sentiment.items():
                    if date in signal.index:
                        signal.loc[date, :] = score
        else:
            # Generate mock sentiment based on returns
            returns = price_data.get_returns(periods=1)
            # Mock: sentiment follows returns with noise
            signal = np.tanh(returns * 5 + np.random.randn(*returns.shape) * 0.1)
            signal = pd.DataFrame(signal, index=close_prices.index, columns=close_prices.columns)
        
        # Apply smoothing
        if self.smoothing_window > 1:
            signal = signal.rolling(window=self.smoothing_window).mean()
        
        # Apply threshold
        signal[signal.abs() < self.sentiment_threshold] = 0.0
        
        return signal.fillna(0)
    
    def get_metadata(self) -> SignalMetadata:
        """Get signal metadata."""
        return SignalMetadata(
            name=self.name,
            signal_type=self.signal_type,
            description=self.description,
            parameters=self.parameters,
            lookback_period=self.smoothing_window,
            requires_volume=False,
        )


class SentimentMomentumSignal(Signal):
    """
    Sentiment momentum signal.
    
    Generates signals based on sentiment trends - going long when
    sentiment is positive and rising, short when negative and falling.
    
    Example:
        >>> signal = SentimentMomentumSignal(
        ...     lookback=7,
        ...     momentum_threshold=0.2
        ... )
        >>> signals = signal.generate(price_data, sentiment_data=sentiment_df)
    """
    
    def __init__(
        self,
        lookback: int = 7,
        momentum_threshold: float = 0.1,
        sentiment_threshold: float = 0.3,
        model_type: str = "simple",
        name: str = "sentiment_momentum",
    ):
        """
        Initialize sentiment momentum signal.
        
        Args:
            lookback: Lookback period for momentum calculation
            momentum_threshold: Minimum momentum change for signal
            sentiment_threshold: Minimum sentiment level for signal
            model_type: 'simple' or 'finbert' for sentiment analysis
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.SENTIMENT,
            description=f"Sentiment momentum ({lookback}-day lookback)",
            lookback=lookback,
            momentum_threshold=momentum_threshold,
            sentiment_threshold=sentiment_threshold,
        )
        self.lookback = lookback
        self.momentum_threshold = momentum_threshold
        self.sentiment_threshold = sentiment_threshold
        self.model_type = model_type
        self._text_signal = TextSentimentSignal(model_type=model_type)
    
    def generate(
        self,
        price_data: PriceData,
        sentiment_data: Optional[pd.DataFrame] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Generate sentiment momentum signal.
        
        Logic:
        - BUY: sentiment > threshold AND sentiment rising
        - SELL: sentiment < -threshold AND sentiment falling
        - NEUTRAL: otherwise
        
        Args:
            price_data: Price data
            sentiment_data: Optional sentiment DataFrame
            **kwargs: Additional arguments
            
        Returns:
            Signal DataFrame
        """
        # Get base sentiment
        base_sentiment = self._text_signal.generate(
            price_data, 
            sentiment_data=sentiment_data
        )
        
        # Calculate sentiment momentum (rate of change)
        sentiment_ma = base_sentiment.rolling(window=self.lookback).mean()
        sentiment_momentum = sentiment_ma.diff()
        
        # Generate signal
        signal = pd.DataFrame(
            0.0,
            index=base_sentiment.index,
            columns=base_sentiment.columns,
        )
        
        # Bullish: positive sentiment + rising momentum
        bullish = (
            (sentiment_ma > self.sentiment_threshold) & 
            (sentiment_momentum > self.momentum_threshold)
        )
        signal[bullish] = 1.0
        
        # Bearish: negative sentiment + falling momentum
        bearish = (
            (sentiment_ma < -self.sentiment_threshold) & 
            (sentiment_momentum < -self.momentum_threshold)
        )
        signal[bearish] = -1.0
        
        return signal.fillna(0)
    
    def get_metadata(self) -> SignalMetadata:
        """Get signal metadata."""
        return SignalMetadata(
            name=self.name,
            signal_type=self.signal_type,
            description=self.description,
            parameters=self.parameters,
            lookback_period=self.lookback,
            requires_volume=False,
        )


class SentimentDivergenceSignal(Signal):
    """
    Price-sentiment divergence signal.
    
    Detects when price and sentiment are moving in opposite directions,
    which can signal potential reversals.
    
    Divergence patterns:
    - Bullish divergence: Price falling, sentiment rising → BUY
    - Bearish divergence: Price rising, sentiment falling → SELL
    
    Example:
        >>> signal = SentimentDivergenceSignal(lookback=10)
        >>> signals = signal.generate(price_data, sentiment_data=sentiment_df)
    """
    
    def __init__(
        self,
        lookback: int = 10,
        divergence_threshold: float = 0.15,
        model_type: str = "simple",
        name: str = "sentiment_divergence",
    ):
        """
        Initialize sentiment divergence signal.
        
        Args:
            lookback: Lookback for divergence detection
            divergence_threshold: Minimum divergence level for signal
            model_type: 'simple' or 'finbert' for sentiment analysis
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.SENTIMENT,
            description=f"Price-sentiment divergence ({lookback}-period)",
            lookback=lookback,
            divergence_threshold=divergence_threshold,
        )
        self.lookback = lookback
        self.divergence_threshold = divergence_threshold
        self.model_type = model_type
        self._text_signal = TextSentimentSignal(model_type=model_type)
    
    def generate(
        self,
        price_data: PriceData,
        sentiment_data: Optional[pd.DataFrame] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Generate sentiment divergence signal.
        
        Args:
            price_data: Price data
            sentiment_data: Optional sentiment DataFrame
            **kwargs: Additional arguments
            
        Returns:
            Signal DataFrame
        """
        close_prices = price_data.get_close_prices()
        
        # Get sentiment signal
        sentiment = self._text_signal.generate(
            price_data, 
            sentiment_data=sentiment_data
        )
        
        # Calculate price momentum (normalized returns)
        price_returns = close_prices.pct_change(periods=self.lookback)
        price_momentum = np.tanh(price_returns * 5)  # Normalize to [-1, 1]
        
        # Calculate sentiment momentum
        sentiment_momentum = sentiment.rolling(window=self.lookback).mean().diff()
        sentiment_momentum = np.tanh(sentiment_momentum * 3)
        
        # Detect divergence
        signal = pd.DataFrame(
            0.0,
            index=close_prices.index,
            columns=close_prices.columns,
        )
        
        # Bullish divergence: price down, sentiment up
        bullish_divergence = (
            (price_momentum < -self.divergence_threshold) & 
            (sentiment_momentum > self.divergence_threshold)
        )
        signal[bullish_divergence] = 1.0
        
        # Bearish divergence: price up, sentiment down
        bearish_divergence = (
            (price_momentum > self.divergence_threshold) & 
            (sentiment_momentum < -self.divergence_threshold)
        )
        signal[bearish_divergence] = -1.0
        
        return signal.fillna(0)
    
    def get_metadata(self) -> SignalMetadata:
        """Get signal metadata."""
        return SignalMetadata(
            name=self.name,
            signal_type=self.signal_type,
            description=self.description,
            parameters=self.parameters,
            lookback_period=self.lookback,
            requires_volume=False,
        )


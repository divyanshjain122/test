"""Sentiment and market regime signals.

This module implements signals based on market sentiment, regime detection,
and behavioral patterns.
"""

from typing import Optional, Any

import pandas as pd
import numpy as np

from jsf.signals.base import Signal, SignalType, SignalMetadata
from jsf.data import PriceData
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


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

"""Technical indicator signals.

This module implements common technical analysis indicators as trading signals.
"""

from typing import Optional, Any

import pandas as pd
import numpy as np

from jsf.signals.base import Signal, SignalType, SignalMetadata, SignalError
from jsf.data import PriceData
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class MomentumSignal(Signal):
    """
    Momentum signal based on rate of change.
    
    Generates long signals when price momentum is positive,
    short signals when negative.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        normalize: bool = True,
        name: str = "momentum",
    ):
        """
        Initialize momentum signal.
        
        Args:
            lookback: Number of periods for momentum calculation
            normalize: Whether to normalize signal to [-1, 1]
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.TECHNICAL,
            description=f"{lookback}-period momentum signal",
            lookback=lookback,
            normalize=normalize,
        )
        self.lookback = lookback
        self.normalize = normalize
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate momentum signal."""
        close_prices = price_data.get_close_prices()
        
        # Calculate momentum (rate of change)
        momentum = close_prices.pct_change(periods=self.lookback)
        
        if self.normalize:
            # Normalize to [-1, 1] using tanh
            signal = np.tanh(momentum * 10)  # Scale factor for tanh
        else:
            signal = momentum
        
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


class MovingAverageCrossSignal(Signal):
    """
    Moving average crossover signal.
    
    Generates long signals when fast MA crosses above slow MA,
    short signals when fast MA crosses below slow MA.
    """
    
    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        name: str = "ma_cross",
    ):
        """
        Initialize MA cross signal.
        
        Args:
            fast_period: Fast moving average period
            slow_period: Slow moving average period
            name: Signal name
        """
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        
        super().__init__(
            name=name,
            signal_type=SignalType.TECHNICAL,
            description=f"MA({fast_period}/{slow_period}) crossover signal",
            fast_period=fast_period,
            slow_period=slow_period,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate MA crossover signal."""
        close_prices = price_data.get_close_prices()
        
        # Calculate moving averages
        fast_ma = close_prices.rolling(window=self.fast_period).mean()
        slow_ma = close_prices.rolling(window=self.slow_period).mean()
        
        # Generate signal: 1 when fast > slow, -1 when fast < slow
        signal = pd.DataFrame(
            np.where(fast_ma > slow_ma, 1.0, -1.0),
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
            lookback_period=self.slow_period,
            requires_volume=False,
        )


class RSISignal(Signal):
    """
    Relative Strength Index (RSI) signal.
    
    Generates signals based on overbought/oversold conditions.
    """
    
    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        name: str = "rsi",
    ):
        """
        Initialize RSI signal.
        
        Args:
            period: RSI calculation period
            overbought: Overbought threshold (generates short signal)
            oversold: Oversold threshold (generates long signal)
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.TECHNICAL,
            description=f"{period}-period RSI signal",
            period=period,
            overbought=overbought,
            oversold=oversold,
        )
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate RSI signal."""
        close_prices = price_data.get_close_prices()
        
        # Calculate price changes
        delta = close_prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=self.period).mean()
        avg_losses = losses.rolling(window=self.period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signal: -1 when overbought, +1 when oversold, 0 otherwise
        signal = pd.DataFrame(
            np.where(
                rsi > self.overbought,
                -1.0,
                np.where(rsi < self.oversold, 1.0, 0.0)
            ),
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
            lookback_period=self.period,
            requires_volume=False,
        )


class BollingerBandsSignal(Signal):
    """
    Bollinger Bands signal.
    
    Generates signals based on price position relative to bands.
    """
    
    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        name: str = "bollinger",
    ):
        """
        Initialize Bollinger Bands signal.
        
        Args:
            period: Moving average period
            num_std: Number of standard deviations for bands
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.TECHNICAL,
            description=f"Bollinger Bands ({period}, {num_std}σ) signal",
            period=period,
            num_std=num_std,
        )
        self.period = period
        self.num_std = num_std
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate Bollinger Bands signal."""
        close_prices = price_data.get_close_prices()
        
        # Calculate middle band (SMA)
        middle_band = close_prices.rolling(window=self.period).mean()
        
        # Calculate standard deviation
        std = close_prices.rolling(window=self.period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * self.num_std)
        lower_band = middle_band - (std * self.num_std)
        
        # Calculate position within bands [-1, 1]
        # -1 when at lower band, +1 when at upper band, 0 at middle
        band_width = upper_band - lower_band
        position = (close_prices - middle_band) / (band_width / 2)
        
        # Clip to [-1, 1]
        signal = position.clip(-1.0, 1.0)
        
        # Invert: generate long signal when near lower band
        signal = -signal
        
        return signal
    
    def get_metadata(self) -> SignalMetadata:
        """Get signal metadata."""
        return SignalMetadata(
            name=self.name,
            signal_type=self.signal_type,
            description=self.description,
            parameters=self.parameters,
            lookback_period=self.period,
            requires_volume=False,
        )


class MACDSignal(Signal):
    """
    Moving Average Convergence Divergence (MACD) signal.
    
    Generates signals based on MACD line and signal line crossovers.
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        name: str = "macd",
    ):
        """
        Initialize MACD signal.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.TECHNICAL,
            description=f"MACD({fast_period},{slow_period},{signal_period}) signal",
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate MACD signal."""
        close_prices = price_data.get_close_prices()
        
        # Calculate EMAs
        fast_ema = close_prices.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close_prices.ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        
        # Generate signal: 1 when MACD > signal, -1 when MACD < signal
        signal = pd.DataFrame(
            np.where(macd_line > signal_line, 1.0, -1.0),
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
            lookback_period=self.slow_period + self.signal_period,
            requires_volume=False,
        )


class VolumeWeightedSignal(Signal):
    """
    Volume-weighted momentum signal.
    
    Combines price momentum with volume confirmation.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        name: str = "volume_weighted",
    ):
        """
        Initialize volume-weighted signal.
        
        Args:
            lookback: Lookback period for momentum
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.TECHNICAL,
            description=f"{lookback}-period volume-weighted momentum",
            lookback=lookback,
        )
        self.lookback = lookback
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate volume-weighted signal."""
        close_prices = price_data.get_close_prices()
        
        # Get volume data
        volume_data = price_data.get_field("volume")
        
        # Calculate price momentum
        returns = close_prices.pct_change(periods=self.lookback)
        
        # Calculate volume ratio (current vs average)
        avg_volume = volume_data.rolling(window=self.lookback).mean()
        volume_ratio = volume_data / avg_volume
        
        # Combine: scale momentum by volume ratio
        signal = returns * volume_ratio
        
        # Normalize using tanh
        signal = np.tanh(signal * 5)
        
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

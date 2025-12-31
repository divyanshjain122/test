"""Statistical signals based on mathematical models.

This module implements statistical and quantitative signals.
"""

from typing import Optional, Any

import pandas as pd
import numpy as np
from scipy import stats

from jsf.signals.base import Signal, SignalType, SignalMetadata
from jsf.data import PriceData
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class MeanReversionSignal(Signal):
    """
    Mean reversion signal based on z-score.
    
    Generates long signals when price is below mean,
    short signals when above mean.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        entry_threshold: float = 1.5,
        name: str = "mean_reversion",
    ):
        """
        Initialize mean reversion signal.
        
        Args:
            lookback: Lookback period for mean/std calculation
            entry_threshold: Z-score threshold for signal generation
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.STATISTICAL,
            description=f"{lookback}-period mean reversion (threshold={entry_threshold})",
            lookback=lookback,
            entry_threshold=entry_threshold,
        )
        self.lookback = lookback
        self.entry_threshold = entry_threshold
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate mean reversion signal."""
        close_prices = price_data.get_close_prices()
        
        # Calculate rolling mean and std
        rolling_mean = close_prices.rolling(window=self.lookback).mean()
        rolling_std = close_prices.rolling(window=self.lookback).std()
        
        # Calculate z-score
        z_score = (close_prices - rolling_mean) / rolling_std
        
        # Generate signal: invert z-score (buy when low, sell when high)
        # Normalize to [-1, 1]
        signal = -np.tanh(z_score / self.entry_threshold)
        
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


class PairsSignal(Signal):
    """
    Pairs trading signal based on spread z-score.
    
    Trades the spread between two cointegrated assets.
    """
    
    def __init__(
        self,
        lookback: int = 60,
        entry_threshold: float = 2.0,
        name: str = "pairs",
    ):
        """
        Initialize pairs signal.
        
        Args:
            lookback: Lookback period for spread statistics
            entry_threshold: Z-score threshold for entries
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.STATISTICAL,
            description=f"Pairs trading signal (lookback={lookback})",
            lookback=lookback,
            entry_threshold=entry_threshold,
        )
        self.lookback = lookback
        self.entry_threshold = entry_threshold
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate pairs trading signal."""
        close_prices = price_data.get_close_prices()
        
        # For each pair of symbols, calculate spread signal
        symbols = close_prices.columns
        signals = pd.DataFrame(
            0.0,
            index=close_prices.index,
            columns=symbols,
        )
        
        # Simple implementation: compare each symbol to market average
        market_avg = close_prices.mean(axis=1)
        
        for symbol in symbols:
            # Calculate spread
            spread = close_prices[symbol] - market_avg
            
            # Calculate spread z-score
            rolling_mean = spread.rolling(window=self.lookback).mean()
            rolling_std = spread.rolling(window=self.lookback).std()
            z_score = (spread - rolling_mean) / rolling_std
            
            # Invert z-score for mean reversion
            signals[symbol] = -np.tanh(z_score / self.entry_threshold)
        
        return signals
    
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


class TrendStrengthSignal(Signal):
    """
    Trend strength signal based on linear regression.
    
    Measures the strength and direction of price trends.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        min_r_squared: float = 0.7,
        name: str = "trend_strength",
    ):
        """
        Initialize trend strength signal.
        
        Args:
            lookback: Lookback period for regression
            min_r_squared: Minimum R² for valid trend
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.STATISTICAL,
            description=f"{lookback}-period trend strength (R²>{min_r_squared})",
            lookback=lookback,
            min_r_squared=min_r_squared,
        )
        self.lookback = lookback
        self.min_r_squared = min_r_squared
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate trend strength signal."""
        close_prices = price_data.get_close_prices()
        
        signals = pd.DataFrame(
            0.0,
            index=close_prices.index,
            columns=close_prices.columns,
        )
        
        # Calculate for each symbol
        for symbol in close_prices.columns:
            prices = close_prices[symbol]
            
            for i in range(self.lookback, len(prices)):
                # Get window
                window = prices.iloc[i - self.lookback:i].values
                x = np.arange(len(window))
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, window)
                r_squared = r_value ** 2
                
                # Generate signal if trend is strong enough
                if r_squared >= self.min_r_squared:
                    # Normalize slope
                    normalized_slope = np.tanh(slope / window.mean() * 100)
                    signals.iloc[i, signals.columns.get_loc(symbol)] = normalized_slope
        
        return signals
    
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


class VolatilitySignal(Signal):
    """
    Volatility-based signal.
    
    Generates signals based on volatility regimes.
    Low volatility → increase positions, high volatility → reduce positions.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        vol_lookback: int = 60,
        invert: bool = False,
        name: str = "volatility",
    ):
        """
        Initialize volatility signal.
        
        Args:
            lookback: Period for volatility calculation
            vol_lookback: Period for volatility percentile
            invert: If True, favor high volatility
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.STATISTICAL,
            description=f"Volatility-based signal (lookback={lookback})",
            lookback=lookback,
            vol_lookback=vol_lookback,
            invert=invert,
        )
        self.lookback = lookback
        self.vol_lookback = vol_lookback
        self.invert = invert
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate volatility signal."""
        # Calculate returns
        returns = price_data.get_returns(periods=1)
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=self.lookback).std()
        
        # Calculate volatility percentile
        vol_percentile = volatility.rolling(window=self.vol_lookback).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
        )
        
        # Generate signal: -1 in low vol, +1 in high vol (if inverted)
        # Or: +1 in low vol, -1 in high vol (default)
        signal = 2 * vol_percentile - 1  # Scale to [-1, 1]
        
        if not self.invert:
            signal = -signal
        
        return signal
    
    def get_metadata(self) -> SignalMetadata:
        """Get signal metadata."""
        return SignalMetadata(
            name=self.name,
            signal_type=self.signal_type,
            description=self.description,
            parameters=self.parameters,
            lookback_period=max(self.lookback, self.vol_lookback),
            requires_volume=False,
        )


class CorrelationSignal(Signal):
    """
    Correlation-based signal.
    
    Generates signals based on rolling correlation with a benchmark.
    """
    
    def __init__(
        self,
        lookback: int = 60,
        benchmark_symbol: Optional[str] = None,
        name: str = "correlation",
    ):
        """
        Initialize correlation signal.
        
        Args:
            lookback: Correlation calculation period
            benchmark_symbol: Symbol to use as benchmark (None = market average)
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.STATISTICAL,
            description=f"{lookback}-period correlation signal",
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
        """Generate correlation signal."""
        returns = price_data.get_returns(periods=1)
        
        # Get benchmark returns
        if self.benchmark_symbol and self.benchmark_symbol in returns.columns:
            benchmark = returns[self.benchmark_symbol]
        else:
            # Use market average as benchmark
            benchmark = returns.mean(axis=1)
        
        signals = pd.DataFrame(
            0.0,
            index=returns.index,
            columns=returns.columns,
        )
        
        # Calculate rolling correlation for each symbol
        for symbol in returns.columns:
            if symbol == self.benchmark_symbol:
                signals[symbol] = 0.0
                continue
            
            # Rolling correlation
            correlation = returns[symbol].rolling(window=self.lookback).corr(benchmark)
            
            # Use correlation change as signal
            # Positive correlation change → long, negative → short
            signals[symbol] = correlation.diff()
        
        # Normalize
        signals = np.tanh(signals * 10)
        
        return signals
    
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

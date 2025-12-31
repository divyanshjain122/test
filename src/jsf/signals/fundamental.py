"""Fundamental analysis signals.

This module implements signals based on fundamental data and financial metrics.
"""

from typing import Optional, Any, Dict

import pandas as pd
import numpy as np

from jsf.signals.base import Signal, SignalType, SignalMetadata
from jsf.data import PriceData
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class ValueSignal(Signal):
    """
    Value-based signal using price-to-book or similar metrics.
    
    In absence of actual fundamental data, uses price-based proxies.
    """
    
    def __init__(
        self,
        lookback: int = 252,
        method: str = "price_momentum",
        name: str = "value",
    ):
        """
        Initialize value signal.
        
        Args:
            lookback: Lookback period for value assessment
            method: Method for value calculation
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.FUNDAMENTAL,
            description=f"Value signal using {method} method",
            lookback=lookback,
            method=method,
        )
        self.lookback = lookback
        self.method = method
    
    def generate(
        self,
        price_data: PriceData,
        fundamental_data: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Generate value signal.
        
        Args:
            price_data: Price data
            fundamental_data: Optional fundamental metrics (P/E, P/B, etc.)
            **kwargs: Additional parameters
        
        Returns:
            Value signal DataFrame
        """
        close_prices = price_data.get_close_prices()
        
        if fundamental_data and 'price_to_book' in fundamental_data:
            # Use actual P/B ratio if available
            pb_ratio = fundamental_data['price_to_book']
            
            # Lower P/B = more value = positive signal
            # Calculate percentile rank and invert
            signal = pb_ratio.rank(axis=1, pct=True)
            signal = 1 - signal  # Invert: low P/B = high signal
            signal = 2 * signal - 1  # Scale to [-1, 1]
        
        else:
            # Use price-based proxy: relative price position
            # Lower relative price = potential value
            price_rank = close_prices.rank(axis=1, pct=True)
            
            # Calculate vs historical position
            rolling_min = close_prices.rolling(window=self.lookback).min()
            rolling_max = close_prices.rolling(window=self.lookback).max()
            
            # Position in range [0, 1]
            position = (close_prices - rolling_min) / (rolling_max - rolling_min + 1e-10)
            
            # Invert: low position = value opportunity
            signal = 1 - position
            signal = 2 * signal - 1  # Scale to [-1, 1]
        
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
            requires_fundamentals=True,
        )


class QualitySignal(Signal):
    """
    Quality signal based on consistent performance.
    
    Measures stability and consistency of returns.
    """
    
    def __init__(
        self,
        lookback: int = 252,
        stability_weight: float = 0.7,
        name: str = "quality",
    ):
        """
        Initialize quality signal.
        
        Args:
            lookback: Lookback period for quality metrics
            stability_weight: Weight for stability vs returns
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.FUNDAMENTAL,
            description=f"Quality signal based on {lookback}-period stability",
            lookback=lookback,
            stability_weight=stability_weight,
        )
        self.lookback = lookback
        self.stability_weight = stability_weight
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate quality signal."""
        returns = price_data.get_returns(periods=1)
        
        # Calculate quality metrics
        # 1. Return consistency (Sharpe-like ratio)
        rolling_mean = returns.rolling(window=self.lookback).mean()
        rolling_std = returns.rolling(window=self.lookback).std()
        sharpe_proxy = rolling_mean / (rolling_std + 1e-10)
        
        # 2. Downside deviation (lower is better)
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0
        downside_std = downside_returns.rolling(window=self.lookback).std()
        
        # 3. Stability score (negative of downside deviation)
        stability = -downside_std
        
        # Combine metrics
        quality_score = (
            self.stability_weight * stability + 
            (1 - self.stability_weight) * sharpe_proxy
        )
        
        # Normalize using tanh
        signal = np.tanh(quality_score * 2)
        
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
            requires_fundamentals=False,
        )


class GrowthSignal(Signal):
    """
    Growth signal based on price and volume growth trends.
    
    Identifies assets with strong growth characteristics.
    """
    
    def __init__(
        self,
        lookback: int = 60,
        growth_periods: tuple = (20, 60, 120),
        name: str = "growth",
    ):
        """
        Initialize growth signal.
        
        Args:
            lookback: Base lookback period
            growth_periods: Periods to measure growth over
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.FUNDAMENTAL,
            description=f"Growth signal over multiple periods",
            lookback=lookback,
            growth_periods=growth_periods,
        )
        self.lookback = lookback
        self.growth_periods = growth_periods
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate growth signal."""
        close_prices = price_data.get_close_prices()
        volume = price_data.get_field("volume")
        
        # Calculate growth rates over multiple periods
        growth_signals = []
        
        for period in self.growth_periods:
            # Price growth
            price_growth = close_prices.pct_change(periods=period)
            
            # Volume growth (indicator of increased interest)
            volume_growth = volume.pct_change(periods=period)
            
            # Combined growth score
            combined = (price_growth + volume_growth * 0.3) / 1.3
            growth_signals.append(combined)
        
        # Average across periods
        avg_growth = sum(growth_signals) / len(growth_signals)
        
        # Normalize
        signal = np.tanh(avg_growth * 5)
        
        return signal
    
    def get_metadata(self) -> SignalMetadata:
        """Get signal metadata."""
        return SignalMetadata(
            name=self.name,
            signal_type=self.signal_type,
            description=self.description,
            parameters=self.parameters,
            lookback_period=max(self.growth_periods),
            requires_volume=True,
            requires_fundamentals=False,
        )


class SizeSignal(Signal):
    """
    Size-based signal (small-cap vs large-cap effect).
    
    Generates signals based on relative market capitalization proxies.
    """
    
    def __init__(
        self,
        favor_small: bool = True,
        lookback: int = 60,
        name: str = "size",
    ):
        """
        Initialize size signal.
        
        Args:
            favor_small: If True, favor smaller caps
            lookback: Lookback for volume-based size proxy
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.FUNDAMENTAL,
            description=f"Size signal ({'small' if favor_small else 'large'}-cap bias)",
            favor_small=favor_small,
            lookback=lookback,
        )
        self.favor_small = favor_small
        self.lookback = lookback
    
    def generate(
        self,
        price_data: PriceData,
        market_caps: Optional[pd.DataFrame] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate size signal."""
        
        if market_caps is not None:
            # Use actual market cap data if available
            size_proxy = market_caps
        else:
            # Use price * volume as proxy for market cap
            close_prices = price_data.get_close_prices()
            volume = price_data.get_field("volume")
            size_proxy = close_prices * volume
        
        # Calculate relative size (percentile rank across symbols)
        size_rank = size_proxy.rank(axis=1, pct=True)
        
        # Generate signal based on preference
        if self.favor_small:
            # Small caps = high signal
            signal = 1 - size_rank  # Invert
        else:
            # Large caps = high signal
            signal = size_rank
        
        # Scale to [-1, 1]
        signal = 2 * signal - 1
        
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
            requires_fundamentals=True,
        )


class DividendSignal(Signal):
    """
    Dividend yield signal.
    
    Favors stocks with higher dividend yields.
    """
    
    def __init__(
        self,
        lookback: int = 252,
        yield_threshold: float = 0.02,
        name: str = "dividend",
    ):
        """
        Initialize dividend signal.
        
        Args:
            lookback: Lookback period
            yield_threshold: Minimum yield threshold
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.FUNDAMENTAL,
            description=f"Dividend yield signal (threshold={yield_threshold})",
            lookback=lookback,
            yield_threshold=yield_threshold,
        )
        self.lookback = lookback
        self.yield_threshold = yield_threshold
    
    def generate(
        self,
        price_data: PriceData,
        dividend_yields: Optional[pd.DataFrame] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate dividend signal."""
        
        if dividend_yields is not None:
            # Use actual dividend yield data
            yields = dividend_yields
        else:
            # Create placeholder (all zeros)
            close_prices = price_data.get_close_prices()
            yields = pd.DataFrame(
                0.0,
                index=close_prices.index,
                columns=close_prices.columns,
            )
            logger.warning(
                f"{self.name}: No dividend data provided, using zeros"
            )
        
        # Rank by yield
        yield_rank = yields.rank(axis=1, pct=True)
        
        # Convert to signal [-1, 1]
        signal = 2 * yield_rank - 1
        
        # Apply threshold: only positive signals for yields above threshold
        signal[yields < self.yield_threshold] = -0.5
        
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
            requires_fundamentals=True,
        )

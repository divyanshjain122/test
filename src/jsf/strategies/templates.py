"""Common strategy templates."""

from typing import List, Optional, Any, Dict

import pandas as pd
import numpy as np

from jsf.strategies.base import Strategy, StrategyType
from jsf.signals.base import Signal
from jsf.signals import (
    MomentumSignal,
    MeanReversionSignal,
    MovingAverageCrossSignal,
    TrendStrengthSignal,
)
from jsf.portfolio import (
    Portfolio,
    PortfolioConstructor,
    SimplePortfolioConstructor,
    EqualWeightSizer,
)
from jsf.data import PriceData
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class MomentumStrategy(Strategy):
    """
    Classic momentum strategy.
    
    Buys past winners and shorts (or avoids) past losers.
    """
    
    def __init__(
        self,
        name: str = "momentum",
        lookback: int = 60,
        long_only: bool = True,
        portfolio_constructor: Optional[PortfolioConstructor] = None,
        **params: Any
    ):
        """
        Initialize momentum strategy.
        
        Args:
            name: Strategy name
            lookback: Momentum lookback period
            long_only: Long-only or long-short
            portfolio_constructor: Portfolio construction method
            **params: Additional parameters
        """
        # Create signal
        signal = MomentumSignal(lookback=lookback)
        
        # Create portfolio constructor if not provided
        if portfolio_constructor is None:
            sizer = EqualWeightSizer(long_only=long_only)
            portfolio_constructor = SimplePortfolioConstructor(
                position_sizer=sizer,
                name=f"{name}_constructor"
            )
        
        super().__init__(
            name=name,
            signals=[signal],
            portfolio_constructor=portfolio_constructor,
            lookback=lookback,
            long_only=long_only,
            **params
        )
    
    def generate_signals(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate momentum signals."""
        return self.signals[0].generate(price_data)
    
    def construct_portfolio(
        self,
        signals: pd.DataFrame,
        price_data: PriceData,
        **kwargs: Any
    ) -> Portfolio:
        """Construct portfolio from momentum signals."""
        return self.portfolio_constructor.construct(signals, price_data, **kwargs)


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy.
    
    Buys oversold assets and shorts (or avoids) overbought assets.
    """
    
    def __init__(
        self,
        name: str = "mean_reversion",
        lookback: int = 20,
        entry_threshold: float = 2.0,
        long_only: bool = True,
        portfolio_constructor: Optional[PortfolioConstructor] = None,
        **params: Any
    ):
        """
        Initialize mean reversion strategy.
        
        Args:
            name: Strategy name
            lookback: Mean reversion lookback
            entry_threshold: Z-score threshold for entry
            long_only: Long-only or long-short
            portfolio_constructor: Portfolio construction method
            **params: Additional parameters
        """
        signal = MeanReversionSignal(
            lookback=lookback,
            entry_threshold=entry_threshold
        )
        
        if portfolio_constructor is None:
            sizer = EqualWeightSizer(long_only=long_only)
            portfolio_constructor = SimplePortfolioConstructor(
                position_sizer=sizer,
                name=f"{name}_constructor"
            )
        
        super().__init__(
            name=name,
            signals=[signal],
            portfolio_constructor=portfolio_constructor,
            lookback=lookback,
            entry_threshold=entry_threshold,
            long_only=long_only,
            **params
        )
    
    def generate_signals(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate mean reversion signals."""
        return self.signals[0].generate(price_data)
    
    def construct_portfolio(
        self,
        signals: pd.DataFrame,
        price_data: PriceData,
        **kwargs: Any
    ) -> Portfolio:
        """Construct portfolio from mean reversion signals."""
        return self.portfolio_constructor.construct(signals, price_data, **kwargs)


class TrendFollowingStrategy(Strategy):
    """
    Trend following strategy.
    
    Uses moving average crossovers and trend strength.
    """
    
    def __init__(
        self,
        name: str = "trend_following",
        fast_period: int = 50,
        slow_period: int = 200,
        trend_lookback: int = 60,
        long_only: bool = True,
        portfolio_constructor: Optional[PortfolioConstructor] = None,
        **params: Any
    ):
        """
        Initialize trend following strategy.
        
        Args:
            name: Strategy name
            fast_period: Fast MA period
            slow_period: Slow MA period
            trend_lookback: Trend strength lookback
            long_only: Long-only or long-short
            portfolio_constructor: Portfolio construction method
            **params: Additional parameters
        """
        # Multiple signals
        ma_cross = MovingAverageCrossSignal(
            fast_period=fast_period,
            slow_period=slow_period
        )
        trend_strength = TrendStrengthSignal(lookback=trend_lookback)
        
        if portfolio_constructor is None:
            sizer = EqualWeightSizer(long_only=long_only)
            portfolio_constructor = SimplePortfolioConstructor(
                position_sizer=sizer,
                name=f"{name}_constructor"
            )
        
        super().__init__(
            name=name,
            signals=[ma_cross, trend_strength],
            portfolio_constructor=portfolio_constructor,
            fast_period=fast_period,
            slow_period=slow_period,
            trend_lookback=trend_lookback,
            long_only=long_only,
            **params
        )
    
    def generate_signals(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate trend following signals."""
        # Combine MA cross and trend strength
        ma_signals = self.signals[0].generate(price_data)
        trend_signals = self.signals[1].generate(price_data)
        
        # Average the signals (both should agree)
        combined = (ma_signals + trend_signals) / 2
        return combined
    
    def construct_portfolio(
        self,
        signals: pd.DataFrame,
        price_data: PriceData,
        **kwargs: Any
    ) -> Portfolio:
        """Construct portfolio from trend signals."""
        return self.portfolio_constructor.construct(signals, price_data, **kwargs)

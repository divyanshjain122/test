"""Base strategy template infrastructure.

This module provides the foundation for building quantitative trading strategies,
combining signals and portfolio construction into complete trading systems.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from enum import Enum

import pandas as pd

from jsf.data import PriceData
from jsf.signals.base import Signal
from jsf.portfolio import Portfolio, PortfolioConstructor
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class StrategyType(Enum):
    """Strategy classification types."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    FACTOR_BASED = "factor_based"
    MULTI_STRATEGY = "multi_strategy"
    CUSTOM = "custom"


class StrategyMetadata:
    """Metadata about a strategy."""
    
    def __init__(
        self,
        name: str,
        strategy_type: StrategyType,
        description: str,
        version: str = "1.0.0",
        author: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize strategy metadata.
        
        Args:
            name: Strategy name
            strategy_type: Type of strategy
            description: Strategy description
            version: Strategy version
            author: Strategy author
            parameters: Strategy parameters
        """
        self.name = name
        self.strategy_type = strategy_type
        self.description = description
        self.version = version
        self.author = author
        self.parameters = parameters or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "parameters": self.parameters,
        }


class Strategy(ABC):
    """
    Base class for all trading strategies.
    
    A strategy combines signal generation with portfolio construction
    to produce actionable trading portfolios.
    """
    
    def __init__(
        self,
        name: str,
        signals: List[Signal],
        portfolio_constructor: PortfolioConstructor,
        **params: Any
    ):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            signals: List of signals to use
            portfolio_constructor: Portfolio construction method
            **params: Additional strategy parameters
        """
        self.name = name
        self.signals = signals
        self.portfolio_constructor = portfolio_constructor
        self.params = params
        
        logger.info(f"Initialized {self.__class__.__name__}: {name}")
    
    @abstractmethod
    def generate_signals(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Args:
            price_data: Price data
            **kwargs: Additional arguments
        
        Returns:
            DataFrame of signals (date x symbol)
        """
        pass
    
    @abstractmethod
    def construct_portfolio(
        self,
        signals: pd.DataFrame,
        price_data: PriceData,
        **kwargs: Any
    ) -> Portfolio:
        """
        Construct portfolio from signals.
        
        Args:
            signals: Trading signals
            price_data: Price data
            **kwargs: Additional arguments
        
        Returns:
            Portfolio object
        """
        pass
    
    def run(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> Portfolio:
        """
        Run complete strategy pipeline.
        
        Args:
            price_data: Price data
            **kwargs: Additional arguments
        
        Returns:
            Final portfolio
        """
        logger.info(f"Running strategy: {self.name}")
        
        # Generate signals
        signals = self.generate_signals(price_data, **kwargs)
        logger.info(f"Generated signals: {signals.shape}")
        
        # Construct portfolio
        portfolio = self.construct_portfolio(signals, price_data, **kwargs)
        logger.info(f"Constructed portfolio: {len(portfolio.weights)} periods")
        
        return portfolio
    
    def get_metadata(self) -> StrategyMetadata:
        """Get strategy metadata."""
        return StrategyMetadata(
            name=self.name,
            strategy_type=StrategyType.CUSTOM,
            description=self.__doc__ or "No description",
            parameters=self.params,
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

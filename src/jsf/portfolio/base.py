"""Base classes for portfolio construction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from datetime import datetime
from enum import Enum

import pandas as pd
import numpy as np

from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    CUSTOM = "custom"


@dataclass
class Portfolio:
    """
    Portfolio container holding positions and weights.
    
    Attributes:
        weights: DataFrame of portfolio weights (date x symbol)
        metadata: Additional portfolio metadata
        cash: Cash position over time
        leverage: Leverage ratio over time
    """
    weights: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    cash: Optional[pd.Series] = None
    leverage: Optional[pd.Series] = None
    
    def __post_init__(self):
        """Validate portfolio data."""
        if not isinstance(self.weights, pd.DataFrame):
            raise ValueError("weights must be a pandas DataFrame")
        
        if not isinstance(self.weights.index, pd.DatetimeIndex):
            raise ValueError("weights index must be DatetimeIndex")
    
    def get_positions(self, date: Optional[datetime] = None) -> pd.Series:
        """
        Get portfolio positions at a specific date.
        
        Args:
            date: Date to get positions for (None = latest)
        
        Returns:
            Series of positions (symbol -> weight)
        """
        if date is None:
            return self.weights.iloc[-1]
        
        if date not in self.weights.index:
            # Find nearest date
            idx = self.weights.index.asof(date)
            if pd.isna(idx):
                raise ValueError(f"No data available for date {date}")
            return self.weights.loc[idx]
        
        return self.weights.loc[date]
    
    def get_turnover(self) -> pd.Series:
        """
        Calculate portfolio turnover over time.
        
        Returns:
            Series of turnover values
        """
        position_changes = self.weights.diff().abs()
        turnover = position_changes.sum(axis=1) / 2  # Divide by 2 to avoid double counting
        return turnover
    
    def get_exposure(self) -> pd.Series:
        """
        Calculate gross exposure over time.
        
        Returns:
            Series of gross exposure
        """
        return self.weights.abs().sum(axis=1)
    
    def get_net_exposure(self) -> pd.Series:
        """
        Calculate net exposure over time.
        
        Returns:
            Series of net exposure
        """
        return self.weights.sum(axis=1)
    
    def get_long_exposure(self) -> pd.Series:
        """Get long exposure over time."""
        return self.weights.clip(lower=0).sum(axis=1)
    
    def get_short_exposure(self) -> pd.Series:
        """Get short exposure over time."""
        return self.weights.clip(upper=0).abs().sum(axis=1)
    
    def summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary statistics.
        
        Returns:
            Dictionary of summary statistics
        """
        return {
            "num_periods": len(self.weights),
            "num_symbols": len(self.weights.columns),
            "avg_positions": (self.weights != 0).sum(axis=1).mean(),
            "avg_turnover": self.get_turnover().mean(),
            "avg_gross_exposure": self.get_exposure().mean(),
            "avg_net_exposure": self.get_net_exposure().mean(),
            "avg_long_exposure": self.get_long_exposure().mean(),
            "avg_short_exposure": self.get_short_exposure().mean(),
            "start_date": self.weights.index[0],
            "end_date": self.weights.index[-1],
        }


class PortfolioConstructor(ABC):
    """
    Abstract base class for portfolio construction.
    
    Combines signals with position sizing, optimization, and rebalancing
    to produce final portfolio weights.
    """
    
    def __init__(
        self,
        name: str = "portfolio_constructor",
        **kwargs: Any
    ):
        """
        Initialize portfolio constructor.
        
        Args:
            name: Constructor name
            **kwargs: Additional parameters
        """
        self.name = name
        self.parameters = kwargs
        logger.info(f"Initialized {self.__class__.__name__}: {name}")
    
    @abstractmethod
    def construct(
        self,
        signals: pd.DataFrame,
        price_data: Any,
        **kwargs: Any
    ) -> Portfolio:
        """
        Construct portfolio from signals.
        
        Args:
            signals: Signal DataFrame (date x symbol)
            price_data: Price data for risk calculations
            **kwargs: Additional parameters
        
        Returns:
            Portfolio object with weights
        """
        pass
    
    def validate_signals(self, signals: pd.DataFrame) -> None:
        """Validate signal input."""
        if not isinstance(signals, pd.DataFrame):
            raise ValueError("signals must be a pandas DataFrame")
        
        if not isinstance(signals.index, pd.DatetimeIndex):
            raise ValueError("signals index must be DatetimeIndex")
        
        if signals.empty:
            raise ValueError("signals DataFrame is empty")


class PositionSizer(ABC):
    """
    Abstract base class for position sizing.
    
    Converts signals to position sizes/weights.
    """
    
    def __init__(
        self,
        name: str = "position_sizer",
        **kwargs: Any
    ):
        """
        Initialize position sizer.
        
        Args:
            name: Sizer name
            **kwargs: Additional parameters
        """
        self.name = name
        self.parameters = kwargs
        logger.info(f"Initialized {self.__class__.__name__}: {name}")
    
    @abstractmethod
    def size(
        self,
        signals: pd.DataFrame,
        price_data: Any,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Calculate position sizes from signals.
        
        Args:
            signals: Signal DataFrame
            price_data: Price data for calculations
            **kwargs: Additional parameters
        
        Returns:
            DataFrame of position weights
        """
        pass


class WeightOptimizer(ABC):
    """
    Abstract base class for weight optimization.
    
    Optimizes portfolio weights subject to constraints.
    """
    
    def __init__(
        self,
        name: str = "weight_optimizer",
        **kwargs: Any
    ):
        """
        Initialize weight optimizer.
        
        Args:
            name: Optimizer name
            **kwargs: Additional parameters
        """
        self.name = name
        self.parameters = kwargs
        logger.info(f"Initialized {self.__class__.__name__}: {name}")
    
    @abstractmethod
    def optimize(
        self,
        signals: pd.DataFrame,
        covariance: pd.DataFrame,
        expected_returns: Optional[pd.Series] = None,
        **kwargs: Any
    ) -> pd.Series:
        """
        Optimize portfolio weights.
        
        Args:
            signals: Signal strengths
            covariance: Covariance matrix
            expected_returns: Expected returns (optional)
            **kwargs: Additional parameters
        
        Returns:
            Series of optimized weights
        """
        pass


class Rebalancer(ABC):
    """
    Abstract base class for portfolio rebalancing.
    
    Determines when and how to rebalance portfolio.
    """
    
    def __init__(
        self,
        name: str = "rebalancer",
        **kwargs: Any
    ):
        """
        Initialize rebalancer.
        
        Args:
            name: Rebalancer name
            **kwargs: Additional parameters
        """
        self.name = name
        self.parameters = kwargs
        logger.info(f"Initialized {self.__class__.__name__}: {name}")
    
    @abstractmethod
    def should_rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        date: datetime,
        **kwargs: Any
    ) -> bool:
        """
        Determine if portfolio should be rebalanced.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            date: Current date
            **kwargs: Additional context
        
        Returns:
            True if should rebalance
        """
        pass
    
    @abstractmethod
    def rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        **kwargs: Any
    ) -> pd.Series:
        """
        Calculate rebalanced weights.
        
        Args:
            current_weights: Current weights
            target_weights: Target weights
            **kwargs: Additional parameters
        
        Returns:
            Rebalanced weights
        """
        pass

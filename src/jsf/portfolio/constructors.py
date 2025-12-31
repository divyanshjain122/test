"""Main portfolio constructor combining all components."""

from typing import Optional, Any, List

import pandas as pd
import numpy as np

from jsf.portfolio.base import (
    Portfolio,
    PortfolioConstructor,
    PositionSizer,
    WeightOptimizer,
    Rebalancer,
)
from jsf.portfolio.constraints import PortfolioConstraints, Constraint
from jsf.data import PriceData
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class SimplePortfolioConstructor(PortfolioConstructor):
    """
    Simple portfolio constructor.
    
    Combines position sizing and optional rebalancing.
    """
    
    def __init__(
        self,
        position_sizer: PositionSizer,
        rebalancer: Optional[Rebalancer] = None,
        constraints: Optional[PortfolioConstraints] = None,
        name: str = "simple_portfolio",
    ):
        """
        Initialize simple portfolio constructor.
        
        Args:
            position_sizer: Position sizing method
            rebalancer: Rebalancing strategy (optional)
            constraints: Portfolio constraints (optional)
            name: Constructor name
        """
        super().__init__(name=name)
        self.position_sizer = position_sizer
        self.rebalancer = rebalancer
        self.constraints = constraints or PortfolioConstraints()
    
    def construct(
        self,
        signals: pd.DataFrame,
        price_data: PriceData,
        **kwargs: Any
    ) -> Portfolio:
        """Construct portfolio from signals."""
        self.validate_signals(signals)
        
        # Generate initial weights from signals
        weights = self.position_sizer.size(signals, price_data, **kwargs)
        
        # Apply constraints
        if self.constraints:
            for idx in weights.index:
                weights.loc[idx] = self.constraints.enforce_all(
                    weights.loc[idx],
                    **kwargs
                )
        
        # Apply rebalancing if specified
        if self.rebalancer:
            rebalanced_weights = pd.DataFrame(
                index=weights.index,
                columns=weights.columns,
                dtype=float,
            )
            
            previous_weights = None
            for date in weights.index:
                target_weights = weights.loc[date]
                
                if previous_weights is None:
                    # First period, use target weights
                    rebalanced_weights.loc[date] = target_weights
                else:
                    # Check if should rebalance
                    if self.rebalancer.should_rebalance(
                        previous_weights,
                        target_weights,
                        date,
                        **kwargs
                    ):
                        rebalanced_weights.loc[date] = self.rebalancer.rebalance(
                            previous_weights,
                            target_weights,
                            **kwargs
                        )
                    else:
                        # Keep previous weights
                        rebalanced_weights.loc[date] = previous_weights
                
                previous_weights = rebalanced_weights.loc[date]
            
            weights = rebalanced_weights
        
        return Portfolio(weights=weights, metadata={
            "constructor": self.name,
            "position_sizer": self.position_sizer.name,
            "rebalancer": self.rebalancer.name if self.rebalancer else None,
        })


class OptimizedPortfolioConstructor(PortfolioConstructor):
    """
    Optimized portfolio constructor.
    
    Uses optimization to determine weights.
    """
    
    def __init__(
        self,
        optimizer: WeightOptimizer,
        lookback: int = 60,
        rebalancer: Optional[Rebalancer] = None,
        constraints: Optional[PortfolioConstraints] = None,
        name: str = "optimized_portfolio",
    ):
        """
        Initialize optimized portfolio constructor.
        
        Args:
            optimizer: Weight optimization method
            lookback: Lookback for covariance/return estimation
            rebalancer: Rebalancing strategy (optional)
            constraints: Portfolio constraints (optional)
            name: Constructor name
        """
        super().__init__(name=name)
        self.optimizer = optimizer
        self.lookback = lookback
        self.rebalancer = rebalancer
        self.constraints = constraints or PortfolioConstraints()
    
    def construct(
        self,
        signals: pd.DataFrame,
        price_data: PriceData,
        **kwargs: Any
    ) -> Portfolio:
        """Construct optimized portfolio."""
        self.validate_signals(signals)
        
        returns = price_data.get_returns(periods=1)
        
        weights = pd.DataFrame(
            index=signals.index,
            columns=signals.columns,
            dtype=float,
        )
        
        for i, date in enumerate(signals.index):
            # Get historical window
            hist_start = max(0, i - self.lookback)
            hist_returns = returns.iloc[hist_start:i]
            
            if len(hist_returns) < 10:
                # Not enough data, use equal weight
                sig_row = signals.loc[date]
                active = sig_row != 0
                if active.sum() > 0:
                    weights.loc[date, active] = 1.0 / active.sum()
                continue
            
            # Get active signals
            sig_row = signals.loc[date]
            active = sig_row != 0
            
            if active.sum() == 0:
                weights.loc[date] = 0.0
                continue
            
            # Calculate covariance and expected returns
            active_returns = hist_returns.loc[:, active]
            cov_matrix = active_returns.cov()
            expected_rets = active_returns.mean()
            
            # Optimize weights
            try:
                optimal_weights = self.optimizer.optimize(
                    sig_row[active],
                    cov_matrix,
                    expected_rets,
                    **kwargs
                )
                
                weights.loc[date, active] = optimal_weights.values
                weights.loc[date, ~active] = 0.0
            except Exception as e:
                logger.warning(f"Optimization failed at {date}: {e}")
                # Fallback to equal weight
                weights.loc[date, active] = 1.0 / active.sum()
        
        # Apply constraints
        if self.constraints:
            for idx in weights.index:
                weights.loc[idx] = self.constraints.enforce_all(
                    weights.loc[idx],
                    **kwargs
                )
        
        # Apply rebalancing if specified
        if self.rebalancer:
            rebalanced_weights = weights.copy()
            previous_weights = None
            
            for date in weights.index:
                target_weights = weights.loc[date]
                
                if previous_weights is not None:
                    if not self.rebalancer.should_rebalance(
                        previous_weights,
                        target_weights,
                        date,
                        **kwargs
                    ):
                        rebalanced_weights.loc[date] = previous_weights
                
                previous_weights = rebalanced_weights.loc[date]
            
            weights = rebalanced_weights
        
        return Portfolio(weights=weights, metadata={
            "constructor": self.name,
            "optimizer": self.optimizer.name,
            "lookback": self.lookback,
            "rebalancer": self.rebalancer.name if self.rebalancer else None,
        })


class HybridPortfolioConstructor(PortfolioConstructor):
    """
    Hybrid portfolio constructor.
    
    Combines signal-based sizing with risk-based optimization.
    """
    
    def __init__(
        self,
        position_sizer: PositionSizer,
        optimizer: Optional[WeightOptimizer] = None,
        signal_weight: float = 0.5,
        lookback: int = 60,
        rebalancer: Optional[Rebalancer] = None,
        constraints: Optional[PortfolioConstraints] = None,
        name: str = "hybrid_portfolio",
    ):
        """
        Initialize hybrid portfolio constructor.
        
        Args:
            position_sizer: Position sizing method
            optimizer: Weight optimizer (optional)
            signal_weight: Weight for signal-based allocation (0-1)
            lookback: Lookback for risk calculations
            rebalancer: Rebalancing strategy (optional)
            constraints: Portfolio constraints (optional)
            name: Constructor name
        """
        super().__init__(name=name)
        self.position_sizer = position_sizer
        self.optimizer = optimizer
        self.signal_weight = signal_weight
        self.lookback = lookback
        self.rebalancer = rebalancer
        self.constraints = constraints or PortfolioConstraints()
    
    def construct(
        self,
        signals: pd.DataFrame,
        price_data: PriceData,
        **kwargs: Any
    ) -> Portfolio:
        """Construct hybrid portfolio."""
        self.validate_signals(signals)
        
        # Get signal-based weights
        signal_weights = self.position_sizer.size(signals, price_data, **kwargs)
        
        if self.optimizer is None:
            # No optimizer, just use signal weights
            final_weights = signal_weights
        else:
            # Blend signal weights with optimized weights
            returns = price_data.get_returns(periods=1)
            
            optimized_weights = pd.DataFrame(
                index=signals.index,
                columns=signals.columns,
                dtype=float,
            )
            
            for i, date in enumerate(signals.index):
                hist_start = max(0, i - self.lookback)
                hist_returns = returns.iloc[hist_start:i]
                
                if len(hist_returns) < 10:
                    optimized_weights.loc[date] = signal_weights.loc[date]
                    continue
                
                # Get active positions from signal weights
                sig_row = signal_weights.loc[date]
                active = sig_row != 0
                
                if active.sum() == 0:
                    optimized_weights.loc[date] = 0.0
                    continue
                
                # Optimize within active set
                active_returns = hist_returns.loc[:, active]
                cov_matrix = active_returns.cov()
                expected_rets = active_returns.mean()
                
                try:
                    opt_weights = self.optimizer.optimize(
                        signals.loc[date, active],
                        cov_matrix,
                        expected_rets,
                        **kwargs
                    )
                    optimized_weights.loc[date, active] = opt_weights.values
                except Exception as e:
                    logger.warning(f"Optimization failed at {date}: {e}")
                    optimized_weights.loc[date] = signal_weights.loc[date]
            
            # Blend signal and optimized weights
            final_weights = (
                self.signal_weight * signal_weights +
                (1 - self.signal_weight) * optimized_weights
            )
            
            # Renormalize
            row_sums = final_weights.abs().sum(axis=1)
            final_weights = final_weights.div(row_sums, axis=0).fillna(0)
        
        # Apply constraints
        if self.constraints:
            for idx in final_weights.index:
                final_weights.loc[idx] = self.constraints.enforce_all(
                    final_weights.loc[idx],
                    **kwargs
                )
        
        # Apply rebalancing
        if self.rebalancer:
            rebalanced_weights = final_weights.copy()
            previous_weights = None
            
            for date in final_weights.index:
                target_weights = final_weights.loc[date]
                
                if previous_weights is not None:
                    if not self.rebalancer.should_rebalance(
                        previous_weights,
                        target_weights,
                        date,
                        **kwargs
                    ):
                        rebalanced_weights.loc[date] = previous_weights
                
                previous_weights = rebalanced_weights.loc[date]
            
            final_weights = rebalanced_weights
        
        return Portfolio(weights=final_weights, metadata={
            "constructor": self.name,
            "position_sizer": self.position_sizer.name,
            "optimizer": self.optimizer.name if self.optimizer else None,
            "signal_weight": self.signal_weight,
        })

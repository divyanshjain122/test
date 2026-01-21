"""Backtesting and portfolio simulation engine.

This module simulates portfolio performance over time, including:
- Returns calculation
- Transaction costs and slippage
- Rebalancing mechanics
- Performance tracking
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

import pandas as pd
import numpy as np

from jsf.data import PriceData
from jsf.portfolio import Portfolio
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 10 bps
    slippage: float = 0.0005  # 5 bps
    margin_requirement: float = 1.0  # 100% (no leverage)
    compound_returns: bool = True
    rebalance_on_signal: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.transaction_cost < 0:
            raise ValueError("transaction_cost must be non-negative")
        if self.slippage < 0:
            raise ValueError("slippage must be non-negative")


@dataclass
class BacktestResult:
    """Results from a backtest."""
    
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_return(self) -> float:
        """Total return over backtest period."""
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
    
    @property
    def cagr(self) -> float:
        """Compound annual growth rate."""
        years = len(self.returns) / 252  # Assume daily data
        if years <= 0:
            return 0.0
        return (1 + self.total_return) ** (1 / years) - 1
    
    @property
    def volatility(self) -> float:
        """Annualized volatility."""
        return self.returns.std() * np.sqrt(252)
    
    @property
    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Sharpe ratio."""
        excess_returns = self.returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown."""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @property
    def calmar_ratio(self) -> float:
        """Calmar ratio (CAGR / abs(max_drawdown))."""
        md = abs(self.max_drawdown)
        if md == 0:
            return 0.0
        return self.cagr / md
    
    def summary(self) -> pd.Series:
        """Get summary statistics."""
        return pd.Series({
            'Total Return': f"{self.total_return:.2%}",
            'CAGR': f"{self.cagr:.2%}",
            'Volatility': f"{self.volatility:.2%}",
            'Sharpe Ratio': f"{self.sharpe_ratio:.2f}",
            'Max Drawdown': f"{self.max_drawdown:.2%}",
            'Start Date': str(self.equity_curve.index[0].date()),
            'End Date': str(self.equity_curve.index[-1].date()),
            'Days': len(self.equity_curve),
        })


class BacktestEngine:
    """
    Backtesting engine for portfolio strategies.
    
    Simulates portfolio performance with realistic trading mechanics.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        logger.info(f"Initialized BacktestEngine with ${self.config.initial_capital:,.0f} capital")
    
    def run(
        self,
        portfolio: Portfolio,
        price_data: PriceData,
        **kwargs: Any
    ) -> BacktestResult:
        """
        Run backtest on a portfolio.
        
        Args:
            portfolio: Portfolio with weights over time
            price_data: Historical price data
            **kwargs: Additional arguments
        
        Returns:
            BacktestResult with performance metrics
        """
        logger.info("Starting backtest simulation")
        
        # Align portfolio weights with price data
        weights = portfolio.weights
        prices = price_data.get_close_prices()
        
        # Ensure we have overlapping dates
        common_dates = weights.index.intersection(prices.index)
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between portfolio and price data")
        
        weights = weights.loc[common_dates]
        prices = prices.loc[common_dates]
        
        logger.info(f"Simulating {len(common_dates)} periods")
        
        # Calculate returns for each asset
        asset_returns = prices.pct_change()
        
        # Initialize tracking
        equity = self.config.initial_capital
        equity_curve = []
        portfolio_returns = []
        trades_list = []
        
        previous_weights = pd.Series(0, index=weights.columns)
        
        for date in common_dates:
            target_weights = weights.loc[date]
            
            # Calculate turnover
            position_changes = (target_weights - previous_weights).abs()
            turnover = position_changes.sum() / 2
            
            # Apply transaction costs
            if turnover > 0:
                cost = equity * turnover * (
                    self.config.transaction_cost + self.config.slippage
                )
                equity -= cost
                
                # Record trades
                for symbol in position_changes[position_changes > 0].index:
                    trades_list.append({
                        'date': date,
                        'symbol': symbol,
                        'change': position_changes[symbol],
                        'cost': cost * (position_changes[symbol] / position_changes.sum()),
                    })
            
            # Calculate portfolio return
            if date in asset_returns.index:
                period_returns = asset_returns.loc[date]
                # Weight by previous positions (returns realized on held positions)
                portfolio_return = (previous_weights * period_returns).sum()
                
                # Update equity
                if self.config.compound_returns:
                    equity *= (1 + portfolio_return)
                else:
                    equity += self.config.initial_capital * portfolio_return
                
                portfolio_returns.append(portfolio_return)
            else:
                portfolio_returns.append(0.0)
            
            equity_curve.append(equity)
            previous_weights = target_weights.copy()
        
        # Build result
        equity_series = pd.Series(equity_curve, index=common_dates)
        returns_series = pd.Series(portfolio_returns, index=common_dates)
        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
        
        result = BacktestResult(
            equity_curve=equity_series,
            returns=returns_series,
            positions=weights,
            trades=trades_df,
            metadata={
                'initial_capital': self.config.initial_capital,
                'transaction_cost': self.config.transaction_cost,
                'slippage': self.config.slippage,
                'periods': len(common_dates),
            }
        )
        
        # Calculate metrics
        result.metrics = {
            'total_return': result.total_return,
            'cagr': result.cagr,
            'volatility': result.volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'total_trades': len(trades_df),
            'avg_turnover': position_changes.sum().mean() / 2 if len(common_dates) > 0 else 0,
        }
        
        logger.info(f"Backtest complete: Return={result.total_return:.2%}, Sharpe={result.sharpe_ratio:.2f}")
        
        return result
    
    def run_strategy(
        self,
        strategy,  # Strategy instance
        price_data: PriceData,
        **kwargs: Any
    ) -> BacktestResult:
        """
        Run backtest on a strategy.
        
        Args:
            strategy: Strategy instance
            price_data: Historical price data
            **kwargs: Additional arguments
        
        Returns:
            BacktestResult
        """
        logger.info(f"Running backtest for strategy: {strategy.name}")
        
        # Generate portfolio from strategy
        portfolio = strategy.run(price_data, **kwargs)
        
        # Run backtest
        return self.run(portfolio, price_data, **kwargs)

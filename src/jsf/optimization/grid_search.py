"""
Grid search optimization for strategy parameters.

This module provides tools for systematic parameter optimization using
grid search across multiple parameter combinations.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from itertools import product
import pandas as pd
import numpy as np
from tqdm import tqdm

from jsf.data.base import PriceData
from jsf.strategies.base import Strategy
from jsf.simulation.backtest import BacktestEngine, BacktestConfig, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class ParameterGrid:
    """
    Parameter grid for optimization.
    
    Attributes:
        parameters: Dictionary mapping parameter names to lists of values
    """
    parameters: Dict[str, List[Any]]
    
    def __iter__(self):
        """Iterate over all parameter combinations."""
        keys = list(self.parameters.keys())
        values = [self.parameters[k] for k in keys]
        
        for combination in product(*values):
            yield dict(zip(keys, combination))
    
    def __len__(self) -> int:
        """Get total number of combinations."""
        count = 1
        for values in self.parameters.values():
            count *= len(values)
        return count


@dataclass
class OptimizationResult:
    """
    Results from parameter optimization.
    
    Attributes:
        best_params: Best parameter combination
        best_score: Best score achieved
        best_result: Full backtest result for best parameters
        all_results: All backtest results
        summary: Summary DataFrame of all results
    """
    best_params: Dict[str, Any]
    best_score: float
    best_result: BacktestResult
    all_results: List[Tuple[Dict[str, Any], BacktestResult]]
    summary: pd.DataFrame
    
    def __str__(self) -> str:
        """String representation."""
        lines = [
            "Optimization Results:",
            f"  Best Score: {self.best_score:.4f}",
            f"  Best Parameters: {self.best_params}",
            f"  Total Combinations Tested: {len(self.all_results)}",
        ]
        return "\n".join(lines)


class GridSearchOptimizer:
    """
    Grid search optimizer for strategy parameters.
    
    This optimizer systematically tests all combinations of parameters
    from a predefined grid and selects the best based on a metric.
    
    Args:
        backtest_config: Configuration for backtesting
        metric: Metric to optimize ('sharpe_ratio', 'total_return', 'calmar_ratio', etc.)
        maximize: Whether to maximize (True) or minimize (False) the metric
    """
    
    def __init__(
        self,
        backtest_config: Optional[BacktestConfig] = None,
        metric: str = "sharpe_ratio",
        maximize: bool = True,
    ):
        """Initialize grid search optimizer."""
        self.backtest_config = backtest_config or BacktestConfig()
        self.metric = metric
        self.maximize = maximize
        self.engine = BacktestEngine(self.backtest_config)
        
        logger.info(f"Initialized GridSearchOptimizer: metric={metric}, "
                   f"maximize={maximize}")
    
    def optimize(
        self,
        strategy_class: type,
        param_grid: ParameterGrid,
        data: PriceData,
        fixed_params: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> OptimizationResult:
        """
        Run grid search optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            param_grid: Grid of parameters to search
            data: Price data for backtesting
            fixed_params: Fixed parameters (not optimized)
            verbose: Whether to show progress bar
            
        Returns:
            OptimizationResult with best parameters and all results
        """
        fixed_params = fixed_params or {}
        all_results = []
        best_score = float('-inf') if self.maximize else float('inf')
        best_params = None
        best_result = None
        
        total_combinations = len(param_grid)
        logger.info(f"Starting grid search over {total_combinations} combinations")
        
        # Iterate over parameter combinations
        iterator = tqdm(param_grid, desc="Grid Search", disable=not verbose)
        
        for params in iterator:
            # Merge with fixed parameters
            full_params = {**fixed_params, **params}
            
            try:
                # Create strategy with these parameters
                strategy = strategy_class(**full_params)
                
                # Run backtest
                result = self.engine.run_strategy(strategy, data)
                
                # Get metric value
                score = self._get_metric_value(result)
                
                # Store result
                all_results.append((params.copy(), result))
                
                # Check if best
                is_better = (
                    (self.maximize and score > best_score) or
                    (not self.maximize and score < best_score)
                )
                
                if is_better:
                    best_score = score
                    best_params = params.copy()
                    best_result = result
                    
                    if verbose:
                        iterator.set_postfix({
                            self.metric: f"{score:.4f}",
                            'params': str(params)
                        })
                        
            except Exception as e:
                logger.warning(f"Failed to test {params}: {e}")
                continue
        
        logger.info(f"Grid search complete. Best {self.metric}: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Create summary DataFrame
        summary = self._create_summary(all_results)
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_result=best_result,
            all_results=all_results,
            summary=summary,
        )
    
    def _get_metric_value(self, result: BacktestResult) -> float:
        """Get metric value from backtest result."""
        # Check properties first
        if hasattr(result, self.metric):
            return getattr(result, self.metric)
        
        # Check metrics dict
        if result.metrics and self.metric in result.metrics:
            return result.metrics[self.metric]
        
        raise ValueError(f"Metric '{self.metric}' not found in backtest result")
    
    def _create_summary(
        self,
        all_results: List[Tuple[Dict[str, Any], BacktestResult]]
    ) -> pd.DataFrame:
        """Create summary DataFrame from all results."""
        rows = []
        
        for params, result in all_results:
            row = params.copy()
            row.update({
                'total_return': result.total_return,
                'cagr': result.cagr,
                'volatility': result.volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'num_trades': len(result.trades),
            })
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by metric
        df = df.sort_values(self.metric, ascending=not self.maximize)
        
        return df


def optimize_strategy(
    strategy_class: type,
    param_grid: Dict[str, List[Any]],
    data: PriceData,
    metric: str = "sharpe_ratio",
    backtest_config: Optional[BacktestConfig] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Convenience function for grid search optimization.
    
    Args:
        strategy_class: Strategy class to optimize
        param_grid: Dictionary mapping parameter names to value lists
        data: Price data for backtesting
        metric: Metric to optimize
        backtest_config: Configuration for backtesting
        fixed_params: Fixed parameters (not optimized)
        verbose: Whether to show progress
        
    Returns:
        OptimizationResult with best parameters and all results
        
    Example:
        >>> from jsf.strategies import MomentumStrategy
        >>> from jsf.data import load_data
        >>> from jsf.optimization import optimize_strategy
        >>> 
        >>> data = load_data(source='synthetic', symbols=['AAPL', 'GOOGL'])
        >>> result = optimize_strategy(
        ...     MomentumStrategy,
        ...     param_grid={'lookback': [30, 60, 90], 'long_only': [True, False]},
        ...     data=data,
        ...     metric='sharpe_ratio'
        ... )
        >>> print(f"Best parameters: {result.best_params}")
        >>> print(f"Best Sharpe: {result.best_score:.2f}")
    """
    grid = ParameterGrid(param_grid)
    optimizer = GridSearchOptimizer(
        backtest_config=backtest_config,
        metric=metric,
        maximize=True,
    )
    
    return optimizer.optimize(
        strategy_class=strategy_class,
        param_grid=grid,
        data=data,
        fixed_params=fixed_params,
        verbose=verbose,
    )

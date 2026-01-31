"""
Walk-forward analysis and out-of-sample validation.

This module provides robust strategy validation using walk-forward analysis
to prevent overfitting and ensure strategies work on unseen data.

Walk-forward analysis works by:
1. Splitting data into in-sample (IS) and out-of-sample (OOS) periods
2. Optimizing parameters on IS period
3. Testing on OOS period
4. Rolling forward and repeating
5. Aggregating OOS results to measure true performance

This is the gold standard for validating quantitative trading strategies.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

from jsf.data.base import PriceData
from jsf.strategies.base import Strategy
from jsf.simulation.backtest import BacktestEngine, BacktestConfig, BacktestResult
from jsf.optimization.grid_search import (
    GridSearchOptimizer,
    ParameterGrid,
    OptimizationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """
    A single walk-forward window with IS and OOS periods.
    
    Attributes:
        window_id: Sequential window number
        is_start: In-sample period start date
        is_end: In-sample period end date
        oos_start: Out-of-sample period start date
        oos_end: Out-of-sample period end date
        best_params: Best parameters found in IS optimization
        is_result: Backtest result for IS period
        oos_result: Backtest result for OOS period
        is_score: IS optimization metric score
        oos_score: OOS metric score
    """
    window_id: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    best_params: Optional[Dict[str, Any]] = None
    is_result: Optional[BacktestResult] = None
    oos_result: Optional[BacktestResult] = None
    is_score: Optional[float] = None
    oos_score: Optional[float] = None
    
    @property
    def is_days(self) -> int:
        """Number of days in IS period."""
        return (self.is_end - self.is_start).days
    
    @property
    def oos_days(self) -> int:
        """Number of days in OOS period."""
        return (self.oos_end - self.oos_start).days
    
    @property
    def efficiency_ratio(self) -> Optional[float]:
        """
        OOS score / IS score ratio.
        
        A ratio close to 1.0 indicates consistent performance.
        A ratio << 1.0 indicates overfitting.
        """
        if self.is_score and self.oos_score and self.is_score != 0:
            return self.oos_score / self.is_score
        return None
    
    def __repr__(self) -> str:
        is_score_str = f"{self.is_score:.3f}" if self.is_score is not None else "N/A"
        oos_score_str = f"{self.oos_score:.3f}" if self.oos_score is not None else "N/A"
        return (
            f"WalkForwardWindow(id={self.window_id}, "
            f"IS={self.is_start.date()}→{self.is_end.date()}, "
            f"OOS={self.oos_start.date()}→{self.oos_end.date()}, "
            f"IS_score={is_score_str}, "
            f"OOS_score={oos_score_str})"
        )


@dataclass
class WalkForwardResult:
    """
    Complete results from walk-forward analysis.
    
    Attributes:
        windows: List of individual walk-forward windows
        combined_oos_result: Combined OOS equity curve and metrics
        metric: Optimization metric used
        strategy_class: Strategy class tested
        param_grid: Parameters searched
        summary: Summary statistics DataFrame
    """
    windows: List[WalkForwardWindow]
    combined_oos_equity: pd.Series
    combined_oos_returns: pd.Series
    metric: str
    strategy_class: type
    param_grid: Dict[str, List[Any]]
    
    @property
    def n_windows(self) -> int:
        """Number of walk-forward windows."""
        return len(self.windows)
    
    @property
    def avg_is_score(self) -> float:
        """Average in-sample score across all windows."""
        scores = [w.is_score for w in self.windows if w.is_score is not None]
        return np.mean(scores) if scores else 0.0
    
    @property
    def avg_oos_score(self) -> float:
        """Average out-of-sample score across all windows."""
        scores = [w.oos_score for w in self.windows if w.oos_score is not None]
        return np.mean(scores) if scores else 0.0
    
    @property
    def oos_total_return(self) -> float:
        """Total return from combined OOS equity curve."""
        if len(self.combined_oos_equity) == 0:
            return 0.0
        return (self.combined_oos_equity.iloc[-1] / self.combined_oos_equity.iloc[0]) - 1
    
    @property
    def oos_sharpe_ratio(self) -> float:
        """Sharpe ratio from combined OOS returns."""
        if len(self.combined_oos_returns) == 0:
            return 0.0
        if self.combined_oos_returns.std() == 0:
            return 0.0
        return (self.combined_oos_returns.mean() / self.combined_oos_returns.std()) * np.sqrt(252)
    
    @property
    def oos_max_drawdown(self) -> float:
        """Max drawdown from combined OOS equity curve."""
        if len(self.combined_oos_equity) == 0:
            return 0.0
        cumulative = self.combined_oos_equity / self.combined_oos_equity.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @property
    def efficiency_ratio(self) -> float:
        """
        Overall efficiency ratio (OOS / IS performance).
        
        Interpretation:
        - > 0.8: Excellent - strategy generalizes well
        - 0.5 - 0.8: Good - acceptable generalization
        - 0.3 - 0.5: Moderate - some overfitting
        - < 0.3: Poor - significant overfitting detected
        """
        if self.avg_is_score == 0:
            return 0.0
        return self.avg_oos_score / self.avg_is_score
    
    @property
    def parameter_stability(self) -> float:
        """
        Measure of how consistent optimal parameters are across windows.
        
        Returns value between 0 and 1:
        - 1.0: Same parameters chosen in every window
        - 0.0: Different parameters in every window
        """
        if len(self.windows) < 2:
            return 1.0
        
        # Count how often each parameter value was chosen
        param_counts = {}
        for window in self.windows:
            if window.best_params:
                for param, value in window.best_params.items():
                    if param not in param_counts:
                        param_counts[param] = {}
                    value_str = str(value)
                    param_counts[param][value_str] = param_counts[param].get(value_str, 0) + 1
        
        # Calculate stability score (mode frequency / n_windows)
        if not param_counts:
            return 0.0
        
        stabilities = []
        for param, value_counts in param_counts.items():
            max_count = max(value_counts.values())
            stabilities.append(max_count / len(self.windows))
        
        return np.mean(stabilities)
    
    @property
    def is_overfitted(self) -> bool:
        """
        Check if strategy shows signs of overfitting.
        
        Returns True if efficiency ratio < 0.5 or parameter stability < 0.3
        """
        return self.efficiency_ratio < 0.5 or self.parameter_stability < 0.3
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all windows."""
        rows = []
        for w in self.windows:
            rows.append({
                'window': w.window_id,
                'is_start': w.is_start.date(),
                'is_end': w.is_end.date(),
                'oos_start': w.oos_start.date(),
                'oos_end': w.oos_end.date(),
                'is_days': w.is_days,
                'oos_days': w.oos_days,
                'is_score': w.is_score,
                'oos_score': w.oos_score,
                'efficiency': w.efficiency_ratio,
                'best_params': str(w.best_params),
            })
        return pd.DataFrame(rows)
    
    def get_parameter_summary(self) -> pd.DataFrame:
        """Get summary of parameters chosen in each window."""
        rows = []
        for w in self.windows:
            if w.best_params:
                row = {'window': w.window_id}
                row.update(w.best_params)
                rows.append(row)
        return pd.DataFrame(rows)
    
    def __str__(self) -> str:
        """String representation with key metrics."""
        lines = [
            "=" * 60,
            "WALK-FORWARD ANALYSIS RESULTS",
            "=" * 60,
            f"Strategy: {self.strategy_class.__name__}",
            f"Metric: {self.metric}",
            f"Windows: {self.n_windows}",
            "",
            "IN-SAMPLE PERFORMANCE:",
            f"  Average {self.metric}: {self.avg_is_score:.4f}",
            "",
            "OUT-OF-SAMPLE PERFORMANCE:",
            f"  Average {self.metric}: {self.avg_oos_score:.4f}",
            f"  Total Return: {self.oos_total_return:.2%}",
            f"  Sharpe Ratio: {self.oos_sharpe_ratio:.2f}",
            f"  Max Drawdown: {self.oos_max_drawdown:.2%}",
            "",
            "VALIDATION METRICS:",
            f"  Efficiency Ratio: {self.efficiency_ratio:.2f} {'✓' if self.efficiency_ratio >= 0.5 else '✗'}",
            f"  Parameter Stability: {self.parameter_stability:.2f} {'✓' if self.parameter_stability >= 0.3 else '✗'}",
            f"  Overfitting Detected: {'YES ⚠️' if self.is_overfitted else 'NO ✓'}",
            "=" * 60,
        ]
        return "\n".join(lines)


class WalkForwardOptimizer:
    """
    Walk-forward analysis optimizer.
    
    Implements robust out-of-sample validation by:
    1. Dividing data into rolling/expanding windows
    2. Optimizing on in-sample period
    3. Testing on out-of-sample period
    4. Aggregating results to measure true strategy performance
    
    Args:
        is_days: Number of days in each in-sample period (default: 252 = 1 year)
        oos_days: Number of days in each out-of-sample period (default: 63 = 1 quarter)
        step_days: Number of days to step forward (default: oos_days for non-overlapping)
        expanding: If True, use expanding window (all prior data as IS)
        backtest_config: Configuration for backtesting
        metric: Metric to optimize
        min_is_days: Minimum IS days required (for expanding window)
    
    Example:
        >>> from jsf.optimization import WalkForwardOptimizer
        >>> from jsf.strategies import MomentumStrategy
        >>> 
        >>> optimizer = WalkForwardOptimizer(
        ...     is_days=252,  # 1 year in-sample
        ...     oos_days=63,  # 1 quarter out-of-sample
        ... )
        >>> 
        >>> result = optimizer.optimize(
        ...     strategy_class=MomentumStrategy,
        ...     param_grid={'lookback': [30, 60, 90]},
        ...     data=data,
        ... )
        >>> 
        >>> print(result)
        >>> print(f"Is overfitted? {result.is_overfitted}")
    """
    
    def __init__(
        self,
        is_days: int = 252,
        oos_days: int = 63,
        step_days: Optional[int] = None,
        expanding: bool = False,
        backtest_config: Optional[BacktestConfig] = None,
        metric: str = "sharpe_ratio",
        min_is_days: int = 126,
    ):
        """Initialize walk-forward optimizer."""
        self.is_days = is_days
        self.oos_days = oos_days
        self.step_days = step_days or oos_days  # Default: non-overlapping OOS
        self.expanding = expanding
        self.backtest_config = backtest_config or BacktestConfig()
        self.metric = metric
        self.min_is_days = min_is_days
        
        logger.info(
            f"Initialized WalkForwardOptimizer: "
            f"IS={is_days}d, OOS={oos_days}d, step={self.step_days}d, "
            f"expanding={expanding}, metric={metric}"
        )
    
    def _generate_windows(
        self,
        data: PriceData,
    ) -> List[WalkForwardWindow]:
        """
        Generate walk-forward windows from data.
        
        Args:
            data: Price data to split into windows
            
        Returns:
            List of WalkForwardWindow objects
        """
        dates = data.dates
        total_days = len(dates)
        
        logger.info(f"Generating walk-forward windows from {total_days} days of data")
        
        windows = []
        window_id = 0
        
        if self.expanding:
            # Expanding window: IS grows over time
            current_oos_start_idx = self.min_is_days
            
            while current_oos_start_idx + self.oos_days <= total_days:
                is_start = dates[0]
                is_end = dates[current_oos_start_idx - 1]
                oos_start = dates[current_oos_start_idx]
                oos_end_idx = min(current_oos_start_idx + self.oos_days - 1, total_days - 1)
                oos_end = dates[oos_end_idx]
                
                windows.append(WalkForwardWindow(
                    window_id=window_id,
                    is_start=is_start,
                    is_end=is_end,
                    oos_start=oos_start,
                    oos_end=oos_end,
                ))
                
                window_id += 1
                current_oos_start_idx += self.step_days
        else:
            # Rolling window: fixed IS size
            current_is_start_idx = 0
            
            while current_is_start_idx + self.is_days + self.oos_days <= total_days:
                is_start = dates[current_is_start_idx]
                is_end = dates[current_is_start_idx + self.is_days - 1]
                oos_start = dates[current_is_start_idx + self.is_days]
                oos_end_idx = min(current_is_start_idx + self.is_days + self.oos_days - 1, total_days - 1)
                oos_end = dates[oos_end_idx]
                
                windows.append(WalkForwardWindow(
                    window_id=window_id,
                    is_start=is_start,
                    is_end=is_end,
                    oos_start=oos_start,
                    oos_end=oos_end,
                ))
                
                window_id += 1
                current_is_start_idx += self.step_days
        
        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows
    
    def _filter_data_by_dates(
        self,
        data: PriceData,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> PriceData:
        """Filter PriceData to a specific date range."""
        mask = (data.dates >= start) & (data.dates <= end)
        valid_dates = data.dates[mask]
        
        if isinstance(data.data.index, pd.MultiIndex):
            # MultiIndex: filter by date level
            filtered = data.data[
                data.data.index.get_level_values(0).isin(valid_dates)
            ]
        else:
            # DatetimeIndex
            filtered = data.data[
                data.data.index.isin(valid_dates)
            ]
        
        return PriceData(filtered)
    
    def _get_metric_value(self, result: BacktestResult) -> float:
        """Get metric value from backtest result."""
        if hasattr(result, self.metric):
            return getattr(result, self.metric)
        if result.metrics and self.metric in result.metrics:
            return result.metrics[self.metric]
        raise ValueError(f"Metric '{self.metric}' not found in backtest result")
    
    def optimize(
        self,
        strategy_class: type,
        param_grid: Dict[str, List[Any]],
        data: PriceData,
        fixed_params: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            param_grid: Dictionary mapping parameter names to value lists
            data: Price data for backtesting
            fixed_params: Fixed parameters (not optimized)
            verbose: Whether to show progress
            
        Returns:
            WalkForwardResult with IS/OOS performance analysis
        """
        fixed_params = fixed_params or {}
        
        # Generate windows
        windows = self._generate_windows(data)
        
        if len(windows) == 0:
            raise ValueError(
                f"Insufficient data for walk-forward analysis. "
                f"Need at least {self.is_days + self.oos_days} days, "
                f"have {len(data.dates)} days."
            )
        
        logger.info(f"Starting walk-forward analysis with {len(windows)} windows")
        
        # Process each window
        all_oos_equity = []
        all_oos_returns = []
        
        iterator = tqdm(windows, desc="Walk-Forward", disable=not verbose)
        
        for window in iterator:
            try:
                # Filter data for IS period
                is_data = self._filter_data_by_dates(data, window.is_start, window.is_end)
                
                # Filter data for OOS period
                oos_data = self._filter_data_by_dates(data, window.oos_start, window.oos_end)
                
                # Optimize on IS data
                grid_optimizer = GridSearchOptimizer(
                    backtest_config=self.backtest_config,
                    metric=self.metric,
                    maximize=True,
                )
                
                is_opt_result = grid_optimizer.optimize(
                    strategy_class=strategy_class,
                    param_grid=ParameterGrid(param_grid),
                    data=is_data,
                    fixed_params=fixed_params,
                    verbose=False,  # Suppress inner progress
                )
                
                # Store IS results
                window.best_params = is_opt_result.best_params
                window.is_result = is_opt_result.best_result
                window.is_score = is_opt_result.best_score
                
                # Test best params on OOS data
                full_params = {**fixed_params, **window.best_params}
                strategy = strategy_class(**full_params)
                
                engine = BacktestEngine(self.backtest_config)
                oos_result = engine.run_strategy(strategy, oos_data)
                
                # Store OOS results
                window.oos_result = oos_result
                window.oos_score = self._get_metric_value(oos_result)
                
                # Collect OOS equity and returns for combined analysis
                all_oos_equity.append(oos_result.equity_curve)
                all_oos_returns.append(oos_result.returns)
                
                if verbose:
                    iterator.set_postfix({
                        'IS': f"{window.is_score:.3f}",
                        'OOS': f"{window.oos_score:.3f}",
                        'eff': f"{window.efficiency_ratio:.2f}" if window.efficiency_ratio else "N/A",
                    })
                
            except Exception as e:
                logger.warning(f"Window {window.window_id} failed: {e}")
                continue
        
        # Combine OOS results
        combined_oos_equity = self._combine_equity_curves(all_oos_equity)
        combined_oos_returns = pd.concat(all_oos_returns) if all_oos_returns else pd.Series()
        
        result = WalkForwardResult(
            windows=windows,
            combined_oos_equity=combined_oos_equity,
            combined_oos_returns=combined_oos_returns,
            metric=self.metric,
            strategy_class=strategy_class,
            param_grid=param_grid,
        )
        
        logger.info(f"Walk-forward complete: efficiency={result.efficiency_ratio:.2f}, "
                   f"overfitted={result.is_overfitted}")
        
        return result
    
    def _combine_equity_curves(
        self,
        equity_curves: List[pd.Series],
    ) -> pd.Series:
        """
        Combine OOS equity curves into a single continuous curve.
        
        Normalizes each curve to start where the previous ended.
        """
        if not equity_curves:
            return pd.Series()
        
        combined = []
        scale_factor = 1.0
        
        for i, curve in enumerate(equity_curves):
            if len(curve) == 0:
                continue
            
            # Scale to continue from previous equity
            scaled_curve = curve * scale_factor / curve.iloc[0]
            combined.append(scaled_curve)
            
            # Update scale factor for next curve
            scale_factor = scaled_curve.iloc[-1]
        
        if not combined:
            return pd.Series()
        
        return pd.concat(combined)


def walk_forward_analysis(
    strategy_class: type,
    param_grid: Dict[str, List[Any]],
    data: PriceData,
    is_days: int = 252,
    oos_days: int = 63,
    metric: str = "sharpe_ratio",
    backtest_config: Optional[BacktestConfig] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    expanding: bool = False,
    verbose: bool = True,
) -> WalkForwardResult:
    """
    Convenience function for walk-forward analysis.
    
    Args:
        strategy_class: Strategy class to optimize
        param_grid: Dictionary mapping parameter names to value lists
        data: Price data for backtesting
        is_days: Number of days in each in-sample period
        oos_days: Number of days in each out-of-sample period
        metric: Metric to optimize
        backtest_config: Configuration for backtesting
        fixed_params: Fixed parameters (not optimized)
        expanding: Use expanding window instead of rolling
        verbose: Whether to show progress
        
    Returns:
        WalkForwardResult with IS/OOS performance analysis
        
    Example:
        >>> from jsf.optimization import walk_forward_analysis
        >>> from jsf.strategies import MomentumStrategy
        >>> from jsf.data import load_data
        >>> 
        >>> data = load_data(source='synthetic', symbols=['AAPL', 'GOOGL'])
        >>> 
        >>> result = walk_forward_analysis(
        ...     strategy_class=MomentumStrategy,
        ...     param_grid={'lookback': [30, 60, 90]},
        ...     data=data,
        ...     is_days=252,  # 1 year IS
        ...     oos_days=63,  # 1 quarter OOS
        ... )
        >>> 
        >>> print(result)
        >>> 
        >>> if result.is_overfitted:
        ...     print("WARNING: Strategy shows signs of overfitting!")
        >>> else:
        ...     print("Strategy generalizes well to out-of-sample data.")
    """
    optimizer = WalkForwardOptimizer(
        is_days=is_days,
        oos_days=oos_days,
        expanding=expanding,
        backtest_config=backtest_config,
        metric=metric,
    )
    
    return optimizer.optimize(
        strategy_class=strategy_class,
        param_grid=param_grid,
        data=data,
        fixed_params=fixed_params,
        verbose=verbose,
    )

"""Metrics Calculator for Dashboard

Calculates performance and risk metrics from historical data.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import math

import numpy as np
import pandas as pd

from .models import (
    RiskMetrics,
    PerformanceMetrics,
    PortfolioSnapshot,
)


# Risk-free rate assumption (annual)
DEFAULT_RISK_FREE_RATE = 0.05  # 5%

# Trading days per year
TRADING_DAYS_PER_YEAR = 252


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate simple returns from price series.
    
    Args:
        prices: Price series
        
    Returns:
        Returns series
    """
    return prices.pct_change().dropna()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns from price series.
    
    Args:
        prices: Price series
        
    Returns:
        Log returns series
    """
    return np.log(prices / prices.shift(1)).dropna()


def calculate_drawdown(equity: pd.Series) -> Tuple[pd.Series, float, float]:
    """Calculate drawdown series and metrics.
    
    Args:
        equity: Equity time series
        
    Returns:
        Tuple of (drawdown_series, max_drawdown, current_drawdown)
    """
    if equity.empty or len(equity) < 2:
        return pd.Series(dtype=float), 0.0, 0.0
    
    # Running maximum
    running_max = equity.expanding().max()
    
    # Drawdown as percentage
    drawdown = (equity - running_max) / running_max * 100
    
    max_drawdown = abs(drawdown.min())
    current_drawdown = abs(drawdown.iloc[-1])
    
    return drawdown, max_drawdown, current_drawdown


def calculate_volatility(
    returns: pd.Series,
    annualize: bool = True,
    window: Optional[int] = None,
) -> Union[float, pd.Series]:
    """Calculate volatility (standard deviation of returns).
    
    Args:
        returns: Returns series
        annualize: Whether to annualize
        window: Rolling window size (returns series if set)
        
    Returns:
        Volatility value or series
    """
    if returns.empty:
        return 0.0
    
    if window:
        vol = returns.rolling(window).std()
        if annualize:
            vol *= np.sqrt(TRADING_DAYS_PER_YEAR)
        return vol
    
    vol = returns.std()
    if annualize:
        vol *= np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return float(vol)


def calculate_sharpe(
    returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    annualize: bool = True,
) -> float:
    """Calculate Sharpe ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        annualize: Whether returns are daily (need annualization)
        
    Returns:
        Sharpe ratio
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    
    # Daily risk-free rate
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    
    # Excess returns
    excess_returns = returns - daily_rf
    
    mean_excess = excess_returns.mean()
    std = excess_returns.std()
    
    if std == 0 or np.isnan(std):
        return 0.0
    
    sharpe = mean_excess / std
    
    if annualize:
        sharpe *= np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return float(sharpe)


def calculate_sortino(
    returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    annualize: bool = True,
) -> float:
    """Calculate Sortino ratio (downside risk adjusted).
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        annualize: Whether returns are daily
        
    Returns:
        Sortino ratio
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    
    # Daily risk-free rate
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    
    # Excess returns
    excess_returns = returns - daily_rf
    
    # Downside returns only
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    downside_std = downside_returns.std()
    
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    
    sortino = excess_returns.mean() / downside_std
    
    if annualize:
        sortino *= np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return float(sortino)


def calculate_calmar(
    returns: pd.Series,
    max_drawdown: float,
    annualize: bool = True,
) -> float:
    """Calculate Calmar ratio (return / max drawdown).
    
    Args:
        returns: Returns series
        max_drawdown: Maximum drawdown percentage
        annualize: Whether to annualize returns
        
    Returns:
        Calmar ratio
    """
    if returns.empty or max_drawdown == 0:
        return 0.0
    
    # Total return
    total_return = (1 + returns).prod() - 1
    
    if annualize and len(returns) > 0:
        # Annualize based on number of periods
        years = len(returns) / TRADING_DAYS_PER_YEAR
        if years > 0:
            total_return = (1 + total_return) ** (1 / years) - 1
    
    calmar = (total_return * 100) / max_drawdown
    
    return float(calmar)


def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Calculate Value at Risk.
    
    Args:
        returns: Returns series
        confidence: Confidence level (e.g., 0.95 for 95%)
        method: 'historical' or 'parametric'
        
    Returns:
        VaR as a positive percentage
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    
    if method == "historical":
        # Historical VaR - percentile of returns
        var = np.percentile(returns, (1 - confidence) * 100)
    else:
        # Parametric VaR - assumes normal distribution
        mean = returns.mean()
        std = returns.std()
        z_score = {0.95: 1.645, 0.99: 2.326}.get(confidence, 1.645)
        var = mean - z_score * std
    
    return abs(float(var)) * 100  # Return as positive percentage


def calculate_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall).
    
    Args:
        returns: Returns series
        confidence: Confidence level
        
    Returns:
        CVaR as a positive percentage
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    
    # Get the VaR threshold
    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    
    # Average of returns below VaR
    tail_returns = returns[returns <= var_threshold]
    
    if len(tail_returns) == 0:
        return calculate_var(returns, confidence)
    
    cvar = tail_returns.mean()
    
    return abs(float(cvar)) * 100


def calculate_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate beta relative to benchmark.
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Beta coefficient
    """
    if returns.empty or benchmark_returns.empty:
        return 1.0
    
    # Align series
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    
    if len(aligned) < 2:
        return 1.0
    
    port_ret = aligned.iloc[:, 0]
    bench_ret = aligned.iloc[:, 1]
    
    covariance = port_ret.cov(bench_ret)
    variance = bench_ret.var()
    
    if variance == 0 or np.isnan(variance):
        return 1.0
    
    return float(covariance / variance)


def calculate_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    beta: float,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """Calculate Jensen's alpha.
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        beta: Portfolio beta
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Alpha (annualized)
    """
    if returns.empty or benchmark_returns.empty:
        return 0.0
    
    # Daily risk-free rate
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    
    # Annualized returns
    port_annual = returns.mean() * TRADING_DAYS_PER_YEAR
    bench_annual = benchmark_returns.mean() * TRADING_DAYS_PER_YEAR
    
    # CAPM expected return
    expected = risk_free_rate + beta * (bench_annual - risk_free_rate)
    
    # Alpha = actual - expected
    alpha = port_annual - expected
    
    return float(alpha)


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate information ratio.
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Information ratio
    """
    if returns.empty or benchmark_returns.empty:
        return 0.0
    
    # Align series
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    
    if len(aligned) < 2:
        return 0.0
    
    # Active return (tracking difference)
    active_return = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    
    # Tracking error
    tracking_error = active_return.std()
    
    if tracking_error == 0 or np.isnan(tracking_error):
        return 0.0
    
    ir = (active_return.mean() / tracking_error) * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return float(ir)


def calculate_win_rate(returns: pd.Series) -> float:
    """Calculate win rate (percentage of positive returns).
    
    Args:
        returns: Returns series
        
    Returns:
        Win rate as percentage
    """
    if returns.empty:
        return 0.0
    
    wins = (returns > 0).sum()
    total = len(returns)
    
    return (wins / total) * 100


def calculate_profit_factor(returns: pd.Series) -> float:
    """Calculate profit factor (gross profit / gross loss).
    
    Args:
        returns: Returns series
        
    Returns:
        Profit factor
    """
    if returns.empty:
        return 0.0
    
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    
    return float(gains / losses)


def calculate_return_for_period(
    equity: pd.Series,
    days: int,
) -> float:
    """Calculate return for a specific period.
    
    Args:
        equity: Equity series
        days: Number of days to look back
        
    Returns:
        Return as percentage
    """
    if equity.empty or len(equity) < 2:
        return 0.0
    
    # Find data point closest to N days ago
    target_date = equity.index[-1] - timedelta(days=days)
    
    # Get value at or before target date
    mask = equity.index <= target_date
    if not mask.any():
        start_value = equity.iloc[0]
    else:
        start_value = equity[mask].iloc[-1]
    
    end_value = equity.iloc[-1]
    
    if start_value == 0:
        return 0.0
    
    return ((end_value - start_value) / start_value) * 100


class MetricsCalculator:
    """Calculates comprehensive portfolio metrics.
    
    Provides methods to compute risk and performance metrics
    from equity and returns data.
    
    Attributes:
        risk_free_rate: Annual risk-free rate
        benchmark: Optional benchmark returns for relative metrics
    """
    
    def __init__(
        self,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        benchmark_returns: Optional[pd.Series] = None,
    ):
        """Initialize calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate
            benchmark_returns: Optional benchmark returns series
        """
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns
    
    def calculate_risk_metrics(
        self,
        equity: pd.Series,
        positions: List[Dict] = None,
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics.
        
        Args:
            equity: Equity time series
            positions: Optional list of position data
            
        Returns:
            RiskMetrics object
        """
        timestamp = datetime.now()
        
        # Calculate returns
        returns = calculate_returns(equity)
        
        # Drawdown metrics
        _, max_dd, current_dd = calculate_drawdown(equity)
        
        # VaR
        var_95 = calculate_var(returns, 0.95)
        var_99 = calculate_var(returns, 0.99)
        
        # Volatility
        volatility = calculate_volatility(returns)
        
        # Risk-adjusted metrics
        sharpe = calculate_sharpe(returns, self.risk_free_rate)
        sortino = calculate_sortino(returns, self.risk_free_rate)
        calmar = calculate_calmar(returns, max_dd) if max_dd > 0 else 0.0
        
        # Benchmark-relative metrics
        beta = None
        correlation = None
        if self.benchmark_returns is not None and not self.benchmark_returns.empty:
            beta = calculate_beta(returns, self.benchmark_returns)
            # Correlation
            aligned = pd.concat([returns, self.benchmark_returns], axis=1).dropna()
            if len(aligned) > 1:
                correlation = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        
        # Position-based metrics
        max_position = 0.0
        gross_exposure = 0.0
        net_exposure = 0.0
        
        if positions:
            weights = [abs(p.get('weight', 0)) for p in positions]
            if weights:
                max_position = max(weights)
                gross_exposure = sum(weights)
                
                # Net exposure (long - short)
                long_weight = sum(p.get('weight', 0) for p in positions 
                                if p.get('side', 'long') == 'long')
                short_weight = sum(abs(p.get('weight', 0)) for p in positions 
                                 if p.get('side') == 'short')
                net_exposure = long_weight - short_weight
        
        return RiskMetrics(
            timestamp=timestamp,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            var_95=var_95,
            var_99=var_99,
            volatility=volatility * 100,  # As percentage
            beta=beta,
            correlation=correlation,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_position_size=max_position,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
        )
    
    def calculate_performance_metrics(
        self,
        equity: pd.Series,
        initial_capital: float,
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics.
        
        Args:
            equity: Equity time series
            initial_capital: Starting capital
            
        Returns:
            PerformanceMetrics object
        """
        timestamp = datetime.now()
        returns = calculate_returns(equity)
        
        # Period returns
        return_1d = calculate_return_for_period(equity, 1)
        return_1w = calculate_return_for_period(equity, 7)
        return_1m = calculate_return_for_period(equity, 30)
        return_3m = calculate_return_for_period(equity, 90)
        return_1y = calculate_return_for_period(equity, 365)
        
        # YTD return
        if not equity.empty:
            year_start = datetime(equity.index[-1].year, 1, 1)
            mask = equity.index >= year_start
            if mask.any():
                ytd_equity = equity[mask]
                return_ytd = ((ytd_equity.iloc[-1] - ytd_equity.iloc[0]) / 
                             ytd_equity.iloc[0] * 100) if len(ytd_equity) > 0 else 0.0
            else:
                return_ytd = 0.0
        else:
            return_ytd = 0.0
        
        # Total return
        if not equity.empty and initial_capital > 0:
            return_total = ((equity.iloc[-1] - initial_capital) / initial_capital) * 100
        else:
            return_total = 0.0
        
        # Daily stats
        if not returns.empty:
            best_day = float(returns.max() * 100)
            worst_day = float(returns.min() * 100)
            win_rate = calculate_win_rate(returns)
            profit_factor = calculate_profit_factor(returns)
            
            # Win/loss averages
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            avg_win = float(wins.mean() * 100) if len(wins) > 0 else 0.0
            avg_loss = float(losses.mean() * 100) if len(losses) > 0 else 0.0
            largest_win = float(wins.max() * 100) if len(wins) > 0 else 0.0
            largest_loss = float(losses.min() * 100) if len(losses) > 0 else 0.0
        else:
            best_day = worst_day = win_rate = profit_factor = 0.0
            avg_win = avg_loss = largest_win = largest_loss = 0.0
        
        return PerformanceMetrics(
            timestamp=timestamp,
            return_1d=return_1d,
            return_1w=return_1w,
            return_1m=return_1m,
            return_3m=return_3m,
            return_ytd=return_ytd,
            return_1y=return_1y,
            return_total=return_total,
            best_day=best_day,
            worst_day=worst_day,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
        )
    
    def get_rolling_metrics(
        self,
        equity: pd.Series,
        window: int = 20,
    ) -> pd.DataFrame:
        """Calculate rolling metrics.
        
        Args:
            equity: Equity series
            window: Rolling window size
            
        Returns:
            DataFrame with rolling metrics
        """
        returns = calculate_returns(equity)
        
        if len(returns) < window:
            return pd.DataFrame()
        
        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        
        # Rolling Sharpe
        daily_rf = self.risk_free_rate / TRADING_DAYS_PER_YEAR
        excess_returns = returns - daily_rf
        rolling_sharpe = (
            excess_returns.rolling(window).mean() / 
            excess_returns.rolling(window).std()
        ) * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # Rolling return
        rolling_return = returns.rolling(window).sum() * 100
        
        return pd.DataFrame({
            'Volatility': rolling_vol,
            'Sharpe': rolling_sharpe,
            'Return': rolling_return,
        }).dropna()
    
    def get_monthly_returns(self, equity: pd.Series) -> pd.DataFrame:
        """Calculate monthly returns matrix.
        
        Args:
            equity: Equity series
            
        Returns:
            DataFrame with years as rows, months as columns
        """
        if equity.empty:
            return pd.DataFrame()
        
        # Resample to month-end
        monthly = equity.resample('M').last()
        monthly_returns = monthly.pct_change() * 100
        
        # Create year-month matrix
        df = monthly_returns.to_frame('return')
        df['year'] = df.index.year
        df['month'] = df.index.month
        
        # Pivot
        pivot = df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        return pivot

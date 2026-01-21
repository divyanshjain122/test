"""Performance metrics calculation.

Comprehensive set of metrics for evaluating trading strategies.
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year
    
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    if excess_returns.std() == 0:
        return 0.0
    return excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (downside deviation).
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    downside_std = downside_returns.std()
    return excess_returns.mean() / downside_std * np.sqrt(periods_per_year)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Return series
    
    Returns:
        Maximum drawdown (negative value)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (CAGR / Max Drawdown).
    
    Args:
        returns: Return series
        periods_per_year: Number of periods per year
    
    Returns:
        Calmar ratio
    """
    years = len(returns) / periods_per_year
    if years <= 0:
        return 0.0
    
    total_return = (1 + returns).prod() - 1
    cagr = (1 + total_return) ** (1 / years) - 1
    max_dd = abs(calculate_max_drawdown(returns))
    
    if max_dd == 0:
        return 0.0
    
    return cagr / max_dd


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate (% of positive periods).
    
    Args:
        returns: Return series
    
    Returns:
        Win rate (0 to 1)
    """
    if len(returns) == 0:
        return 0.0
    return (returns > 0).sum() / len(returns)


def calculate_profit_factor(returns: pd.Series) -> float:
    """
    Calculate profit factor (gross profits / gross losses).
    
    Args:
        returns: Return series
    
    Returns:
        Profit factor
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    
    return gains / losses


def calculate_value_at_risk(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Return series
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        VaR (positive number representing loss)
    """
    return -returns.quantile(1 - confidence_level)


def calculate_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
    
    Args:
        returns: Return series
        confidence_level: Confidence level
    
    Returns:
        CVaR (positive number)
    """
    var = -returns.quantile(1 - confidence_level)
    return -returns[returns <= -var].mean()


def calculate_all_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate all performance metrics.
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate
        periods_per_year: Periods per year
    
    Returns:
        Dictionary of metrics
    """
    total_return = (1 + returns).prod() - 1
    years = len(returns) / periods_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    
    return {
        # Return metrics
        'total_return': total_return,
        'cagr': cagr,
        'mean_return': returns.mean() * periods_per_year,
        'median_return': returns.median() * periods_per_year,
        
        # Risk metrics
        'volatility': returns.std() * np.sqrt(periods_per_year),
        'downside_deviation': returns[returns < 0].std() * np.sqrt(periods_per_year),
        'max_drawdown': calculate_max_drawdown(returns),
        'var_95': calculate_value_at_risk(returns, 0.95),
        'cvar_95': calculate_cvar(returns, 0.95),
        
        # Risk-adjusted returns
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        'calmar_ratio': calculate_calmar_ratio(returns, periods_per_year),
        
        # Trading metrics
        'win_rate': calculate_win_rate(returns),
        'profit_factor': calculate_profit_factor(returns),
        'best_day': returns.max(),
        'worst_day': returns.min(),
        'avg_win': returns[returns > 0].mean() if (returns > 0).any() else 0.0,
        'avg_loss': returns[returns < 0].mean() if (returns < 0).any() else 0.0,
        
        # Other
        'num_periods': len(returns),
        'num_positive': (returns > 0).sum(),
        'num_negative': (returns < 0).sum(),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
    }

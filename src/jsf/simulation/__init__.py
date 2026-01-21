"""Simulation module exports."""

from jsf.simulation.backtest import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
)
from jsf.simulation.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_value_at_risk,
    calculate_cvar,
    calculate_all_metrics,
)

__all__ = [
    # Backtesting
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    # Metrics
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_calmar_ratio",
    "calculate_win_rate",
    "calculate_profit_factor",
    "calculate_value_at_risk",
    "calculate_cvar",
    "calculate_all_metrics",
]

"""
Visualization module for JSF-Core.

This module provides plotting functions for backtesting results,
including equity curves, drawdowns, and performance analysis.
"""

from jsf.visualization.plots import (
    plot_equity_curve,
    plot_drawdown,
    plot_returns_distribution,
    plot_rolling_metrics,
    plot_monthly_returns,
    plot_performance_summary,
)

__all__ = [
    "plot_equity_curve",
    "plot_drawdown",
    "plot_returns_distribution",
    "plot_rolling_metrics",
    "plot_monthly_returns",
    "plot_performance_summary",
]

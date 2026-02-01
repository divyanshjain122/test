"""Dashboard Module

Real-time monitoring dashboard for live trading.

This module provides:
- Data models for dashboard state and snapshots
- Data collectors for gathering metrics from broker/engine
- Metrics calculators for performance and risk analysis
- Streamlit-based dashboard pages

Components:
    - DashboardState: Current state of dashboard session
    - PortfolioSnapshot: Point-in-time portfolio snapshot
    - TradeRecord: Individual trade record for display
    - RiskMetrics: Calculated risk metrics
    - DataCollector: Gathers data from broker and engine
    - MetricsCalculator: Calculates performance and risk metrics
"""

from .models import (
    DashboardState,
    PortfolioSnapshot,
    PositionSnapshot,
    TradeRecord,
    RiskMetrics,
    PerformanceMetrics,
    DashboardConfig,
)

from .collectors import (
    DataCollector,
    SnapshotHistory,
)

from .metrics import (
    MetricsCalculator,
    calculate_drawdown,
    calculate_sharpe,
    calculate_sortino,
    calculate_var,
)

__all__ = [
    # Models
    "DashboardState",
    "PortfolioSnapshot",
    "PositionSnapshot",
    "TradeRecord",
    "RiskMetrics",
    "PerformanceMetrics",
    "DashboardConfig",
    # Collectors
    "DataCollector",
    "SnapshotHistory",
    # Metrics
    "MetricsCalculator",
    "calculate_drawdown",
    "calculate_sharpe",
    "calculate_sortino",
    "calculate_var",
]

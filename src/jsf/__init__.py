"""
JSF-Core: JBAC Strategy Foundry

A production-grade quantitative research engine for building, backtesting,
and optimizing trading strategies.

Version: 0.6.0-dev
Author: JBAC EdTech
License: MIT
"""

__version__ = "0.6.0-dev"
__author__ = "JBAC EdTech"
__license__ = "MIT"

# Configuration exports
from .config import (
    ExperimentConfig,
    StrategyConfig,
    CostConfig,
    RiskConfig,
    OptimizationConfig,
    DataConfig,
    StrategyType,
    UniverseType,
    create_experiment_config,
    get_default_strategy_config,
    quick_config,
)

# Broker exports
from .broker import (
    # Base classes
    Broker,
    # Implementations
    PaperBroker,
    AlpacaBroker,
    # Models
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    Position,
    Fill,
    Trade,
    AccountInfo,
    OrderResult,
    # Exceptions
    BrokerError,
    OrderError,
    ConnectionError,
    InsufficientFundsError,
)

# Live Trading exports
from .live import (
    # Data handlers
    DataHandler,
    PriceUpdate,
    BarData,
    PollingDataHandler,
    RealtimeDataHandler,
    SimulatedDataHandler,
    # Order management
    OrderManager,
    OrderTracker,
    # Engine
    LiveTradingEngine,
    TradingState,
    EngineConfig,
)

# Dashboard exports
from .dashboard import (
    # Models
    DashboardState,
    DashboardConfig,
    PortfolioSnapshot,
    PositionSnapshot,
    TradeRecord,
    RiskMetrics,
    PerformanceMetrics,
    # Collectors
    DataCollector,
    MockDataCollector,
    SnapshotHistory,
    # Metrics
    MetricsCalculator,
    calculate_drawdown,
    calculate_sharpe,
    calculate_sortino,
    calculate_var,
)
__all__ = [
    "__version__",
    "__author__",
    "__license__",
    # Configuration
    "ExperimentConfig",
    "StrategyConfig",
    "CostConfig",
    "RiskConfig",
    "OptimizationConfig",
    "DataConfig",
    "StrategyType",
    "UniverseType",
    "create_experiment_config",
    "get_default_strategy_config",
    "quick_config",
    # Broker
    "Broker",
    "PaperBroker",
    "AlpacaBroker",
    "Order",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "Position",
    "Fill",
    "Trade",
    "AccountInfo",
    "OrderResult",
    "BrokerError",
    "OrderError",
    "ConnectionError",
    "InsufficientFundsError",
    # Live Trading
    "DataHandler",
    "PriceUpdate",
    "BarData",
    "PollingDataHandler",
    "RealtimeDataHandler",
    "SimulatedDataHandler",
    "OrderManager",
    "OrderTracker",
    "LiveTradingEngine",
    "TradingState",
    "EngineConfig",
    # Dashboard
    "DashboardState",
    "DashboardConfig",
    "PortfolioSnapshot",
    "PositionSnapshot",
    "TradeRecord",
    "RiskMetrics",
    "PerformanceMetrics",
    "DataCollector",
    "MockDataCollector",
    "SnapshotHistory",
    "MetricsCalculator",
    "calculate_drawdown",
    "calculate_sharpe",
    "calculate_sortino",
    "calculate_var",
]

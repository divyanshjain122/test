"""
JSF-Core: JBAC Strategy Foundry

A production-grade quantitative research engine for building, backtesting,
and optimizing trading strategies.

Version: 0.7.0-dev
Author: JBAC EdTech
License: MIT

⚠️ EDUCATIONAL PURPOSE ONLY ⚠️
==============================

This software is designed for LEARNING and RESEARCH purposes only.

IMPORTANT NOTICES:
- NOT FINANCIAL ADVICE: Nothing in this software constitutes investment advice.
- NOT FOR REAL TRADING: This is not intended for trading with real money.
- PAST PERFORMANCE: Backtested results do NOT guarantee future results.
- SUBSTANTIAL RISK: Trading involves substantial risk of financial loss.
- NO WARRANTY: Provided "AS IS" without warranty. Use at your own risk.

By importing this module, you acknowledge that you understand and accept
these risks. The authors and JBAC EdTech are NOT liable for any losses.

For full disclaimer, see: https://github.com/JaiAnshSB26/JBAC-Strategy-Foundry#disclaimer
"""

__version__ = "0.7.0-dev"
__author__ = "JBAC EdTech"
__license__ = "MIT"
__disclaimer__ = "EDUCATIONAL PURPOSE ONLY - NOT FOR REAL TRADING"

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

# ML Integration exports (Phase 19)
from .ml import (
    # Features
    FeatureExtractor,
    FeatureConfig,
    create_feature_extractor,
    FEATURE_GROUPS,
    # Models
    MLModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    EnsembleModel,
    ModelConfig,
    # Strategy
    MLStrategy,
    MLStrategyConfig,
    # Validation
    WalkForwardMLValidator,
    MLValidationResult,
    validate_ml_strategy,
    # Preprocessing
    prepare_ml_data,
    create_target_variable,
    split_train_test,
    handle_missing_features,
    MultiIndexConverter,
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
    # ML Integration (Phase 19)
    "FeatureExtractor",
    "FeatureConfig",
    "create_feature_extractor",
    "FEATURE_GROUPS",
    "MLModel",
    "RandomForestModel",
    "XGBoostModel",
    "LightGBMModel",
    "EnsembleModel",
    "ModelConfig",
    "MLStrategy",
    "MLStrategyConfig",
    "WalkForwardMLValidator",
    "MLValidationResult",
    "validate_ml_strategy",
    "prepare_ml_data",
    "create_target_variable",
    "split_train_test",
    "handle_missing_features",
    "MultiIndexConverter",
]

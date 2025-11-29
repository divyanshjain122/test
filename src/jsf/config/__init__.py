"""Configuration module for JSF-Core."""

from .base import JSFBaseConfig, DateRangeConfig
from .enums import (
    StrategyType,
    UniverseType,
    FrequencyType,
    RebalanceFrequency,
    OptimizationMethod,
    CostModel,
    PositionSizing,
    RiskMetric,
    DataSource,
)
from .schemas import (
    CostConfig,
    RiskConfig,
    StrategyConfig,
    DataConfig,
    OptimizationConfig,
    ExperimentConfig,
    create_experiment_config,
)
from .defaults import (
    STRATEGY_DEFAULTS,
    OPTIMIZATION_GRIDS,
    COST_PRESETS,
    RISK_PRESETS,
    get_default_strategy_config,
    get_optimization_grid,
    get_cost_preset,
    get_risk_preset,
    quick_config,
)

__all__ = [
    # Base classes
    "JSFBaseConfig",
    "DateRangeConfig",
    # Enums
    "StrategyType",
    "UniverseType",
    "FrequencyType",
    "RebalanceFrequency",
    "OptimizationMethod",
    "CostModel",
    "PositionSizing",
    "RiskMetric",
    "DataSource",
    # Config schemas
    "CostConfig",
    "RiskConfig",
    "StrategyConfig",
    "DataConfig",
    "OptimizationConfig",
    "ExperimentConfig",
    "create_experiment_config",
    # Defaults and presets
    "STRATEGY_DEFAULTS",
    "OPTIMIZATION_GRIDS",
    "COST_PRESETS",
    "RISK_PRESETS",
    "get_default_strategy_config",
    "get_optimization_grid",
    "get_cost_preset",
    "get_risk_preset",
    "quick_config",
]

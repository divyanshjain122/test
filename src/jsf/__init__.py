"""
JSF-Core: JBAC Strategy Foundry

A production-grade quantitative research engine for building, backtesting,
and optimizing trading strategies.

Version: 0.5.0-dev
Author: JBAC EdTech
License: MIT
"""

__version__ = "0.5.0-dev"
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

# Core exports will be added as modules are developed
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
]

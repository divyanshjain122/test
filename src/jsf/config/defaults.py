"""
Default parameter presets for strategies.

Provides sensible defaults and common configurations for each strategy type.
"""

from typing import Dict, Any

from .enums import StrategyType, RebalanceFrequency, PositionSizing
from .schemas import (
    StrategyConfig,
    CostConfig,
    RiskConfig,
    DataConfig,
    OptimizationConfig,
)


# ============================================================================
# STRATEGY PARAMETER DEFAULTS
# ============================================================================

STRATEGY_DEFAULTS: Dict[StrategyType, Dict[str, Any]] = {
    # Time-series momentum
    StrategyType.TS_MOMENTUM: {
        "lookback": 60,
        "holding_period": 20,
        "entry_threshold": 0.0,
        "volatility_scaling": True,
        "skip_period": 1,
    },
    
    # Time-series mean reversion
    StrategyType.TS_MEAN_REVERSION: {
        "lookback": 20,
        "entry_threshold": 2.0,  # z-score threshold
        "exit_threshold": 0.5,
        "holding_period": 5,
        "volatility_scaling": True,
    },
    
    # Time-series breakout
    StrategyType.TS_BREAKOUT: {
        "lookback": 50,
        "upper_percentile": 0.9,
        "lower_percentile": 0.1,
        "holding_period": 10,
    },
    
    # Cross-sectional momentum
    StrategyType.CS_MOMENTUM: {
        "lookback": 126,  # ~6 months
        "skip_period": 21,  # Skip last month
        "long_pct": 0.2,  # Top 20%
        "short_pct": 0.2,  # Bottom 20%
        "rebalance_frequency": "monthly",
    },
    
    # Cross-sectional value
    StrategyType.CS_VALUE: {
        "metrics": ["pe_ratio", "pb_ratio", "dividend_yield"],
        "long_pct": 0.3,
        "short_pct": 0.3,
        "rebalance_frequency": "quarterly",
    },
    
    # Cross-sectional quality
    StrategyType.CS_QUALITY: {
        "metrics": ["roa", "roe", "margin"],
        "long_pct": 0.3,
        "short_pct": 0.0,  # Long-only
        "rebalance_frequency": "quarterly",
    },
    
    # Pairs trading
    StrategyType.PAIRS_TRADING: {
        "formation_period": 252,
        "trading_period": 21,
        "entry_threshold": 2.0,
        "exit_threshold": 0.5,
        "correlation_threshold": 0.7,
    },
    
    # Statistical arbitrage
    StrategyType.STAT_ARB: {
        "lookback": 60,
        "half_life": 20,
        "entry_threshold": 1.5,
        "exit_threshold": 0.0,
    },
    
    # ML strategy
    StrategyType.ML_STRATEGY: {
        "lookback": 60,
        "prediction_horizon": 5,
        "model_type": "random_forest",
        "retrain_frequency": 60,
    },
}


# ============================================================================
# OPTIMIZATION PARAMETER GRIDS
# ============================================================================

OPTIMIZATION_GRIDS: Dict[StrategyType, Dict[str, list]] = {
    StrategyType.TS_MOMENTUM: {
        "lookback": [20, 40, 60, 90, 120],
        "holding_period": [5, 10, 20, 30],
        "entry_threshold": [0.0, 0.5, 1.0],
    },
    
    StrategyType.TS_MEAN_REVERSION: {
        "lookback": [10, 20, 30, 40],
        "entry_threshold": [1.5, 2.0, 2.5, 3.0],
        "exit_threshold": [0.0, 0.5, 1.0],
    },
    
    StrategyType.CS_MOMENTUM: {
        "lookback": [63, 126, 189, 252],  # 3, 6, 9, 12 months
        "skip_period": [0, 21, 42],  # 0, 1, 2 months
        "long_pct": [0.1, 0.2, 0.3],
        "short_pct": [0.0, 0.1, 0.2, 0.3],
    },
}


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

def get_default_strategy_config(strategy_type: StrategyType) -> StrategyConfig:
    """
    Get default configuration for a strategy type.
    
    Args:
        strategy_type: Strategy type
        
    Returns:
        StrategyConfig with defaults
    """
    params = STRATEGY_DEFAULTS.get(strategy_type, {}).copy()
    
    return StrategyConfig(
        name=strategy_type,
        parameters=params,
        rebalance_frequency=RebalanceFrequency.DAILY,
    )


def get_optimization_grid(strategy_type: StrategyType) -> Dict[str, list]:
    """
    Get default optimization grid for a strategy type.
    
    Args:
        strategy_type: Strategy type
        
    Returns:
        Parameter grid dictionary
    """
    return OPTIMIZATION_GRIDS.get(strategy_type, {}).copy()


# ============================================================================
# COST MODEL PRESETS
# ============================================================================

COST_PRESETS = {
    "conservative": CostConfig(
        slippage=0.002,  # 20 bps
        commission=0.001,  # 10 bps
        market_impact_coefficient=0.0001,
    ),
    
    "moderate": CostConfig(
        slippage=0.001,  # 10 bps
        commission=0.0005,  # 5 bps
        market_impact_coefficient=0.00005,
    ),
    
    "aggressive": CostConfig(
        slippage=0.0005,  # 5 bps
        commission=0.0001,  # 1 bp
        market_impact_coefficient=0.0,
    ),
    
    "zero_cost": CostConfig(
        slippage=0.0,
        commission=0.0,
        market_impact_coefficient=0.0,
    ),
}


def get_cost_preset(preset_name: str = "moderate") -> CostConfig:
    """
    Get cost configuration preset.
    
    Args:
        preset_name: Preset name (conservative, moderate, aggressive, zero_cost)
        
    Returns:
        CostConfig instance
    """
    if preset_name not in COST_PRESETS:
        raise ValueError(
            f"Unknown cost preset: {preset_name}. "
            f"Available: {list(COST_PRESETS.keys())}"
        )
    return COST_PRESETS[preset_name]


# ============================================================================
# RISK MODEL PRESETS
# ============================================================================

RISK_PRESETS = {
    "conservative": RiskConfig(
        max_leverage=1.0,
        max_position_size=0.05,  # 5% per position
        target_volatility=0.10,  # 10% annual vol
        position_sizing=PositionSizing.VOLATILITY_TARGET,
    ),
    
    "moderate": RiskConfig(
        max_leverage=1.5,
        max_position_size=0.10,  # 10% per position
        target_volatility=0.15,  # 15% annual vol
        position_sizing=PositionSizing.RISK_PARITY,
    ),
    
    "aggressive": RiskConfig(
        max_leverage=2.0,
        max_position_size=0.20,  # 20% per position
        target_volatility=0.25,  # 25% annual vol
        position_sizing=PositionSizing.SIGNAL_STRENGTH,
    ),
    
    "long_only": RiskConfig(
        max_leverage=1.0,
        max_position_size=0.10,
        position_sizing=PositionSizing.EQUAL_WEIGHT,
    ),
}


def get_risk_preset(preset_name: str = "moderate") -> RiskConfig:
    """
    Get risk configuration preset.
    
    Args:
        preset_name: Preset name (conservative, moderate, aggressive, long_only)
        
    Returns:
        RiskConfig instance
    """
    if preset_name not in RISK_PRESETS:
        raise ValueError(
            f"Unknown risk preset: {preset_name}. "
            f"Available: {list(RISK_PRESETS.keys())}"
        )
    return RISK_PRESETS[preset_name]


# ============================================================================
# QUICK CONFIG BUILDERS
# ============================================================================

def quick_config(
    strategy: str,
    universe: str = "SP500",
    start_date: str = "2015-01-01",
    end_date: str = "2023-12-31",
    cost_preset: str = "moderate",
    risk_preset: str = "moderate",
    **strategy_params: Any,
) -> Dict[str, Any]:
    """
    Quickly build a config dictionary with common defaults.
    
    Args:
        strategy: Strategy type name
        universe: Universe identifier
        start_date: Start date
        end_date: End date
        cost_preset: Cost preset name
        risk_preset: Risk preset name
        **strategy_params: Override strategy parameters
        
    Returns:
        Config dictionary ready for ExperimentConfig
        
    Examples:
        >>> config = quick_config("ts_momentum", lookback=90)
    """
    strategy_type = StrategyType(strategy)
    
    # Get defaults and override with custom params
    params = STRATEGY_DEFAULTS.get(strategy_type, {}).copy()
    params.update(strategy_params)
    
    return {
        "universe": universe,
        "start_date": start_date,
        "end_date": end_date,
        "strategy": {
            "name": strategy,
            "parameters": params,
        },
        "costs": get_cost_preset(cost_preset).to_dict(),
        "risk": get_risk_preset(risk_preset).to_dict(),
    }

"""
Enumerations for configuration options.

All enums used in configuration schemas.
"""

from enum import Enum


class StrategyType(str, Enum):
    """Available strategy types."""
    
    # Time-series strategies
    TS_MOMENTUM = "ts_momentum"
    TS_MEAN_REVERSION = "ts_meanrev"
    TS_BREAKOUT = "ts_breakout"
    
    # Cross-sectional strategies
    CS_MOMENTUM = "cs_momentum"
    CS_VALUE = "cs_value"
    CS_QUALITY = "cs_quality"
    
    # Other strategies
    PAIRS_TRADING = "pairs_trading"
    STAT_ARB = "stat_arb"
    ML_STRATEGY = "ml_strategy"


class UniverseType(str, Enum):
    """Predefined universe types."""
    
    SP500 = "SP500"
    SP400 = "SP400"  # Mid-cap
    SP600 = "SP600"  # Small-cap
    RUSSELL_1000 = "RUSSELL_1000"
    RUSSELL_2000 = "RUSSELL_2000"
    NASDAQ_100 = "NASDAQ_100"
    DOW_30 = "DOW_30"
    CUSTOM = "CUSTOM"


class FrequencyType(str, Enum):
    """Data frequency types."""
    
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    BUSINESS_DAILY = "B"
    HOURLY = "H"
    MINUTE = "T"


class RebalanceFrequency(str, Enum):
    """Portfolio rebalancing frequency."""
    
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    ON_SIGNAL = "on_signal"


class OptimizationMethod(str, Enum):
    """Optimization methods."""
    
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"


class CostModel(str, Enum):
    """Transaction cost models."""
    
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    TIERED = "tiered"
    MARKET_IMPACT = "market_impact"


class PositionSizing(str, Enum):
    """Position sizing methods."""
    
    EQUAL_WEIGHT = "equal_weight"
    VOLATILITY_TARGET = "volatility_target"
    RISK_PARITY = "risk_parity"
    KELLY = "kelly"
    FIXED_DOLLAR = "fixed_dollar"
    SIGNAL_STRENGTH = "signal_strength"


class RiskMetric(str, Enum):
    """Risk metrics for constraints."""
    
    VOLATILITY = "volatility"
    VAR = "var"  # Value at Risk
    CVAR = "cvar"  # Conditional VaR
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"


class DataSource(str, Enum):
    """Data source types."""
    
    CSV = "csv"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    IN_MEMORY = "in_memory"

"""
Core configuration schemas for JSF-Core.

Defines the main configuration objects:
- ExperimentConfig: Complete experiment specification
- StrategyConfig: Strategy parameters
- OptimizationConfig: Optimization settings
- CostConfig: Transaction cost settings
- RiskConfig: Risk management settings
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from .base import JSFBaseConfig, DateRangeConfig
from .enums import (
    StrategyType,
    UniverseType,
    FrequencyType,
    RebalanceFrequency,
    OptimizationMethod,
    CostModel,
    PositionSizing,
    DataSource,
)


class CostConfig(JSFBaseConfig):
    """Transaction cost configuration."""
    
    model: CostModel = Field(
        default=CostModel.PERCENTAGE,
        description="Cost model type",
    )
    
    slippage: float = Field(
        default=0.001,
        ge=0.0,
        le=0.1,
        description="Slippage in decimal (e.g., 0.001 = 10 bps)",
    )
    
    commission: float = Field(
        default=0.0005,
        ge=0.0,
        le=0.1,
        description="Commission in decimal (e.g., 0.0005 = 5 bps)",
    )
    
    market_impact_coefficient: float = Field(
        default=0.0,
        ge=0.0,
        description="Market impact coefficient for large orders",
    )
    
    min_commission: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum commission per trade (in dollars)",
    )


class RiskConfig(JSFBaseConfig):
    """Risk management configuration."""
    
    max_leverage: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Maximum portfolio leverage",
    )
    
    max_position_size: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Maximum position size as fraction of portfolio (None = no limit)",
    )
    
    max_sector_exposure: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Maximum sector exposure as fraction of portfolio",
    )
    
    target_volatility: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Target annualized volatility (e.g., 0.15 = 15%)",
    )
    
    stop_loss: Optional[float] = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Stop loss as decimal drawdown (e.g., 0.2 = 20%)",
    )
    
    position_sizing: PositionSizing = Field(
        default=PositionSizing.EQUAL_WEIGHT,
        description="Position sizing method",
    )


class StrategyConfig(JSFBaseConfig):
    """
    Strategy configuration.
    
    Defines strategy type and its specific parameters.
    """
    
    name: StrategyType = Field(
        ...,
        description="Strategy type identifier",
    )
    
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific parameters",
    )
    
    rebalance_frequency: RebalanceFrequency = Field(
        default=RebalanceFrequency.DAILY,
        description="How often to rebalance portfolio",
    )
    
    lookback_period: Optional[int] = Field(
        default=None,
        ge=1,
        description="Lookback period in bars (strategy-dependent)",
    )
    
    holding_period: Optional[int] = Field(
        default=None,
        ge=1,
        description="Minimum holding period in bars",
    )
    
    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate strategy parameters are JSON-serializable."""
        try:
            import json
            json.dumps(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Parameters must be JSON-serializable: {e}")
        return v


class DataConfig(JSFBaseConfig):
    """Data configuration."""
    
    source: DataSource = Field(
        default=DataSource.CSV,
        description="Data source type",
    )
    
    data_path: Optional[Path] = Field(
        default=None,
        description="Path to data file or directory",
    )
    
    frequency: FrequencyType = Field(
        default=FrequencyType.DAILY,
        description="Data frequency",
    )
    
    fields: List[str] = Field(
        default=["open", "high", "low", "close", "volume"],
        description="Required data fields",
    )
    
    adjust_prices: bool = Field(
        default=True,
        description="Whether to adjust prices for splits/dividends",
    )


class OptimizationConfig(JSFBaseConfig):
    """Optimization configuration."""
    
    method: OptimizationMethod = Field(
        default=OptimizationMethod.GRID_SEARCH,
        description="Optimization method",
    )
    
    parameter_grid: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="Parameter grid for optimization",
        examples=[{"lookback": [20, 40, 60], "threshold": [0.5, 1.0, 1.5]}],
    )
    
    objective: str = Field(
        default="sharpe_ratio",
        description="Objective metric to optimize",
    )
    
    maximize: bool = Field(
        default=True,
        description="Whether to maximize (True) or minimize (False) objective",
    )
    
    n_trials: int = Field(
        default=100,
        ge=1,
        description="Number of trials for random/Bayesian search",
    )
    
    n_jobs: int = Field(
        default=1,
        ge=-1,
        description="Number of parallel jobs (-1 = use all CPUs)",
    )
    
    cv_splits: int = Field(
        default=5,
        ge=2,
        description="Number of cross-validation splits",
    )
    
    walk_forward: bool = Field(
        default=False,
        description="Use walk-forward optimization",
    )
    
    @model_validator(mode="after")
    def validate_optimization(self) -> "OptimizationConfig":
        """Validate optimization configuration."""
        if self.method == OptimizationMethod.GRID_SEARCH and not self.parameter_grid:
            raise ValueError("parameter_grid is required for grid search")
        return self


class ExperimentConfig(JSFBaseConfig):
    """
    Complete experiment configuration.
    
    This is the main config object that defines an entire backtest experiment.
    """
    
    # Experiment metadata
    name: str = Field(
        default="experiment",
        description="Experiment name for identification",
    )
    
    description: Optional[str] = Field(
        default=None,
        description="Optional experiment description",
    )
    
    # Universe and time period
    universe: Union[UniverseType, List[str]] = Field(
        ...,
        description="Universe of assets (predefined or custom list)",
        examples=["SP500", ["AAPL", "GOOGL", "MSFT"]],
    )
    
    start_date: str = Field(
        ...,
        description="Backtest start date (YYYY-MM-DD)",
        examples=["2015-01-01"],
    )
    
    end_date: str = Field(
        ...,
        description="Backtest end date (YYYY-MM-DD)",
        examples=["2023-12-31"],
    )
    
    # Strategy configuration
    strategy: StrategyConfig = Field(
        ...,
        description="Strategy configuration",
    )
    
    # Data configuration
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data source configuration",
    )
    
    # Portfolio settings
    initial_capital: float = Field(
        default=100000.0,
        gt=0,
        description="Initial portfolio capital",
    )
    
    # Cost and risk
    costs: CostConfig = Field(
        default_factory=CostConfig,
        description="Transaction cost configuration",
    )
    
    risk: RiskConfig = Field(
        default_factory=RiskConfig,
        description="Risk management configuration",
    )
    
    # Optimization (optional)
    optimization: Optional[OptimizationConfig] = Field(
        default=None,
        description="Optional optimization configuration",
    )
    
    # Execution settings
    random_seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    
    output_dir: Optional[Path] = Field(
        default=None,
        description="Directory for output files",
    )
    
    save_trades: bool = Field(
        default=True,
        description="Whether to save individual trades",
    )
    
    save_positions: bool = Field(
        default=True,
        description="Whether to save position history",
    )
    
    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date format."""
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}")
        return v
    
    @model_validator(mode="after")
    def validate_dates(self) -> "ExperimentConfig":
        """Validate end date is after start date."""
        from datetime import datetime
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        if end <= start:
            raise ValueError(f"end_date must be after start_date")
        return self
    
    @model_validator(mode="after")
    def validate_universe(self) -> "ExperimentConfig":
        """Validate universe configuration."""
        if isinstance(self.universe, list) and len(self.universe) == 0:
            raise ValueError("Custom universe cannot be empty")
        return self
    
    def get_date_range(self) -> DateRangeConfig:
        """Get date range as DateRangeConfig object."""
        return DateRangeConfig(start_date=self.start_date, end_date=self.end_date)


# Convenience function for creating configs
def create_experiment_config(
    strategy_name: str | StrategyType,
    universe: str | List[str],
    start_date: str,
    end_date: str,
    **kwargs: Any,
) -> ExperimentConfig:
    """
    Convenience function to create ExperimentConfig.
    
    Args:
        strategy_name: Strategy type
        universe: Universe identifier or list of symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        **kwargs: Additional config parameters
        
    Returns:
        ExperimentConfig instance
        
    Examples:
        >>> config = create_experiment_config(
        ...     strategy_name="ts_momentum",
        ...     universe="SP500",
        ...     start_date="2015-01-01",
        ...     end_date="2023-01-01",
        ...     parameters={"lookback": 60},
        ... )
    """
    # Extract strategy parameters
    parameters = kwargs.pop("parameters", {})
    
    # Create strategy config
    if isinstance(strategy_name, str):
        strategy_name = StrategyType(strategy_name)
    
    strategy_config = StrategyConfig(
        name=strategy_name,
        parameters=parameters,
    )
    
    # Handle universe
    if isinstance(universe, str):
        try:
            universe = UniverseType(universe)
        except ValueError:
            # Treat as single symbol
            universe = [universe]
    
    return ExperimentConfig(
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        strategy=strategy_config,
        **kwargs,
    )

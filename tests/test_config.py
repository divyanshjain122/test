"""Unit tests for the config module."""

import pytest
import json
from pathlib import Path
from pydantic import ValidationError

from jsf.config import (
    ExperimentConfig,
    StrategyConfig,
    CostConfig,
    RiskConfig,
    OptimizationConfig,
    DataConfig,
    StrategyType,
    UniverseType,
    RebalanceFrequency,
    PositionSizing,
    create_experiment_config,
    get_default_strategy_config,
    get_cost_preset,
    get_risk_preset,
    quick_config,
)


# ============================================================================
# Test Base Configuration
# ============================================================================

class TestCostConfig:
    """Test CostConfig schema."""
    
    def test_default_creation(self) -> None:
        """Test creation with defaults."""
        config = CostConfig()
        assert config.slippage == 0.001
        assert config.commission == 0.0005
        assert config.market_impact_coefficient == 0.0
    
    def test_custom_values(self) -> None:
        """Test creation with custom values."""
        config = CostConfig(slippage=0.002, commission=0.001)
        assert config.slippage == 0.002
        assert config.commission == 0.001
    
    def test_validation_negative_slippage(self) -> None:
        """Test validation rejects negative slippage."""
        with pytest.raises(ValidationError):
            CostConfig(slippage=-0.001)
    
    def test_validation_excessive_slippage(self) -> None:
        """Test validation rejects excessive slippage."""
        with pytest.raises(ValidationError):
            CostConfig(slippage=0.5)
    
    def test_serialization(self) -> None:
        """Test JSON serialization."""
        config = CostConfig(slippage=0.002)
        json_str = config.to_json()
        data = json.loads(json_str)
        assert data["slippage"] == 0.002


class TestRiskConfig:
    """Test RiskConfig schema."""
    
    def test_default_creation(self) -> None:
        """Test creation with defaults."""
        config = RiskConfig()
        assert config.max_leverage == 1.0
        assert config.position_sizing == PositionSizing.EQUAL_WEIGHT
    
    def test_custom_values(self) -> None:
        """Test creation with custom values."""
        config = RiskConfig(
            max_leverage=2.0,
            max_position_size=0.1,
            target_volatility=0.15,
        )
        assert config.max_leverage == 2.0
        assert config.max_position_size == 0.1
        assert config.target_volatility == 0.15
    
    def test_validation_leverage(self) -> None:
        """Test leverage validation."""
        with pytest.raises(ValidationError):
            RiskConfig(max_leverage=-1.0)
        
        with pytest.raises(ValidationError):
            RiskConfig(max_leverage=20.0)


class TestStrategyConfig:
    """Test StrategyConfig schema."""
    
    def test_creation_with_name(self) -> None:
        """Test creation with strategy name."""
        config = StrategyConfig(
            name=StrategyType.TS_MOMENTUM,
            parameters={"lookback": 60},
        )
        assert config.name == StrategyType.TS_MOMENTUM
        assert config.parameters["lookback"] == 60
    
    def test_rebalance_frequency(self) -> None:
        """Test rebalance frequency setting."""
        config = StrategyConfig(
            name=StrategyType.CS_MOMENTUM,
            rebalance_frequency=RebalanceFrequency.MONTHLY,
        )
        assert config.rebalance_frequency == RebalanceFrequency.MONTHLY
    
    def test_parameters_validation(self) -> None:
        """Test parameters are JSON-serializable."""
        # Should work
        config = StrategyConfig(
            name=StrategyType.TS_MOMENTUM,
            parameters={"lookback": 60, "threshold": 1.5},
        )
        assert config.parameters["lookback"] == 60
        
        # Should fail with non-serializable object
        with pytest.raises(ValidationError):
            StrategyConfig(
                name=StrategyType.TS_MOMENTUM,
                parameters={"func": lambda x: x},
            )


# ============================================================================
# Test ExperimentConfig
# ============================================================================

class TestExperimentConfig:
    """Test ExperimentConfig schema."""
    
    def test_minimal_creation(self) -> None:
        """Test creation with minimal required fields."""
        config = ExperimentConfig(
            universe=UniverseType.SP500,
            start_date="2020-01-01",
            end_date="2023-12-31",
            strategy=StrategyConfig(
                name=StrategyType.TS_MOMENTUM,
                parameters={"lookback": 60},
            ),
        )
        assert config.universe == UniverseType.SP500
        assert config.start_date == "2020-01-01"
        assert config.end_date == "2023-12-31"
        assert config.initial_capital == 100000.0
    
    def test_custom_universe_list(self) -> None:
        """Test creation with custom symbol list."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        config = ExperimentConfig(
            universe=symbols,
            start_date="2020-01-01",
            end_date="2023-12-31",
            strategy=StrategyConfig(
                name=StrategyType.TS_MOMENTUM,
            ),
        )
        assert config.universe == symbols
    
    def test_date_validation(self) -> None:
        """Test date format validation."""
        # Invalid format
        with pytest.raises(ValidationError):
            ExperimentConfig(
                universe=UniverseType.SP500,
                start_date="01-01-2020",  # Wrong format
                end_date="2023-12-31",
                strategy=StrategyConfig(name=StrategyType.TS_MOMENTUM),
            )
    
    def test_end_before_start_validation(self) -> None:
        """Test end date must be after start date."""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                universe=UniverseType.SP500,
                start_date="2023-12-31",
                end_date="2020-01-01",  # Before start
                strategy=StrategyConfig(name=StrategyType.TS_MOMENTUM),
            )
    
    def test_empty_universe_validation(self) -> None:
        """Test empty custom universe is rejected."""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                universe=[],  # Empty list
                start_date="2020-01-01",
                end_date="2023-12-31",
                strategy=StrategyConfig(name=StrategyType.TS_MOMENTUM),
            )
    
    def test_complete_config(self) -> None:
        """Test creation with all optional fields."""
        config = ExperimentConfig(
            name="test_experiment",
            description="Test description",
            universe=UniverseType.SP500,
            start_date="2020-01-01",
            end_date="2023-12-31",
            strategy=StrategyConfig(
                name=StrategyType.TS_MOMENTUM,
                parameters={"lookback": 60},
            ),
            initial_capital=50000.0,
            costs=CostConfig(slippage=0.002),
            risk=RiskConfig(max_leverage=1.5),
            random_seed=123,
        )
        assert config.name == "test_experiment"
        assert config.description == "Test description"
        assert config.initial_capital == 50000.0
        assert config.costs.slippage == 0.002
        assert config.risk.max_leverage == 1.5
        assert config.random_seed == 123
    
    def test_serialization_roundtrip(self) -> None:
        """Test config can be serialized and deserialized."""
        original = ExperimentConfig(
            universe=UniverseType.SP500,
            start_date="2020-01-01",
            end_date="2023-12-31",
            strategy=StrategyConfig(
                name=StrategyType.TS_MOMENTUM,
                parameters={"lookback": 60},
            ),
        )
        
        # To dict and back
        data = original.to_dict()
        restored = ExperimentConfig.from_dict(data)
        assert restored.universe == original.universe
        assert restored.start_date == original.start_date
        
        # To JSON and back
        json_str = original.to_json()
        restored_json = ExperimentConfig.from_json(json_str)
        assert restored_json.universe == original.universe


class TestOptimizationConfig:
    """Test OptimizationConfig schema."""
    
    def test_default_creation(self) -> None:
        """Test creation with defaults."""
        config = OptimizationConfig()
        assert config.n_trials == 100
        assert config.maximize is True
    
    def test_grid_search_validation(self) -> None:
        """Test grid search requires parameter grid."""
        # Should fail without grid
        with pytest.raises(ValidationError):
            OptimizationConfig(
                method="grid_search",
                parameter_grid={},
            )
        
        # Should work with grid
        config = OptimizationConfig(
            method="grid_search",
            parameter_grid={"lookback": [20, 40, 60]},
        )
        assert len(config.parameter_grid["lookback"]) == 3


# ============================================================================
# Test Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience functions for creating configs."""
    
    def test_create_experiment_config(self) -> None:
        """Test create_experiment_config function."""
        config = create_experiment_config(
            strategy_name="ts_momentum",
            universe="SP500",
            start_date="2020-01-01",
            end_date="2023-12-31",
            parameters={"lookback": 90},
        )
        assert isinstance(config, ExperimentConfig)
        assert config.strategy.name == StrategyType.TS_MOMENTUM
        assert config.strategy.parameters["lookback"] == 90
    
    def test_quick_config(self) -> None:
        """Test quick_config helper."""
        config_dict = quick_config(
            strategy="ts_momentum",
            lookback=120,
        )
        assert config_dict["strategy"]["name"] == "ts_momentum"
        assert config_dict["strategy"]["parameters"]["lookback"] == 120


class TestPresets:
    """Test preset configurations."""
    
    def test_get_default_strategy_config(self) -> None:
        """Test getting default strategy config."""
        config = get_default_strategy_config(StrategyType.TS_MOMENTUM)
        assert isinstance(config, StrategyConfig)
        assert config.name == StrategyType.TS_MOMENTUM
        assert "lookback" in config.parameters
    
    def test_get_cost_preset(self) -> None:
        """Test getting cost preset."""
        config = get_cost_preset("conservative")
        assert isinstance(config, CostConfig)
        assert config.slippage == 0.002
        
        with pytest.raises(ValueError):
            get_cost_preset("nonexistent")
    
    def test_get_risk_preset(self) -> None:
        """Test getting risk preset."""
        config = get_risk_preset("aggressive")
        assert isinstance(config, RiskConfig)
        assert config.max_leverage == 2.0
        
        with pytest.raises(ValueError):
            get_risk_preset("nonexistent")


# ============================================================================
# Test File I/O
# ============================================================================

class TestConfigIO:
    """Test configuration file I/O."""
    
    def test_save_and_load_json(self, tmp_path: Path) -> None:
        """Test saving and loading config from JSON file."""
        config = ExperimentConfig(
            universe=UniverseType.SP500,
            start_date="2020-01-01",
            end_date="2023-12-31",
            strategy=StrategyConfig(
                name=StrategyType.TS_MOMENTUM,
                parameters={"lookback": 60},
            ),
        )
        
        # Save to file
        filepath = tmp_path / "config.json"
        config.to_json(filepath)
        
        # Load from file
        loaded = ExperimentConfig.from_json(filepath)
        assert loaded.universe == config.universe
        assert loaded.start_date == config.start_date
        assert loaded.strategy.name == config.strategy.name
    
    def test_copy_with(self) -> None:
        """Test copying config with modifications."""
        original = ExperimentConfig(
            universe=UniverseType.SP500,
            start_date="2020-01-01",
            end_date="2023-12-31",
            strategy=StrategyConfig(
                name=StrategyType.TS_MOMENTUM,
            ),
            initial_capital=100000,
        )
        
        modified = original.copy_with(initial_capital=200000)
        assert modified.initial_capital == 200000
        assert modified.universe == original.universe
        assert original.initial_capital == 100000  # Original unchanged

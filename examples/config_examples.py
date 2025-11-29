"""
Configuration System Examples

Demonstrates how to use JSF-Core's configuration system.
"""

from jsf.config import (
    ExperimentConfig,
    StrategyConfig,
    StrategyType,
    UniverseType,
    create_experiment_config,
    get_default_strategy_config,
    get_cost_preset,
    get_risk_preset,
    quick_config,
)


def example_1_minimal_config():
    """Example 1: Minimal configuration."""
    print("=" * 60)
    print("EXAMPLE 1: Minimal Configuration")
    print("=" * 60)
    
    config = ExperimentConfig(
        universe=UniverseType.SP500,
        start_date="2020-01-01",
        end_date="2023-12-31",
        strategy=StrategyConfig(
            name=StrategyType.TS_MOMENTUM,
            parameters={"lookback": 60},
        ),
    )
    
    print(f"Universe: {config.universe}")
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Strategy: {config.strategy.name}")
    print(f"Lookback: {config.strategy.parameters['lookback']}")
    print(f"Initial Capital: ${config.initial_capital:,.0f}")
    print()


def example_2_complete_config():
    """Example 2: Complete configuration with all options."""
    print("=" * 60)
    print("EXAMPLE 2: Complete Configuration")
    print("=" * 60)
    
    config = ExperimentConfig(
        name="momentum_backtest_2020_2023",
        description="Testing momentum strategy with conservative settings",
        universe=UniverseType.SP500,
        start_date="2020-01-01",
        end_date="2023-12-31",
        strategy=StrategyConfig(
            name=StrategyType.TS_MOMENTUM,
            parameters={
                "lookback": 90,
                "holding_period": 20,
                "volatility_scaling": True,
            },
        ),
        initial_capital=500000.0,
        costs=get_cost_preset("conservative"),
        risk=get_risk_preset("moderate"),
        random_seed=42,
        save_trades=True,
        save_positions=True,
    )
    
    print(f"Experiment: {config.name}")
    print(f"Description: {config.description}")
    print(f"Capital: ${config.initial_capital:,.0f}")
    print(f"Slippage: {config.costs.slippage:.4f} ({config.costs.slippage*10000:.1f} bps)")
    print(f"Max Leverage: {config.risk.max_leverage}x")
    print(f"Position Sizing: {config.risk.position_sizing}")
    print()


def example_3_convenience_function():
    """Example 3: Using convenience function."""
    print("=" * 60)
    print("EXAMPLE 3: Convenience Function")
    print("=" * 60)
    
    config = create_experiment_config(
        strategy_name="cs_momentum",
        universe="NASDAQ_100",
        start_date="2018-01-01",
        end_date="2023-12-31",
        parameters={
            "lookback": 126,
            "long_pct": 0.2,
            "short_pct": 0.2,
        },
        initial_capital=1000000,
    )
    
    print(f"Strategy: {config.strategy.name}")
    print(f"Universe: {config.universe}")
    print(f"Parameters: {config.strategy.parameters}")
    print()


def example_4_custom_universe():
    """Example 4: Custom symbol list."""
    print("=" * 60)
    print("EXAMPLE 4: Custom Universe")
    print("=" * 60)
    
    tech_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA"]
    
    config = ExperimentConfig(
        universe=tech_stocks,
        start_date="2020-01-01",
        end_date="2023-12-31",
        strategy=StrategyConfig(
            name=StrategyType.PAIRS_TRADING,
            parameters={
                "formation_period": 252,
                "entry_threshold": 2.0,
            },
        ),
    )
    
    print(f"Custom Universe: {config.universe}")
    print(f"Number of symbols: {len(config.universe)}")
    print(f"Strategy: {config.strategy.name}")
    print()


def example_5_serialization():
    """Example 5: Saving and loading configurations."""
    print("=" * 60)
    print("EXAMPLE 5: Serialization")
    print("=" * 60)
    
    config = ExperimentConfig(
        name="test_config",
        universe=UniverseType.SP500,
        start_date="2020-01-01",
        end_date="2023-12-31",
        strategy=get_default_strategy_config(StrategyType.TS_MOMENTUM),
    )
    
    # To JSON string
    json_str = config.to_json()
    print("JSON representation (first 200 chars):")
    print(json_str[:200] + "...")
    print()
    
    # To dictionary
    config_dict = config.to_dict()
    print("Dictionary keys:")
    print(list(config_dict.keys()))
    print()
    
    # Round-trip test
    restored = ExperimentConfig.from_json(json_str)
    print(f"Restored config name: {restored.name}")
    print(f"Configs match: {config.to_dict() == restored.to_dict()}")
    print()


def example_6_quick_config():
    """Example 6: Quick config builder."""
    print("=" * 60)
    print("EXAMPLE 6: Quick Config Builder")
    print("=" * 60)
    
    config_dict = quick_config(
        strategy="ts_momentum",
        universe="SP500",
        start_date="2020-01-01",
        end_date="2023-12-31",
        cost_preset="aggressive",
        risk_preset="conservative",
        lookback=120,
        holding_period=30,
    )
    
    # Convert to ExperimentConfig
    config = ExperimentConfig(**config_dict)
    
    print(f"Strategy: {config.strategy.name}")
    print(f"Lookback: {config.strategy.parameters['lookback']}")
    print(f"Holding Period: {config.strategy.parameters['holding_period']}")
    print(f"Slippage: {config.costs.slippage:.4f}")
    print(f"Max Leverage: {config.risk.max_leverage}x")
    print()


def example_7_copy_with_modifications():
    """Example 7: Copying configs with modifications."""
    print("=" * 60)
    print("EXAMPLE 7: Copy with Modifications")
    print("=" * 60)
    
    base_config = ExperimentConfig(
        name="base_experiment",
        universe=UniverseType.SP500,
        start_date="2020-01-01",
        end_date="2023-12-31",
        strategy=StrategyConfig(
            name=StrategyType.TS_MOMENTUM,
            parameters={"lookback": 60},
        ),
        initial_capital=100000,
    )
    
    # Create variations
    config_2x = base_config.copy_with(
        name="2x_capital",
        initial_capital=200000,
    )
    
    config_different_period = base_config.copy_with(
        name="different_period",
        start_date="2021-01-01",
        end_date="2022-12-31",
    )
    
    print(f"Base config: {base_config.name}, capital=${base_config.initial_capital:,.0f}")
    print(f"Modified #1: {config_2x.name}, capital=${config_2x.initial_capital:,.0f}")
    print(f"Modified #2: {config_different_period.name}, period={config_different_period.start_date} to {config_different_period.end_date}")
    print()


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("JSF-CORE CONFIGURATION SYSTEM EXAMPLES")
    print("="*60 + "\n")
    
    example_1_minimal_config()
    example_2_complete_config()
    example_3_convenience_function()
    example_4_custom_universe()
    example_5_serialization()
    example_6_quick_config()
    example_7_copy_with_modifications()
    
    print("="*60)
    print("All examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

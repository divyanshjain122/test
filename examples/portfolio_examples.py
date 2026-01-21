"""
Comprehensive examples for Phase 7: Portfolio Construction

This file demonstrates how to use the portfolio construction framework
to build and manage quantitative trading portfolios.
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Import data loading
from jsf.data import load_data

# Import portfolio components
from jsf.portfolio import (
    # Base
    Portfolio,
    RebalanceFrequency,
    # Position Sizing
    EqualWeightSizer,
    SignalWeightedSizer,
    VolatilityScaledSizer,
    RiskParitySizer,
    KellyCriterionSizer,
    # Optimization
    MinimumVarianceOptimizer,
    MaxSharpeOptimizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    MaxDiversificationOptimizer,
    # Rebalancing
    PeriodicRebalancer,
    ThresholdRebalancer,
    VolatilityTargetRebalancer,
    BandRebalancer,
    SmartRebalancer,
    # Constraints
    PortfolioConstraints,
    PositionLimitConstraint,
    SectorConstraint,
    TurnoverConstraint,
    LeverageConstraint,
    ConcentrationConstraint,
    # Constructors
    SimplePortfolioConstructor,
    OptimizedPortfolioConstructor,
    HybridPortfolioConstructor,
)


def example_1_basic_equal_weight():
    """
    Example 1: Basic Equal Weight Portfolio
    
    The simplest portfolio construction approach - equal allocation
    to all assets.
    """
    print("=" * 70)
    print("Example 1: Basic Equal Weight Portfolio")
    print("=" * 70)
    
    # Load sample data
    price_data = load_data(
        source="synthetic",
        symbols=["AAPL", "GOOGL", "MSFT", "AMZN"],
        start_date="2022-01-01",
        end_date="2023-12-31",
        annual_return=0.12,
        annual_volatility=0.20,
        seed=42,
    )
    
    print(f"\nLoaded data: {len(price_data.data)} rows, {len(price_data.symbols)} symbols")
    print(f"Date range: {price_data.start_date} to {price_data.end_date}")
    
    # Create simple signals (all 1.0 = bullish on everything)
    signals = pd.DataFrame(
        1.0,
        index=price_data.data.index,
        columns=price_data.symbols
    )
    
    # Create equal weight sizer
    sizer = EqualWeightSizer(long_only=True)
    
    # Size positions
    weights = sizer.size(signals, price_data)
    
    print(f"\nPortfolio weights (first day):")
    print(weights.iloc[0])
    print(f"Total weight: {weights.iloc[0].sum():.4f}")
    
    # Create Portfolio object (takes DataFrame)
    portfolio = Portfolio(
        weights=weights.iloc[:10],  # First 10 days
        metadata={"strategy": "equal_weight"}
    )
    
    print(f"\nPortfolio created with {len(portfolio.weights)} days")
    print(f"First day positions:")
    print(portfolio.get_positions(portfolio.weights.index[0]))
    
    # Calculate turnover
    turnover = portfolio.get_turnover()
    print(f"\nAverage daily turnover: {turnover.mean():.6f}")
    

def example_2_signal_weighted():
    """
    Example 2: Signal-Weighted Portfolio
    
    Allocate capital proportionally to signal strength.
    """
    print("\n" + "=" * 70)
    print("Example 2: Signal-Weighted Portfolio")
    print("=" * 70)
    
    # Load data
    price_data = load_data(
        source="synthetic",
        symbols=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        seed=123,
    )
    
    # Create varied signals (different conviction levels)
    signals = pd.DataFrame({
        "AAPL": [0.8] * len(price_data.data),
        "GOOGL": [0.6] * len(price_data.data),
        "MSFT": [1.0] * len(price_data.data),  # Highest conviction
        "AMZN": [0.4] * len(price_data.data),
        "TSLA": [0.2] * len(price_data.data),
    }, index=price_data.data.index)
    
    # Create signal-weighted sizer
    sizer = SignalWeightedSizer(normalize=True, long_only=True)
    
    # Size positions
    weights = sizer.size(signals, price_data)
    
    print(f"\nSignal strengths:")
    print(signals.iloc[0])
    
    print(f"\nResulting weights:")
    print(weights.iloc[0])
    print(f"Total: {weights.iloc[0].sum():.4f}")
    
    # Note how weights are proportional to signals
    print(f"\nMSFT has highest signal ({signals.iloc[0]['MSFT']:.2f}) "
          f"and highest weight ({weights.iloc[0]['MSFT']:.4f})")


def example_3_volatility_scaled():
    """
    Example 3: Volatility-Scaled Portfolio
    
    Allocate inversely to volatility - less weight to riskier assets.
    """
    print("\n" + "=" * 70)
    print("Example 3: Volatility-Scaled Portfolio")
    print("=" * 70)
    
    # Load data with different volatilities
    price_data = load_data(
        source="synthetic",
        symbols=["LOW_VOL", "MED_VOL", "HIGH_VOL"],
        start_date="2022-01-01",
        end_date="2023-12-31",
        annual_volatility=0.15,  # Will vary by symbol
        seed=456,
    )
    
    # Uniform signals
    signals = pd.DataFrame(
        1.0,
        index=price_data.data.index,
        columns=price_data.symbols
    )
    
    # Create volatility-scaled sizer
    sizer = VolatilityScaledSizer(
        lookback=60,
        target_volatility=0.12,
        long_only=True
    )
    
    # Size positions
    weights = sizer.size(signals, price_data)
    
    print(f"\nVolatility-scaled weights (first valid day):")
    # Find first row with valid weights
    valid_idx = weights.apply(lambda x: x.sum() > 0, axis=1).idxmax()
    print(weights.loc[valid_idx])
    print(f"Total: {weights.loc[valid_idx].sum():.4f}")
    
    print(f"\nLower volatility assets receive higher weights")


def example_4_minimum_variance_optimization():
    """
    Example 4: Minimum Variance Optimization
    
    Find the portfolio with lowest volatility.
    """
    print("\n" + "=" * 70)
    print("Example 4: Minimum Variance Optimization")
    print("=" * 70)
    
    # Load correlated assets
    price_data = load_data(
        source="synthetic",
        symbols=["STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D"],
        start_date="2022-01-01",
        end_date="2023-12-31",
        annual_return=0.10,
        annual_volatility=0.25,
        seed=789,
    )
    
    # Universe of potential holdings
    signals = pd.DataFrame(
        1.0,
        index=price_data.data.index,
        columns=price_data.symbols
    )
    
    # Create minimum variance optimizer
    optimizer = MinimumVarianceOptimizer()
    
    # Optimize for specific date
    target_date = pd.Timestamp("2023-06-01")
    optimized_weights = optimizer.optimize(
        signals.loc[target_date],
        price_data
    )
    
    print(f"\nMinimum variance weights:")
    print(optimized_weights)
    print(f"Total: {optimized_weights.sum():.4f}")
    
    print(f"\nThese weights minimize portfolio volatility")
    print(f"while maintaining full investment")


def example_5_maximum_sharpe_optimization():
    """
    Example 5: Maximum Sharpe Ratio Optimization
    
    Find the portfolio with best risk-adjusted returns.
    """
    print("\n" + "=" * 70)
    print("Example 5: Maximum Sharpe Ratio Optimization")
    print("=" * 70)
    
    # Load data
    price_data = load_data(
        source="synthetic",
        symbols=["TECH", "FINANCE", "HEALTHCARE", "ENERGY"],
        start_date="2022-01-01",
        end_date="2023-12-31",
        annual_return=0.15,
        annual_volatility=0.30,
        seed=101,
    )
    
    signals = pd.DataFrame(
        1.0,
        index=price_data.data.index,
        columns=price_data.symbols
    )
    
    # Create max Sharpe optimizer
    optimizer = MaxSharpeOptimizer(risk_free_rate=0.03)
    
    # Optimize
    target_date = pd.Timestamp("2023-06-01")
    optimized_weights = optimizer.optimize(
        signals.loc[target_date],
        price_data
    )
    
    print(f"\nMaximum Sharpe ratio weights:")
    print(optimized_weights)
    print(f"Total: {optimized_weights.sum():.4f}")
    
    print(f"\nThese weights maximize risk-adjusted returns")


def example_6_periodic_rebalancing():
    """
    Example 6: Periodic Rebalancing Strategy
    
    Rebalance portfolio on a fixed schedule.
    """
    print("\n" + "=" * 70)
    print("Example 6: Periodic Rebalancing (Monthly)")
    print("=" * 70)
    
    # Create periodic rebalancer
    rebalancer = PeriodicRebalancer(frequency=RebalanceFrequency.MONTHLY)
    
    # Test rebalancing dates
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    
    last_rebalance = pd.Timestamp("2023-01-01")
    rebalance_dates = []
    
    for date in dates:
        if rebalancer.should_rebalance(date, None, last_rebalance):
            rebalance_dates.append(date)
            last_rebalance = date
    
    print(f"\nRebalancing {len(rebalance_dates)} times in 2023:")
    for i, date in enumerate(rebalance_dates[:6], 1):
        print(f"  {i}. {date.strftime('%Y-%m-%d')}")
    print("  ...")
    
    print(f"\nMonthly rebalancing ensures periodic portfolio review")


def example_7_threshold_rebalancing():
    """
    Example 7: Threshold-Based Rebalancing
    
    Rebalance only when drift exceeds threshold.
    """
    print("\n" + "=" * 70)
    print("Example 7: Threshold-Based Rebalancing")
    print("=" * 70)
    
    # Create threshold rebalancer (5% drift triggers rebalance)
    rebalancer = ThresholdRebalancer(threshold=0.05)
    
    # Target weights
    target_weights = pd.Series({
        "AAPL": 0.25,
        "GOOGL": 0.25,
        "MSFT": 0.25,
        "AMZN": 0.25,
    })
    
    # Simulate market movements causing drift
    scenarios = [
        ("Small drift", pd.Series({
            "AAPL": 0.26, "GOOGL": 0.24, "MSFT": 0.25, "AMZN": 0.25
        })),
        ("Large drift", pd.Series({
            "AAPL": 0.40, "GOOGL": 0.20, "MSFT": 0.20, "AMZN": 0.20
        })),
    ]
    
    for scenario_name, current_weights in scenarios:
        should_rebal = rebalancer.should_rebalance(
            current_time=pd.Timestamp.now(),
            current_weights=current_weights,
            last_rebalance=pd.Timestamp.now() - pd.Timedelta(days=30),
            target_weights=target_weights
        )
        
        drift = (current_weights - target_weights).abs().sum() / 2
        
        print(f"\n{scenario_name}:")
        print(f"  Drift: {drift:.4f}")
        print(f"  Rebalance: {'YES' if should_rebal else 'NO'}")
    
    print(f"\nRebalancing minimizes trading costs")


def example_8_position_constraints():
    """
    Example 8: Position Limit Constraints
    
    Enforce maximum position sizes for risk management.
    """
    print("\n" + "=" * 70)
    print("Example 8: Position Limit Constraints")
    print("=" * 70)
    
    # Create position limit constraint (max 30% per position)
    constraint = PositionLimitConstraint(
        min_weight=0.05,
        max_weight=0.30
    )
    
    # Concentrated portfolio (violates constraint)
    concentrated_weights = pd.Series({
        "AAPL": 0.60,  # Too large!
        "GOOGL": 0.30,
        "MSFT": 0.10,
    })
    
    print(f"\nOriginal (concentrated) weights:")
    print(concentrated_weights)
    
    # Check if valid
    is_valid = constraint.check(concentrated_weights)
    print(f"\nValid: {is_valid}")
    
    # Enforce constraint
    adjusted_weights = constraint.enforce(concentrated_weights)
    
    print(f"\nAdjusted weights:")
    print(adjusted_weights)
    print(f"Total: {adjusted_weights.sum():.4f}")
    
    print(f"\nMax position reduced from 60% to 30%")


def example_9_leverage_constraints():
    """
    Example 9: Leverage Constraints
    
    Control gross and net exposure.
    """
    print("\n" + "=" * 70)
    print("Example 9: Leverage Constraints")
    print("=" * 70)
    
    # Create leverage constraint (max 1.5x gross, 1.0x net)
    constraint = LeverageConstraint(
        max_gross_leverage=1.5,
        max_net_leverage=1.0
    )
    
    # Long-short portfolio
    ls_weights = pd.Series({
        "LONG_1": 0.50,
        "LONG_2": 0.50,
        "SHORT_1": -0.30,
        "SHORT_2": -0.20,
    })
    
    print(f"\nLong-short weights:")
    print(ls_weights)
    
    gross = ls_weights.abs().sum()
    net = ls_weights.sum()
    
    print(f"\nGross leverage: {gross:.2f}x")
    print(f"Net leverage: {net:.2f}x")
    
    is_valid = constraint.check(ls_weights)
    print(f"Valid: {is_valid}")
    
    print(f"\nGross exposure of {gross:.2f}x is within 1.5x limit")


def example_10_multi_constraint_portfolio():
    """
    Example 10: Multiple Constraints
    
    Combine multiple constraints for comprehensive risk management.
    """
    print("\n" + "=" * 70)
    print("Example 10: Multi-Constraint Portfolio")
    print("=" * 70)
    
    # Create constraint manager
    constraints = PortfolioConstraints()
    
    # Add multiple constraints
    constraints.add_constraint(
        PositionLimitConstraint(min_weight=0.05, max_weight=0.35)
    )
    constraints.add_constraint(
        LeverageConstraint(max_gross_leverage=1.0)
    )
    constraints.add_constraint(
        ConcentrationConstraint(max_hhi=0.25)
    )
    
    print(f"\nApplied {len(constraints.constraints)} constraints:")
    for c in constraints.constraints:
        print(f"  - {c.name}")
    
    # Test portfolio
    test_weights = pd.Series({
        "A": 0.40,  # Violates 35% limit
        "B": 0.30,
        "C": 0.30,
    })
    
    print(f"\nTest weights:")
    print(test_weights)
    
    # Validate
    valid, violations = constraints.validate(test_weights)
    print(f"\nValid: {valid}")
    if violations:
        print(f"Violations: {', '.join(violations)}")
    
    # Enforce all constraints
    adjusted = constraints.enforce(test_weights)
    print(f"\nAdjusted weights:")
    print(adjusted)
    
    # Revalidate
    valid, _ = constraints.validate(adjusted)
    print(f"Valid after adjustment: {valid}")


def example_11_simple_constructor():
    """
    Example 11: Simple Portfolio Constructor
    
    Basic end-to-end portfolio construction.
    """
    print("\n" + "=" * 70)
    print("Example 11: Simple Portfolio Constructor")
    print("=" * 70)
    
    # Load data
    price_data = load_data(
        source="synthetic",
        symbols=["AAPL", "GOOGL", "MSFT", "AMZN"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        seed=999,
    )
    
    # Create simple signals
    signals = pd.DataFrame(
        np.random.randn(len(price_data.data), len(price_data.symbols)),
        index=price_data.data.index,
        columns=price_data.symbols
    )
    
    # Create constructor with sizer
    sizer = EqualWeightSizer(long_only=True)
    constructor = SimplePortfolioConstructor(
        position_sizer=sizer,
        name="simple_strategy"
    )
    
    # Construct portfolio for specific date
    target_date = pd.Timestamp("2023-06-01")
    portfolio = constructor.construct(
        signals=signals.loc[target_date],
        price_data=price_data
    )
    
    print(f"\nConstructed portfolio for {target_date.date()}:")
    print(portfolio.weights)
    print(f"\nTotal weight: {portfolio.weights.sum():.4f}")
    print(f"Long exposure: {portfolio.long_exposure:.4f}")


def example_12_optimized_constructor():
    """
    Example 12: Optimized Portfolio Constructor
    
    Portfolio construction with optimization.
    """
    print("\n" + "=" * 70)
    print("Example 12: Optimized Portfolio Constructor")
    print("=" * 70)
    
    # Load data
    price_data = load_data(
        source="synthetic",
        symbols=["TECH", "FINANCE", "HEALTHCARE", "CONSUMER"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        seed=777,
    )
    
    signals = pd.DataFrame(
        1.0,
        index=price_data.data.index,
        columns=price_data.symbols
    )
    
    # Create constructor with optimizer
    optimizer = MinimumVarianceOptimizer()
    constructor = OptimizedPortfolioConstructor(
        optimizer=optimizer,
        name="min_var_strategy"
    )
    
    # Construct optimized portfolio
    target_date = pd.Timestamp("2023-06-01")
    portfolio = constructor.construct(
        signals=signals.loc[target_date],
        price_data=price_data
    )
    
    print(f"\nOptimized portfolio weights:")
    print(portfolio.weights)
    print(f"\nThese are minimum variance weights")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  JBAC Strategy Foundry - Phase 7 Portfolio Examples".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    examples = [
        example_1_basic_equal_weight,
        example_2_signal_weighted,
        example_3_volatility_scaled,
        example_4_minimum_variance_optimization,
        example_5_maximum_sharpe_optimization,
        example_6_periodic_rebalancing,
        example_7_threshold_rebalancing,
        example_8_position_constraints,
        example_9_leverage_constraints,
        example_10_multi_constraint_portfolio,
        example_11_simple_constructor,
        example_12_optimized_constructor,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nExample encountered an issue: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Customize parameters for your strategy")
    print("  - Combine components for advanced workflows")
    print("  - Backtest with real market data")
    print("\n")


if __name__ == "__main__":
    main()

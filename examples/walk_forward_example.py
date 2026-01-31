"""
Walk-Forward Analysis Example

This example demonstrates how to use walk-forward analysis
to validate strategies and detect overfitting.

Walk-forward analysis is the gold standard for strategy validation because:
1. It tests on truly unseen data (out-of-sample)
2. It measures how well optimized parameters generalize
3. It detects overfitting before real money is risked

Example Output:
    Window 0: IS=2020-01-01→2020-12-31, OOS=2021-01-04→2021-04-01
    Window 1: IS=2020-04-01→2021-03-31, OOS=2021-04-01→2021-07-01
    ...
    
    Efficiency Ratio: 0.75 ✓ (Good generalization)
    Parameter Stability: 0.67 ✓ (Consistent parameters)
    Overfitting Detected: NO ✓
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from jsf.data import load_data
from jsf.strategies import MomentumStrategy, MeanReversionStrategy
from jsf.simulation import BacktestConfig
from jsf.optimization import (
    walk_forward_analysis,
    WalkForwardOptimizer,
    optimize_strategy,
)


def main():
    """Run walk-forward analysis examples."""
    
    print("=" * 70)
    print("WALK-FORWARD ANALYSIS EXAMPLES")
    print("=" * 70)
    
    # =========================================================================
    # Example 1: Basic Walk-Forward Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("Example 1: Basic Walk-Forward Analysis")
    print("=" * 70)
    
    # Load 4 years of data (need enough for multiple windows)
    data = load_data(
        source='synthetic',
        symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        start_date='2020-01-01',
        end_date='2023-12-31',
    )
    
    print(f"\nLoaded data: {len(data.dates)} days, {len(data.symbols)} symbols")
    print(f"Period: {data.start_date.date()} to {data.end_date.date()}")
    
    # Run walk-forward analysis with standard settings
    result = walk_forward_analysis(
        strategy_class=MomentumStrategy,
        param_grid={'lookback': [30, 60, 90, 120]},
        data=data,
        is_days=252,  # 1 year in-sample
        oos_days=63,  # 1 quarter out-of-sample
        metric='sharpe_ratio',
    )
    
    # Print comprehensive results
    print(result)
    
    # Show window-by-window breakdown
    print("\nWindow-by-Window Breakdown:")
    print("-" * 70)
    summary = result.get_summary()
    print(summary.to_string(index=False))
    
    # Show parameter choices
    print("\nParameter Choices per Window:")
    print("-" * 70)
    param_summary = result.get_parameter_summary()
    print(param_summary.to_string(index=False))
    
    # =========================================================================
    # Example 2: Compare Grid Search vs Walk-Forward
    # =========================================================================
    print("\n" + "=" * 70)
    print("Example 2: Grid Search vs Walk-Forward Comparison")
    print("=" * 70)
    
    # First, run regular grid search (potentially overfitted)
    grid_result = optimize_strategy(
        strategy_class=MomentumStrategy,
        param_grid={'lookback': [30, 60, 90, 120]},
        data=data,
        metric='sharpe_ratio',
    )
    
    print(f"\n📊 GRID SEARCH (In-Sample Only):")
    print(f"   Best Sharpe: {grid_result.best_score:.3f}")
    print(f"   Best params: {grid_result.best_params}")
    print(f"   ⚠️ WARNING: This may be overfitted!")
    
    # Walk-forward gives true OOS performance
    print(f"\n🔍 WALK-FORWARD (Out-of-Sample):")
    print(f"   Avg IS Sharpe:  {result.avg_is_score:.3f}")
    print(f"   Avg OOS Sharpe: {result.avg_oos_score:.3f}")
    print(f"   Efficiency:     {result.efficiency_ratio:.2%}")
    print(f"   OOS Total Return: {result.oos_total_return:.2%}")
    
    # Analysis
    degradation = (result.avg_is_score - result.avg_oos_score) / result.avg_is_score * 100
    print(f"\n📉 Performance Degradation IS→OOS: {degradation:.1f}%")
    
    if result.is_overfitted:
        print("⚠️  CONCLUSION: Strategy shows signs of OVERFITTING")
        print("    Recommendation: Simplify parameters or use more robust signals")
    else:
        print("✅ CONCLUSION: Strategy generalizes WELL to unseen data")
        print("    Recommendation: Safe to proceed with this strategy")
    
    # =========================================================================
    # Example 3: Expanding vs Rolling Windows
    # =========================================================================
    print("\n" + "=" * 70)
    print("Example 3: Expanding vs Rolling Windows")
    print("=" * 70)
    
    # Rolling window: fixed IS period size
    rolling_optimizer = WalkForwardOptimizer(
        is_days=252,
        oos_days=63,
        expanding=False,  # Rolling window
        metric='sharpe_ratio',
    )
    
    rolling_result = rolling_optimizer.optimize(
        strategy_class=MomentumStrategy,
        param_grid={'lookback': [30, 60, 90]},
        data=data,
    )
    
    # Expanding window: IS grows over time
    expanding_optimizer = WalkForwardOptimizer(
        is_days=252,
        oos_days=63,
        expanding=True,  # Expanding window
        min_is_days=126,  # Start with at least 6 months
        metric='sharpe_ratio',
    )
    
    expanding_result = expanding_optimizer.optimize(
        strategy_class=MomentumStrategy,
        param_grid={'lookback': [30, 60, 90]},
        data=data,
    )
    
    print("\nComparison:")
    print("-" * 50)
    print(f"{'Metric':<25} {'Rolling':<15} {'Expanding':<15}")
    print("-" * 50)
    print(f"{'Windows':<25} {rolling_result.n_windows:<15} {expanding_result.n_windows:<15}")
    print(f"{'Avg IS Score':<25} {rolling_result.avg_is_score:<15.3f} {expanding_result.avg_is_score:<15.3f}")
    print(f"{'Avg OOS Score':<25} {rolling_result.avg_oos_score:<15.3f} {expanding_result.avg_oos_score:<15.3f}")
    print(f"{'Efficiency Ratio':<25} {rolling_result.efficiency_ratio:<15.2%} {expanding_result.efficiency_ratio:<15.2%}")
    print(f"{'Param Stability':<25} {rolling_result.parameter_stability:<15.2%} {expanding_result.parameter_stability:<15.2%}")
    print(f"{'OOS Total Return':<25} {rolling_result.oos_total_return:<15.2%} {expanding_result.oos_total_return:<15.2%}")
    
    # =========================================================================
    # Example 4: Detecting Overfitting
    # =========================================================================
    print("\n" + "=" * 70)
    print("Example 4: Detecting Overfitting")
    print("=" * 70)
    
    # Test with many parameters (more likely to overfit)
    overfit_result = walk_forward_analysis(
        strategy_class=MomentumStrategy,
        param_grid={
            'lookback': [10, 20, 30, 40, 50, 60, 90, 120, 180, 252],
        },
        data=data,
        is_days=252,
        oos_days=63,
    )
    
    print("\n📊 Many Parameters Test (10 lookback values):")
    print(f"   Avg IS Sharpe:  {overfit_result.avg_is_score:.3f}")
    print(f"   Avg OOS Sharpe: {overfit_result.avg_oos_score:.3f}")
    print(f"   Efficiency:     {overfit_result.efficiency_ratio:.2%}")
    print(f"   Param Stability: {overfit_result.parameter_stability:.2%}")
    print(f"   Overfitted?     {'YES ⚠️' if overfit_result.is_overfitted else 'NO ✓'}")
    
    # Test with fewer parameters (more robust)
    robust_result = walk_forward_analysis(
        strategy_class=MomentumStrategy,
        param_grid={
            'lookback': [60, 120],  # Only 2 sensible values
        },
        data=data,
        is_days=252,
        oos_days=63,
    )
    
    print("\n📊 Fewer Parameters Test (2 lookback values):")
    print(f"   Avg IS Sharpe:  {robust_result.avg_is_score:.3f}")
    print(f"   Avg OOS Sharpe: {robust_result.avg_oos_score:.3f}")
    print(f"   Efficiency:     {robust_result.efficiency_ratio:.2%}")
    print(f"   Param Stability: {robust_result.parameter_stability:.2%}")
    print(f"   Overfitted?     {'YES ⚠️' if robust_result.is_overfitted else 'NO ✓'}")
    
    # =========================================================================
    # Example 5: Multiple Strategies Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("Example 5: Strategy Comparison via Walk-Forward")
    print("=" * 70)
    
    strategies = [
        (MomentumStrategy, {'lookback': [30, 60, 90]}),
        (MeanReversionStrategy, {'lookback': [10, 20, 30]}),
    ]
    
    print("\nComparing strategies with walk-forward validation:")
    print("-" * 70)
    
    for strategy_class, param_grid in strategies:
        result = walk_forward_analysis(
            strategy_class=strategy_class,
            param_grid=param_grid,
            data=data,
            is_days=252,
            oos_days=63,
            verbose=False,
        )
        
        status = '✓' if not result.is_overfitted else '⚠️'
        print(f"\n{strategy_class.__name__}:")
        print(f"  OOS Sharpe:      {result.oos_sharpe_ratio:.3f}")
        print(f"  OOS Return:      {result.oos_total_return:.2%}")
        print(f"  OOS Max DD:      {result.oos_max_drawdown:.2%}")
        print(f"  Efficiency:      {result.efficiency_ratio:.2%} {status}")
        print(f"  Param Stability: {result.parameter_stability:.2%}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("WALK-FORWARD ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("""
Key Takeaways:
1. Walk-forward analysis provides TRUE out-of-sample performance
2. Efficiency ratio measures how well parameters generalize
3. Parameter stability shows if optimal params are consistent
4. Use fewer parameters to reduce overfitting risk
5. Always validate with walk-forward before live trading!

Efficiency Ratio Interpretation:
  > 0.8  : Excellent - strategy generalizes very well
  0.5-0.8: Good - acceptable for trading
  0.3-0.5: Moderate - some overfitting, use with caution
  < 0.3  : Poor - significant overfitting, do not trade

Parameter Stability Interpretation:
  > 0.7  : Excellent - same params work across periods
  0.5-0.7: Good - reasonable consistency
  0.3-0.5: Moderate - parameters vary significantly
  < 0.3  : Poor - parameters change every window (bad sign)
""")
    
    print("\n✅ Walk-forward analysis examples completed!")


if __name__ == "__main__":
    main()

"""
Parameter Optimization Example

This example demonstrates how to use grid search to find optimal
strategy parameters.
"""

from jsf.data import load_data
from jsf.strategies import MomentumStrategy, MeanReversionStrategy
from jsf.optimization import optimize_strategy, ParameterGrid, GridSearchOptimizer
from jsf.simulation import BacktestConfig


def example_1_simple_optimization():
    """Example 1: Simple grid search for momentum lookback."""
    print("=" * 70)
    print("Example 1: Optimize Momentum Lookback Period")
    print("=" * 70)
    
    # Load data
    data = load_data(
        source='synthetic',
        symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        start_date='2020-01-01',
        end_date='2023-12-31',
        seed=42
    )
    print(f"\nLoaded {len(data.symbols)} symbols")
    
    # Define parameter grid
    param_grid = {
        'lookback': [30, 60, 90, 120, 180],
    }
    
    print(f"Testing {len(param_grid['lookback'])} lookback periods...")
    
    # Run optimization
    result = optimize_strategy(
        strategy_class=MomentumStrategy,
        param_grid=param_grid,
        data=data,
        metric='sharpe_ratio',
        fixed_params={'long_only': True},
        verbose=True,
    )
    
    # Display results
    print("\n" + "-" * 70)
    print("OPTIMIZATION RESULTS")
    print("-" * 70)
    print(f"Best Lookback: {result.best_params['lookback']} days")
    print(f"Best Sharpe Ratio: {result.best_score:.3f}")
    print(f"Total Return: {result.best_result.total_return:.2%}")
    print(f"Max Drawdown: {result.best_result.max_drawdown:.2%}")
    
    print("\n" + "-" * 70)
    print("ALL RESULTS:")
    print("-" * 70)
    print(result.summary[['lookback', 'sharpe_ratio', 'total_return', 'max_drawdown']])
    
    return result


def example_2_multi_parameter():
    """Example 2: Optimize multiple parameters."""
    print("\n\n" + "=" * 70)
    print("Example 2: Multi-Parameter Optimization")
    print("=" * 70)
    
    # Load data
    data = load_data(
        source='synthetic',
        symbols=['SPY', 'QQQ', 'IWM', 'DIA'],
        start_date='2021-01-01',
        end_date='2023-12-31',
        seed=123
    )
    
    # Define parameter grid
    param_grid = {
        'lookback': [60, 90, 120],
        'long_only': [True, False],
    }
    
    total_combinations = len(param_grid['lookback']) * len(param_grid['long_only'])
    print(f"\nTesting {total_combinations} parameter combinations...")
    
    # Run optimization
    result = optimize_strategy(
        strategy_class=MomentumStrategy,
        param_grid=param_grid,
        data=data,
        metric='sharpe_ratio',
        verbose=True,
    )
    
    # Display results
    print("\n" + "-" * 70)
    print("BEST CONFIGURATION:")
    print("-" * 70)
    for param, value in result.best_params.items():
        print(f"  {param}: {value}")
    print(f"\nPerformance:")
    print(f"  Sharpe Ratio: {result.best_score:.3f}")
    print(f"  Total Return: {result.best_result.total_return:.2%}")
    print(f"  Volatility: {result.best_result.volatility:.2%}")
    print(f"  Max Drawdown: {result.best_result.max_drawdown:.2%}")
    
    print("\n" + "-" * 70)
    print("TOP 3 CONFIGURATIONS:")
    print("-" * 70)
    top3 = result.summary.head(3)[['lookback', 'long_only', 'sharpe_ratio', 
                                     'total_return', 'max_drawdown']]
    print(top3.to_string(index=False))
    
    return result


def example_3_compare_strategies():
    """Example 3: Optimize and compare different strategies."""
    print("\n\n" + "=" * 70)
    print("Example 3: Compare Optimized Strategies")
    print("=" * 70)
    
    # Load data
    data = load_data(
        source='synthetic',
        symbols=['TECH', 'FINANCE', 'HEALTHCARE'],
        start_date='2022-01-01',
        end_date='2023-12-31',
        seed=456
    )
    
    strategies = {
        'Momentum': (MomentumStrategy, {'lookback': [30, 60, 90, 120]}),
        'Mean Reversion': (MeanReversionStrategy, {'lookback': [10, 20, 30, 40]}),
    }
    
    results = {}
    
    for name, (strategy_class, param_grid) in strategies.items():
        print(f"\nOptimizing {name}...")
        result = optimize_strategy(
            strategy_class=strategy_class,
            param_grid=param_grid,
            data=data,
            metric='sharpe_ratio',
            verbose=False,
        )
        results[name] = result
        print(f"  Best Sharpe: {result.best_score:.3f}")
        print(f"  Best Params: {result.best_params}")
    
    # Compare results
    print("\n" + "-" * 70)
    print("STRATEGY COMPARISON (After Optimization):")
    print("-" * 70)
    print(f"{'Strategy':<20} {'Sharpe':>8} {'Return':>10} {'Max DD':>10} {'Params':<20}")
    print("-" * 70)
    
    for name, result in results.items():
        params_str = str(result.best_params)[:20]
        print(f"{name:<20} {result.best_score:>8.2f} "
              f"{result.best_result.total_return:>9.2%} "
              f"{result.best_result.max_drawdown:>9.2%} "
              f"{params_str:<20}")
    
    # Find best overall
    best_strategy = max(results.items(), key=lambda x: x[1].best_score)
    print("\n" + "-" * 70)
    print(f"Winner: {best_strategy[0]} (Sharpe: {best_strategy[1].best_score:.2f})")
    
    return results


def example_4_custom_metric():
    """Example 4: Optimize using different metrics."""
    print("\n\n" + "=" * 70)
    print("Example 4: Optimization with Different Metrics")
    print("=" * 70)
    
    # Load data
    data = load_data(
        source='synthetic',
        symbols=['AAPL', 'GOOGL'],
        start_date='2022-01-01',
        end_date='2023-12-31',
        seed=789
    )
    
    param_grid = {'lookback': [30, 60, 90, 120]}
    metrics = ['sharpe_ratio', 'total_return', 'calmar_ratio']
    
    results = {}
    
    for metric in metrics:
        print(f"\nOptimizing for {metric}...")
        result = optimize_strategy(
            strategy_class=MomentumStrategy,
            param_grid=param_grid,
            data=data,
            metric=metric,
            fixed_params={'long_only': True},
            verbose=False,
        )
        results[metric] = result
    
    # Compare
    print("\n" + "-" * 70)
    print("DIFFERENT OPTIMIZATION TARGETS:")
    print("-" * 70)
    print(f"{'Metric':<20} {'Best Lookback':>15} {'Sharpe':>8} {'Return':>10}")
    print("-" * 70)
    
    for metric, result in results.items():
        print(f"{metric:<20} {result.best_params['lookback']:>15} "
              f"{result.best_result.sharpe_ratio:>8.2f} "
              f"{result.best_result.total_return:>9.2%}")
    
    print("\nNote: Different metrics can lead to different optimal parameters!")
    
    return results


def main():
    """Run all optimization examples."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  JSF-Core Parameter Optimization Examples".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    try:
        example_1_simple_optimization()
        example_2_multi_parameter()
        example_3_compare_strategies()
        example_4_custom_metric()
        
        print("\n" + "=" * 70)
        print("ALL OPTIMIZATION EXAMPLES COMPLETE!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  - Grid search tests all parameter combinations systematically")
        print("  - Different metrics (Sharpe, return, Calmar) may favor different parameters")
        print("  - Multi-parameter optimization finds best combinations")
        print("  - Compare optimized strategies to select best approach")
        print("\nNext steps:")
        print("  - Use walk-forward analysis for out-of-sample validation")
        print("  - Try Bayesian optimization for larger parameter spaces")
        print("  - Add custom metrics based on your objectives")
        print("\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

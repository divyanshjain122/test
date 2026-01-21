"""
Complete End-to-End Backtesting Example

This example demonstrates the full workflow from data loading through
strategy execution to performance analysis.
"""

from jsf.data import load_data
from jsf.strategies import MomentumStrategy, MeanReversionStrategy, TrendFollowingStrategy
from jsf.simulation import BacktestEngine, BacktestConfig
from jsf.simulation import calculate_all_metrics
from jsf.visualization import plot_performance_summary

def example_1_basic_backtest():
    """Example 1: Basic momentum strategy backtest."""
    print("=" * 70)
    print("Example 1: Basic Momentum Strategy Backtest")
    print("=" * 70)
    
    # Load historical data
    data = load_data(
        source='synthetic',
        symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        start_date='2020-01-01',
        end_date='2023-12-31',
        annual_return=0.12,
        annual_volatility=0.25,
        seed=42
    )
    print(f"\nLoaded {len(data.data)} days of data for {len(data.symbols)} symbols")
    
    # Create strategy
    strategy = MomentumStrategy(
        name="momentum_60d",
        lookback=60,
        long_only=True
    )
    print(f"Created strategy: {strategy.name}")
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000,
        transaction_cost=0.001,  # 10 bps
        slippage=0.0005,  # 5 bps
    )
    
    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run_strategy(strategy, data)
    
    # Display results
    print("\n" + "-" * 70)
    print("BACKTEST RESULTS")
    print("-" * 70)
    print(result.summary())
    
    print(f"\nFinal Portfolio Value: ${result.equity_curve.iloc[-1]:,.2f}")
    print(f"Total Trades: {len(result.trades)}")
    
    return result


def example_2_compare_strategies():
    """Example 2: Compare multiple strategies."""
    print("\n\n" + "=" * 70)
    print("Example 2: Strategy Comparison")
    print("=" * 70)
    
    # Load data
    data = load_data(
        source='synthetic',
        symbols=['TECH', 'FINANCE', 'HEALTHCARE', 'ENERGY', 'CONSUMER'],
        start_date='2021-01-01',
        end_date='2023-12-31',
        seed=123
    )
    
    # Create multiple strategies
    strategies = {
        'Momentum (60d)': MomentumStrategy(lookback=60),
        'Momentum (120d)': MomentumStrategy(lookback=120),
        'Mean Reversion': MeanReversionStrategy(lookback=20),
        'Trend Following': TrendFollowingStrategy(fast_period=50, slow_period=200),
    }
    
    # Run backtests
    engine = BacktestEngine(BacktestConfig(initial_capital=100000))
    results = {}
    
    print(f"\nTesting {len(strategies)} strategies...")
    for name, strategy in strategies.items():
        result = engine.run_strategy(strategy, data)
        results[name] = result
    
    # Compare results
    print("\n" + "-" * 70)
    print("STRATEGY COMPARISON")
    print("-" * 70)
    print(f"{'Strategy':<20} {'Return':>10} {'Sharpe':>8} {'Max DD':>10}")
    print("-" * 70)
    
    for name, result in results.items():
        print(f"{name:<20} {result.total_return:>9.2%} {result.sharpe_ratio:>8.2f} {result.max_drawdown:>9.2%}")
    
    # Find best strategy
    best_sharpe = max(results.items(), key=lambda x: x[1].sharpe_ratio)
    best_return = max(results.items(), key=lambda x: x[1].total_return)
    
    print("\n" + "-" * 70)
    print(f"Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1].sharpe_ratio:.2f})")
    print(f"Best Total Return: {best_return[0]} ({best_return[1].total_return:.2%})")
    
    return results


def example_3_detailed_metrics():
    """Example 3: Detailed performance metrics analysis."""
    print("\n\n" + "=" * 70)
    print("Example 3: Detailed Performance Metrics")
    print("=" * 70)
    
    # Load data
    data = load_data(
        source='synthetic',
        symbols=['SPY', 'QQQ', 'IWM'],
        start_date='2022-01-01',
        end_date='2023-12-31',
        seed=456
    )
    
    # Create and run strategy
    strategy = MomentumStrategy(lookback=90)
    engine = BacktestEngine(BacktestConfig(initial_capital=100000))
    result = engine.run_strategy(strategy, data)
    
    # Calculate all metrics
    all_metrics = calculate_all_metrics(result.returns)
    
    print("\n" + "-" * 70)
    print("COMPREHENSIVE METRICS")
    print("-" * 70)
    
    print("\nReturn Metrics:")
    print(f"  Total Return:        {all_metrics['total_return']:>10.2%}")
    print(f"  CAGR:                {all_metrics['cagr']:>10.2%}")
    print(f"  Mean Daily Return:   {all_metrics['mean_return']:>10.2%}")
    
    print("\nRisk Metrics:")
    print(f"  Volatility:          {all_metrics['volatility']:>10.2%}")
    print(f"  Downside Deviation:  {all_metrics['downside_deviation']:>10.2%}")
    print(f"  Max Drawdown:        {all_metrics['max_drawdown']:>10.2%}")
    print(f"  VaR (95%):           {all_metrics['var_95']:>10.2%}")
    print(f"  CVaR (95%):          {all_metrics['cvar_95']:>10.2%}")
    
    print("\nRisk-Adjusted Returns:")
    print(f"  Sharpe Ratio:        {all_metrics['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio:       {all_metrics['sortino_ratio']:>10.2f}")
    print(f"  Calmar Ratio:        {all_metrics['calmar_ratio']:>10.2f}")
    
    print("\nTrading Metrics:")
    print(f"  Win Rate:            {all_metrics['win_rate']:>10.2%}")
    print(f"  Profit Factor:       {all_metrics['profit_factor']:>10.2f}")
    print(f"  Best Day:            {all_metrics['best_day']:>10.2%}")
    print(f"  Worst Day:           {all_metrics['worst_day']:>10.2%}")
    print(f"  Avg Win:             {all_metrics['avg_win']:>10.2%}")
    print(f"  Avg Loss:            {all_metrics['avg_loss']:>10.2%}")
    
    print("\nDistribution:")
    print(f"  Skewness:            {all_metrics['skewness']:>10.2f}")
    print(f"  Kurtosis:            {all_metrics['kurtosis']:>10.2f}")
    
    return all_metrics


def example_4_transaction_costs():
    """Example 4: Impact of transaction costs."""
    print("\n\n" + "=" * 70)
    print("Example 4: Transaction Cost Analysis")
    print("=" * 70)
    
    # Load data
    data = load_data(
        source='synthetic',
        symbols=['STOCK_A', 'STOCK_B', 'STOCK_C'],
        start_date='2022-01-01',
        end_date='2023-12-31',
        seed=789
    )
    
    # Test different cost levels
    strategy = MomentumStrategy(lookback=30)  # Shorter lookback = more trading
    
    cost_scenarios = {
        'No Costs': (0.0000, 0.0000),
        'Low Costs (5bps)': (0.0005, 0.0000),
        'Medium Costs (15bps)': (0.0010, 0.0005),
        'High Costs (30bps)': (0.0020, 0.0010),
    }
    
    print("\n" + "-" * 70)
    print("TRANSACTION COST IMPACT")
    print("-" * 70)
    print(f"{'Scenario':<20} {'Return':>10} {'Sharpe':>8} {'# Trades':>10}")
    print("-" * 70)
    
    for scenario, (txn_cost, slippage) in cost_scenarios.items():
        config = BacktestConfig(
            initial_capital=100000,
            transaction_cost=txn_cost,
            slippage=slippage
        )
        engine = BacktestEngine(config)
        result = engine.run_strategy(strategy, data)
        
        print(f"{scenario:<20} {result.total_return:>9.2%} {result.sharpe_ratio:>8.2f} {len(result.trades):>10}")
    
    print("\nNote: Higher transaction costs reduce returns proportionally to trading frequency")


def example_5_visualization():
    """Example 5: Visualizing backtest results."""
    print("\n\n" + "=" * 70)
    print("Example 5: Visualization")
    print("=" * 70)
    
    # Load data and run backtest
    data = load_data(
        source='synthetic',
        symbols=['TECH_A', 'TECH_B', 'TECH_C'],
        start_date='2021-01-01',
        end_date='2023-12-31',
        seed=999
    )
    
    strategy = MomentumStrategy(lookback=90)
    engine = BacktestEngine(BacktestConfig(initial_capital=100000))
    result = engine.run_strategy(strategy, data)
    
    print(f"\nBacktest complete: {result.total_return:.2%} return, {result.sharpe_ratio:.2f} Sharpe")
    
    # Generate comprehensive performance dashboard
    print("\nGenerating performance dashboard...")
    plot_performance_summary(
        result,
        title="Example 5: Momentum Strategy Performance",
        show=False,
        save_path="example_5_dashboard.png"
    )
    
    print("  ✓ Saved: example_5_dashboard.png")
    print("\nThe dashboard includes:")
    print("  - Equity curve over time")
    print("  - Drawdown chart")
    print("  - Returns distribution")
    print("  - Rolling 60-day Sharpe ratio")
    print("  - Complete performance metrics table")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  JSF-Core Complete Backtesting Examples".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    try:
        example_1_basic_backtest()
        example_2_compare_strategies()
        example_3_detailed_metrics()
        example_4_transaction_costs()
        example_5_visualization()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETE!")
        print("=" * 70)
        print("\nYou now have a complete quantitative trading system with:")
        print("  - Data loading and preprocessing")
        print("  - Signal generation")
        print("  - Portfolio construction")
        print("  - Strategy templates")
        print("  - Backtesting engine")
        print("  - Performance metrics")
        print("  - Professional visualizations")
        print("\nGenerated files:")
        print("  - example_5_dashboard.png (comprehensive performance analysis)")
        print("\nNext steps:")
        print("  - Test with real market data")
        print("  - Optimize strategy parameters")
        print("  - Add custom strategies")
        print("  - Implement walk-forward analysis")
        print("\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

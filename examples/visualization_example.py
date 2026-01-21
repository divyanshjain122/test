"""
Visualization Examples for JSF-Core

This example demonstrates all available plotting functions
for analyzing backtest results.
"""

from jsf.data import load_data
from jsf.strategies import MomentumStrategy
from jsf.simulation import BacktestEngine, BacktestConfig
from jsf.visualization import (
    plot_equity_curve,
    plot_drawdown,
    plot_returns_distribution,
    plot_rolling_metrics,
    plot_monthly_returns,
    plot_performance_summary,
)


def main():
    """Run visualization examples."""
    print("=" * 70)
    print("JSF-Core Visualization Examples")
    print("=" * 70)
    
    # Load data and run backtest
    print("\n1. Loading data and running backtest...")
    data = load_data(
        source='synthetic',
        symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
        start_date='2020-01-01',
        end_date='2023-12-31',
        annual_return=0.15,
        annual_volatility=0.30,
        seed=42
    )
    
    strategy = MomentumStrategy(lookback=90)
    config = BacktestConfig(initial_capital=100000)
    engine = BacktestEngine(config)
    result = engine.run_strategy(strategy, data)
    
    print(f"   Backtest complete: {result.total_return:.2%} return, "
          f"{result.sharpe_ratio:.2f} Sharpe")
    
    # Example 1: Equity Curve
    print("\n2. Plotting equity curve...")
    plot_equity_curve(
        result,
        title="Momentum Strategy - Equity Curve",
        show=False,
        save_path="equity_curve.png"
    )
    print("   Saved: equity_curve.png")
    
    # Example 2: Drawdown Chart
    print("\n3. Plotting drawdown chart...")
    plot_drawdown(
        result,
        title="Momentum Strategy - Drawdown Analysis",
        show=False,
        save_path="drawdown.png"
    )
    print("   Saved: drawdown.png")
    
    # Example 3: Returns Distribution
    print("\n4. Plotting returns distribution...")
    plot_returns_distribution(
        result,
        bins=50,
        title="Momentum Strategy - Returns Distribution",
        show=False,
        save_path="returns_distribution.png"
    )
    print("   Saved: returns_distribution.png")
    
    # Example 4: Rolling Metrics
    print("\n5. Plotting rolling metrics...")
    plot_rolling_metrics(
        result,
        window=60,
        title="Momentum Strategy - 60-Day Rolling Metrics",
        show=False,
        save_path="rolling_metrics.png"
    )
    print("   Saved: rolling_metrics.png")
    
    # Example 5: Monthly Returns Heatmap
    print("\n6. Plotting monthly returns heatmap...")
    plot_monthly_returns(
        result,
        title="Momentum Strategy - Monthly Returns",
        show=False,
        save_path="monthly_returns.png"
    )
    print("   Saved: monthly_returns.png")
    
    # Example 6: Performance Summary Dashboard
    print("\n7. Plotting performance summary dashboard...")
    plot_performance_summary(
        result,
        title="Momentum Strategy - Complete Performance Analysis",
        show=False,
        save_path="performance_summary.png"
    )
    print("   Saved: performance_summary.png")
    
    print("\n" + "=" * 70)
    print("All visualizations generated successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - equity_curve.png")
    print("  - drawdown.png")
    print("  - returns_distribution.png")
    print("  - rolling_metrics.png")
    print("  - monthly_returns.png")
    print("  - performance_summary.png")
    print("\nThese plots provide comprehensive analysis of:")
    print("  - Portfolio performance over time")
    print("  - Risk metrics and drawdowns")
    print("  - Returns distribution and normality")
    print("  - Rolling performance metrics")
    print("  - Monthly/periodic performance")
    print("  - Complete dashboard summary")
    print()


if __name__ == "__main__":
    main()

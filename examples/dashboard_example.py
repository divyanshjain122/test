"""Dashboard Example

This example demonstrates how to use the monitoring dashboard
with live trading or paper trading.

Features demonstrated:
1. Connecting dashboard to a broker
2. Using demo mode with mock data
3. Running the Streamlit dashboard
4. Programmatic access to metrics
"""

import time
from datetime import datetime

# Dashboard components
from jsf.dashboard import (
    DashboardState,
    DashboardConfig,
    PortfolioSnapshot,
    PositionSnapshot,
    TradeRecord,
    RiskMetrics,
    PerformanceMetrics,
    DataCollector,
    MockDataCollector,
    SnapshotHistory,
    MetricsCalculator,
    calculate_drawdown,
    calculate_sharpe,
    calculate_sortino,
    calculate_var,
)


def demo_mock_collector():
    """Demonstrate using MockDataCollector for testing."""
    print("=" * 60)
    print("MockDataCollector Demo")
    print("=" * 60)
    
    # Create mock collector with initial settings
    collector = MockDataCollector(
        initial_capital=100000.0,
        symbols=["AAPL", "GOOGL", "MSFT", "AMZN"],
    )
    
    print(f"Initial capital: ${collector.initial_capital:,.2f}")
    print(f"Symbols: {collector.symbols}")
    
    # Collect some snapshots
    print("\nCollecting 10 snapshots...")
    for i in range(10):
        snapshot = collector.collect_snapshot()
        print(f"  [{i+1}] Equity: ${snapshot.equity:,.2f}, "
              f"Positions: {snapshot.num_positions}, "
              f"Daily P&L: ${snapshot.daily_pnl:+,.2f}")
        time.sleep(0.1)
    
    # Get equity series
    equity = collector.history.get_equity_series()
    print(f"\nEquity series: {len(equity)} data points")
    print(f"  Start: ${equity.iloc[0]:,.2f}")
    print(f"  End: ${equity.iloc[-1]:,.2f}")
    print(f"  Change: ${equity.iloc[-1] - equity.iloc[0]:+,.2f}")


def demo_metrics_calculator():
    """Demonstrate using MetricsCalculator."""
    print("\n" + "=" * 60)
    print("MetricsCalculator Demo")
    print("=" * 60)
    
    # Generate sample data using mock collector
    collector = MockDataCollector(initial_capital=100000.0)
    
    print("Generating 100 snapshots for analysis...")
    for _ in range(100):
        collector.collect_snapshot()
    
    # Get equity series
    equity = collector.history.get_equity_series()
    
    # Create calculator
    calculator = MetricsCalculator(risk_free_rate=0.05)
    
    # Calculate risk metrics
    print("\n--- Risk Metrics ---")
    risk_metrics = calculator.calculate_risk_metrics(equity)
    
    print(f"Max Drawdown: {risk_metrics.max_drawdown:.2f}%")
    print(f"Current Drawdown: {risk_metrics.current_drawdown:.2f}%")
    print(f"VaR 95%: {risk_metrics.var_95:.2f}%")
    print(f"Volatility (Ann.): {risk_metrics.volatility:.2f}%")
    print(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {risk_metrics.sortino_ratio:.2f}")
    
    # Calculate performance metrics
    print("\n--- Performance Metrics ---")
    perf = calculator.calculate_performance_metrics(equity, initial_capital=100000.0)
    
    print(f"1-Day Return: {perf.return_1d:+.2f}%")
    print(f"Total Return: {perf.return_total:+.2f}%")
    print(f"Best Day: {perf.best_day:+.2f}%")
    print(f"Worst Day: {perf.worst_day:+.2f}%")
    print(f"Win Rate: {perf.win_rate:.1f}%")
    print(f"Profit Factor: {perf.profit_factor:.2f}")


def demo_individual_metrics():
    """Demonstrate individual metric functions."""
    print("\n" + "=" * 60)
    print("Individual Metrics Demo")
    print("=" * 60)
    
    import pandas as pd
    import numpy as np
    
    # Generate sample equity curve
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    
    # Simulate random walk with slight upward drift
    returns = np.random.normal(0.0005, 0.015, 252)
    equity = pd.Series(
        100000 * (1 + returns).cumprod(),
        index=dates
    )
    
    print(f"Sample equity curve: {len(equity)} days")
    print(f"  Start: ${equity.iloc[0]:,.2f}")
    print(f"  End: ${equity.iloc[-1]:,.2f}")
    
    # Calculate individual metrics
    print("\n--- Drawdown ---")
    dd_series, max_dd, current_dd = calculate_drawdown(equity)
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Current Drawdown: {current_dd:.2f}%")
    
    print("\n--- Risk Metrics ---")
    from jsf.dashboard.metrics import calculate_returns
    rets = calculate_returns(equity)
    
    sharpe = calculate_sharpe(rets)
    sortino = calculate_sortino(rets)
    var_95 = calculate_var(rets, confidence=0.95)
    var_99 = calculate_var(rets, confidence=0.99)
    
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Sortino Ratio: {sortino:.2f}")
    print(f"VaR 95%: {var_95:.2f}%")
    print(f"VaR 99%: {var_99:.2f}%")


def demo_dashboard_state():
    """Demonstrate using DashboardState."""
    print("\n" + "=" * 60)
    print("DashboardState Demo")
    print("=" * 60)
    
    # Create dashboard state
    state = DashboardState(
        config=DashboardConfig(
            theme="dark",
            show_notifications=True,
            max_trade_history=100,
        ),
        initial_capital=100000.0,
    )
    
    print(f"Theme: {state.config.theme}")
    print(f"Connected: {state.is_connected}")
    
    # Add some alerts
    state.add_alert("Dashboard started", "info")
    state.add_alert("Large trade detected", "warning")
    
    print(f"Alerts: {len(state.alerts)}")
    for alert in state.alerts:
        print(f"  [{alert['level']}] {alert['message']}")
    
    # Add equity history
    now = datetime.now()
    state.update_equity_history(now, 100000.0)
    state.update_equity_history(now, 100500.0)
    state.update_equity_history(now, 101000.0)
    
    print(f"\nEquity history points: {len(state.equity_history)}")
    
    # Get returns
    returns = state.get_returns_series()
    print(f"Returns: {len(returns)} data points")


def demo_with_broker():
    """Demonstrate connecting dashboard to a broker."""
    print("\n" + "=" * 60)
    print("Broker Integration Demo")
    print("=" * 60)
    
    from jsf.broker import PaperBroker, Order, OrderSide, OrderType
    
    # Create paper broker
    broker = PaperBroker(initial_cash=100000)
    broker.connect()
    
    # Set some prices
    broker.set_price("AAPL", 150.0)
    broker.set_price("GOOGL", 140.0)
    broker.set_price("MSFT", 380.0)
    
    # Execute some trades
    orders = [
        Order(symbol="AAPL", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET),
        Order(symbol="GOOGL", side=OrderSide.BUY, quantity=50, order_type=OrderType.MARKET),
        Order(symbol="MSFT", side=OrderSide.BUY, quantity=25, order_type=OrderType.MARKET),
    ]
    
    for order in orders:
        result = broker.submit_order(order)
        print(f"Submitted: {order.side.value} {order.quantity} {order.symbol}")
    
    # Create data collector connected to broker
    collector = DataCollector(
        broker=broker,
        initial_capital=100000.0,
    )
    
    # Collect snapshot
    snapshot = collector.collect_snapshot()
    
    print(f"\n--- Portfolio Snapshot ---")
    print(f"Equity: ${snapshot.equity:,.2f}")
    print(f"Cash: ${snapshot.cash:,.2f}")
    print(f"Portfolio Value: ${snapshot.portfolio_value:,.2f}")
    print(f"Positions: {snapshot.num_positions}")
    
    print("\n--- Positions ---")
    for pos in snapshot.positions:
        print(f"  {pos.symbol}: {pos.quantity} shares @ ${pos.current_price:.2f} "
              f"(Weight: {pos.weight:.1f}%)")
    
    broker.disconnect()


def how_to_run_dashboard():
    """Instructions for running the Streamlit dashboard."""
    print("\n" + "=" * 60)
    print("How to Run the Streamlit Dashboard")
    print("=" * 60)
    
    print("""
To run the interactive Streamlit dashboard:

1. Make sure Streamlit and Plotly are installed:
   pip install streamlit plotly

2. Run the dashboard in demo mode:
   streamlit run src/jsf/dashboard/app.py

3. The dashboard will open in your browser automatically.

4. Click "Start Demo Mode" in settings to see sample data.

5. To connect to a real broker, use the dashboard module programmatically:

   from jsf.dashboard.app import connect_broker
   from jsf.broker import PaperBroker
   
   broker = PaperBroker(initial_cash=100000)
   broker.connect()
   
   # In your Streamlit app or before running:
   connect_broker(broker, initial_capital=100000)

Dashboard Features:
- Real-time portfolio monitoring
- P&L tracking with equity curve
- Trade history and analysis
- Risk metrics and drawdown charts
- Position concentration analysis
- Auto-refresh with configurable intervals
""")


def main():
    """Run all demos."""
    print("\nJSF Dashboard Module Examples")
    print("=" * 60)
    
    demo_mock_collector()
    demo_metrics_calculator()
    demo_individual_metrics()
    demo_dashboard_state()
    demo_with_broker()
    how_to_run_dashboard()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

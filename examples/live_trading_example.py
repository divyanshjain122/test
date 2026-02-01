"""Live Trading Example

This example demonstrates how to use the live trading module
for paper trading with simulated or real-time data.

Features demonstrated:
1. Setting up a paper broker with the data handler
2. Creating and configuring the live trading engine
3. Implementing a simple trading strategy
4. Monitoring positions and account status
5. Handling callbacks for trades and state changes
"""

import time
from datetime import datetime
from typing import Dict

from jsf.broker import PaperBroker, Order, OrderSide, OrderType
from jsf.live import (
    LiveTradingEngine,
    SimulatedDataHandler,
    EngineConfig,
    TradingState,
)


def basic_paper_trading():
    """Basic paper trading example."""
    print("=" * 60)
    print("Basic Paper Trading Example")
    print("=" * 60)
    
    # Create a paper broker with initial capital
    broker = PaperBroker(initial_cash=100000)
    broker.connect()
    
    # Set initial prices (required for paper broker)
    broker.set_price("AAPL", 150.0)
    broker.set_price("GOOGL", 140.0)
    broker.set_price("MSFT", 380.0)
    
    # Create a simulated data handler
    data_handler = SimulatedDataHandler(
        initial_prices={"AAPL": 150.0, "GOOGL": 140.0, "MSFT": 380.0},
        volatility=0.005,  # 0.5% price volatility
        update_interval=0.5,  # Update every 0.5 seconds
        seed=42,  # For reproducibility
    )
    
    # Configure the engine
    config = EngineConfig(
        initial_capital=100000,
        max_position_size=0.3,  # Max 30% per position
        max_total_exposure=1.0,  # Max 100% invested
        trading_interval=1.0,  # Trade every 1 second
        data_warmup_seconds=2.0,  # Wait 2 seconds for data
        max_daily_loss=0.02,  # Stop if down 2%
    )
    
    # Create the live trading engine
    engine = LiveTradingEngine(
        broker=broker,
        data_handler=data_handler,
        symbols=["AAPL", "GOOGL", "MSFT"],
        config=config,
    )
    
    # Define a simple equal-weight strategy
    def equal_weight_strategy(engine: LiveTradingEngine, prices: Dict[str, float]) -> Dict[str, float]:
        """Simple equal-weight allocation strategy."""
        if not prices:
            return {}
        
        # Get current positions
        positions = engine.get_positions()
        
        # If we don't have positions yet, buy equal weights
        if not positions:
            symbols = list(prices.keys())
            weight = 0.9 / len(symbols)  # 90% invested, 10% cash
            return {s: weight for s in symbols}
        
        # Otherwise, no rebalancing needed
        return {}
    
    engine.set_strategy(equal_weight_strategy)
    
    # Start trading
    print("Starting live trading engine...")
    engine.start()
    
    # Run for 10 seconds
    for i in range(5):
        time.sleep(2)
        status = engine.get_status()
        print(f"[{i*2}s] State: {status['state']}, "
              f"Equity: ${status['equity']:,.2f}, "
              f"Positions: {status['positions']}, "
              f"Daily P&L: {status['daily_return']}")
    
    # Stop trading
    print("\nStopping engine...")
    engine.stop()
    
    # Final summary
    print("\n" + "-" * 40)
    print("Final Account Summary:")
    account = broker.get_account()
    print(f"  Cash: ${account.cash:,.2f}")
    print(f"  Portfolio Value: ${account.portfolio_value:,.2f}")
    print(f"  Equity: ${account.equity:,.2f}")
    
    positions = broker.get_positions()
    if positions:
        print("\nPositions:")
        for pos in positions:
            print(f"  {pos.symbol}: {pos.quantity} shares @ ${pos.avg_cost:.2f}")


def strategy_with_callbacks():
    """Example with event callbacks."""
    print("\n" + "=" * 60)
    print("Strategy with Callbacks Example")
    print("=" * 60)
    
    # Setup
    broker = PaperBroker(initial_cash=50000)
    broker.connect()
    broker.set_price("SPY", 450.0)
    broker.set_price("QQQ", 380.0)
    
    data_handler = SimulatedDataHandler(
        initial_prices={"SPY": 450.0, "QQQ": 380.0},
        volatility=0.003,
        update_interval=0.3,
    )
    
    config = EngineConfig(
        initial_capital=50000,
        trading_interval=0.5,
        data_warmup_seconds=1.0,
    )
    
    engine = LiveTradingEngine(
        broker=broker,
        data_handler=data_handler,
        symbols=["SPY", "QQQ"],
        config=config,
    )
    
    # Track events
    trade_count = [0]
    
    def on_trade(tracker):
        trade_count[0] += 1
        print(f"  Trade #{trade_count[0]}: {tracker.order.side.value} "
              f"{tracker.order.quantity} {tracker.symbol}")
    
    def on_state_change(data):
        print(f"  State changed: {data['old'].value} -> {data['new'].value}")
    
    def on_cycle(data):
        # Print every 5th cycle
        if hasattr(on_cycle, 'count'):
            on_cycle.count += 1
        else:
            on_cycle.count = 1
        
        if on_cycle.count % 5 == 0:
            prices = data['prices']
            print(f"  Cycle {on_cycle.count}: SPY=${prices.get('SPY', 0):.2f}, "
                  f"QQQ=${prices.get('QQQ', 0):.2f}")
    
    # Register callbacks
    engine.register_callback("on_trade", on_trade)
    engine.register_callback("on_state_change", on_state_change)
    engine.register_callback("on_cycle", on_cycle)
    
    # Simple momentum strategy
    price_history = {"SPY": [], "QQQ": []}
    
    def momentum_strategy(engine: LiveTradingEngine, prices: Dict[str, float]) -> Dict[str, float]:
        """Buy symbols with positive momentum."""
        targets = {}
        
        for symbol, price in prices.items():
            price_history[symbol].append(price)
            
            # Need at least 5 prices for momentum
            if len(price_history[symbol]) >= 5:
                recent = price_history[symbol][-5:]
                momentum = (recent[-1] - recent[0]) / recent[0]
                
                # Buy if positive momentum
                if momentum > 0:
                    targets[symbol] = 0.4
                else:
                    targets[symbol] = 0.0
        
        return targets
    
    engine.set_strategy(momentum_strategy)
    
    print("Starting with callbacks...")
    engine.start()
    
    time.sleep(8)
    
    print("\nPausing engine...")
    engine.pause()
    time.sleep(2)
    
    print("Resuming engine...")
    engine.resume()
    time.sleep(3)
    
    engine.stop()
    
    print(f"\nTotal trades executed: {trade_count[0]}")


def manual_order_management():
    """Example of manual order management."""
    print("\n" + "=" * 60)
    print("Manual Order Management Example")
    print("=" * 60)
    
    broker = PaperBroker(initial_cash=100000)
    broker.connect()
    broker.set_price("NVDA", 500.0)
    broker.set_price("AMD", 150.0)
    
    data_handler = SimulatedDataHandler(
        initial_prices={"NVDA": 500.0, "AMD": 150.0},
        volatility=0.01,
        update_interval=0.5,
    )
    
    config = EngineConfig(
        initial_capital=100000,
        trading_interval=1.0,
        data_warmup_seconds=1.0,
    )
    
    engine = LiveTradingEngine(
        broker=broker,
        data_handler=data_handler,
        symbols=["NVDA", "AMD"],
        config=config,
    )
    
    # Start engine WITHOUT a strategy (manual trading)
    engine.start()
    time.sleep(2)  # Wait for data
    
    # Manual order submission through the order manager
    print("\nSubmitting manual orders...")
    
    # Buy NVDA
    order1 = Order(
        symbol="NVDA",
        side=OrderSide.BUY,
        quantity=50,
        order_type=OrderType.MARKET,
    )
    tracker1 = engine.order_manager.submit_order(order1)
    print(f"Order 1: {tracker1.order.side.value} {tracker1.order.quantity} {tracker1.symbol} "
          f"- Status: {tracker1.state.value}")
    
    # Buy AMD
    order2 = Order(
        symbol="AMD",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
    )
    tracker2 = engine.order_manager.submit_order(order2)
    print(f"Order 2: {tracker2.order.side.value} {tracker2.order.quantity} {tracker2.symbol} "
          f"- Status: {tracker2.state.value}")
    
    # Check positions
    time.sleep(1)
    print("\nCurrent positions:")
    for symbol, pos in engine.get_positions().items():
        print(f"  {symbol}: {pos.quantity} shares")
    
    # Close one position
    print("\nClosing NVDA position...")
    close_tracker = engine.close_position("NVDA")
    if close_tracker:
        print(f"Close order: {close_tracker.state.value}")
    
    time.sleep(1)
    
    # Check positions again
    print("\nPositions after close:")
    for symbol, pos in engine.get_positions().items():
        print(f"  {symbol}: {pos.quantity} shares")
    
    # Close all remaining
    print("\nClosing all positions...")
    close_trackers = engine.close_all_positions()
    print(f"Closed {len(close_trackers)} positions")
    
    engine.stop()
    
    # Final account state
    account = broker.get_account()
    print(f"\nFinal equity: ${account.equity:,.2f}")


def monitoring_example():
    """Example of monitoring trading activity."""
    print("\n" + "=" * 60)
    print("Monitoring Example")
    print("=" * 60)
    
    broker = PaperBroker(initial_cash=100000)
    broker.connect()
    broker.set_price("TSLA", 250.0)
    
    data_handler = SimulatedDataHandler(
        initial_prices={"TSLA": 250.0},
        volatility=0.02,  # Higher volatility for TSLA
        update_interval=0.2,
    )
    
    config = EngineConfig(
        initial_capital=100000,
        trading_interval=0.5,
        data_warmup_seconds=1.0,
        max_daily_loss=0.01,  # 1% daily loss limit
    )
    
    engine = LiveTradingEngine(
        broker=broker,
        data_handler=data_handler,
        symbols=["TSLA"],
        config=config,
    )
    
    # Strategy that buys and holds
    bought = [False]
    
    def buy_and_hold(engine: LiveTradingEngine, prices: Dict[str, float]) -> Dict[str, float]:
        if not bought[0]:
            bought[0] = True
            return {"TSLA": 0.8}  # 80% allocation
        return {}
    
    engine.set_strategy(buy_and_hold)
    
    print("Starting engine with monitoring...")
    engine.start()
    
    # Monitor for 10 seconds
    for i in range(10):
        time.sleep(1)
        
        # Get current state
        status = engine.get_status()
        account = broker.get_account()
        
        print(f"[{i+1}s] Equity: ${account.equity:,.2f} | "
              f"Daily P&L: ${engine.daily_pnl:,.2f} ({engine.daily_return:.2%}) | "
              f"State: {status['state']}")
        
        # Check if engine paused due to loss limit
        if engine.state == TradingState.PAUSED:
            print("    Engine paused due to risk limit!")
            break
    
    engine.stop()
    
    # Get equity curve
    curve = engine.get_equity_curve()
    if len(curve) > 0:
        print(f"\nEquity curve: {len(curve)} data points")
        print(f"  Start: ${curve.iloc[0]:,.2f}")
        print(f"  End: ${curve.iloc[-1]:,.2f}")
        print(f"  Min: ${curve.min():,.2f}")
        print(f"  Max: ${curve.max():,.2f}")


def main():
    """Run all examples."""
    print("\nJSF Live Trading Module Examples")
    print("================================\n")
    
    # Run examples
    basic_paper_trading()
    strategy_with_callbacks()
    manual_order_management()
    monitoring_example()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Broker Integration Example (Phase 14).

This example demonstrates how to use the broker module for paper trading,
including order submission, position management, and account tracking.

The broker module provides:
- Abstract Broker interface for consistent API across brokers
- PaperBroker for simulated paper trading
- AlpacaBroker for live paper/real trading via Alpaca API

This example focuses on PaperBroker, which requires no external API keys
and is perfect for testing strategies before going live.
"""

from datetime import datetime
import sys
sys.path.insert(0, 'src')

from jsf.broker import (
    PaperBroker,
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)


def basic_paper_trading():
    """Demonstrate basic paper trading functionality."""
    print("=" * 60)
    print("BASIC PAPER TRADING EXAMPLE")
    print("=" * 60)
    
    # Create a paper broker with $100,000 initial capital
    broker = PaperBroker(
        initial_cash=100000,
        commission=0.0,      # No commission for this example
        slippage=0.0,        # No slippage for this example
        name="basic_demo"
    )
    
    # Connect to the broker
    broker.connect()
    print(f"\n[OK] Connected to {broker.name}")
    
    # Set current market prices (required for paper trading)
    broker.set_price("AAPL", 150.0)
    broker.set_price("GOOGL", 175.0)
    broker.set_price("MSFT", 400.0)
    print("[OK] Set prices: AAPL=$150, GOOGL=$175, MSFT=$400")
    
    # Check initial account state
    account = broker.get_account()
    print(f"\nInitial Account State:")
    print(f"   Cash: ${account.cash:,.2f}")
    print(f"   Portfolio Value: ${account.portfolio_value:,.2f}")
    
    # Submit some buy orders
    print("\nSubmitting Orders...")
    
    result1 = broker.buy("AAPL", 100)  # Buy 100 shares of AAPL
    status1 = "[OK] Filled" if result1.success else "[FAIL]"
    print(f"   BUY 100 AAPL @ ${result1.order.avg_fill_price:.2f} - {status1}")
    
    result2 = broker.buy("GOOGL", 50)  # Buy 50 shares of GOOGL
    status2 = "[OK] Filled" if result2.success else "[FAIL]"
    print(f"   BUY 50 GOOGL @ ${result2.order.avg_fill_price:.2f} - {status2}")
    
    result3 = broker.buy("MSFT", 25)   # Buy 25 shares of MSFT
    status3 = "[OK] Filled" if result3.success else "[FAIL]"
    print(f"   BUY 25 MSFT @ ${result3.order.avg_fill_price:.2f} - {status3}")
    
    # Check positions
    print("\nCurrent Positions:")
    positions = broker.get_positions()
    for pos in positions:
        print(f"   {pos.symbol}: {pos.quantity} shares @ ${pos.avg_cost:.2f} "
              f"(Value: ${pos.market_value:,.2f})")
    
    # Check account after trades
    account = broker.get_account()
    print(f"\nAccount After Trades:")
    print(f"   Cash: ${account.cash:,.2f}")
    print(f"   Portfolio Value: ${account.portfolio_value:,.2f}")
    
    # Simulate market movement
    print("\nSimulating market movement...")
    broker.set_price("AAPL", 165.0)   # AAPL up 10%
    broker.set_price("GOOGL", 168.0)  # GOOGL down 4%
    broker.set_price("MSFT", 420.0)   # MSFT up 5%
    
    # Check updated positions with P&L
    print("\nUpdated Positions with P&L:")
    for pos in broker.get_positions():
        pnl = pos.unrealized_pnl
        pnl_pct = (pnl / pos.cost_basis) * 100 if pos.cost_basis else 0
        print(f"   {pos.symbol}: {pos.quantity} shares @ ${pos.avg_cost:.2f} "
              f"-> ${pos.current_price:.2f} "
              f"(P&L: ${pnl:+,.2f} / {pnl_pct:+.2f}%)")
    
    # Check updated portfolio value
    account = broker.get_account()
    total_return = (account.portfolio_value / 100000 - 1) * 100
    print(f"\nUpdated Account:")
    print(f"   Portfolio Value: ${account.portfolio_value:,.2f}")
    print(f"   Total Return: {total_return:+.2f}%")
    
    # Disconnect
    broker.disconnect()
    print(f"\n[OK] Disconnected from broker")
    
    return broker


def trading_with_costs():
    """Demonstrate paper trading with transaction costs."""
    print("\n" + "=" * 60)
    print("TRADING WITH TRANSACTION COSTS")
    print("=" * 60)
    
    # Create broker with realistic costs
    broker = PaperBroker(
        initial_cash=200000,      # Need more cash for this example
        commission=1.0,           # $1 per trade
        commission_per_share=0.005,  # $0.005 per share
        slippage=0.001,           # 10 basis points slippage
        name="costs_demo"
    )
    broker.connect()
    
    broker.set_price("AAPL", 150.0)
    
    print("\nCost Structure:")
    print(f"   Commission: ${broker.commission:.2f} + ${broker.commission_per_share:.4f}/share")
    print(f"   Slippage: {broker.slippage:.2%}")
    
    # Execute a trade
    print("\nExecuting: BUY 1000 AAPL @ $150.00 (market)")
    result = broker.buy("AAPL", 1000)
    
    # Check if order was successful
    if not result.success:
        print(f"   Order failed: {result.message}")
        broker.disconnect()
        return
    
    # Calculate expected costs
    base_price = 150.0
    slippage_price = base_price * (1 + broker.slippage)
    commission = broker.commission + (broker.commission_per_share * 1000)
    
    print("\nTrade Details:")
    print(f"   Fill Price: ${result.order.avg_fill_price:.4f} (with slippage)")
    print(f"   Commission: ${commission:.2f}")
    print(f"   Total Cost: ${(result.order.avg_fill_price * 1000 + commission):,.2f}")
    
    account = broker.get_account()
    print(f"\nAccount After Trade:")
    print(f"   Remaining Cash: ${account.cash:,.2f}")
    print(f"   Expected Cash: ${(200000 - slippage_price * 1000 - commission):,.2f}")
    
    broker.disconnect()


def order_types_demo():
    """Demonstrate different order types."""
    print("\n" + "=" * 60)
    print("ORDER TYPES DEMONSTRATION")
    print("=" * 60)
    
    broker = PaperBroker(initial_cash=100000)
    broker.connect()
    broker.set_price("AAPL", 150.0)
    
    # 1. Market Order (immediate execution)
    print("\n1. MARKET ORDER")
    print(f"   Current Price: $150.00")
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
    )
    result = broker.submit_order(order)
    status = "[OK] Filled" if result.order.is_filled else "[PENDING]"
    print(f"   Result: {status} @ ${result.order.avg_fill_price:.2f}")
    
    # 2. Limit Order (fills if price favorable)
    print("\n2. LIMIT ORDER (price below market)")
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=145.0,  # Below current price
    )
    result = broker.submit_order(order)
    status = "[OK] Filled" if result.order.is_filled else "[PENDING] (price not reached)"
    print(f"   Limit Price: $145.00")
    print(f"   Result: {status}")
    
    print("\n3. LIMIT ORDER (price above market)")
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=160.0,  # Above current price
    )
    result = broker.submit_order(order)
    status = "[OK] Filled" if result.order.is_filled else "[PENDING]"
    print(f"   Limit Price: $160.00")
    print(f"   Result: {status} @ ${result.order.avg_fill_price:.2f}")
    
    # 3. Stop Order
    print("\n4. STOP ORDER (sell when price falls)")
    # First need a position to sell
    broker.buy("AAPL", 20)
    order = Order(
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=10,
        order_type=OrderType.STOP,
        stop_price=140.0,  # Trigger when price falls to $140
    )
    result = broker.submit_order(order)
    status = "[OK] Filled" if result.order.is_filled else "[PENDING] (stop not triggered)"
    print(f"   Stop Price: $140.00")
    print(f"   Current Price: $150.00")
    print(f"   Result: {status}")
    
    broker.disconnect()


def position_management_demo():
    """Demonstrate position management features."""
    print("\n" + "=" * 60)
    print("POSITION MANAGEMENT DEMONSTRATION")
    print("=" * 60)
    
    broker = PaperBroker(initial_cash=100000)
    broker.connect()
    
    # Set prices for multiple symbols
    prices = {
        "AAPL": 150.0,
        "GOOGL": 175.0,
        "MSFT": 400.0,
        "AMZN": 185.0,
        "TSLA": 250.0,
    }
    broker.set_prices(prices)
    
    # Build a portfolio
    print("\nBuilding Portfolio...")
    broker.buy("AAPL", 100)
    broker.buy("GOOGL", 50)
    broker.buy("MSFT", 25)
    broker.buy("AMZN", 40)
    broker.buy("TSLA", 30)
    
    # Display portfolio
    print("\nPortfolio Summary:")
    print("-" * 50)
    positions = broker.get_positions()
    for pos in positions:
        weight = (pos.market_value / broker.portfolio_value) * 100
        print(f"   {pos.symbol:6s} | {pos.quantity:5.0f} shares | "
              f"${pos.market_value:>10,.2f} | {weight:5.1f}%")
    print("-" * 50)
    print(f"   {'TOTAL':6s} |       |"
          f"${broker.portfolio_value:>10,.2f} | 100.0%")
    
    # Close specific position
    print("\nClosing TSLA position...")
    result = broker.close_position("TSLA")
    status = "[OK] Closed" if result.success else "[FAIL]"
    print(f"   Result: {status}")
    
    # Verify position is closed
    tsla_pos = broker.get_position("TSLA")
    print(f"   TSLA position after close: {tsla_pos.quantity if tsla_pos else 0} shares")
    
    # Close all positions
    print("\nClosing all remaining positions...")
    results = broker.close_all_positions()
    for symbol, result in results.items():
        status = "[OK] Closed" if result.success else "[FAIL]"
        print(f"   {symbol}: {status}")
    
    # Final account state
    account = broker.get_account()
    print(f"\nFinal Account State:")
    print(f"   Cash: ${account.cash:,.2f}")
    print(f"   Positions: {len(broker.get_positions())}")
    
    broker.disconnect()


def event_callbacks_demo():
    """Demonstrate event callbacks."""
    print("\n" + "=" * 60)
    print("EVENT CALLBACKS DEMONSTRATION")
    print("=" * 60)
    
    broker = PaperBroker(initial_cash=100000)
    broker.connect()
    broker.set_price("AAPL", 150.0)
    
    # Register callbacks
    fill_count = [0]  # Use list to allow modification in closure
    order_count = [0]
    
    def on_fill(fill):
        fill_count[0] += 1
        print(f"   [EVENT] Fill: {fill.side.value.upper()} {fill.quantity} "
              f"{fill.symbol} @ ${fill.price:.2f}")
    
    def on_order_update(order):
        order_count[0] += 1
        print(f"   [EVENT] Order update: {order.order_id[:12]}... -> {order.status.value}")
    
    broker.on_fill(on_fill)
    broker.on_order_update(on_order_update)
    
    print("\nExecuting trades with callbacks registered...")
    broker.buy("AAPL", 50)
    broker.buy("AAPL", 25)
    broker.sell("AAPL", 30)
    
    print(f"\nCallback Summary:")
    print(f"   Fill callbacks triggered: {fill_count[0]}")
    print(f"   Order callbacks triggered: {order_count[0]}")
    
    broker.disconnect()


def full_workflow_example():
    """Complete trading workflow example."""
    print("\n" + "=" * 60)
    print("COMPLETE TRADING WORKFLOW EXAMPLE")
    print("=" * 60)
    
    # Initialize broker with realistic settings
    broker = PaperBroker(
        initial_cash=100000,
        commission=0.50,
        slippage=0.0005,  # 5 bps
        name="full_demo"
    )
    
    # Use context manager for automatic connect/disconnect
    with broker:
        print(f"\n[OK] Connected to {broker.name}")
        
        # Day 1: Initial positions
        print("\n--- Day 1: Building initial positions ---")
        broker.set_prices({
            "AAPL": 150.0,
            "MSFT": 400.0,
            "NVDA": 900.0,
        })
        
        broker.buy("AAPL", 100)
        broker.buy("MSFT", 25)
        broker.buy("NVDA", 10)
        
        summary = broker.get_summary()
        print(f"   Portfolio Value: ${summary['portfolio_value']:,.2f}")
        print(f"   Cash Remaining: ${summary['current_cash']:,.2f}")
        
        # Day 2: Market moves
        print("\n--- Day 2: Market moves ---")
        broker.set_prices({
            "AAPL": 155.0,   # +3.3%
            "MSFT": 410.0,   # +2.5%
            "NVDA": 850.0,   # -5.5%
        })
        
        print("   New prices: AAPL=$155 (+3.3%), MSFT=$410 (+2.5%), NVDA=$850 (-5.5%)")
        
        for pos in broker.get_positions():
            pnl_pct = (pos.unrealized_pnl / pos.cost_basis) * 100
            print(f"   {pos.symbol}: ${pos.unrealized_pnl:+,.2f} ({pnl_pct:+.1f}%)")
        
        # Day 3: Rebalance
        print("\n--- Day 3: Rebalancing portfolio ---")
        
        # Reduce NVDA position (taking the loss)
        broker.sell("NVDA", 5)
        print("   Sold 5 NVDA")
        
        # Add to winning positions
        broker.buy("AAPL", 50)
        print("   Bought 50 more AAPL")
        
        # Day 4: Exit all
        print("\n--- Day 4: Closing all positions ---")
        broker.set_prices({
            "AAPL": 160.0,
            "MSFT": 405.0,
            "NVDA": 875.0,
        })
        
        broker.close_all_positions()
        
        # Final summary
        print("\n" + "=" * 50)
        broker.print_summary()
    
    print("[OK] Disconnected from broker")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("JSF-CORE: BROKER INTEGRATION EXAMPLES (PHASE 14)")
    print("=" * 60)
    
    # Run examples
    basic_paper_trading()
    trading_with_costs()
    order_types_demo()
    position_management_demo()
    event_callbacks_demo()
    full_workflow_example()
    
    print("\n" + "=" * 60)
    print("[OK] All examples completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Try the PaperBroker with your own strategies")
    print("  2. Set up Alpaca API credentials for real paper trading")
    print("  3. Integrate with BacktestEngine for signal-to-trade workflow")


if __name__ == "__main__":
    main()

"""
Alert System Examples
=====================

This module demonstrates how to use the JSF-Core alert system
for real-time trading notifications.

Examples include:
1. Basic console alerts
2. Telegram integration
3. Alert manager with multiple channels
4. Trading event alerts
5. Alert filtering and routing
6. Custom alert types

Author: JBAC Team
Phase: 17 - Alert System
"""

from datetime import datetime
import time

from jsf.alerts import (
    Alert,
    AlertType,
    AlertSeverity,
    AlertManager,
    ConsoleAlerter,
    create_default_manager,
    create_order_alert,
    create_position_alert,
    create_risk_alert,
    create_strategy_alert,
)
from jsf.broker.models import Order, OrderSide, OrderType, OrderStatus, Position, PositionSide


# =============================================================================
# Example 1: Basic Console Alerts
# =============================================================================

def example_1_console_alerts():
    """
    Demonstrate basic console alerting.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Console Alerts")
    print("=" * 70)
    
    # Create console alerter
    console = ConsoleAlerter(use_colors=True)
    
    # Send alerts of different severities
    print("\n1.1 Different Severity Levels:")
    print("-" * 50)
    
    severities = [
        (AlertSeverity.DEBUG, "Debug message for detailed logging"),
        (AlertSeverity.INFO, "Informational message about system state"),
        (AlertSeverity.WARNING, "Warning about potential issues"),
        (AlertSeverity.ERROR, "Error that occurred during execution"),
        (AlertSeverity.CRITICAL, "Critical issue requiring immediate attention"),
    ]
    
    for severity, message in severities:
        alert = Alert(
            title=f"{severity.value.upper()} Alert",
            message=message,
            severity=severity,
        )
        console.send(alert)
        time.sleep(0.5)
    
    # Send alert with metadata
    print("\n1.2 Alert with Metadata:")
    print("-" * 50)
    
    alert = Alert(
        title="Trade Executed",
        message="Successfully bought 100 shares of AAPL",
        severity=AlertSeverity.INFO,
        alert_type=AlertType.ORDER_FILLED,
        metadata={
            "Symbol": "AAPL",
            "Quantity": 100,
            "Price": 150.50,
            "Total Value": "$15,050.00",
            "Commission": "$1.00",
        }
    )
    console.send(alert)


# =============================================================================
# Example 2: Telegram Integration (Optional)
# =============================================================================

def example_2_telegram_setup():
    """
    Demonstrate Telegram alerter setup.
    
    Note: Requires python-telegram-bot package and bot setup.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Telegram Integration Setup")
    print("=" * 70)
    
    print("""
    To use Telegram alerts:
    
    1. Create a Bot:
       - Message @BotFather on Telegram
       - Send /newbot command
       - Choose a name and username
       - Copy the bot token (looks like: 123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11)
    
    2. Get Your Chat ID:
       - Message @userinfobot on Telegram
       - It will reply with your chat_id (looks like: 123456789)
    
    3. Start Chat with Your Bot:
       - Search for your bot on Telegram
       - Click "Start" button
    
    4. Install python-telegram-bot:
       - pip install python-telegram-bot
    
    5. Create the alerter:
       ```python
       from jsf.alerts import TelegramAlerter
       
       telegram = TelegramAlerter(
           bot_token="YOUR_BOT_TOKEN",
           chat_id="YOUR_CHAT_ID"
       )
       
       # Test the connection
       telegram.test()
       ```
    """)


# =============================================================================
# Example 3: Alert Manager with Multiple Channels
# =============================================================================

def example_3_alert_manager():
    """
    Demonstrate using AlertManager with multiple alerters.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Alert Manager with Multiple Channels")
    print("=" * 70)
    
    # Create alert manager
    manager = AlertManager()
    
    # Add console alerter for INFO and above
    console = ConsoleAlerter(
        name="console_main",
        min_severity=AlertSeverity.INFO,
        use_colors=True,
    )
    manager.add_alerter(console)
    
    # Add another console alerter for CRITICAL only
    console_critical = ConsoleAlerter(
        name="console_critical",
        min_severity=AlertSeverity.CRITICAL,
        use_colors=True,
    )
    manager.add_alerter(console_critical)
    
    print("\n3.1 Sending alerts through manager:")
    print("-" * 50)
    
    # INFO alert - will be shown by console_main only
    manager.send_alert(
        title="System Started",
        message="Trading system initialized successfully",
        severity=AlertSeverity.INFO,
    )
    
    time.sleep(0.5)
    
    # CRITICAL alert - will be shown by both alerters
    manager.send_alert(
        title="CRITICAL: Connection Lost",
        message="Lost connection to broker, stopping all trading",
        severity=AlertSeverity.CRITICAL,
        alert_type=AlertType.CONNECTION_LOST,
    )
    
    # Get statistics
    print("\n3.2 Alert Statistics:")
    print("-" * 50)
    stats = manager.get_stats()
    print(f"Total Sent: {stats['total_sent']}")
    print(f"Total Failed: {stats['total_failed']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"\nBy Severity:")
    for severity, count in stats['by_severity'].items():
        if count > 0:
            print(f"  {severity}: {count}")


# =============================================================================
# Example 4: Trading Event Alerts
# =============================================================================

def example_4_trading_alerts():
    """
    Demonstrate alerts for trading events.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Trading Event Alerts")
    print("=" * 70)
    
    manager = AlertManager()
    manager.add_alerter(ConsoleAlerter(use_colors=True))
    
    print("\n4.1 Order Alerts:")
    print("-" * 50)
    
    # Order submitted
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
        order_id="ORD_001",
        status=OrderStatus.SUBMITTED,
    )
    
    alert = create_order_alert(order, OrderStatus.SUBMITTED)
    manager.send(alert)
    
    time.sleep(0.5)
    
    # Order filled
    order.status = OrderStatus.FILLED
    order.filled_quantity = 100
    order.avg_fill_price = 150.25
    
    alert = create_order_alert(order, OrderStatus.FILLED)
    manager.send(alert)
    
    print("\n4.2 Position Alerts:")
    print("-" * 50)
    
    # Position opened
    position = Position(
        symbol="AAPL",
        side=PositionSide.LONG,
        quantity=100,
        avg_cost=150.25,
        current_price=152.00,
    )
    
    alert = create_position_alert(position, "opened")
    manager.send(alert)
    
    time.sleep(0.5)
    
    # Position closed with profit
    position.realized_pnl = 175.00
    alert = create_position_alert(position, "closed")
    manager.send(alert)
    
    print("\n4.3 Risk Alerts:")
    print("-" * 50)
    
    # Drawdown alert
    alert = create_risk_alert(
        alert_type=AlertType.DRAWDOWN_THRESHOLD,
        title="Drawdown Warning",
        message="Portfolio drawdown exceeded 5% threshold",
        severity=AlertSeverity.WARNING,
        current_drawdown=-0.052,
        threshold=-0.05,
        portfolio_value=94800,
    )
    manager.send(alert)
    
    print("\n4.4 Strategy Alerts:")
    print("-" * 50)
    
    # Strategy started
    alert = create_strategy_alert(
        strategy_name="Momentum Strategy",
        event="started",
        message="Strategy initialized with 5 symbols",
        num_symbols=5,
        capital=100000,
    )
    manager.send(alert)
    
    time.sleep(0.5)
    
    # Signal generated
    alert = create_strategy_alert(
        strategy_name="Momentum Strategy",
        event="signal",
        message="Strong buy signal detected for GOOGL",
        symbol="GOOGL",
        signal_strength=0.85,
    )
    manager.send(alert)


# =============================================================================
# Example 5: Alert Callbacks
# =============================================================================

def example_5_alert_callbacks():
    """
    Demonstrate using alert callbacks.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Alert Callbacks")
    print("=" * 70)
    
    manager = AlertManager()
    manager.add_alerter(ConsoleAlerter(use_colors=True))
    
    # Track critical alerts
    critical_alerts = []
    
    def log_critical(alert: Alert):
        """Log critical alerts to a list."""
        if alert.severity == AlertSeverity.CRITICAL:
            critical_alerts.append(alert)
            print(f"\n>>> CRITICAL ALERT LOGGED: {alert.title}")
    
    # Add callback
    manager.add_callback(log_critical)
    
    print("\n5.1 Sending alerts with callback:")
    print("-" * 50)
    
    # Send some alerts
    manager.send_alert("Info Alert", "This is informational", AlertSeverity.INFO)
    manager.send_alert("Warning Alert", "This is a warning", AlertSeverity.WARNING)
    manager.send_alert("CRITICAL Alert", "This is critical!", AlertSeverity.CRITICAL)
    
    print(f"\n5.2 Critical alerts captured: {len(critical_alerts)}")
    for alert in critical_alerts:
        print(f"  - {alert.title} at {alert.timestamp}")


# =============================================================================
# Example 6: Using Create Default Manager
# =============================================================================

def example_6_default_manager():
    """
    Demonstrate create_default_manager helper.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Using create_default_manager")
    print("=" * 70)
    
    # Create manager with console only
    manager = create_default_manager(include_console=True)
    
    print(f"\n6.1 Manager created with {len(manager.alerters)} alerter(s)")
    
    # Send test alert
    manager.send_alert(
        title="System Ready",
        message="Alert system configured and ready",
        severity=AlertSeverity.INFO,
    )
    
    # To add Telegram (uncomment if you have credentials):
    # manager = create_default_manager(
    #     telegram_token="YOUR_BOT_TOKEN",
    #     telegram_chat_id="YOUR_CHAT_ID",
    #     include_console=True,
    # )


# =============================================================================
# Example 7: Async Alert Processing
# =============================================================================

def example_7_async_alerts():
    """
    Demonstrate async alert processing.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Async Alert Processing")
    print("=" * 70)
    
    # Create manager with async mode
    manager = AlertManager(async_mode=True, queue_size=100)
    manager.add_alerter(ConsoleAlerter(use_colors=True))
    
    print("\n7.1 Sending alerts asynchronously:")
    print("-" * 50)
    
    # Send multiple alerts quickly
    for i in range(5):
        manager.send_alert(
            title=f"Alert {i+1}",
            message=f"This is async alert number {i+1}",
            severity=AlertSeverity.INFO,
        )
    
    # Wait a bit for async processing
    print("\nWaiting for async processing...")
    time.sleep(1)
    
    # Shutdown cleanly
    manager.shutdown()
    
    print("\nAsync alert processing complete")


# =============================================================================
# Run All Examples
# =============================================================================

def main():
    """Run all examples."""
    print("\n")
    print("=" * 70)
    print("JSF-CORE ALERT SYSTEM EXAMPLES")
    print("=" * 70)
    print()
    
    try:
        example_1_console_alerts()
        example_2_telegram_setup()
        example_3_alert_manager()
        example_4_trading_alerts()
        example_5_alert_callbacks()
        example_6_default_manager()
        example_7_async_alerts()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETE")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Set up Telegram bot for mobile notifications")
        print("2. Integrate alerts with your trading strategies")
        print("3. Configure alert severity levels for your needs")
        print("4. Add custom callbacks for alert processing")
        print()
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        raise


if __name__ == "__main__":
    main()

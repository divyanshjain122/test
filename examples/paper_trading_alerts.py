"""
Example: Paper Trading with Telegram Alerts

Demonstrates how to receive real-time alerts for paper trading strategies.
Shows [PAPER] tag in alerts to distinguish from live trading.
"""

from jsf.alerts import create_alert_manager_from_config, Alert, AlertSeverity


def main():
    """Run paper trading examples with alerts."""
    
    print("=" * 70)
    print("Paper Trading Alerts Demo")
    print("=" * 70)
    print("\nThis demonstrates paper trading alerts with [PAPER] indicators.\n")
    
    # Create alert manager (auto-loads from .env)
    manager = create_alert_manager_from_config()
    
    print(f"Alert manager configured with {len(manager.alerters)} alerter(s)")
    print("\nSending paper trading alerts...\n")
    
    # Example 1: Strategy started
    manager.send(Alert(
        title="📊 Strategy Started",
        message="Paper trading mode activated. Testing MA Crossover strategy.",
        severity=AlertSeverity.INFO,
        metadata={'mode': 'paper', 'strategy': 'MA_Crossover'}
    ))
    print("✅ Sent: Strategy started alert")
    
    # Example 2: Buy signal
    manager.send(Alert(
        title="📈 Buy Signal",
        message="Strong buy signal detected for AAPL\nConfidence: 87%",
        severity=AlertSeverity.WARNING,
        metadata={
            'mode': 'paper',
            'symbol': 'AAPL',
            'signal_strength': 0.87,
            'entry_price': 150.25
        }
    ))
    print("✅ Sent: Buy signal alert")
    
    # Example 3: Order filled
    manager.send(Alert(
        title="✅ Order Filled",
        message="Bought 100 shares of AAPL at $150.25",
        severity=AlertSeverity.INFO,
        metadata={
            'mode': 'paper',
            'symbol': 'AAPL',
            'quantity': 100,
            'price': 150.25,
            'total_value': 15025.00,
            'order_type': 'market'
        }
    ))
    print("✅ Sent: Order filled alert")
    
    # Example 4: Risk warning
    manager.send(Alert(
        title="⚠️ Risk Alert",
        message="Position size approaching limit (85% of max)",
        severity=AlertSeverity.WARNING,
        metadata={
            'mode': 'paper',
            'current_exposure': 0.85,
            'max_exposure': 1.0,
            'action': 'review_positions'
        }
    ))
    print("✅ Sent: Risk warning alert")
    
    # Example 5: Profit update
    manager.send(Alert(
        title="💰 Profit Update",
        message="AAPL position up 2.5% (+$375.62)",
        severity=AlertSeverity.INFO,
        metadata={
            'mode': 'paper',
            'symbol': 'AAPL',
            'profit_pct': 0.025,
            'profit_usd': 375.62,
            'current_price': 154.01
        }
    ))
    print("✅ Sent: Profit update alert")
    
    # Example 6: Live trading warning (for comparison)
    print("\n" + "-" * 70)
    print("COMPARISON: How a LIVE trading alert would look:")
    print("-" * 70 + "\n")
    
    manager.send(Alert(
        title="🚨 LIVE Order Executed",
        message="REAL MONEY: Bought 100 shares of MSFT at $380.50",
        severity=AlertSeverity.CRITICAL,
        metadata={
            'mode': 'live',  # ← This shows [LIVE] tag
            'symbol': 'MSFT',
            'quantity': 100,
            'price': 380.50,
            'total_value': 38050.00
        }
    ))
    print("✅ Sent: Live trading alert (notice the [LIVE] tag)\n")
    
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nCheck your Telegram to see the difference between:")
    print("  📊 [PAPER] alerts - Safe for learning")
    print("  🚨 [LIVE] alerts - Real money warnings")
    print("\n💡 Always test with [PAPER] mode first!")
    print()


if __name__ == "__main__":
    main()

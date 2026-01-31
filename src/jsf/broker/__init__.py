"""Broker integration module.

This module provides broker integrations for paper trading and live trading,
including abstract base classes and concrete implementations.

Supported Brokers:
    - PaperBroker: Simulated broker for testing and development
    - AlpacaBroker: Alpaca Markets API integration

Example Usage:
    ```python
    from jsf.broker import PaperBroker, Order, OrderSide, OrderType
    
    # Create and connect to paper broker
    broker = PaperBroker(initial_cash=100000)
    broker.connect()
    
    # Set prices for simulation
    broker.set_price("AAPL", 150.0)
    broker.set_price("GOOGL", 2800.0)
    
    # Submit orders
    result = broker.buy("AAPL", 100)  # Buy 100 shares
    print(f"Order filled: {result.order.is_filled}")
    
    # Check positions
    positions = broker.get_positions()
    for pos in positions:
        print(f"{pos.symbol}: {pos.quantity} shares @ ${pos.avg_cost:.2f}")
    
    # Get account info
    account = broker.get_account()
    print(f"Portfolio value: ${account.portfolio_value:,.2f}")
    
    # Clean up
    broker.disconnect()
    ```

For Alpaca integration:
    ```python
    from jsf.broker import AlpacaBroker
    
    # Create broker with API credentials
    broker = AlpacaBroker(
        api_key="your_api_key",
        api_secret="your_api_secret",
        paper=True  # Use paper trading
    )
    broker.connect()
    
    # Trade real markets (paper money)
    result = broker.buy("AAPL", 10)
    ```
"""

# Models and enums
from jsf.broker.models import (
    # Enums
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    PositionSide,
    AssetClass,
    # Data classes
    Order,
    Fill,
    Position,
    Trade,
    AccountInfo,
    OrderResult,
)

# Base classes and exceptions
from jsf.broker.base import (
    Broker,
    BrokerError,
    ConnectionError,
    OrderError,
    InsufficientFundsError,
    PositionError,
)

# Concrete implementations
from jsf.broker.paper import PaperBroker

# Optional Alpaca integration
try:
    from jsf.broker.alpaca import AlpacaBroker, ALPACA_AVAILABLE
except ImportError:
    AlpacaBroker = None
    ALPACA_AVAILABLE = False


__all__ = [
    # Enums
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "PositionSide",
    "AssetClass",
    # Data classes
    "Order",
    "Fill",
    "Position",
    "Trade",
    "AccountInfo",
    "OrderResult",
    # Base classes
    "Broker",
    "BrokerError",
    "ConnectionError",
    "OrderError",
    "InsufficientFundsError",
    "PositionError",
    # Implementations
    "PaperBroker",
    "AlpacaBroker",
    "ALPACA_AVAILABLE",
]

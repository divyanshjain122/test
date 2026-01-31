"""Abstract broker interface.

This module defines the abstract base class for all broker implementations,
providing a consistent interface for order submission, position management,
and account information retrieval.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import logging

from jsf.broker.models import (
    Order,
    OrderResult,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    Position,
    Fill,
    Trade,
    AccountInfo,
)
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class BrokerError(Exception):
    """Base exception for broker-related errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code


class ConnectionError(BrokerError):
    """Raised when broker connection fails."""
    pass


class OrderError(BrokerError):
    """Raised when order submission or modification fails."""
    pass


class InsufficientFundsError(BrokerError):
    """Raised when account has insufficient funds for an order."""
    pass


class PositionError(BrokerError):
    """Raised when position-related operations fail."""
    pass


class Broker(ABC):
    """
    Abstract base class for broker implementations.
    
    This class defines the interface that all broker implementations
    must follow. It provides methods for:
    - Connecting and disconnecting from the broker
    - Submitting and managing orders
    - Querying positions and account information
    - Subscribing to market data and order updates
    
    Subclasses must implement all abstract methods.
    
    Example usage:
        ```python
        broker = PaperBroker(initial_cash=100000)
        broker.connect()
        
        # Submit an order
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        result = broker.submit_order(order)
        
        # Check positions
        positions = broker.get_positions()
        
        # Disconnect
        broker.disconnect()
        ```
    """
    
    def __init__(self, name: str = "broker", **kwargs):
        """
        Initialize broker.
        
        Args:
            name: Broker name for identification
            **kwargs: Additional broker-specific parameters
        """
        self.name = name
        self._connected = False
        self._callbacks: Dict[str, List[Callable]] = {
            "on_fill": [],
            "on_order_update": [],
            "on_position_update": [],
            "on_error": [],
        }
        logger.info(f"Initialized broker: {name}")
    
    # ==========================================================================
    # Connection Management
    # ==========================================================================
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the broker.
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            ConnectionError: If connection fails
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the broker.
        
        Returns:
            True if disconnection successful
        """
        pass
    
    @property
    def is_connected(self) -> bool:
        """Check if broker is connected."""
        return self._connected
    
    # ==========================================================================
    # Order Management
    # ==========================================================================
    
    @abstractmethod
    def submit_order(self, order: Order) -> OrderResult:
        """
        Submit an order to the broker.
        
        Args:
            order: Order to submit
            
        Returns:
            OrderResult with submission status
            
        Raises:
            OrderError: If order submission fails
            InsufficientFundsError: If insufficient buying power
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancellation request was accepted
            
        Raises:
            OrderError: If cancellation fails
        """
        pass
    
    @abstractmethod
    def modify_order(
        self,
        order_id: str,
        quantity: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> OrderResult:
        """
        Modify an existing order.
        
        Args:
            order_id: ID of order to modify
            quantity: New quantity (optional)
            limit_price: New limit price (optional)
            stop_price: New stop price (optional)
            
        Returns:
            OrderResult with modification status
            
        Raises:
            OrderError: If modification fails
        """
        pass
    
    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.
        
        Args:
            order_id: Order ID to look up
            
        Returns:
            Order if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Order]:
        """
        Get list of orders.
        
        Args:
            status: Filter by order status
            symbol: Filter by symbol
            since: Get orders since this timestamp
            
        Returns:
            List of orders matching filters
        """
        pass
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get list of open (active) orders.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of open orders
        """
        orders = []
        for status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, 
                       OrderStatus.ACCEPTED, OrderStatus.PARTIAL]:
            orders.extend(self.get_orders(status=status, symbol=symbol))
        return orders
    
    # ==========================================================================
    # Convenience Order Methods
    # ==========================================================================
    
    def buy(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        **kwargs,
    ) -> OrderResult:
        """
        Submit a buy order.
        
        Args:
            symbol: Ticker symbol
            quantity: Number of shares
            order_type: Type of order
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Order duration
            **kwargs: Additional order parameters
            
        Returns:
            OrderResult with submission status
        """
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            **kwargs,
        )
        return self.submit_order(order)
    
    def sell(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        **kwargs,
    ) -> OrderResult:
        """
        Submit a sell order.
        
        Args:
            symbol: Ticker symbol
            quantity: Number of shares
            order_type: Type of order
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Order duration
            **kwargs: Additional order parameters
            
        Returns:
            OrderResult with submission status
        """
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            **kwargs,
        )
        return self.submit_order(order)
    
    def close_position(self, symbol: str) -> OrderResult:
        """
        Close an entire position in a symbol.
        
        Args:
            symbol: Symbol to close position for
            
        Returns:
            OrderResult with submission status
        """
        position = self.get_position(symbol)
        if position is None or position.quantity == 0:
            return OrderResult(
                success=False,
                message=f"No position in {symbol}",
                error_code="NO_POSITION"
            )
        
        if position.quantity > 0:
            return self.sell(symbol, position.quantity)
        else:
            return self.buy(symbol, abs(position.quantity))
    
    def close_all_positions(self) -> Dict[str, OrderResult]:
        """
        Close all open positions.
        
        Returns:
            Dictionary of symbol -> OrderResult
        """
        results = {}
        positions = self.get_positions()
        for position in positions:
            if position.quantity != 0:
                results[position.symbol] = self.close_position(position.symbol)
        return results
    
    # ==========================================================================
    # Position Management
    # ==========================================================================
    
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Symbol to look up
            
        Returns:
            Position if exists, None otherwise
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get all current positions.
        
        Returns:
            List of all positions (including zero positions may be omitted)
        """
        pass
    
    # ==========================================================================
    # Account Information
    # ==========================================================================
    
    @abstractmethod
    def get_account(self) -> AccountInfo:
        """
        Get account information.
        
        Returns:
            AccountInfo with current account state
        """
        pass
    
    @property
    def cash(self) -> float:
        """Get available cash balance."""
        return self.get_account().cash
    
    @property
    def portfolio_value(self) -> float:
        """Get total portfolio value."""
        return self.get_account().portfolio_value
    
    @property
    def buying_power(self) -> float:
        """Get available buying power."""
        account = self.get_account()
        return account.buying_power if account.buying_power else account.cash
    
    # ==========================================================================
    # Trade History
    # ==========================================================================
    
    @abstractmethod
    def get_trades(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Trade]:
        """
        Get trade history.
        
        Args:
            symbol: Filter by symbol
            since: Start timestamp
            until: End timestamp
            
        Returns:
            List of completed trades
        """
        pass
    
    @abstractmethod
    def get_fills(
        self,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Fill]:
        """
        Get fill (execution) history.
        
        Args:
            order_id: Filter by order ID
            symbol: Filter by symbol
            since: Start timestamp
            
        Returns:
            List of fills
        """
        pass
    
    # ==========================================================================
    # Event Callbacks
    # ==========================================================================
    
    def on_fill(self, callback: Callable[[Fill], None]) -> None:
        """
        Register callback for fill events.
        
        Args:
            callback: Function to call when a fill occurs
        """
        self._callbacks["on_fill"].append(callback)
    
    def on_order_update(self, callback: Callable[[Order], None]) -> None:
        """
        Register callback for order update events.
        
        Args:
            callback: Function to call when order status changes
        """
        self._callbacks["on_order_update"].append(callback)
    
    def on_position_update(self, callback: Callable[[Position], None]) -> None:
        """
        Register callback for position update events.
        
        Args:
            callback: Function to call when position changes
        """
        self._callbacks["on_position_update"].append(callback)
    
    def on_error(self, callback: Callable[[BrokerError], None]) -> None:
        """
        Register callback for error events.
        
        Args:
            callback: Function to call when an error occurs
        """
        self._callbacks["on_error"].append(callback)
    
    def _emit(self, event: str, data: Any) -> None:
        """
        Emit an event to registered callbacks.
        
        Args:
            event: Event name
            data: Event data
        """
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in callback for {event}: {e}")
    
    # ==========================================================================
    # Market Data (Optional - not all brokers support this)
    # ==========================================================================
    
    def get_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get current quote for a symbol.
        
        Args:
            symbol: Symbol to get quote for
            
        Returns:
            Dictionary with 'bid', 'ask', 'last' prices, or None if not supported
        """
        return None  # Override in subclasses that support quotes
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """
        Get last trade price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Last price or None if not available
        """
        quote = self.get_quote(symbol)
        return quote.get("last") if quote else None
    
    # ==========================================================================
    # Utility Methods
    # ==========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        status = "connected" if self._connected else "disconnected"
        return f"{self.__class__.__name__}(name='{self.name}', status={status})"
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False

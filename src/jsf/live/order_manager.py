"""Order management for live trading.

This module provides order tracking, execution management,
and order lifecycle handling for live trading operations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import threading
import logging
from collections import defaultdict

from jsf.broker import (
    Broker,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    Position,
    Fill,
    OrderResult,
    BrokerError,
    OrderError,
)
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class OrderManagerError(Exception):
    """Exception raised for order manager errors."""
    
    def __init__(self, message: str, order_id: Optional[str] = None):
        super().__init__(message)
        self.order_id = order_id


class OrderState(Enum):
    """Internal order tracking states."""
    PENDING_SUBMIT = "pending_submit"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    PENDING_CANCEL = "pending_cancel"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OrderTracker:
    """
    Tracks an order through its lifecycle.
    
    Maintains state, fill information, and timing data for a single order.
    """
    
    order: Order
    state: OrderState = OrderState.PENDING_SUBMIT
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    fills: List[Fill] = field(default_factory=list)
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    error_message: Optional[str] = None
    broker_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def order_id(self) -> str:
        """Get order ID."""
        return self.order.order_id
    
    @property
    def symbol(self) -> str:
        """Get symbol."""
        return self.order.symbol
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining unfilled quantity."""
        return self.order.quantity - self.filled_quantity
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active (can be filled)."""
        return self.state in (
            OrderState.PENDING_SUBMIT,
            OrderState.SUBMITTED,
            OrderState.PARTIALLY_FILLED,
        )
    
    @property
    def is_complete(self) -> bool:
        """Check if order is in a terminal state."""
        return self.state in (
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        )
    
    @property
    def fill_ratio(self) -> float:
        """Get ratio of filled to total quantity."""
        if self.order.quantity == 0:
            return 0.0
        return self.filled_quantity / self.order.quantity
    
    def add_fill(self, fill: Fill) -> None:
        """
        Add a fill to this order.
        
        Args:
            fill: Fill to add
        """
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        
        # Update average fill price
        total_value = sum(f.price * f.quantity for f in self.fills)
        total_qty = sum(f.quantity for f in self.fills)
        if total_qty > 0:
            self.average_fill_price = total_value / total_qty
        
        # Update state
        if self.filled_quantity >= self.order.quantity:
            self.state = OrderState.FILLED
            self.filled_at = fill.timestamp
        else:
            self.state = OrderState.PARTIALLY_FILLED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.order.side.value,
            "quantity": self.order.quantity,
            "order_type": self.order.order_type.value,
            "state": self.state.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_fill_price": self.average_fill_price,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "fill_count": len(self.fills),
        }


class OrderManager:
    """
    Manages orders for live trading.
    
    Provides functionality for:
    - Submitting orders to brokers
    - Tracking order state
    - Managing pending and active orders
    - Order cancellation and modification
    - Position reconciliation
    
    Example usage:
        ```python
        broker = PaperBroker(initial_cash=100000)
        manager = OrderManager(broker)
        
        # Submit an order
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        tracker = manager.submit_order(order)
        
        # Check order status
        status = manager.get_order_status(tracker.order_id)
        
        # Get all active orders
        active = manager.get_active_orders()
        
        # Cancel an order
        manager.cancel_order(tracker.order_id)
        ```
    """
    
    def __init__(
        self,
        broker: Broker,
        max_pending_orders: int = 1000,
        order_timeout: Optional[timedelta] = None,
    ):
        """
        Initialize order manager.
        
        Args:
            broker: Broker instance for order execution
            max_pending_orders: Maximum number of pending orders
            order_timeout: Auto-cancel orders after this duration
        """
        self.broker = broker
        self._max_pending_orders = max_pending_orders
        self._order_timeout = order_timeout
        
        self._orders: Dict[str, OrderTracker] = {}
        self._orders_by_symbol: Dict[str, List[str]] = defaultdict(list)
        self._active_order_ids: Set[str] = set()
        
        self._callbacks: Dict[str, List[Callable]] = {
            "on_order_submitted": [],
            "on_order_filled": [],
            "on_order_partially_filled": [],
            "on_order_cancelled": [],
            "on_order_rejected": [],
            "on_order_expired": [],
        }
        
        self._lock = threading.RLock()  # Use RLock for reentrant safety
        
        # Register for broker callbacks
        self._setup_broker_callbacks()
        
        logger.info("Initialized OrderManager")
    
    def _setup_broker_callbacks(self) -> None:
        """Set up callbacks from broker."""
        self.broker.on_fill(self._handle_fill)
        self.broker.on_order_update(self._handle_order_update)
    
    # ==========================================================================
    # Order Submission
    # ==========================================================================
    
    def submit_order(self, order: Order) -> OrderTracker:
        """
        Submit an order to the broker.
        
        Args:
            order: Order to submit
            
        Returns:
            OrderTracker for tracking the order
            
        Raises:
            OrderManagerError: If order cannot be submitted
        """
        # Create tracker FIRST (before broker call) so fills can find it
        tracker = OrderTracker(order=order)
        
        with self._lock:
            # Check pending order limit
            if len(self._active_order_ids) >= self._max_pending_orders:
                raise OrderManagerError(
                    f"Maximum pending orders ({self._max_pending_orders}) reached"
                )
            
            # Register tracker before broker submission
            # This ensures fill callbacks can find the tracker
            self._orders[order.order_id] = tracker
            self._orders_by_symbol[order.symbol].append(order.order_id)
            self._active_order_ids.add(order.order_id)
        
        # Submit to broker (outside lock)
        # Note: For paper broker, this may fill synchronously and trigger
        # the fill callback before returning
        try:
            result = self.broker.submit_order(order)
            
            with self._lock:
                # Only update state if not already filled by callback
                if tracker.state == OrderState.PENDING_SUBMIT:
                    tracker.state = OrderState.SUBMITTED
                tracker.submitted_at = datetime.now()
                # Store broker's order ID if available
                if result.order:
                    tracker.broker_order_id = result.order.order_id
                    # Check if order was already filled (market orders)
                    if result.order.status == OrderStatus.FILLED:
                        if tracker.state != OrderState.FILLED:
                            # Create fill from order info if callback didn't fire
                            tracker.filled_quantity = result.order.filled_quantity
                            tracker.average_fill_price = result.order.avg_fill_price
                            tracker.state = OrderState.FILLED
                            tracker.filled_at = datetime.now()
                            self._active_order_ids.discard(order.order_id)
            
            logger.info(
                f"Submitted order {order.order_id}: "
                f"{order.side.value} {order.quantity} {order.symbol}"
            )
            
            self._emit("on_order_submitted", tracker)
            return tracker
            
        except (BrokerError, OrderError) as e:
            with self._lock:
                tracker.state = OrderState.REJECTED
                tracker.error_message = str(e)
                self._active_order_ids.discard(order.order_id)
            
            logger.error(f"Order rejected: {order.order_id} - {e}")
            self._emit("on_order_rejected", tracker)
            raise OrderManagerError(f"Order rejected: {e}", order.order_id)
    
    def submit_orders(self, orders: List[Order]) -> List[OrderTracker]:
        """
        Submit multiple orders.
        
        Args:
            orders: Orders to submit
            
        Returns:
            List of order trackers
        """
        trackers = []
        for order in orders:
            try:
                tracker = self.submit_order(order)
                trackers.append(tracker)
            except OrderManagerError as e:
                logger.error(f"Failed to submit order: {e}")
        return trackers
    
    # ==========================================================================
    # Order Cancellation
    # ==========================================================================
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancellation was successful
        """
        with self._lock:
            tracker = self._orders.get(order_id)
            if tracker is None:
                logger.warning(f"Order not found: {order_id}")
                return False
            
            if not tracker.is_active:
                logger.warning(f"Order {order_id} is not active (state={tracker.state})")
                return False
            
            tracker.state = OrderState.PENDING_CANCEL
        
        # Cancel with broker
        try:
            success = self.broker.cancel_order(order_id)
            
            if success:
                with self._lock:
                    tracker.state = OrderState.CANCELLED
                    tracker.cancelled_at = datetime.now()
                    self._active_order_ids.discard(order_id)
                
                logger.info(f"Cancelled order: {order_id}")
                self._emit("on_order_cancelled", tracker)
            else:
                with self._lock:
                    # Revert to previous state
                    if tracker.filled_quantity > 0:
                        tracker.state = OrderState.PARTIALLY_FILLED
                    else:
                        tracker.state = OrderState.SUBMITTED
            
            return success
            
        except BrokerError as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            with self._lock:
                if tracker.filled_quantity > 0:
                    tracker.state = OrderState.PARTIALLY_FILLED
                else:
                    tracker.state = OrderState.SUBMITTED
            return False
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all active orders.
        
        Args:
            symbol: Only cancel orders for this symbol (optional)
            
        Returns:
            Number of orders cancelled
        """
        cancelled = 0
        
        with self._lock:
            order_ids = list(self._active_order_ids)
            if symbol:
                order_ids = [
                    oid for oid in order_ids
                    if self._orders[oid].symbol == symbol.upper()
                ]
        
        for order_id in order_ids:
            if self.cancel_order(order_id):
                cancelled += 1
        
        logger.info(f"Cancelled {cancelled} orders")
        return cancelled
    
    # ==========================================================================
    # Order Queries
    # ==========================================================================
    
    def get_order(self, order_id: str) -> Optional[OrderTracker]:
        """
        Get order tracker by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            OrderTracker or None if not found
        """
        with self._lock:
            return self._orders.get(order_id)
    
    def get_order_status(self, order_id: str) -> Optional[OrderState]:
        """
        Get order state.
        
        Args:
            order_id: Order ID
            
        Returns:
            OrderState or None if not found
        """
        tracker = self.get_order(order_id)
        return tracker.state if tracker else None
    
    def get_active_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[OrderTracker]:
        """
        Get all active orders.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of active order trackers
        """
        with self._lock:
            trackers = [
                self._orders[oid] 
                for oid in self._active_order_ids
            ]
            
            if symbol:
                trackers = [t for t in trackers if t.symbol == symbol.upper()]
            
            return trackers
    
    def get_orders_by_symbol(self, symbol: str) -> List[OrderTracker]:
        """
        Get all orders for a symbol.
        
        Args:
            symbol: Symbol to filter by
            
        Returns:
            List of order trackers
        """
        with self._lock:
            order_ids = self._orders_by_symbol.get(symbol.upper(), [])
            return [self._orders[oid] for oid in order_ids if oid in self._orders]
    
    def get_filled_orders(
        self,
        since: Optional[datetime] = None
    ) -> List[OrderTracker]:
        """
        Get all filled orders.
        
        Args:
            since: Only orders filled after this time
            
        Returns:
            List of filled order trackers
        """
        with self._lock:
            filled = [
                t for t in self._orders.values()
                if t.state == OrderState.FILLED
            ]
            
            if since:
                filled = [
                    t for t in filled
                    if t.filled_at and t.filled_at >= since
                ]
            
            return filled
    
    def get_pending_quantity(self, symbol: str, side: OrderSide) -> float:
        """
        Get total pending quantity for a symbol and side.
        
        Args:
            symbol: Symbol to check
            side: Order side (BUY or SELL)
            
        Returns:
            Total pending quantity
        """
        active = self.get_active_orders(symbol)
        return sum(
            t.remaining_quantity 
            for t in active 
            if t.order.side == side
        )
    
    # ==========================================================================
    # Order Statistics
    # ==========================================================================
    
    @property
    def active_order_count(self) -> int:
        """Get number of active orders."""
        with self._lock:
            return len(self._active_order_ids)
    
    @property
    def total_order_count(self) -> int:
        """Get total number of orders tracked."""
        with self._lock:
            return len(self._orders)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get order statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            states = defaultdict(int)
            for tracker in self._orders.values():
                states[tracker.state.value] += 1
            
            total_filled_value = sum(
                (t.average_fill_price or 0) * t.filled_quantity
                for t in self._orders.values()
                if t.state == OrderState.FILLED
            )
            
            return {
                "total_orders": len(self._orders),
                "active_orders": len(self._active_order_ids),
                "orders_by_state": dict(states),
                "total_filled_value": total_filled_value,
                "unique_symbols": len(self._orders_by_symbol),
            }
    
    # ==========================================================================
    # Broker Callback Handlers
    # ==========================================================================
    
    def _handle_fill(self, fill: Fill) -> None:
        """Handle fill from broker."""
        with self._lock:
            tracker = self._orders.get(fill.order_id)
            if tracker is None:
                logger.warning(f"Received fill for unknown order: {fill.order_id}")
                return
            
            tracker.add_fill(fill)
            
            if tracker.state == OrderState.FILLED:
                self._active_order_ids.discard(fill.order_id)
        
        if tracker.state == OrderState.FILLED:
            logger.info(
                f"Order filled: {fill.order_id} at ${tracker.average_fill_price:.2f}"
            )
            self._emit("on_order_filled", tracker)
        else:
            logger.info(
                f"Order partially filled: {fill.order_id} "
                f"({tracker.fill_ratio:.1%})"
            )
            self._emit("on_order_partially_filled", tracker)
    
    def _handle_order_update(self, order: Order) -> None:
        """Handle order update from broker."""
        with self._lock:
            tracker = self._orders.get(order.order_id)
            if tracker is None:
                return
            
            # Map broker status to our state
            if order.status == OrderStatus.CANCELLED:
                tracker.state = OrderState.CANCELLED
                tracker.cancelled_at = datetime.now()
                self._active_order_ids.discard(order.order_id)
                self._emit("on_order_cancelled", tracker)
            
            elif order.status == OrderStatus.REJECTED:
                tracker.state = OrderState.REJECTED
                self._active_order_ids.discard(order.order_id)
                self._emit("on_order_rejected", tracker)
            
            elif order.status == OrderStatus.EXPIRED:
                tracker.state = OrderState.EXPIRED
                self._active_order_ids.discard(order.order_id)
                self._emit("on_order_expired", tracker)
    
    # ==========================================================================
    # Event Callbacks
    # ==========================================================================
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for an event.
        
        Args:
            event: Event name
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        else:
            raise ValueError(f"Unknown event: {event}")
    
    def unregister_callback(self, event: str, callback: Callable) -> None:
        """Unregister a callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
    
    def _emit(self, event: str, data: Any) -> None:
        """Emit an event to callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    # ==========================================================================
    # Cleanup
    # ==========================================================================
    
    def clear_completed_orders(self, keep_recent: int = 100) -> int:
        """
        Clear completed orders from memory.
        
        Args:
            keep_recent: Keep this many recent completed orders
            
        Returns:
            Number of orders cleared
        """
        with self._lock:
            completed = [
                (oid, t) for oid, t in self._orders.items()
                if t.is_complete
            ]
            
            # Sort by completion time
            completed.sort(
                key=lambda x: x[1].filled_at or x[1].cancelled_at or datetime.min
            )
            
            # Remove oldest
            to_remove = completed[:-keep_recent] if keep_recent > 0 else completed
            
            for order_id, tracker in to_remove:
                del self._orders[order_id]
                if tracker.symbol in self._orders_by_symbol:
                    if order_id in self._orders_by_symbol[tracker.symbol]:
                        self._orders_by_symbol[tracker.symbol].remove(order_id)
            
            return len(to_remove)

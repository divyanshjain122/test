"""Paper trading broker implementation.

This module provides a simulated broker for paper trading,
allowing strategy testing without real money risk.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
import uuid
import copy

from jsf.broker.base import (
    Broker,
    BrokerError,
    OrderError,
    InsufficientFundsError,
    PositionError,
)
from jsf.broker.models import (
    Order,
    OrderResult,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    Position,
    PositionSide,
    Fill,
    Trade,
    AccountInfo,
)
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class PaperBroker(Broker):
    """
    Paper trading broker for simulated trading.
    
    This broker simulates order execution without real money,
    perfect for strategy testing and development.
    
    Features:
    - Simulated order execution with configurable fill model
    - Position tracking with average cost basis
    - P&L calculation (realized and unrealized)
    - Transaction cost modeling (commissions, slippage)
    - Trade history tracking
    
    Example:
        ```python
        broker = PaperBroker(
            initial_cash=100000,
            commission=1.0,  # $1 per trade
            slippage=0.001,  # 10 bps slippage
        )
        broker.connect()
        
        # Set price for simulation
        broker.set_price("AAPL", 150.0)
        
        # Submit a buy order
        result = broker.buy("AAPL", 100)
        print(f"Order filled: {result.order.is_filled}")
        
        # Check position
        pos = broker.get_position("AAPL")
        print(f"Position: {pos.quantity} shares @ ${pos.avg_cost:.2f}")
        ```
    
    Attributes:
        initial_cash: Starting cash balance
        commission: Commission per trade (flat rate)
        commission_per_share: Commission per share
        slippage: Slippage as fraction of price
        fill_model: How orders are filled ('immediate', 'next_bar')
    """
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission: float = 0.0,
        commission_per_share: float = 0.0,
        slippage: float = 0.0,
        fill_model: str = "immediate",
        name: str = "paper_broker",
        **kwargs,
    ):
        """
        Initialize paper broker.
        
        Args:
            initial_cash: Starting cash balance
            commission: Commission per trade (flat rate)
            commission_per_share: Commission per share (in addition to flat rate)
            slippage: Slippage as decimal fraction (e.g., 0.001 = 10 bps)
            fill_model: Order fill model ('immediate' or 'next_bar')
            name: Broker identifier
            **kwargs: Additional parameters
        """
        super().__init__(name=name, **kwargs)
        
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if commission < 0:
            raise ValueError("commission must be non-negative")
        if commission_per_share < 0:
            raise ValueError("commission_per_share must be non-negative")
        if slippage < 0 or slippage > 1:
            raise ValueError("slippage must be between 0 and 1")
        if fill_model not in ("immediate", "next_bar"):
            raise ValueError("fill_model must be 'immediate' or 'next_bar'")
        
        self.initial_cash = initial_cash
        self.commission = commission
        self.commission_per_share = commission_per_share
        self.slippage = slippage
        self.fill_model = fill_model
        
        # Internal state
        self._cash = initial_cash
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._fills: List[Fill] = []
        self._trades: List[Trade] = []
        self._prices: Dict[str, float] = {}
        self._pending_orders: List[str] = []  # Orders waiting for next bar
        
        # Trade tracking for P&L
        self._open_trades: Dict[str, List[Dict]] = defaultdict(list)
        
        # Account info
        self._account_id = f"paper_{uuid.uuid4().hex[:8]}"
        self._created_at = datetime.now()
        
        logger.info(
            f"Initialized PaperBroker with ${initial_cash:,.2f} cash, "
            f"commission=${commission:.2f} + ${commission_per_share:.4f}/share, "
            f"slippage={slippage:.4%}"
        )
    
    # ==========================================================================
    # Connection Management
    # ==========================================================================
    
    def connect(self) -> bool:
        """Connect to paper broker (always succeeds)."""
        self._connected = True
        logger.info(f"Connected to paper broker: {self.name}")
        return True
    
    def disconnect(self) -> bool:
        """Disconnect from paper broker."""
        self._connected = False
        logger.info(f"Disconnected from paper broker: {self.name}")
        return True
    
    # ==========================================================================
    # Price Management (Paper Trading Specific)
    # ==========================================================================
    
    def set_price(self, symbol: str, price: float) -> None:
        """
        Set the current price for a symbol.
        
        This is used to simulate market prices for paper trading.
        
        Args:
            symbol: Ticker symbol
            price: Current price
        """
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        
        old_price = self._prices.get(symbol)
        self._prices[symbol] = price
        
        # Update position if exists
        if symbol in self._positions:
            self._positions[symbol].update_price(price)
            self._emit("on_position_update", self._positions[symbol])
        
        # Process pending orders at new price
        if self.fill_model == "next_bar" and old_price is not None:
            self._process_pending_orders(symbol, price)
        
        logger.debug(f"Set price for {symbol}: ${price:.2f}")
    
    def set_prices(self, prices: Dict[str, float]) -> None:
        """
        Set prices for multiple symbols.
        
        Args:
            prices: Dictionary of symbol -> price
        """
        for symbol, price in prices.items():
            self.set_price(symbol, price)
    
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price for a symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Current price or None if not set
        """
        return self._prices.get(symbol)
    
    # ==========================================================================
    # Order Execution
    # ==========================================================================
    
    def submit_order(self, order: Order) -> OrderResult:
        """
        Submit an order to the paper broker.
        
        Market orders are executed immediately at the current price
        (plus slippage). Limit orders are checked against the current
        price and executed if conditions are met.
        
        Args:
            order: Order to submit
            
        Returns:
            OrderResult with execution status
        """
        if not self._connected:
            return OrderResult(
                success=False,
                message="Broker not connected",
                error_code="NOT_CONNECTED"
            )
        
        # Validate symbol price is set
        if order.symbol not in self._prices:
            return OrderResult(
                success=False,
                message=f"No price set for {order.symbol}. Use set_price() first.",
                error_code="NO_PRICE"
            )
        
        # Assign order ID if not already set
        if order.order_id is None:
            order.order_id = f"ord_{uuid.uuid4().hex[:12]}"
        order.status = OrderStatus.SUBMITTED
        order.updated_at = datetime.now()
        
        # Store order
        self._orders[order.order_id] = order
        
        # Check buying power for buy orders
        current_price = self._prices[order.symbol]
        if order.side == OrderSide.BUY:
            estimated_cost = self._estimate_cost(order, current_price)
            if estimated_cost > self._cash:
                order.status = OrderStatus.REJECTED
                order.updated_at = datetime.now()
                self._emit("on_order_update", order)
                return OrderResult(
                    success=False,
                    order=order,
                    message=f"Insufficient funds. Need ${estimated_cost:,.2f}, have ${self._cash:,.2f}",
                    error_code="INSUFFICIENT_FUNDS"
                )
        
        # For sell orders, check if we have enough shares
        if order.side == OrderSide.SELL:
            position = self._positions.get(order.symbol)
            available = position.quantity if position else 0
            if available < order.quantity:
                order.status = OrderStatus.REJECTED
                order.updated_at = datetime.now()
                self._emit("on_order_update", order)
                return OrderResult(
                    success=False,
                    order=order,
                    message=f"Insufficient shares. Have {available}, need {order.quantity}",
                    error_code="INSUFFICIENT_SHARES"
                )
        
        order.status = OrderStatus.ACCEPTED
        order.updated_at = datetime.now()
        self._emit("on_order_update", order)
        
        # Process order based on fill model
        if self.fill_model == "immediate":
            return self._execute_order(order, current_price)
        else:
            # Queue for next bar
            self._pending_orders.append(order.order_id)
            return OrderResult(
                success=True,
                order=order,
                message="Order accepted, pending execution at next bar"
            )
    
    def _estimate_cost(self, order: Order, price: float) -> float:
        """Estimate total cost of an order including fees."""
        fill_price = self._apply_slippage(price, order.side)
        notional = order.quantity * fill_price
        commission = self._calculate_commission(order.quantity)
        return notional + commission
    
    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply slippage to price based on order side."""
        if side == OrderSide.BUY:
            return price * (1 + self.slippage)  # Pay more
        else:
            return price * (1 - self.slippage)  # Receive less
    
    def _calculate_commission(self, quantity: float) -> float:
        """Calculate commission for a trade."""
        return self.commission + (self.commission_per_share * quantity)
    
    def _execute_order(self, order: Order, current_price: float) -> OrderResult:
        """Execute an order at the given price."""
        # Determine execution price based on order type
        if order.order_type == OrderType.MARKET:
            fill_price = self._apply_slippage(current_price, order.side)
        elif order.order_type == OrderType.LIMIT:
            # Check if limit price is favorable
            if order.side == OrderSide.BUY:
                if current_price > order.limit_price:
                    # Current price too high for buy limit
                    return OrderResult(
                        success=True,
                        order=order,
                        message="Limit order pending - price above limit"
                    )
                fill_price = min(current_price, order.limit_price)
            else:  # SELL
                if current_price < order.limit_price:
                    # Current price too low for sell limit
                    return OrderResult(
                        success=True,
                        order=order,
                        message="Limit order pending - price below limit"
                    )
                fill_price = max(current_price, order.limit_price)
            fill_price = self._apply_slippage(fill_price, order.side)
        elif order.order_type == OrderType.STOP:
            # Check if stop is triggered
            if order.side == OrderSide.BUY:
                if current_price < order.stop_price:
                    return OrderResult(
                        success=True,
                        order=order,
                        message="Stop order pending - price below stop"
                    )
            else:  # SELL
                if current_price > order.stop_price:
                    return OrderResult(
                        success=True,
                        order=order,
                        message="Stop order pending - price above stop"
                    )
            fill_price = self._apply_slippage(current_price, order.side)
        else:
            # Default to market execution
            fill_price = self._apply_slippage(current_price, order.side)
        
        # Calculate commission
        commission = self._calculate_commission(order.quantity)
        
        # Create fill
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=datetime.now(),
            commission=commission,
            fill_id=f"fill_{uuid.uuid4().hex[:12]}",
            exchange="PAPER",
        )
        self._fills.append(fill)
        
        # Update order status
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.status = OrderStatus.FILLED
        order.updated_at = datetime.now()
        
        # Update cash and position
        self._update_position(fill)
        
        # Emit events
        self._emit("on_fill", fill)
        self._emit("on_order_update", order)
        
        logger.info(
            f"Executed {order.side.value.upper()} {order.quantity} {order.symbol} "
            f"@ ${fill_price:.2f} (commission: ${commission:.2f})"
        )
        
        return OrderResult(
            success=True,
            order=order,
            message=f"Order filled at ${fill_price:.2f}"
        )
    
    def _update_position(self, fill: Fill) -> None:
        """Update position and cash based on a fill."""
        symbol = fill.symbol
        notional = fill.quantity * fill.price
        
        if fill.side == OrderSide.BUY:
            # Buying: decrease cash, increase position
            self._cash -= notional + fill.commission
            
            if symbol in self._positions:
                # Update existing position
                pos = self._positions[symbol]
                new_quantity = pos.quantity + fill.quantity
                new_cost_basis = (pos.cost_basis or 0) + notional
                new_avg_cost = new_cost_basis / new_quantity if new_quantity > 0 else 0
                
                pos.quantity = new_quantity
                pos.avg_cost = new_avg_cost
                pos.cost_basis = new_cost_basis
            else:
                # Create new position
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=fill.quantity,
                    avg_cost=fill.price,
                    current_price=self._prices.get(symbol),
                )
            
            # Track open trade for P&L
            self._open_trades[symbol].append({
                "quantity": fill.quantity,
                "price": fill.price,
                "timestamp": fill.timestamp,
            })
        
        else:  # SELL
            # Selling: increase cash, decrease position
            self._cash += notional - fill.commission
            
            if symbol in self._positions:
                pos = self._positions[symbol]
                
                # Calculate realized P&L using FIFO
                realized_pnl = self._calculate_realized_pnl(symbol, fill.quantity, fill.price)
                pos.realized_pnl += realized_pnl
                
                # Update position
                pos.quantity -= fill.quantity
                if pos.quantity == 0:
                    # Position closed
                    pos.cost_basis = 0
                    pos.avg_cost = 0
                else:
                    # Recalculate cost basis
                    pos.cost_basis = pos.quantity * pos.avg_cost
                
                # Create trade record if position closed
                if pos.quantity == 0:
                    self._record_trade(symbol, realized_pnl)
        
        # Update position with current price
        if symbol in self._positions:
            self._positions[symbol].update_price(self._prices.get(symbol, 0))
            self._emit("on_position_update", self._positions[symbol])
    
    def _calculate_realized_pnl(
        self, symbol: str, quantity: float, sell_price: float
    ) -> float:
        """Calculate realized P&L using FIFO method."""
        remaining = quantity
        total_pnl = 0.0
        
        while remaining > 0 and self._open_trades[symbol]:
            trade = self._open_trades[symbol][0]
            
            if trade["quantity"] <= remaining:
                # Close entire trade
                pnl = (sell_price - trade["price"]) * trade["quantity"]
                total_pnl += pnl
                remaining -= trade["quantity"]
                self._open_trades[symbol].pop(0)
            else:
                # Partial close
                pnl = (sell_price - trade["price"]) * remaining
                total_pnl += pnl
                trade["quantity"] -= remaining
                remaining = 0
        
        return total_pnl
    
    def _record_trade(self, symbol: str, realized_pnl: float) -> None:
        """Record a completed trade."""
        # This would track entry/exit for reporting
        # Simplified version just logs
        logger.debug(f"Trade closed for {symbol}, realized P&L: ${realized_pnl:.2f}")
    
    def _process_pending_orders(self, symbol: str, price: float) -> None:
        """Process pending orders for a symbol at the new price."""
        orders_to_process = [
            oid for oid in self._pending_orders
            if self._orders[oid].symbol == symbol
        ]
        
        for order_id in orders_to_process:
            order = self._orders[order_id]
            if order.is_active:
                result = self._execute_order(order, price)
                if order.is_filled:
                    self._pending_orders.remove(order_id)
    
    # ==========================================================================
    # Order Management
    # ==========================================================================
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self._orders:
            raise OrderError(f"Order not found: {order_id}")
        
        order = self._orders[order_id]
        if not order.is_active:
            raise OrderError(f"Cannot cancel order in state: {order.status.value}")
        
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        
        if order_id in self._pending_orders:
            self._pending_orders.remove(order_id)
        
        self._emit("on_order_update", order)
        logger.info(f"Cancelled order: {order_id}")
        
        return True
    
    def modify_order(
        self,
        order_id: str,
        quantity: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> OrderResult:
        """Modify an existing order."""
        if order_id not in self._orders:
            raise OrderError(f"Order not found: {order_id}")
        
        order = self._orders[order_id]
        if not order.is_active:
            raise OrderError(f"Cannot modify order in state: {order.status.value}")
        
        if quantity is not None:
            if quantity <= 0:
                raise ValueError("quantity must be positive")
            order.quantity = quantity
        
        if limit_price is not None:
            order.limit_price = limit_price
        
        if stop_price is not None:
            order.stop_price = stop_price
        
        order.updated_at = datetime.now()
        self._emit("on_order_update", order)
        
        logger.info(f"Modified order: {order_id}")
        
        return OrderResult(
            success=True,
            order=order,
            message="Order modified"
        )
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Order]:
        """Get list of orders with optional filters."""
        orders = list(self._orders.values())
        
        if status is not None:
            orders = [o for o in orders if o.status == status]
        
        if symbol is not None:
            orders = [o for o in orders if o.symbol == symbol]
        
        if since is not None:
            orders = [o for o in orders if o.created_at and o.created_at >= since]
        
        return orders
    
    # ==========================================================================
    # Position Management
    # ==========================================================================
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self._positions.get(symbol)
    
    def get_positions(self) -> List[Position]:
        """Get all non-zero positions."""
        return [p for p in self._positions.values() if p.quantity != 0]
    
    # ==========================================================================
    # Account Information
    # ==========================================================================
    
    def get_account(self) -> AccountInfo:
        """Get current account information."""
        # Calculate portfolio value
        positions_value = sum(
            pos.market_value or 0
            for pos in self._positions.values()
        )
        portfolio_value = self._cash + positions_value
        
        return AccountInfo(
            account_id=self._account_id,
            cash=self._cash,
            portfolio_value=portfolio_value,
            equity=portfolio_value,
            buying_power=self._cash,  # No margin in paper trading
            margin_used=0.0,
            margin_available=self._cash,
            currency="USD",
            account_type="paper",
            status="active",
            created_at=self._created_at,
        )
    
    # ==========================================================================
    # Trade History
    # ==========================================================================
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Trade]:
        """Get trade history."""
        trades = self._trades.copy()
        
        if symbol is not None:
            trades = [t for t in trades if t.symbol == symbol]
        
        if since is not None:
            trades = [t for t in trades if t.entry_time >= since]
        
        if until is not None:
            trades = [t for t in trades if t.exit_time <= until]
        
        return trades
    
    def get_fills(
        self,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Fill]:
        """Get fill history."""
        fills = self._fills.copy()
        
        if order_id is not None:
            fills = [f for f in fills if f.order_id == order_id]
        
        if symbol is not None:
            fills = [f for f in fills if f.symbol == symbol]
        
        if since is not None:
            fills = [f for f in fills if f.timestamp >= since]
        
        return fills
    
    # ==========================================================================
    # Market Data
    # ==========================================================================
    
    def get_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get quote for a symbol."""
        price = self._prices.get(symbol)
        if price is None:
            return None
        
        # Simulate bid/ask spread
        spread = price * 0.0001  # 1 bp spread
        return {
            "bid": price - spread,
            "ask": price + spread,
            "last": price,
        }
    
    # ==========================================================================
    # Paper Trading Utilities
    # ==========================================================================
    
    def reset(self) -> None:
        """Reset broker to initial state."""
        self._cash = self.initial_cash
        self._positions.clear()
        self._orders.clear()
        self._fills.clear()
        self._trades.clear()
        self._prices.clear()
        self._pending_orders.clear()
        self._open_trades.clear()
        self._account_id = f"paper_{uuid.uuid4().hex[:8]}"
        self._created_at = datetime.now()
        logger.info(f"Reset paper broker: {self.name}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of broker state."""
        account = self.get_account()
        positions = self.get_positions()
        
        total_realized_pnl = sum(
            pos.realized_pnl for pos in self._positions.values()
        )
        total_unrealized_pnl = sum(
            pos.unrealized_pnl or 0 for pos in positions
        )
        
        return {
            "account_id": account.account_id,
            "initial_cash": self.initial_cash,
            "current_cash": self._cash,
            "portfolio_value": account.portfolio_value,
            "total_return": (account.portfolio_value / self.initial_cash - 1) * 100,
            "realized_pnl": total_realized_pnl,
            "unrealized_pnl": total_unrealized_pnl,
            "total_pnl": total_realized_pnl + total_unrealized_pnl,
            "num_positions": len(positions),
            "num_orders": len(self._orders),
            "num_fills": len(self._fills),
        }
    
    def print_summary(self) -> None:
        """Print formatted summary."""
        summary = self.get_summary()
        print("\n" + "=" * 50)
        print("PAPER BROKER SUMMARY")
        print("=" * 50)
        print(f"Account ID:      {summary['account_id']}")
        print(f"Initial Cash:    ${summary['initial_cash']:>15,.2f}")
        print(f"Current Cash:    ${summary['current_cash']:>15,.2f}")
        print(f"Portfolio Value: ${summary['portfolio_value']:>15,.2f}")
        print(f"Total Return:    {summary['total_return']:>15.2f}%")
        print("-" * 50)
        print(f"Realized P&L:    ${summary['realized_pnl']:>15,.2f}")
        print(f"Unrealized P&L:  ${summary['unrealized_pnl']:>15,.2f}")
        print(f"Total P&L:       ${summary['total_pnl']:>15,.2f}")
        print("-" * 50)
        print(f"Open Positions:  {summary['num_positions']:>15}")
        print(f"Total Orders:    {summary['num_orders']:>15}")
        print(f"Total Fills:     {summary['num_fills']:>15}")
        print("=" * 50 + "\n")

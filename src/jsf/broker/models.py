"""Broker data models and enumerations.

This module defines the core data structures for broker integration:
- Order types and statuses
- Position and trade representations
- Account information
- Execution reports (fills)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from decimal import Decimal


class OrderSide(Enum):
    """Order side (buy/sell)."""
    
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order types supported by brokers."""
    
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    """Order execution status."""
    
    PENDING = "pending"         # Order created, not yet submitted
    SUBMITTED = "submitted"     # Sent to broker
    ACCEPTED = "accepted"       # Accepted by broker
    PARTIAL = "partial"         # Partially filled
    FILLED = "filled"           # Completely filled
    CANCELLED = "cancelled"     # Cancelled by user or broker
    REJECTED = "rejected"       # Rejected by broker
    EXPIRED = "expired"         # Order expired (e.g., day order at EOD)


class TimeInForce(Enum):
    """Order time in force specifications."""
    
    DAY = "day"                 # Good for day only
    GTC = "gtc"                 # Good till cancelled
    IOC = "ioc"                 # Immediate or cancel
    FOK = "fok"                 # Fill or kill
    OPG = "opg"                 # Market on open
    CLS = "cls"                 # Market on close


class PositionSide(Enum):
    """Position side."""
    
    LONG = "long"
    SHORT = "short"


class AssetClass(Enum):
    """Asset class types."""
    
    EQUITY = "equity"
    OPTION = "option"
    FUTURE = "future"
    FOREX = "forex"
    CRYPTO = "crypto"


@dataclass
class Order:
    """
    Represents a trading order.
    
    Orders are instructions to buy or sell a specified quantity
    of a security at a specified price.
    
    Attributes:
        symbol: Ticker symbol
        side: Buy or sell
        quantity: Number of shares/contracts
        order_type: Market, limit, etc.
        limit_price: Price for limit orders
        stop_price: Trigger price for stop orders
        time_in_force: Order duration
        order_id: Unique order identifier
        client_order_id: Client-assigned identifier
        status: Current order status
        filled_quantity: Amount filled so far
        avg_fill_price: Average execution price
        created_at: Order creation timestamp
        updated_at: Last update timestamp
        extended_hours: Allow pre/post market execution
        metadata: Additional order metadata
    """
    
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    extended_hours: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate order parameters and generate IDs if needed."""
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if self.limit_price is None or self.limit_price <= 0:
                raise ValueError(f"{self.order_type.value} order requires positive limit_price")
        
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP):
            if self.stop_price is None or self.stop_price <= 0:
                raise ValueError(f"{self.order_type.value} order requires positive stop_price")
        
        # Auto-generate order_id if not provided
        if self.order_id is None:
            import uuid
            self.order_id = f"ord_{uuid.uuid4().hex[:12]}"
        
        if self.created_at is None:
            self.created_at = datetime.now()
        
        if self.updated_at is None:
            self.updated_at = self.created_at
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active (not terminal state)."""
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIAL
        )
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def remaining_quantity(self) -> float:
        """Get unfilled quantity."""
        return self.quantity - self.filled_quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "extended_hours": self.extended_hours,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        """Create order from dictionary."""
        return cls(
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            quantity=data["quantity"],
            order_type=OrderType(data.get("order_type", "market")),
            limit_price=data.get("limit_price"),
            stop_price=data.get("stop_price"),
            time_in_force=TimeInForce(data.get("time_in_force", "day")),
            order_id=data.get("order_id"),
            client_order_id=data.get("client_order_id"),
            status=OrderStatus(data.get("status", "pending")),
            filled_quantity=data.get("filled_quantity", 0.0),
            avg_fill_price=data.get("avg_fill_price"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            extended_hours=data.get("extended_hours", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Fill:
    """
    Represents an order execution (fill).
    
    A fill is generated when an order is executed (partially or fully).
    
    Attributes:
        order_id: Associated order ID
        symbol: Ticker symbol
        side: Buy or sell
        quantity: Number of shares filled
        price: Execution price
        timestamp: Execution timestamp
        commission: Commission charged
        fill_id: Unique fill identifier
        exchange: Exchange where executed
        metadata: Additional fill metadata
    """
    
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    fill_id: Optional[str] = None
    exchange: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate fill parameters."""
        if self.quantity <= 0:
            raise ValueError("Fill quantity must be positive")
        if self.price <= 0:
            raise ValueError("Fill price must be positive")
        if self.commission < 0:
            raise ValueError("Commission cannot be negative")
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of the fill."""
        return self.quantity * self.price
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost including commission."""
        base_cost = self.notional_value
        if self.side == OrderSide.BUY:
            return base_cost + self.commission
        return base_cost - self.commission
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "commission": self.commission,
            "fill_id": self.fill_id,
            "exchange": self.exchange,
            "metadata": self.metadata,
        }


@dataclass
class Position:
    """
    Represents a position in a security.
    
    Attributes:
        symbol: Ticker symbol
        quantity: Number of shares (positive for long, negative for short)
        avg_cost: Average cost basis per share
        current_price: Current market price
        unrealized_pnl: Unrealized profit/loss
        realized_pnl: Realized profit/loss from closed trades
        market_value: Current market value
        cost_basis: Total cost basis
        asset_class: Type of asset
        exchange: Primary exchange
        side: Long or short position
        metadata: Additional position metadata
    """
    
    symbol: str
    quantity: float
    avg_cost: float
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: float = 0.0
    market_value: Optional[float] = None
    cost_basis: Optional[float] = None
    asset_class: AssetClass = AssetClass.EQUITY
    exchange: Optional[str] = None
    side: Optional[PositionSide] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.cost_basis is None:
            self.cost_basis = abs(self.quantity) * self.avg_cost
        
        if self.side is None:
            self.side = PositionSide.LONG if self.quantity >= 0 else PositionSide.SHORT
        
        if self.current_price is not None:
            if self.market_value is None:
                self.market_value = abs(self.quantity) * self.current_price
            if self.unrealized_pnl is None:
                self.unrealized_pnl = (self.current_price - self.avg_cost) * self.quantity
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0
    
    @property
    def total_pnl(self) -> Optional[float]:
        """Calculate total P&L (realized + unrealized)."""
        if self.unrealized_pnl is None:
            return self.realized_pnl
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def pnl_percent(self) -> Optional[float]:
        """Calculate P&L as percentage of cost basis."""
        if self.cost_basis is None or self.cost_basis == 0:
            return None
        if self.unrealized_pnl is None:
            return None
        return (self.unrealized_pnl / self.cost_basis) * 100
    
    def update_price(self, new_price: float) -> None:
        """Update position with new market price."""
        self.current_price = new_price
        self.market_value = abs(self.quantity) * new_price
        self.unrealized_pnl = (new_price - self.avg_cost) * self.quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "market_value": self.market_value,
            "cost_basis": self.cost_basis,
            "asset_class": self.asset_class.value,
            "exchange": self.exchange,
            "side": self.side.value if self.side else None,
            "metadata": self.metadata,
        }


@dataclass
class AccountInfo:
    """
    Represents broker account information.
    
    Attributes:
        account_id: Unique account identifier
        cash: Available cash balance
        portfolio_value: Total portfolio value
        equity: Account equity (cash + positions)
        buying_power: Available buying power
        margin_used: Margin currently in use
        margin_available: Available margin
        day_trades_remaining: PDT rule remaining day trades
        currency: Account currency (e.g., 'USD')
        account_type: Type of account (cash, margin, etc.)
        status: Account status
        created_at: Account creation date
        updated_at: Last update timestamp
        metadata: Additional account metadata
    """
    
    account_id: str
    cash: float
    portfolio_value: float
    equity: Optional[float] = None
    buying_power: Optional[float] = None
    margin_used: float = 0.0
    margin_available: Optional[float] = None
    day_trades_remaining: Optional[int] = None
    currency: str = "USD"
    account_type: str = "paper"
    status: str = "active"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.equity is None:
            self.equity = self.portfolio_value
        
        if self.buying_power is None:
            # Default to 2x margin for margin accounts
            if self.account_type == "margin":
                self.buying_power = self.cash * 2
            else:
                self.buying_power = self.cash
        
        if self.margin_available is None:
            self.margin_available = self.buying_power - self.margin_used
        
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    @property
    def margin_ratio(self) -> Optional[float]:
        """Calculate margin utilization ratio."""
        if self.buying_power is None or self.buying_power == 0:
            return None
        return self.margin_used / self.buying_power
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "account_id": self.account_id,
            "cash": self.cash,
            "portfolio_value": self.portfolio_value,
            "equity": self.equity,
            "buying_power": self.buying_power,
            "margin_used": self.margin_used,
            "margin_available": self.margin_available,
            "day_trades_remaining": self.day_trades_remaining,
            "currency": self.currency,
            "account_type": self.account_type,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata,
        }


@dataclass
class Trade:
    """
    Represents a completed trade (round trip).
    
    A trade is created when a position is opened and closed.
    
    Attributes:
        trade_id: Unique trade identifier
        symbol: Ticker symbol
        side: Trade direction (long/short)
        entry_price: Average entry price
        exit_price: Average exit price
        quantity: Number of shares
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        pnl: Realized profit/loss
        pnl_percent: P&L as percentage
        commissions: Total commissions paid
        holding_period: Duration held
        metadata: Additional trade metadata
    """
    
    trade_id: str
    symbol: str
    side: PositionSide
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    commissions: float = 0.0
    holding_period: Optional[int] = None  # In seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.holding_period is None:
            delta = self.exit_time - self.entry_time
            self.holding_period = int(delta.total_seconds())
    
    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0
    
    @property
    def gross_pnl(self) -> float:
        """P&L before commissions."""
        return self.pnl + self.commissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "commissions": self.commissions,
            "holding_period": self.holding_period,
            "metadata": self.metadata,
        }


@dataclass
class OrderResult:
    """
    Result of an order submission.
    
    Attributes:
        success: Whether submission was successful
        order: The submitted order (with updated fields)
        message: Status or error message
        error_code: Error code if failed
    """
    
    success: bool
    order: Optional[Order] = None
    message: str = ""
    error_code: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "order": self.order.to_dict() if self.order else None,
            "message": self.message,
            "error_code": self.error_code,
        }

"""Alpaca broker integration.

This module provides integration with the Alpaca Trading API
for paper trading and live trading.

Alpaca offers commission-free trading with a REST API,
making it ideal for algorithmic trading.

API Documentation: https://docs.alpaca.markets/
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import os

from jsf.broker.base import (
    Broker,
    BrokerError,
    ConnectionError,
    OrderError,
    InsufficientFundsError,
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
    AssetClass,
)
from jsf.utils.logging import get_logger

logger = get_logger(__name__)

# Optional dependency
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    tradeapi = None
    APIError = Exception


def require_alpaca():
    """Check if alpaca-trade-api is installed."""
    if not ALPACA_AVAILABLE:
        raise ImportError(
            "alpaca-trade-api is required for AlpacaBroker. "
            "Install with: pip install alpaca-trade-api"
        )


class AlpacaBroker(Broker):
    """
    Alpaca Markets broker integration.
    
    Supports both paper trading and live trading through the Alpaca API.
    
    Setup:
        1. Create an Alpaca account at https://alpaca.markets/
        2. Get your API keys from the dashboard
        3. For paper trading, use the paper trading API keys
    
    Environment Variables:
        APCA_API_KEY_ID: Your Alpaca API key
        APCA_API_SECRET_KEY: Your Alpaca API secret
        APCA_API_BASE_URL: API base URL (paper or live)
    
    Example:
        ```python
        broker = AlpacaBroker(
            api_key="your_api_key",
            api_secret="your_api_secret",
            paper=True  # Use paper trading
        )
        broker.connect()
        
        # Submit a market order
        result = broker.buy("AAPL", 10)
        
        # Check positions
        positions = broker.get_positions()
        ```
    
    Attributes:
        api_key: Alpaca API key ID
        api_secret: Alpaca API secret key
        paper: If True, use paper trading endpoint
        base_url: API base URL (auto-set based on paper flag)
    """
    
    # API endpoints
    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True,
        base_url: Optional[str] = None,
        name: str = "alpaca",
        **kwargs,
    ):
        """
        Initialize Alpaca broker.
        
        Args:
            api_key: Alpaca API key ID (or use APCA_API_KEY_ID env var)
            api_secret: Alpaca API secret key (or use APCA_API_SECRET_KEY env var)
            paper: If True, use paper trading API
            base_url: Custom API base URL (overrides paper flag)
            name: Broker identifier
            **kwargs: Additional parameters
        """
        require_alpaca()
        super().__init__(name=name, **kwargs)
        
        # API credentials — support both naming conventions
        self.api_key = (
            api_key
            or os.getenv("ALPACA_API_KEY")       # jsf-core .env convention
            or os.getenv("APCA_API_KEY_ID")       # alpaca SDK convention
        )
        self.api_secret = (
            api_secret
            or os.getenv("ALPACA_SECRET_KEY")     # jsf-core .env convention
            or os.getenv("APCA_API_SECRET_KEY")   # alpaca SDK convention
        )
        
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca API credentials required. "
                "Set api_key/api_secret or ALPACA_API_KEY/ALPACA_SECRET_KEY env vars."
            )
        
        # API endpoint
        self.paper = paper
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = self.PAPER_URL if paper else self.LIVE_URL
        
        # API client
        self._api: Optional[tradeapi.REST] = None
        
        logger.info(
            f"Initialized AlpacaBroker (paper={paper}) "
            f"with base_url={self.base_url}"
        )
    
    # ==========================================================================
    # Connection Management
    # ==========================================================================
    
    def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            self._api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self.base_url,
            )
            
            # Verify connection by fetching account
            account = self._api.get_account()
            self._connected = True
            
            logger.info(
                f"Connected to Alpaca. Account: {account.account_number}, "
                f"Status: {account.status}, Equity: ${float(account.equity):,.2f}"
            )
            return True
            
        except APIError as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to Alpaca: {e}")
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Unexpected error connecting to Alpaca: {e}")
    
    def disconnect(self) -> bool:
        """Disconnect from Alpaca API."""
        self._api = None
        self._connected = False
        logger.info("Disconnected from Alpaca")
        return True
    
    def _check_connection(self) -> None:
        """Verify broker is connected."""
        if not self._connected or self._api is None:
            raise ConnectionError("Not connected to Alpaca. Call connect() first.")
    
    # ==========================================================================
    # Order Management
    # ==========================================================================
    
    def submit_order(self, order: Order) -> OrderResult:
        """Submit an order to Alpaca."""
        self._check_connection()
        
        try:
            # Map order type
            alpaca_type = self._map_order_type(order.order_type)
            alpaca_side = "buy" if order.side == OrderSide.BUY else "sell"
            alpaca_tif = self._map_time_in_force(order.time_in_force)
            
            # Build order parameters
            params = {
                "symbol": order.symbol,
                "qty": order.quantity,
                "side": alpaca_side,
                "type": alpaca_type,
                "time_in_force": alpaca_tif,
                "extended_hours": order.extended_hours,
            }
            
            # Add price parameters for limit/stop orders
            if order.limit_price is not None:
                params["limit_price"] = str(order.limit_price)
            if order.stop_price is not None:
                params["stop_price"] = str(order.stop_price)
            
            # Add client order ID if provided
            if order.client_order_id:
                params["client_order_id"] = order.client_order_id
            
            # Submit order
            alpaca_order = self._api.submit_order(**params)
            
            # Update order with Alpaca response
            order.order_id = alpaca_order.id
            order.client_order_id = alpaca_order.client_order_id
            order.status = self._map_order_status(alpaca_order.status)
            order.filled_quantity = float(alpaca_order.filled_qty or 0)
            if alpaca_order.filled_avg_price:
                order.avg_fill_price = float(alpaca_order.filled_avg_price)
            order.created_at = alpaca_order.created_at
            order.updated_at = alpaca_order.updated_at
            
            logger.info(f"Submitted order {order.order_id}: {alpaca_side} {order.quantity} {order.symbol}")
            
            return OrderResult(
                success=True,
                order=order,
                message=f"Order submitted: {alpaca_order.status}"
            )
            
        except APIError as e:
            error_code = getattr(e, 'code', None)
            
            # Handle specific error types
            if "insufficient" in str(e).lower():
                return OrderResult(
                    success=False,
                    order=order,
                    message=str(e),
                    error_code="INSUFFICIENT_FUNDS"
                )
            
            order.status = OrderStatus.REJECTED
            return OrderResult(
                success=False,
                order=order,
                message=str(e),
                error_code=str(error_code) if error_code else "API_ERROR"
            )
        except Exception as e:
            order.status = OrderStatus.REJECTED
            return OrderResult(
                success=False,
                order=order,
                message=f"Unexpected error: {e}",
                error_code="UNKNOWN_ERROR"
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        self._check_connection()
        
        try:
            self._api.cancel_order(order_id)
            logger.info(f"Cancelled order: {order_id}")
            return True
        except APIError as e:
            raise OrderError(f"Failed to cancel order {order_id}: {e}")
    
    def modify_order(
        self,
        order_id: str,
        quantity: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> OrderResult:
        """
        Modify an existing order.
        
        Note: Alpaca uses a replace operation for order modifications.
        """
        self._check_connection()
        
        try:
            params = {}
            if quantity is not None:
                params["qty"] = str(int(quantity))
            if limit_price is not None:
                params["limit_price"] = str(limit_price)
            if stop_price is not None:
                params["stop_price"] = str(stop_price)
            
            if not params:
                return OrderResult(
                    success=False,
                    message="No modifications specified"
                )
            
            alpaca_order = self._api.replace_order(order_id, **params)
            
            order = self._alpaca_order_to_order(alpaca_order)
            logger.info(f"Modified order: {order_id}")
            
            return OrderResult(
                success=True,
                order=order,
                message="Order modified"
            )
            
        except APIError as e:
            raise OrderError(f"Failed to modify order {order_id}: {e}")
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        self._check_connection()
        
        try:
            alpaca_order = self._api.get_order(order_id)
            return self._alpaca_order_to_order(alpaca_order)
        except APIError:
            return None
    
    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Order]:
        """Get list of orders."""
        self._check_connection()
        
        try:
            # Map status to Alpaca format
            alpaca_status = None
            if status is not None:
                if status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, 
                             OrderStatus.ACCEPTED, OrderStatus.PARTIAL):
                    alpaca_status = "open"
                elif status == OrderStatus.FILLED:
                    alpaca_status = "closed"
                elif status == OrderStatus.CANCELLED:
                    alpaca_status = "canceled"  # Alpaca uses American spelling
            
            params = {}
            if alpaca_status:
                params["status"] = alpaca_status
            if since:
                params["after"] = since.isoformat()
            
            alpaca_orders = self._api.list_orders(**params)
            orders = [self._alpaca_order_to_order(o) for o in alpaca_orders]
            
            # Additional filtering
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
            if status and alpaca_status is None:
                orders = [o for o in orders if o.status == status]
            
            return orders
            
        except APIError as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    # ==========================================================================
    # Position Management
    # ==========================================================================
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        self._check_connection()
        
        try:
            alpaca_pos = self._api.get_position(symbol)
            return self._alpaca_position_to_position(alpaca_pos)
        except APIError:
            return None
    
    def get_positions(self) -> List[Position]:
        """Get all positions."""
        self._check_connection()
        
        try:
            alpaca_positions = self._api.list_positions()
            return [self._alpaca_position_to_position(p) for p in alpaca_positions]
        except APIError as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    # ==========================================================================
    # Account Information
    # ==========================================================================
    
    def get_account(self) -> AccountInfo:
        """Get account information."""
        self._check_connection()
        
        try:
            account = self._api.get_account()
            
            return AccountInfo(
                account_id=account.account_number,
                cash=float(account.cash),
                portfolio_value=float(account.portfolio_value),
                equity=float(account.equity),
                buying_power=float(account.buying_power),
                margin_used=float(account.initial_margin) if hasattr(account, 'initial_margin') else 0,
                margin_available=float(account.regt_buying_power) if hasattr(account, 'regt_buying_power') else float(account.buying_power),
                day_trades_remaining=int(account.daytrade_count) if hasattr(account, 'daytrade_count') else None,
                currency=account.currency,
                account_type="paper" if self.paper else "live",
                status=account.status,
                created_at=account.created_at if hasattr(account, 'created_at') else None,
            )
            
        except APIError as e:
            raise BrokerError(f"Failed to get account info: {e}")
    
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
        self._check_connection()
        
        # Alpaca doesn't have a direct trades endpoint
        # We need to reconstruct from activities or orders
        # For now, return empty list (can be enhanced)
        logger.warning("Trade history reconstruction not fully implemented for Alpaca")
        return []
    
    def get_fills(
        self,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Fill]:
        """Get fill history from account activities."""
        self._check_connection()
        
        try:
            params = {"activity_types": "FILL"}
            if since:
                params["after"] = since.isoformat()
            
            activities = self._api.get_activities(**params)
            
            fills = []
            for act in activities:
                if hasattr(act, 'order_id'):
                    # Filter by order_id if specified
                    if order_id and act.order_id != order_id:
                        continue
                    # Filter by symbol if specified
                    if symbol and act.symbol != symbol:
                        continue
                    
                    fills.append(Fill(
                        order_id=act.order_id,
                        symbol=act.symbol,
                        side=OrderSide.BUY if act.side == "buy" else OrderSide.SELL,
                        quantity=float(act.qty),
                        price=float(act.price),
                        timestamp=act.transaction_time,
                        fill_id=act.id if hasattr(act, 'id') else None,
                    ))
            
            return fills
            
        except APIError as e:
            logger.error(f"Failed to get fills: {e}")
            return []
    
    # ==========================================================================
    # Market Data
    # ==========================================================================
    
    def get_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current quote for a symbol."""
        self._check_connection()
        
        try:
            quote = self._api.get_latest_quote(symbol)
            return {
                "bid": float(quote.bid_price),
                "ask": float(quote.ask_price),
                "last": float(quote.ask_price),  # Alpaca doesn't always have last price
            }
        except Exception as e:
            logger.debug(f"Failed to get quote for {symbol}: {e}")
            return None
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last trade price for a symbol."""
        self._check_connection()
        
        try:
            trade = self._api.get_latest_trade(symbol)
            return float(trade.price)
        except Exception as e:
            logger.debug(f"Failed to get last price for {symbol}: {e}")
            return None
    
    # ==========================================================================
    # Helper Methods
    # ==========================================================================
    
    def _map_order_type(self, order_type: OrderType) -> str:
        """Map internal order type to Alpaca format."""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop_limit",
            OrderType.TRAILING_STOP: "trailing_stop",
        }
        return mapping.get(order_type, "market")
    
    def _map_time_in_force(self, tif: TimeInForce) -> str:
        """Map internal time in force to Alpaca format."""
        mapping = {
            TimeInForce.DAY: "day",
            TimeInForce.GTC: "gtc",
            TimeInForce.IOC: "ioc",
            TimeInForce.FOK: "fok",
            TimeInForce.OPG: "opg",
            TimeInForce.CLS: "cls",
        }
        return mapping.get(tif, "day")
    
    def _map_order_status(self, alpaca_status: str) -> OrderStatus:
        """Map Alpaca order status to internal format."""
        mapping = {
            "new": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.ACCEPTED,
            "pending_new": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIAL,
            "filled": OrderStatus.FILLED,
            "done_for_day": OrderStatus.EXPIRED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "replaced": OrderStatus.CANCELLED,
            "pending_cancel": OrderStatus.ACCEPTED,
            "pending_replace": OrderStatus.ACCEPTED,
            "rejected": OrderStatus.REJECTED,
        }
        return mapping.get(alpaca_status.lower(), OrderStatus.PENDING)
    
    def _alpaca_order_to_order(self, alpaca_order) -> Order:
        """Convert Alpaca order to internal Order."""
        return Order(
            symbol=alpaca_order.symbol,
            side=OrderSide.BUY if alpaca_order.side == "buy" else OrderSide.SELL,
            quantity=float(alpaca_order.qty),
            order_type=self._reverse_map_order_type(alpaca_order.type),
            limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
            stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
            time_in_force=self._reverse_map_tif(alpaca_order.time_in_force),
            order_id=alpaca_order.id,
            client_order_id=alpaca_order.client_order_id,
            status=self._map_order_status(alpaca_order.status),
            filled_quantity=float(alpaca_order.filled_qty or 0),
            avg_fill_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
            created_at=alpaca_order.created_at,
            updated_at=alpaca_order.updated_at,
            extended_hours=alpaca_order.extended_hours,
        )
    
    def _reverse_map_order_type(self, alpaca_type: str) -> OrderType:
        """Map Alpaca order type to internal format."""
        mapping = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP,
            "stop_limit": OrderType.STOP_LIMIT,
            "trailing_stop": OrderType.TRAILING_STOP,
        }
        return mapping.get(alpaca_type.lower(), OrderType.MARKET)
    
    def _reverse_map_tif(self, alpaca_tif: str) -> TimeInForce:
        """Map Alpaca time in force to internal format."""
        mapping = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
            "opg": TimeInForce.OPG,
            "cls": TimeInForce.CLS,
        }
        return mapping.get(alpaca_tif.lower(), TimeInForce.DAY)
    
    def _alpaca_position_to_position(self, alpaca_pos) -> Position:
        """Convert Alpaca position to internal Position."""
        qty = float(alpaca_pos.qty)
        return Position(
            symbol=alpaca_pos.symbol,
            quantity=qty,
            avg_cost=float(alpaca_pos.avg_entry_price),
            current_price=float(alpaca_pos.current_price),
            unrealized_pnl=float(alpaca_pos.unrealized_pl),
            market_value=float(alpaca_pos.market_value),
            cost_basis=float(alpaca_pos.cost_basis),
            asset_class=AssetClass.EQUITY,
            exchange=alpaca_pos.exchange if hasattr(alpaca_pos, 'exchange') else None,
            side=PositionSide.LONG if qty > 0 else PositionSide.SHORT,
        )

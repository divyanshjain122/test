"""Live trading engine.

This module provides the main orchestration for live trading,
coordinating data handling, signal generation, order management,
and position monitoring.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import threading
import time
import logging

import pandas as pd
import numpy as np

from jsf.broker import (
    Broker,
    PaperBroker,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    Position,
    Fill,
    AccountInfo,
)
from jsf.live.data_handler import (
    DataHandler,
    SimulatedDataHandler,
    PriceUpdate,
)
from jsf.live.order_manager import (
    OrderManager,
    OrderTracker,
    OrderState,
)
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class EngineError(Exception):
    """Exception raised for engine errors."""
    pass


class TradingState(Enum):
    """Live trading engine states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class EngineConfig:
    """Configuration for LiveTradingEngine."""
    
    # Trading parameters
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # Max 10% of portfolio per position
    max_total_exposure: float = 1.0  # Max 100% invested
    
    # Timing
    trading_interval: float = 1.0  # Seconds between trading cycles
    data_warmup_seconds: float = 5.0  # Wait for data before trading
    
    # Risk controls
    max_daily_loss: float = 0.02  # Stop trading if down 2%
    max_order_value: float = 50000.0  # Max single order value
    require_prices_for_all: bool = True  # Need prices for all symbols
    
    # Execution
    use_market_orders: bool = True
    default_time_in_force: TimeInForce = TimeInForce.DAY
    
    def __post_init__(self):
        """Validate configuration."""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0 < self.max_position_size <= 1:
            raise ValueError("max_position_size must be between 0 and 1")
        if not 0 < self.max_total_exposure <= 2:
            raise ValueError("max_total_exposure must be between 0 and 2")


@dataclass
class TradingSnapshot:
    """Snapshot of trading state at a point in time."""
    
    timestamp: datetime
    state: TradingState
    equity: float
    cash: float
    positions: Dict[str, Position]
    prices: Dict[str, float]
    daily_pnl: float
    daily_return: float
    active_orders: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "state": self.state.value,
            "equity": self.equity,
            "cash": self.cash,
            "position_count": len(self.positions),
            "daily_pnl": self.daily_pnl,
            "daily_return": self.daily_return,
            "active_orders": self.active_orders,
        }


class LiveTradingEngine:
    """
    Main live trading engine.
    
    Orchestrates the complete live trading workflow:
    1. Connects to broker and data sources
    2. Manages real-time data subscriptions
    3. Executes trading logic on each cycle
    4. Manages orders and positions
    5. Monitors risk and enforces limits
    
    Example usage:
        ```python
        # Create broker and data handler
        broker = PaperBroker(initial_cash=100000)
        data_handler = SimulatedDataHandler(
            initial_prices={"AAPL": 150, "GOOGL": 2800}
        )
        
        # Create engine
        engine = LiveTradingEngine(
            broker=broker,
            data_handler=data_handler,
            symbols=["AAPL", "GOOGL"],
        )
        
        # Define trading logic
        def my_strategy(engine, prices):
            # Return target weights
            return {"AAPL": 0.5, "GOOGL": 0.5}
        
        engine.set_strategy(my_strategy)
        
        # Start trading
        engine.start()
        time.sleep(60)  # Trade for 1 minute
        engine.stop()
        ```
    """
    
    def __init__(
        self,
        broker: Optional[Broker] = None,
        data_handler: Optional[DataHandler] = None,
        symbols: Optional[List[str]] = None,
        config: Optional[EngineConfig] = None,
    ):
        """
        Initialize live trading engine.
        
        Args:
            broker: Broker for order execution (creates PaperBroker if None)
            data_handler: Handler for price data (creates SimulatedDataHandler if None)
            symbols: Symbols to trade
            config: Engine configuration
        """
        self.config = config or EngineConfig()
        self.symbols = [s.upper() for s in (symbols or [])]
        
        # Create default broker if not provided
        if broker is None:
            self.broker = PaperBroker(initial_cash=self.config.initial_capital)
        else:
            self.broker = broker
        
        # Create default data handler if not provided
        if data_handler is None:
            self.data_handler = SimulatedDataHandler(
                initial_prices={s: 100.0 for s in self.symbols}
            )
        else:
            self.data_handler = data_handler
        
        # Create order manager
        self.order_manager = OrderManager(self.broker)
        
        # State
        self._state = TradingState.STOPPED
        self._strategy: Optional[Callable] = None
        self._start_time: Optional[datetime] = None
        self._start_equity: Optional[float] = None
        self._daily_high_equity: Optional[float] = None
        
        # Threading
        self._trading_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # History
        self._snapshots: List[TradingSnapshot] = []
        self._max_snapshots = 10000
        
        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "on_cycle": [],
            "on_trade": [],
            "on_state_change": [],
            "on_error": [],
        }
        
        logger.info(f"Initialized LiveTradingEngine with {len(self.symbols)} symbols")
    
    # ==========================================================================
    # Strategy Configuration
    # ==========================================================================
    
    def set_strategy(
        self,
        strategy: Callable[["LiveTradingEngine", Dict[str, float]], Dict[str, float]]
    ) -> None:
        """
        Set the trading strategy.
        
        The strategy function receives the engine and current prices,
        and returns target portfolio weights.
        
        Args:
            strategy: Strategy function(engine, prices) -> weights
        """
        self._strategy = strategy
        logger.info("Strategy set")
    
    def set_symbols(self, symbols: List[str]) -> None:
        """
        Set symbols to trade.
        
        Args:
            symbols: List of symbols
        """
        self.symbols = [s.upper() for s in symbols]
        self.data_handler.subscribe(self.symbols)
        logger.info(f"Trading symbols: {self.symbols}")
    
    # ==========================================================================
    # Lifecycle Management
    # ==========================================================================
    
    def start(self) -> None:
        """Start the trading engine."""
        if self._state != TradingState.STOPPED:
            logger.warning(f"Cannot start engine in state: {self._state}")
            return
        
        self._set_state(TradingState.STARTING)
        
        try:
            # Connect broker
            self.broker.connect()
            
            # Subscribe to data
            if self.symbols:
                self.data_handler.subscribe(self.symbols)
            self.data_handler.start()
            
            # Initialize state
            self._start_time = datetime.now()
            account = self.broker.get_account()
            self._start_equity = account.equity
            self._daily_high_equity = account.equity
            
            # Wait for data warmup
            logger.info(f"Warming up data ({self.config.data_warmup_seconds}s)...")
            time.sleep(self.config.data_warmup_seconds)
            
            # Start trading loop
            self._stop_event.clear()
            self._trading_thread = threading.Thread(
                target=self._trading_loop,
                daemon=True,
                name="trading_loop"
            )
            self._trading_thread.start()
            
            self._set_state(TradingState.RUNNING)
            logger.info("Trading engine started")
            
        except Exception as e:
            logger.error(f"Failed to start engine: {e}")
            self._set_state(TradingState.ERROR)
            raise EngineError(f"Failed to start: {e}")
    
    def stop(self) -> None:
        """Stop the trading engine."""
        if self._state not in (TradingState.RUNNING, TradingState.PAUSED):
            return
        
        self._set_state(TradingState.STOPPING)
        logger.info("Stopping trading engine...")
        
        # Signal stop
        self._stop_event.set()
        
        # Wait for trading thread
        if self._trading_thread and self._trading_thread.is_alive():
            self._trading_thread.join(timeout=10.0)
        
        # Cancel pending orders
        cancelled = self.order_manager.cancel_all_orders()
        if cancelled > 0:
            logger.info(f"Cancelled {cancelled} pending orders")
        
        # Stop data handler
        self.data_handler.stop()
        
        # Disconnect broker
        self.broker.disconnect()
        
        self._set_state(TradingState.STOPPED)
        logger.info("Trading engine stopped")
    
    def pause(self) -> None:
        """Pause trading (keeps connections alive)."""
        if self._state != TradingState.RUNNING:
            return
        
        self._set_state(TradingState.PAUSED)
        logger.info("Trading paused")
    
    def resume(self) -> None:
        """Resume trading after pause."""
        if self._state != TradingState.PAUSED:
            return
        
        self._set_state(TradingState.RUNNING)
        logger.info("Trading resumed")
    
    def _set_state(self, new_state: TradingState) -> None:
        """Set engine state and emit callback."""
        old_state = self._state
        self._state = new_state
        self._emit("on_state_change", {"old": old_state, "new": new_state})
    
    @property
    def state(self) -> TradingState:
        """Get current engine state."""
        return self._state
    
    @property
    def is_running(self) -> bool:
        """Check if engine is actively trading."""
        return self._state == TradingState.RUNNING
    
    # ==========================================================================
    # Trading Loop
    # ==========================================================================
    
    def _trading_loop(self) -> None:
        """Main trading loop."""
        logger.info("Trading loop started")
        
        while not self._stop_event.is_set():
            try:
                if self._state == TradingState.RUNNING:
                    self._execute_cycle()
                
                self._stop_event.wait(self.config.trading_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                self._emit("on_error", e)
                
                # Don't stop on individual cycle errors
                continue
        
        logger.info("Trading loop ended")
    
    def _execute_cycle(self) -> None:
        """Execute a single trading cycle."""
        # Get current prices
        prices = self.data_handler.get_prices()
        
        # Check if we have required prices
        if self.config.require_prices_for_all:
            missing = [s for s in self.symbols if s not in prices]
            if missing:
                logger.debug(f"Waiting for prices: {missing}")
                return
        
        # Check risk limits
        if not self._check_risk_limits():
            return
        
        # Execute strategy if defined
        if self._strategy is not None:
            try:
                target_weights = self._strategy(self, prices)
                if target_weights:
                    self._rebalance_to_weights(target_weights, prices)
            except Exception as e:
                logger.error(f"Strategy error: {e}")
                self._emit("on_error", e)
        
        # Take snapshot
        self._take_snapshot(prices)
        
        # Emit cycle callback
        self._emit("on_cycle", {"prices": prices, "timestamp": datetime.now()})
    
    def _check_risk_limits(self) -> bool:
        """
        Check if risk limits are breached.
        
        Returns:
            True if trading should continue, False if limits breached
        """
        account = self.broker.get_account()
        
        # Check daily loss limit
        if self._start_equity is not None:
            daily_return = (account.equity - self._start_equity) / self._start_equity
            
            if daily_return < -self.config.max_daily_loss:
                logger.warning(
                    f"Daily loss limit breached: {daily_return:.2%} "
                    f"(limit: {-self.config.max_daily_loss:.2%})"
                )
                self.pause()
                return False
        
        return True
    
    def _rebalance_to_weights(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float]
    ) -> None:
        """
        Rebalance portfolio to target weights.
        
        Args:
            target_weights: Target weights by symbol
            prices: Current prices
        """
        account = self.broker.get_account()
        positions = self.get_positions()  # Use the dict version
        
        # Calculate current weights
        current_weights = {}
        for symbol, position in positions.items():
            if symbol in prices:
                position_value = position.quantity * prices[symbol]
                current_weights[symbol] = position_value / account.equity
        
        # Calculate required trades
        for symbol, target_weight in target_weights.items():
            if symbol not in prices:
                continue
            
            # Apply position size limit
            target_weight = min(target_weight, self.config.max_position_size)
            target_weight = max(target_weight, -self.config.max_position_size)
            
            current_weight = current_weights.get(symbol, 0.0)
            weight_diff = target_weight - current_weight
            
            # Skip small adjustments (< 1%)
            if abs(weight_diff) < 0.01:
                continue
            
            # Calculate trade
            target_value = target_weight * account.equity
            current_value = current_weight * account.equity
            trade_value = target_value - current_value
            
            # Check max order value
            if abs(trade_value) > self.config.max_order_value:
                trade_value = self.config.max_order_value * (1 if trade_value > 0 else -1)
            
            price = prices[symbol]
            quantity = abs(trade_value) / price
            
            if quantity < 0.01:  # Skip very small trades
                continue
            
            # Create order
            side = OrderSide.BUY if trade_value > 0 else OrderSide.SELL
            order_type = OrderType.MARKET if self.config.use_market_orders else OrderType.LIMIT
            
            order = Order(
                symbol=symbol,
                side=side,
                quantity=round(quantity, 2),
                order_type=order_type,
                limit_price=price if order_type == OrderType.LIMIT else None,
                time_in_force=self.config.default_time_in_force,
            )
            
            try:
                tracker = self.order_manager.submit_order(order)
                self._emit("on_trade", tracker)
            except Exception as e:
                logger.error(f"Failed to submit order for {symbol}: {e}")
    
    # ==========================================================================
    # Position Management
    # ==========================================================================
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions as a dictionary."""
        positions_list = self.broker.get_positions()
        return {p.symbol: p for p in positions_list}
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        positions = self.get_positions()
        return positions.get(symbol.upper())
    
    def close_position(self, symbol: str) -> Optional[OrderTracker]:
        """
        Close a position.
        
        Args:
            symbol: Symbol to close
            
        Returns:
            OrderTracker for the closing order
        """
        symbol = symbol.upper()
        position = self.get_position(symbol)
        
        if position is None or position.quantity == 0:
            logger.info(f"No position to close for {symbol}")
            return None
        
        # Determine side (opposite of position)
        side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
        
        order = Order(
            symbol=symbol,
            side=side,
            quantity=abs(position.quantity),
            order_type=OrderType.MARKET,
        )
        
        return self.order_manager.submit_order(order)
    
    def close_all_positions(self) -> List[OrderTracker]:
        """
        Close all positions.
        
        Returns:
            List of order trackers for closing orders
        """
        trackers = []
        positions = self.get_positions()
        
        for symbol in positions:
            tracker = self.close_position(symbol)
            if tracker:
                trackers.append(tracker)
        
        return trackers
    
    # ==========================================================================
    # Account Information
    # ==========================================================================
    
    def get_account(self) -> AccountInfo:
        """Get account information."""
        return self.broker.get_account()
    
    @property
    def equity(self) -> float:
        """Get current equity."""
        return self.broker.get_account().equity
    
    @property
    def cash(self) -> float:
        """Get available cash."""
        return self.broker.get_account().cash
    
    @property
    def daily_pnl(self) -> float:
        """Get daily P&L."""
        if self._start_equity is None:
            return 0.0
        return self.equity - self._start_equity
    
    @property
    def daily_return(self) -> float:
        """Get daily return."""
        if self._start_equity is None or self._start_equity == 0:
            return 0.0
        return self.daily_pnl / self._start_equity
    
    # ==========================================================================
    # Snapshots and History
    # ==========================================================================
    
    def _take_snapshot(self, prices: Dict[str, float]) -> None:
        """Take a snapshot of current state."""
        account = self.broker.get_account()
        
        snapshot = TradingSnapshot(
            timestamp=datetime.now(),
            state=self._state,
            equity=account.equity,
            cash=account.cash,
            positions=self.get_positions(),
            prices=dict(prices),
            daily_pnl=self.daily_pnl,
            daily_return=self.daily_return,
            active_orders=self.order_manager.active_order_count,
        )
        
        with self._lock:
            self._snapshots.append(snapshot)
            if len(self._snapshots) > self._max_snapshots:
                self._snapshots = self._snapshots[-self._max_snapshots // 2:]
    
    def get_snapshots(
        self,
        n: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> List[TradingSnapshot]:
        """
        Get trading snapshots.
        
        Args:
            n: Number of most recent snapshots
            since: Get snapshots since this time
            
        Returns:
            List of snapshots
        """
        with self._lock:
            snapshots = list(self._snapshots)
        
        if since:
            snapshots = [s for s in snapshots if s.timestamp >= since]
        
        if n:
            snapshots = snapshots[-n:]
        
        return snapshots
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve from snapshots."""
        snapshots = self.get_snapshots()
        if not snapshots:
            return pd.Series(dtype=float)
        
        data = {s.timestamp: s.equity for s in snapshots}
        return pd.Series(data).sort_index()
    
    # ==========================================================================
    # Event Callbacks
    # ==========================================================================
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for an event.
        
        Events:
            - on_cycle: Called each trading cycle
            - on_trade: Called when a trade is executed
            - on_state_change: Called when engine state changes
            - on_error: Called on errors
        
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
    # Summary and Status
    # ==========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get engine status summary.
        
        Returns:
            Dictionary with status information
        """
        account = self.broker.get_account()
        
        return {
            "state": self._state.value,
            "symbols": self.symbols,
            "equity": account.equity,
            "cash": account.cash,
            "positions": len(self.broker.get_positions()),
            "active_orders": self.order_manager.active_order_count,
            "daily_pnl": self.daily_pnl,
            "daily_return": f"{self.daily_return:.2%}",
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "uptime_seconds": (
                (datetime.now() - self._start_time).total_seconds()
                if self._start_time else 0
            ),
        }
    
    def __repr__(self) -> str:
        return (
            f"LiveTradingEngine(state={self._state.value}, "
            f"symbols={len(self.symbols)}, equity=${self.equity:,.2f})"
        )

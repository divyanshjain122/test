"""Data Collectors for Dashboard

Collects and aggregates data from broker and live trading engine
for dashboard display.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import threading
import time
from collections import deque

import pandas as pd

from .models import (
    PortfolioSnapshot,
    PositionSnapshot,
    TradeRecord,
    DashboardState,
)


@dataclass
class SnapshotHistory:
    """Maintains historical snapshots for time-series analysis.
    
    Uses a ring buffer to efficiently store snapshots with
    configurable retention period.
    
    Attributes:
        max_snapshots: Maximum number of snapshots to retain
        retention_hours: Hours to retain snapshots
        snapshots: Deque of historical snapshots
    """
    max_snapshots: int = 10000
    retention_hours: int = 24
    snapshots: deque = field(default_factory=lambda: deque(maxlen=10000))
    
    def __post_init__(self):
        """Initialize with correct maxlen."""
        if not isinstance(self.snapshots, deque):
            self.snapshots = deque(maxlen=self.max_snapshots)
    
    def add(self, snapshot: PortfolioSnapshot):
        """Add a snapshot to history."""
        self.snapshots.append(snapshot)
        self._cleanup_old()
    
    def _cleanup_old(self):
        """Remove snapshots older than retention period."""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        while self.snapshots and self.snapshots[0].timestamp < cutoff:
            self.snapshots.popleft()
    
    def get_equity_series(self) -> pd.Series:
        """Get equity values as time series."""
        if not self.snapshots:
            return pd.Series(dtype=float)
        
        data = [(s.timestamp, s.equity) for s in self.snapshots]
        timestamps, values = zip(*data)
        return pd.Series(values, index=pd.DatetimeIndex(timestamps), name="Equity")
    
    def get_returns_series(self) -> pd.Series:
        """Get returns as time series."""
        equity = self.get_equity_series()
        if len(equity) < 2:
            return pd.Series(dtype=float)
        return equity.pct_change().dropna()
    
    def get_daily_equity(self) -> pd.Series:
        """Get daily equity (end of day values)."""
        equity = self.get_equity_series()
        if equity.empty:
            return equity
        return equity.resample('D').last().dropna()
    
    def get_latest(self, n: int = 1) -> List[PortfolioSnapshot]:
        """Get the n most recent snapshots."""
        return list(self.snapshots)[-n:]
    
    def get_snapshot_at(self, timestamp: datetime) -> Optional[PortfolioSnapshot]:
        """Get snapshot closest to given timestamp."""
        if not self.snapshots:
            return None
        
        # Binary search would be more efficient, but linear is fine for typical sizes
        closest = None
        min_diff = timedelta.max
        
        for snapshot in self.snapshots:
            diff = abs(snapshot.timestamp - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest = snapshot
        
        return closest
    
    def clear(self):
        """Clear all snapshots."""
        self.snapshots.clear()
    
    def __len__(self) -> int:
        return len(self.snapshots)


class DataCollector:
    """Collects data from broker and engine for dashboard.
    
    Provides methods to gather current state and build
    snapshots for dashboard display.
    
    Attributes:
        broker: Broker instance to collect data from
        engine: Optional LiveTradingEngine for additional data
        initial_capital: Starting capital for P&L calculations
        history: SnapshotHistory for historical data
        trade_history: List of recent trades
        _lock: Thread lock for safe access
    """
    
    def __init__(
        self,
        broker: Any,
        engine: Any = None,
        initial_capital: float = 100000.0,
        max_trade_history: int = 1000,
    ):
        """Initialize data collector.
        
        Args:
            broker: Broker instance to collect data from
            engine: Optional LiveTradingEngine
            initial_capital: Starting capital
            max_trade_history: Maximum trades to retain
        """
        self.broker = broker
        self.engine = engine
        self.initial_capital = initial_capital
        self.max_trade_history = max_trade_history
        
        self.history = SnapshotHistory()
        self.trade_history: List[TradeRecord] = []
        self._callbacks: Dict[str, List[Callable]] = {
            "on_snapshot": [],
            "on_trade": [],
        }
        self._lock = threading.RLock()
        self._running = False
        self._collect_thread: Optional[threading.Thread] = None
        
        # Track previous day's equity for daily P&L
        self._day_start_equity: Optional[float] = None
        self._last_day: Optional[datetime] = None
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for events.
        
        Args:
            event: Event name ('on_snapshot' or 'on_trade')
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _fire_callbacks(self, event: str, data: Any):
        """Fire all callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                # Log but don't crash
                pass
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices from broker or engine."""
        prices = {}
        
        # Try engine's data handler first
        if self.engine is not None:
            try:
                if hasattr(self.engine, 'data_handler') and self.engine.data_handler:
                    for symbol in getattr(self.engine, 'symbols', []):
                        price = self.engine.data_handler.get_price(symbol)
                        if price is not None:
                            prices[symbol] = price
            except Exception:
                pass
        
        # Fall back to broker prices
        if not prices and hasattr(self.broker, 'get_price'):
            positions = self.broker.get_positions()
            for pos in positions:
                try:
                    price = self.broker.get_price(pos.symbol)
                    if price is not None:
                        prices[pos.symbol] = price
                except Exception:
                    prices[pos.symbol] = pos.avg_cost  # Fallback
        
        return prices
    
    def collect_snapshot(self) -> PortfolioSnapshot:
        """Collect current portfolio snapshot.
        
        Returns:
            PortfolioSnapshot with current state
        """
        with self._lock:
            timestamp = datetime.now()
            
            # Get account info
            account = self.broker.get_account()
            cash = account.cash
            portfolio_value = account.portfolio_value
            equity = account.equity
            
            # Get prices
            prices = self.get_current_prices()
            
            # Build position snapshots
            positions = []
            broker_positions = self.broker.get_positions()
            
            for pos in broker_positions:
                price = prices.get(pos.symbol, pos.avg_cost)
                pos_snapshot = PositionSnapshot.from_position(pos, price, equity)
                positions.append(pos_snapshot)
            
            # Calculate daily P&L
            today = timestamp.date()
            if self._last_day is None or self._last_day != today:
                self._day_start_equity = equity
                self._last_day = today
            
            daily_pnl = equity - (self._day_start_equity or equity)
            daily_return = (daily_pnl / self._day_start_equity * 100) if self._day_start_equity else 0.0
            
            # Calculate total P&L
            total_pnl = equity - self.initial_capital
            total_return = (total_pnl / self.initial_capital * 100) if self.initial_capital else 0.0
            
            snapshot = PortfolioSnapshot(
                timestamp=timestamp,
                cash=cash,
                portfolio_value=portfolio_value,
                equity=equity,
                positions=positions,
                num_positions=len(positions),
                daily_pnl=daily_pnl,
                daily_return=daily_return,
                total_pnl=total_pnl,
                total_return=total_return,
            )
            
            # Add to history
            self.history.add(snapshot)
            
            # Fire callbacks
            self._fire_callbacks("on_snapshot", snapshot)
            
            return snapshot
    
    def record_trade(self, fill: Any) -> TradeRecord:
        """Record a trade from a fill.
        
        Args:
            fill: Fill object from broker
            
        Returns:
            TradeRecord for the trade
        """
        with self._lock:
            record = TradeRecord.from_fill(fill)
            self.trade_history.append(record)
            
            # Trim to max size
            if len(self.trade_history) > self.max_trade_history:
                self.trade_history = self.trade_history[-self.max_trade_history:]
            
            # Fire callbacks
            self._fire_callbacks("on_trade", record)
            
            return record
    
    def get_recent_trades(self, n: int = 50) -> List[TradeRecord]:
        """Get n most recent trades.
        
        Args:
            n: Number of trades to return
            
        Returns:
            List of TradeRecord
        """
        with self._lock:
            return self.trade_history[-n:]
    
    def get_trades_for_symbol(self, symbol: str) -> List[TradeRecord]:
        """Get all trades for a symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            List of TradeRecord for that symbol
        """
        with self._lock:
            return [t for t in self.trade_history if t.symbol == symbol]
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as a DataFrame."""
        with self._lock:
            if not self.trade_history:
                return pd.DataFrame()
            
            data = [t.to_dict() for t in self.trade_history]
            return pd.DataFrame(data)
    
    def start_collection(self, interval: float = 1.0):
        """Start background collection thread.
        
        Args:
            interval: Collection interval in seconds
        """
        if self._running:
            return
        
        self._running = True
        self._collect_thread = threading.Thread(
            target=self._collection_loop,
            args=(interval,),
            daemon=True,
        )
        self._collect_thread.start()
    
    def stop_collection(self):
        """Stop background collection."""
        self._running = False
        if self._collect_thread:
            self._collect_thread.join(timeout=5.0)
            self._collect_thread = None
    
    def _collection_loop(self, interval: float):
        """Background collection loop."""
        while self._running:
            try:
                self.collect_snapshot()
            except Exception as e:
                # Log but continue
                pass
            time.sleep(interval)
    
    def get_state(self) -> DashboardState:
        """Get current dashboard state.
        
        Returns:
            DashboardState with current data
        """
        with self._lock:
            # Get latest snapshot
            latest = self.history.get_latest(1)
            current_snapshot = latest[0] if latest else None
            
            # Build equity history for state
            equity_history = [
                (s.timestamp, s.equity) for s in self.history.snapshots
            ]
            
            state = DashboardState(
                is_connected=True,
                last_update=datetime.now(),
                initial_capital=self.initial_capital,
                current_snapshot=current_snapshot,
                trade_history=self.trade_history.copy(),
                equity_history=equity_history,
            )
            
            return state
    
    def reset(self):
        """Reset collector state."""
        with self._lock:
            self.history.clear()
            self.trade_history.clear()
            self._day_start_equity = None
            self._last_day = None


class MockDataCollector(DataCollector):
    """Data collector with mock data for testing/demo.
    
    Generates realistic-looking data without requiring
    a real broker connection.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        symbols: List[str] = None,
    ):
        """Initialize mock collector.
        
        Args:
            initial_capital: Starting capital
            symbols: List of symbols to simulate
        """
        self.initial_capital = initial_capital
        self.symbols = symbols or ["AAPL", "GOOGL", "MSFT", "AMZN"]
        self.history = SnapshotHistory()
        self.trade_history: List[TradeRecord] = []
        self._lock = threading.RLock()
        self._running = False
        self._collect_thread = None
        self._callbacks: Dict[str, List[Callable]] = {
            "on_snapshot": [],
            "on_trade": [],
        }
        
        # Mock state
        self._equity = initial_capital
        self._cash = initial_capital * 0.3
        self._positions = self._generate_mock_positions()
        self._day_start_equity = initial_capital
        self._last_day = datetime.now().date()
    
    def _generate_mock_positions(self) -> List[PositionSnapshot]:
        """Generate mock positions."""
        import random
        
        positions = []
        remaining_value = self._equity - self._cash
        
        for i, symbol in enumerate(self.symbols):
            if i == len(self.symbols) - 1:
                # Last position gets remaining value
                value = remaining_value
            else:
                value = remaining_value * random.uniform(0.2, 0.4)
                remaining_value -= value
            
            price = random.uniform(100, 500)
            quantity = int(value / price)
            avg_cost = price * random.uniform(0.95, 1.05)
            market_value = quantity * price
            unrealized_pnl = market_value - (quantity * avg_cost)
            
            positions.append(PositionSnapshot(
                symbol=symbol,
                quantity=quantity,
                avg_cost=avg_cost,
                current_price=price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=(unrealized_pnl / (quantity * avg_cost)) * 100,
                weight=(market_value / self._equity) * 100,
            ))
        
        return positions
    
    def collect_snapshot(self) -> PortfolioSnapshot:
        """Collect mock snapshot with random walk."""
        import random
        
        with self._lock:
            timestamp = datetime.now()
            
            # Random walk equity
            change = random.gauss(0, 0.001)  # 0.1% std dev
            self._equity *= (1 + change)
            
            # Update positions with random price changes
            portfolio_value = 0
            for pos in self._positions:
                price_change = random.gauss(0, 0.002)
                pos.current_price *= (1 + price_change)
                pos.market_value = pos.quantity * pos.current_price
                pos.unrealized_pnl = pos.market_value - (pos.quantity * pos.avg_cost)
                pos.unrealized_pnl_pct = (pos.unrealized_pnl / (pos.quantity * pos.avg_cost)) * 100
                pos.weight = (pos.market_value / self._equity) * 100
                portfolio_value += pos.market_value
            
            self._cash = self._equity - portfolio_value
            
            # Daily P&L
            today = timestamp.date()
            if self._last_day != today:
                self._day_start_equity = self._equity
                self._last_day = today
            
            daily_pnl = self._equity - self._day_start_equity
            daily_return = (daily_pnl / self._day_start_equity) * 100
            total_pnl = self._equity - self.initial_capital
            total_return = (total_pnl / self.initial_capital) * 100
            
            snapshot = PortfolioSnapshot(
                timestamp=timestamp,
                cash=self._cash,
                portfolio_value=portfolio_value,
                equity=self._equity,
                positions=self._positions.copy(),
                num_positions=len(self._positions),
                daily_pnl=daily_pnl,
                daily_return=daily_return,
                total_pnl=total_pnl,
                total_return=total_return,
            )
            
            self.history.add(snapshot)
            self._fire_callbacks("on_snapshot", snapshot)
            
            return snapshot

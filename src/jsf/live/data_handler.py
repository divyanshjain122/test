"""Real-time data handling for live trading.

This module provides handlers for managing real-time price data,
including polling-based and streaming approaches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import time
import threading
from queue import Queue, Empty
import logging

import pandas as pd
import numpy as np

from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class DataHandlerError(Exception):
    """Exception raised for data handler errors."""
    
    def __init__(self, message: str, symbol: Optional[str] = None):
        super().__init__(message)
        self.symbol = symbol


class DataEventType(Enum):
    """Types of data events."""
    PRICE_UPDATE = "price_update"
    BAR_COMPLETE = "bar_complete"
    QUOTE_UPDATE = "quote_update"
    ERROR = "error"


@dataclass
class PriceUpdate:
    """Real-time price update."""
    
    symbol: str
    price: float
    timestamp: datetime
    volume: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price from bid/ask."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "volume": self.volume,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
        }


@dataclass 
class BarData:
    """OHLCV bar data."""
    
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class DataEvent:
    """Data event for the event queue."""
    
    event_type: DataEventType
    symbol: str
    timestamp: datetime
    data: Any
    
    
class DataHandler(ABC):
    """
    Abstract base class for real-time data handlers.
    
    Data handlers are responsible for:
    - Managing subscriptions to price data
    - Providing current prices for symbols
    - Maintaining price history
    - Emitting events on price updates
    
    Example usage:
        ```python
        handler = PollingDataHandler(poll_interval=1.0)
        handler.subscribe(["AAPL", "GOOGL", "MSFT"])
        handler.start()
        
        # Get current prices
        price = handler.get_price("AAPL")
        
        # Get price history
        history = handler.get_history("AAPL", periods=100)
        
        handler.stop()
        ```
    """
    
    def __init__(self, name: str = "data_handler"):
        """
        Initialize data handler.
        
        Args:
            name: Handler name for identification
        """
        self.name = name
        self._subscriptions: Set[str] = set()
        self._prices: Dict[str, float] = {}
        self._last_update: Dict[str, datetime] = {}
        self._history: Dict[str, List[PriceUpdate]] = {}
        self._callbacks: Dict[str, List[Callable]] = {
            "on_price_update": [],
            "on_bar_complete": [],
            "on_error": [],
        }
        self._running = False
        self._lock = threading.Lock()
        logger.info(f"Initialized data handler: {name}")
    
    # ==========================================================================
    # Subscription Management
    # ==========================================================================
    
    def subscribe(self, symbols: List[str]) -> None:
        """
        Subscribe to price updates for symbols.
        
        Args:
            symbols: List of symbols to subscribe to
        """
        with self._lock:
            for symbol in symbols:
                symbol = symbol.upper()
                if symbol not in self._subscriptions:
                    self._subscriptions.add(symbol)
                    self._history[symbol] = []
                    logger.debug(f"Subscribed to {symbol}")
            logger.info(f"Subscribed to {len(symbols)} symbols")
    
    def unsubscribe(self, symbols: List[str]) -> None:
        """
        Unsubscribe from price updates for symbols.
        
        Args:
            symbols: List of symbols to unsubscribe from
        """
        with self._lock:
            for symbol in symbols:
                symbol = symbol.upper()
                self._subscriptions.discard(symbol)
                logger.debug(f"Unsubscribed from {symbol}")
    
    def get_subscriptions(self) -> List[str]:
        """Get list of subscribed symbols."""
        with self._lock:
            return list(self._subscriptions)
    
    # ==========================================================================
    # Price Access
    # ==========================================================================
    
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Current price or None if not available
        """
        with self._lock:
            return self._prices.get(symbol.upper())
    
    def get_prices(self, symbols: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get current prices for multiple symbols.
        
        Args:
            symbols: Symbols to get prices for (None = all subscribed)
            
        Returns:
            Dictionary of symbol -> price
        """
        with self._lock:
            if symbols is None:
                return dict(self._prices)
            return {
                s.upper(): self._prices.get(s.upper())
                for s in symbols
                if s.upper() in self._prices
            }
    
    def get_last_update(self, symbol: str) -> Optional[datetime]:
        """
        Get timestamp of last price update for a symbol.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            Timestamp of last update or None
        """
        with self._lock:
            return self._last_update.get(symbol.upper())
    
    def get_history(
        self,
        symbol: str,
        periods: Optional[int] = None,
        since: Optional[datetime] = None,
    ) -> List[PriceUpdate]:
        """
        Get price history for a symbol.
        
        Args:
            symbol: Symbol to get history for
            periods: Number of most recent updates (None = all)
            since: Get updates since this timestamp
            
        Returns:
            List of price updates
        """
        with self._lock:
            history = self._history.get(symbol.upper(), [])
            
            if since is not None:
                history = [u for u in history if u.timestamp >= since]
            
            if periods is not None:
                history = history[-periods:]
            
            return list(history)
    
    def get_history_df(
        self,
        symbol: str,
        periods: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get price history as a DataFrame.
        
        Args:
            symbol: Symbol to get history for
            periods: Number of most recent updates
            
        Returns:
            DataFrame with price history
        """
        history = self.get_history(symbol, periods=periods)
        if not history:
            return pd.DataFrame()
        
        data = [u.to_dict() for u in history]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df
    
    # ==========================================================================
    # Price Updates
    # ==========================================================================
    
    def update_price(self, update: PriceUpdate) -> None:
        """
        Process a price update.
        
        Args:
            update: Price update to process
        """
        symbol = update.symbol.upper()
        
        with self._lock:
            self._prices[symbol] = update.price
            self._last_update[symbol] = update.timestamp
            
            # Add to history (with size limit)
            if symbol in self._history:
                self._history[symbol].append(update)
                # Keep last 10000 updates per symbol
                if len(self._history[symbol]) > 10000:
                    self._history[symbol] = self._history[symbol][-5000:]
        
        # Notify callbacks
        self._emit("on_price_update", update)
    
    def set_price(self, symbol: str, price: float) -> None:
        """
        Manually set a price (useful for paper trading).
        
        Args:
            symbol: Symbol to set price for
            price: Price to set
        """
        update = PriceUpdate(
            symbol=symbol.upper(),
            price=price,
            timestamp=datetime.now(),
        )
        self.update_price(update)
    
    def set_prices(self, prices: Dict[str, float]) -> None:
        """
        Set multiple prices at once.
        
        Args:
            prices: Dictionary of symbol -> price
        """
        for symbol, price in prices.items():
            self.set_price(symbol, price)
    
    # ==========================================================================
    # Lifecycle Management
    # ==========================================================================
    
    @abstractmethod
    def start(self) -> None:
        """Start the data handler."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the data handler."""
        pass
    
    @property
    def is_running(self) -> bool:
        """Check if handler is running."""
        return self._running
    
    # ==========================================================================
    # Event Callbacks
    # ==========================================================================
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for an event.
        
        Args:
            event: Event name (on_price_update, on_bar_complete, on_error)
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
            logger.debug(f"Registered callback for {event}")
        else:
            raise ValueError(f"Unknown event: {event}")
    
    def unregister_callback(self, event: str, callback: Callable) -> None:
        """
        Unregister a callback.
        
        Args:
            event: Event name
            callback: Callback function to remove
        """
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
    
    def _emit(self, event: str, data: Any) -> None:
        """Emit an event to all registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")


class PollingDataHandler(DataHandler):
    """
    Data handler that polls for price updates at a fixed interval.
    
    This handler is useful for paper trading or when streaming
    data is not available. It requires a price provider function.
    
    Example:
        ```python
        def get_prices(symbols):
            # Return dict of symbol -> price
            return {"AAPL": 150.0, "GOOGL": 2800.0}
        
        handler = PollingDataHandler(
            price_provider=get_prices,
            poll_interval=1.0
        )
        handler.subscribe(["AAPL", "GOOGL"])
        handler.start()
        ```
    """
    
    def __init__(
        self,
        price_provider: Optional[Callable[[List[str]], Dict[str, float]]] = None,
        poll_interval: float = 1.0,
        name: str = "polling_handler",
    ):
        """
        Initialize polling data handler.
        
        Args:
            price_provider: Function that returns prices for symbols
            poll_interval: Seconds between polls
            name: Handler name
        """
        super().__init__(name=name)
        self._price_provider = price_provider
        self._poll_interval = poll_interval
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def set_price_provider(
        self,
        provider: Callable[[List[str]], Dict[str, float]]
    ) -> None:
        """
        Set the price provider function.
        
        Args:
            provider: Function that returns prices for symbols
        """
        self._price_provider = provider
    
    def start(self) -> None:
        """Start polling for price updates."""
        if self._running:
            logger.warning("Handler already running")
            return
        
        self._stop_event.clear()
        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name=f"{self.name}_poll_thread"
        )
        self._poll_thread.start()
        logger.info(f"Started polling handler (interval={self._poll_interval}s)")
    
    def stop(self) -> None:
        """Stop polling."""
        if not self._running:
            return
        
        self._stop_event.set()
        self._running = False
        
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=5.0)
        
        logger.info("Stopped polling handler")
    
    def _poll_loop(self) -> None:
        """Main polling loop."""
        while not self._stop_event.is_set():
            try:
                self._fetch_prices()
            except Exception as e:
                logger.error(f"Error fetching prices: {e}")
                self._emit("on_error", e)
            
            self._stop_event.wait(self._poll_interval)
    
    def _fetch_prices(self) -> None:
        """Fetch prices from provider."""
        if not self._price_provider:
            return
        
        symbols = self.get_subscriptions()
        if not symbols:
            return
        
        try:
            prices = self._price_provider(symbols)
            timestamp = datetime.now()
            
            for symbol, price in prices.items():
                if price is not None:
                    update = PriceUpdate(
                        symbol=symbol.upper(),
                        price=float(price),
                        timestamp=timestamp,
                    )
                    self.update_price(update)
                    
        except Exception as e:
            raise DataHandlerError(f"Failed to fetch prices: {e}")


class RealtimeDataHandler(DataHandler):
    """
    Data handler for real-time streaming data.
    
    This handler manages an event queue for processing
    streaming price updates asynchronously.
    
    Example:
        ```python
        handler = RealtimeDataHandler()
        handler.subscribe(["AAPL", "GOOGL"])
        handler.start()
        
        # Push updates from streaming source
        handler.push_update(PriceUpdate(
            symbol="AAPL",
            price=150.25,
            timestamp=datetime.now()
        ))
        ```
    """
    
    def __init__(
        self,
        queue_size: int = 10000,
        name: str = "realtime_handler",
    ):
        """
        Initialize realtime data handler.
        
        Args:
            queue_size: Maximum size of event queue
            name: Handler name
        """
        super().__init__(name=name)
        self._queue: Queue = Queue(maxsize=queue_size)
        self._process_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def push_update(self, update: PriceUpdate) -> bool:
        """
        Push a price update to the queue.
        
        Args:
            update: Price update to queue
            
        Returns:
            True if queued successfully, False if queue is full
        """
        try:
            self._queue.put_nowait(update)
            return True
        except:
            logger.warning(f"Queue full, dropping update for {update.symbol}")
            return False
    
    def push_bar(self, bar: BarData) -> bool:
        """
        Push a bar update to the queue.
        
        Args:
            bar: Bar data to queue
            
        Returns:
            True if queued successfully
        """
        # Convert bar to price update (use close price)
        update = PriceUpdate(
            symbol=bar.symbol,
            price=bar.close,
            timestamp=bar.timestamp,
            volume=bar.volume,
        )
        return self.push_update(update)
    
    def start(self) -> None:
        """Start processing updates."""
        if self._running:
            logger.warning("Handler already running")
            return
        
        self._stop_event.clear()
        self._running = True
        self._process_thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name=f"{self.name}_process_thread"
        )
        self._process_thread.start()
        logger.info("Started realtime handler")
    
    def stop(self) -> None:
        """Stop processing updates."""
        if not self._running:
            return
        
        self._stop_event.set()
        self._running = False
        
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=5.0)
        
        logger.info("Stopped realtime handler")
    
    def _process_loop(self) -> None:
        """Main processing loop."""
        while not self._stop_event.is_set():
            try:
                update = self._queue.get(timeout=0.1)
                self.update_price(update)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing update: {e}")
                self._emit("on_error", e)
    
    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()


class SimulatedDataHandler(DataHandler):
    """
    Data handler that simulates price movements for testing.
    
    Generates random price updates based on initial prices
    and configurable volatility.
    
    Example:
        ```python
        handler = SimulatedDataHandler(
            initial_prices={"AAPL": 150.0, "GOOGL": 2800.0},
            volatility=0.02,
            update_interval=0.5
        )
        handler.start()
        ```
    """
    
    def __init__(
        self,
        initial_prices: Optional[Dict[str, float]] = None,
        volatility: float = 0.01,
        update_interval: float = 1.0,
        seed: Optional[int] = None,
        name: str = "simulated_handler",
    ):
        """
        Initialize simulated data handler.
        
        Args:
            initial_prices: Starting prices for symbols
            volatility: Price volatility (std dev of returns)
            update_interval: Seconds between updates
            seed: Random seed for reproducibility
            name: Handler name
        """
        super().__init__(name=name)
        self._initial_prices = initial_prices or {}
        self._volatility = volatility
        self._update_interval = update_interval
        self._rng = np.random.default_rng(seed)
        self._sim_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Initialize prices
        for symbol, price in self._initial_prices.items():
            self._prices[symbol.upper()] = price
            self._subscriptions.add(symbol.upper())
    
    def start(self) -> None:
        """Start simulating price updates."""
        if self._running:
            return
        
        self._stop_event.clear()
        self._running = True
        self._sim_thread = threading.Thread(
            target=self._simulate_loop,
            daemon=True,
            name=f"{self.name}_sim_thread"
        )
        self._sim_thread.start()
        logger.info(f"Started simulated handler (volatility={self._volatility})")
    
    def stop(self) -> None:
        """Stop simulation."""
        if not self._running:
            return
        
        self._stop_event.set()
        self._running = False
        
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=5.0)
        
        logger.info("Stopped simulated handler")
    
    def _simulate_loop(self) -> None:
        """Main simulation loop."""
        while not self._stop_event.is_set():
            try:
                self._generate_updates()
            except Exception as e:
                logger.error(f"Error generating updates: {e}")
            
            self._stop_event.wait(self._update_interval)
    
    def _generate_updates(self) -> None:
        """Generate simulated price updates."""
        timestamp = datetime.now()
        
        with self._lock:
            for symbol in list(self._subscriptions):
                current_price = self._prices.get(symbol)
                if current_price is None:
                    continue
                
                # Random walk with drift
                return_pct = self._rng.normal(0, self._volatility)
                new_price = current_price * (1 + return_pct)
                new_price = max(0.01, new_price)  # Floor at 1 cent
                
                update = PriceUpdate(
                    symbol=symbol,
                    price=round(new_price, 2),
                    timestamp=timestamp,
                )
                
                # Update without lock (already held)
                self._prices[symbol] = update.price
                self._last_update[symbol] = update.timestamp
                if symbol in self._history:
                    self._history[symbol].append(update)
        
        # Emit callbacks outside lock
        for symbol in self._subscriptions:
            if symbol in self._last_update:
                update = PriceUpdate(
                    symbol=symbol,
                    price=self._prices[symbol],
                    timestamp=self._last_update[symbol],
                )
                self._emit("on_price_update", update)

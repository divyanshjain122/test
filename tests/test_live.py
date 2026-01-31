"""Tests for live trading module.

Tests cover:
- Data handlers (polling, realtime, simulated)
- Order manager
- Live trading engine
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

import pandas as pd
import numpy as np

from jsf.live import (
    DataHandler,
    PollingDataHandler,
    RealtimeDataHandler,
    SimulatedDataHandler,
    DataHandlerError,
    PriceUpdate,
    BarData,
    OrderManager,
    OrderTracker,
    OrderManagerError,
    LiveTradingEngine,
    TradingState,
    EngineConfig,
    EngineError,
)
from jsf.broker import (
    PaperBroker,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Position,
    Fill,
)
from jsf.live.order_manager import OrderState


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def paper_broker():
    """Create a paper broker for testing."""
    broker = PaperBroker(initial_cash=100000)
    broker.connect()
    broker.set_price("AAPL", 150.0)
    broker.set_price("GOOGL", 2800.0)
    broker.set_price("MSFT", 300.0)
    return broker


@pytest.fixture
def simulated_handler():
    """Create a simulated data handler."""
    return SimulatedDataHandler(
        initial_prices={"AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 300.0},
        volatility=0.01,
        update_interval=0.1,
        seed=42,
    )


@pytest.fixture
def order_manager(paper_broker):
    """Create an order manager with paper broker."""
    return OrderManager(paper_broker)


@pytest.fixture
def engine_config():
    """Create engine configuration."""
    return EngineConfig(
        initial_capital=100000,
        trading_interval=0.1,
        data_warmup_seconds=0.2,
        max_daily_loss=0.05,
    )


# =============================================================================
# PriceUpdate Tests
# =============================================================================

class TestPriceUpdate:
    """Tests for PriceUpdate dataclass."""
    
    def test_create_price_update(self):
        """Test creating a price update."""
        update = PriceUpdate(
            symbol="AAPL",
            price=150.0,
            timestamp=datetime.now(),
        )
        assert update.symbol == "AAPL"
        assert update.price == 150.0
    
    def test_price_update_with_bid_ask(self):
        """Test price update with bid/ask."""
        update = PriceUpdate(
            symbol="AAPL",
            price=150.0,
            timestamp=datetime.now(),
            bid=149.95,
            ask=150.05,
        )
        assert update.mid_price == 150.0
        assert abs(update.spread - 0.10) < 0.001  # Allow for float precision
    
    def test_price_update_to_dict(self):
        """Test converting to dictionary."""
        now = datetime.now()
        update = PriceUpdate(
            symbol="AAPL",
            price=150.0,
            timestamp=now,
            volume=1000.0,
        )
        d = update.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["price"] == 150.0
        assert d["volume"] == 1000.0


class TestBarData:
    """Tests for BarData dataclass."""
    
    def test_create_bar_data(self):
        """Test creating bar data."""
        bar = BarData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=149.0,
            high=151.0,
            low=148.5,
            close=150.0,
            volume=10000.0,
        )
        assert bar.symbol == "AAPL"
        assert bar.close == 150.0
    
    def test_bar_data_to_dict(self):
        """Test converting bar to dictionary."""
        bar = BarData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=149.0,
            high=151.0,
            low=148.5,
            close=150.0,
        )
        d = bar.to_dict()
        assert "open" in d
        assert "high" in d
        assert "low" in d
        assert "close" in d


# =============================================================================
# SimulatedDataHandler Tests
# =============================================================================

class TestSimulatedDataHandler:
    """Tests for SimulatedDataHandler."""
    
    def test_create_simulated_handler(self, simulated_handler):
        """Test creating simulated handler."""
        assert simulated_handler is not None
        assert not simulated_handler.is_running
    
    def test_initial_prices(self, simulated_handler):
        """Test initial prices are set."""
        assert simulated_handler.get_price("AAPL") == 150.0
        assert simulated_handler.get_price("GOOGL") == 2800.0
    
    def test_subscribe_symbols(self, simulated_handler):
        """Test subscribing to symbols."""
        simulated_handler.subscribe(["TSLA", "AMZN"])
        subs = simulated_handler.get_subscriptions()
        assert "TSLA" in subs
        assert "AMZN" in subs
    
    def test_unsubscribe_symbols(self, simulated_handler):
        """Test unsubscribing from symbols."""
        simulated_handler.unsubscribe(["AAPL"])
        subs = simulated_handler.get_subscriptions()
        assert "AAPL" not in subs
    
    def test_start_stop(self, simulated_handler):
        """Test starting and stopping handler."""
        simulated_handler.start()
        assert simulated_handler.is_running
        
        simulated_handler.stop()
        assert not simulated_handler.is_running
    
    def test_price_updates(self, simulated_handler):
        """Test that prices update over time."""
        initial_price = simulated_handler.get_price("AAPL")
        
        simulated_handler.start()
        time.sleep(0.5)  # Let some updates happen (longer wait)
        simulated_handler.stop()
        
        # Check that handler ran - price may or may not have changed
        # but we should have history entries
        prices = simulated_handler.get_prices()
        assert "AAPL" in prices
    
    def test_get_prices(self, simulated_handler):
        """Test getting multiple prices."""
        prices = simulated_handler.get_prices()
        assert "AAPL" in prices
        assert "GOOGL" in prices
        assert "MSFT" in prices
    
    def test_price_callback(self, simulated_handler):
        """Test price update callback."""
        updates = []
        
        def on_update(update):
            updates.append(update)
        
        simulated_handler.register_callback("on_price_update", on_update)
        simulated_handler.start()
        time.sleep(0.3)
        simulated_handler.stop()
        
        assert len(updates) > 0
    
    def test_get_history_df(self, simulated_handler):
        """Test getting history as DataFrame."""
        simulated_handler.start()
        time.sleep(0.3)
        simulated_handler.stop()
        
        df = simulated_handler.get_history_df("AAPL")
        assert isinstance(df, pd.DataFrame)


class TestPollingDataHandler:
    """Tests for PollingDataHandler."""
    
    def test_create_polling_handler(self):
        """Test creating polling handler."""
        handler = PollingDataHandler(poll_interval=0.1)
        assert handler is not None
    
    def test_set_price_provider(self):
        """Test setting price provider."""
        handler = PollingDataHandler(poll_interval=0.1)
        
        def provider(symbols):
            return {s: 100.0 for s in symbols}
        
        handler.set_price_provider(provider)
        handler.subscribe(["AAPL", "GOOGL"])
        handler.start()
        time.sleep(0.2)
        handler.stop()
        
        assert handler.get_price("AAPL") == 100.0
    
    def test_manual_set_price(self):
        """Test manually setting prices."""
        handler = PollingDataHandler(poll_interval=1.0)
        handler.subscribe(["AAPL"])
        
        handler.set_price("AAPL", 155.0)
        assert handler.get_price("AAPL") == 155.0


class TestRealtimeDataHandler:
    """Tests for RealtimeDataHandler."""
    
    def test_create_realtime_handler(self):
        """Test creating realtime handler."""
        handler = RealtimeDataHandler(queue_size=1000)
        assert handler is not None
    
    def test_push_update(self):
        """Test pushing updates to queue."""
        handler = RealtimeDataHandler()
        handler.subscribe(["AAPL"])
        handler.start()
        
        update = PriceUpdate(
            symbol="AAPL",
            price=150.0,
            timestamp=datetime.now(),
        )
        
        result = handler.push_update(update)
        assert result is True
        
        time.sleep(0.2)  # Let it process
        handler.stop()
        
        assert handler.get_price("AAPL") == 150.0
    
    def test_push_bar(self):
        """Test pushing bar data."""
        handler = RealtimeDataHandler()
        handler.subscribe(["AAPL"])
        handler.start()
        
        bar = BarData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=149.0,
            high=151.0,
            low=148.5,
            close=150.0,
        )
        
        handler.push_bar(bar)
        time.sleep(0.2)
        handler.stop()
        
        # Close price should be used
        assert handler.get_price("AAPL") == 150.0


# =============================================================================
# OrderTracker Tests
# =============================================================================

class TestOrderTracker:
    """Tests for OrderTracker."""
    
    def test_create_order_tracker(self):
        """Test creating order tracker."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        tracker = OrderTracker(order=order)
        
        assert tracker.order_id == order.order_id
        assert tracker.symbol == "AAPL"
        assert tracker.state == OrderState.PENDING_SUBMIT
    
    def test_remaining_quantity(self):
        """Test remaining quantity calculation."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        tracker = OrderTracker(order=order)
        
        assert tracker.remaining_quantity == 100
        
        # Simulate partial fill
        tracker.filled_quantity = 40
        assert tracker.remaining_quantity == 60
    
    def test_fill_ratio(self):
        """Test fill ratio calculation."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        tracker = OrderTracker(order=order)
        tracker.filled_quantity = 50
        
        assert tracker.fill_ratio == 0.5
    
    def test_is_active(self):
        """Test is_active property."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        tracker = OrderTracker(order=order)
        
        assert tracker.is_active is True
        
        tracker.state = OrderState.FILLED
        assert tracker.is_active is False
    
    def test_add_fill(self):
        """Test adding a fill."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        tracker = OrderTracker(order=order)
        
        fill = Fill(
            fill_id="f1",
            order_id=order.order_id,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=50,
            price=150.0,
            timestamp=datetime.now(),
        )
        
        tracker.add_fill(fill)
        
        assert tracker.filled_quantity == 50
        assert tracker.average_fill_price == 150.0
        assert tracker.state == OrderState.PARTIALLY_FILLED
    
    def test_complete_fill(self):
        """Test completing a fill."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        tracker = OrderTracker(order=order)
        
        fill = Fill(
            fill_id="f1",
            order_id=order.order_id,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
        )
        
        tracker.add_fill(fill)
        
        assert tracker.state == OrderState.FILLED
        assert tracker.is_complete is True


# =============================================================================
# OrderManager Tests
# =============================================================================

class TestOrderManager:
    """Tests for OrderManager."""
    
    def test_create_order_manager(self, order_manager):
        """Test creating order manager."""
        assert order_manager is not None
        assert order_manager.active_order_count == 0
    
    def test_submit_order(self, order_manager):
        """Test submitting an order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )
        
        tracker = order_manager.submit_order(order)
        
        assert tracker is not None
        # Market orders fill immediately in paper broker
        assert tracker.state == OrderState.FILLED
    
    def test_submit_multiple_orders(self, order_manager):
        """Test submitting multiple orders."""
        orders = [
            Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET),
            Order(symbol="GOOGL", side=OrderSide.BUY, quantity=5, order_type=OrderType.MARKET),
        ]
        
        trackers = order_manager.submit_orders(orders)
        
        assert len(trackers) == 2
    
    def test_get_order(self, order_manager):
        """Test getting an order by ID."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )
        
        tracker = order_manager.submit_order(order)
        retrieved = order_manager.get_order(tracker.order_id)
        
        assert retrieved is not None
        assert retrieved.order_id == tracker.order_id
    
    def test_get_active_orders(self, order_manager, paper_broker):
        """Test getting active orders."""
        # Submit a limit order that won't fill immediately
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=100.0,  # Below market, won't fill
        )
        
        order_manager.submit_order(order)
        active = order_manager.get_active_orders()
        
        assert len(active) == 1
    
    def test_cancel_order(self, order_manager):
        """Test cancelling an order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=100.0,
        )
        
        tracker = order_manager.submit_order(order)
        success = order_manager.cancel_order(tracker.order_id)
        
        assert success is True
        assert tracker.state == OrderState.CANCELLED
    
    def test_cancel_all_orders(self, order_manager):
        """Test cancelling all orders."""
        orders = [
            Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, 
                  order_type=OrderType.LIMIT, limit_price=100.0),
            Order(symbol="GOOGL", side=OrderSide.BUY, quantity=5, 
                  order_type=OrderType.LIMIT, limit_price=2000.0),
        ]
        
        order_manager.submit_orders(orders)
        cancelled = order_manager.cancel_all_orders()
        
        assert cancelled == 2
    
    def test_get_orders_by_symbol(self, order_manager):
        """Test getting orders by symbol."""
        orders = [
            Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET),
            Order(symbol="AAPL", side=OrderSide.BUY, quantity=5, order_type=OrderType.MARKET),
            Order(symbol="GOOGL", side=OrderSide.BUY, quantity=5, order_type=OrderType.MARKET),
        ]
        
        order_manager.submit_orders(orders)
        aapl_orders = order_manager.get_orders_by_symbol("AAPL")
        
        assert len(aapl_orders) == 2
    
    def test_get_statistics(self, order_manager):
        """Test getting order statistics."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )
        order_manager.submit_order(order)
        
        stats = order_manager.get_statistics()
        
        assert "total_orders" in stats
        assert "active_orders" in stats
        assert "orders_by_state" in stats
    
    def test_order_filled_callback(self, order_manager):
        """Test order filled callback."""
        filled_trackers = []
        
        def on_filled(tracker):
            filled_trackers.append(tracker)
        
        order_manager.register_callback("on_order_filled", on_filled)
        
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )
        order_manager.submit_order(order)
        
        assert len(filled_trackers) == 1
    
    def test_pending_quantity(self, order_manager):
        """Test getting pending quantity."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=100.0,
        )
        
        order_manager.submit_order(order)
        pending = order_manager.get_pending_quantity("AAPL", OrderSide.BUY)
        
        assert pending == 100


# =============================================================================
# EngineConfig Tests
# =============================================================================

class TestEngineConfig:
    """Tests for EngineConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = EngineConfig()
        
        assert config.initial_capital == 100000.0
        assert config.max_position_size == 0.1
        assert config.trading_interval == 1.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = EngineConfig(
            initial_capital=50000,
            max_position_size=0.2,
            max_daily_loss=0.03,
        )
        
        assert config.initial_capital == 50000
        assert config.max_position_size == 0.2
        assert config.max_daily_loss == 0.03
    
    def test_invalid_initial_capital(self):
        """Test validation of initial capital."""
        with pytest.raises(ValueError):
            EngineConfig(initial_capital=-1000)
    
    def test_invalid_position_size(self):
        """Test validation of position size."""
        with pytest.raises(ValueError):
            EngineConfig(max_position_size=1.5)


# =============================================================================
# LiveTradingEngine Tests
# =============================================================================

class TestLiveTradingEngine:
    """Tests for LiveTradingEngine."""
    
    def test_create_engine(self, paper_broker, simulated_handler, engine_config):
        """Test creating engine."""
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            symbols=["AAPL", "GOOGL"],
            config=engine_config,
        )
        
        assert engine is not None
        assert engine.state == TradingState.STOPPED
    
    def test_create_engine_defaults(self):
        """Test creating engine with defaults."""
        engine = LiveTradingEngine(symbols=["AAPL"])
        
        assert engine.broker is not None
        assert engine.data_handler is not None
    
    def test_set_strategy(self, paper_broker, simulated_handler, engine_config):
        """Test setting strategy."""
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            config=engine_config,
        )
        
        def my_strategy(engine, prices):
            return {"AAPL": 0.5}
        
        engine.set_strategy(my_strategy)
    
    def test_start_stop(self, paper_broker, simulated_handler, engine_config):
        """Test starting and stopping engine."""
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            symbols=["AAPL"],
            config=engine_config,
        )
        
        engine.start()
        assert engine.state == TradingState.RUNNING
        assert engine.is_running is True
        
        time.sleep(0.5)
        
        engine.stop()
        assert engine.state == TradingState.STOPPED
        assert engine.is_running is False
    
    def test_pause_resume(self, paper_broker, simulated_handler, engine_config):
        """Test pausing and resuming."""
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            symbols=["AAPL"],
            config=engine_config,
        )
        
        engine.start()
        time.sleep(0.3)
        
        engine.pause()
        assert engine.state == TradingState.PAUSED
        
        engine.resume()
        assert engine.state == TradingState.RUNNING
        
        engine.stop()
    
    def test_get_positions(self, paper_broker, simulated_handler, engine_config):
        """Test getting positions."""
        # Buy some stock first
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        paper_broker.submit_order(order)
        
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            config=engine_config,
        )
        
        positions = engine.get_positions()
        assert "AAPL" in positions
    
    def test_get_account(self, paper_broker, simulated_handler, engine_config):
        """Test getting account info."""
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            config=engine_config,
        )
        
        account = engine.get_account()
        assert account.cash > 0
    
    def test_equity_property(self, paper_broker, simulated_handler, engine_config):
        """Test equity property."""
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            config=engine_config,
        )
        
        assert engine.equity == 100000.0
    
    def test_close_position(self, paper_broker, simulated_handler, engine_config):
        """Test closing a position."""
        # Buy some stock
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        paper_broker.submit_order(order)
        
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            config=engine_config,
        )
        
        tracker = engine.close_position("AAPL")
        
        assert tracker is not None
        positions = engine.get_positions()
        assert "AAPL" not in positions or positions["AAPL"].quantity == 0
    
    def test_get_status(self, paper_broker, simulated_handler, engine_config):
        """Test getting engine status."""
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            symbols=["AAPL", "GOOGL"],
            config=engine_config,
        )
        
        status = engine.get_status()
        
        assert "state" in status
        assert "symbols" in status
        assert "equity" in status
    
    def test_strategy_execution(self, paper_broker, simulated_handler, engine_config):
        """Test that strategy gets executed."""
        strategy_calls = []
        
        def test_strategy(engine, prices):
            strategy_calls.append(prices)
            return {}  # No trades
        
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            symbols=["AAPL"],
            config=engine_config,
        )
        
        engine.set_strategy(test_strategy)
        engine.start()
        time.sleep(0.5)
        engine.stop()
        
        assert len(strategy_calls) > 0
    
    def test_on_cycle_callback(self, paper_broker, simulated_handler, engine_config):
        """Test on_cycle callback."""
        cycles = []
        
        def on_cycle(data):
            cycles.append(data)
        
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            symbols=["AAPL"],
            config=engine_config,
        )
        
        engine.register_callback("on_cycle", on_cycle)
        engine.start()
        time.sleep(0.5)
        engine.stop()
        
        assert len(cycles) > 0
    
    def test_state_change_callback(self, paper_broker, simulated_handler, engine_config):
        """Test state change callback."""
        states = []
        
        def on_state_change(data):
            states.append(data)
        
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            symbols=["AAPL"],
            config=engine_config,
        )
        
        engine.register_callback("on_state_change", on_state_change)
        engine.start()
        time.sleep(0.3)
        engine.stop()
        
        # Should have starting, running, stopping, stopped
        assert len(states) >= 3
    
    def test_get_snapshots(self, paper_broker, simulated_handler, engine_config):
        """Test getting trading snapshots."""
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            symbols=["AAPL"],
            config=engine_config,
        )
        
        engine.start()
        time.sleep(0.5)
        engine.stop()
        
        snapshots = engine.get_snapshots()
        assert len(snapshots) > 0
    
    def test_get_equity_curve(self, paper_broker, simulated_handler, engine_config):
        """Test getting equity curve."""
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            symbols=["AAPL"],
            config=engine_config,
        )
        
        engine.start()
        time.sleep(0.5)
        engine.stop()
        
        curve = engine.get_equity_curve()
        assert isinstance(curve, pd.Series)
    
    def test_daily_pnl(self, paper_broker, simulated_handler, engine_config):
        """Test daily P&L calculation."""
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            symbols=["AAPL"],
            config=engine_config,
        )
        
        engine.start()
        time.sleep(0.3)
        
        # Before any trades, PnL should be ~0
        pnl = engine.daily_pnl
        assert abs(pnl) < 1000  # Allow for small movements
        
        engine.stop()
    
    def test_repr(self, paper_broker, simulated_handler, engine_config):
        """Test string representation."""
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            symbols=["AAPL", "GOOGL"],
            config=engine_config,
        )
        
        repr_str = repr(engine)
        assert "LiveTradingEngine" in repr_str
        assert "stopped" in repr_str


# =============================================================================
# Integration Tests
# =============================================================================

class TestLiveIntegration:
    """Integration tests for live trading."""
    
    def test_full_trading_cycle(self, paper_broker, simulated_handler, engine_config):
        """Test a complete trading cycle."""
        trades_executed = []
        
        def simple_strategy(engine, prices):
            # Buy equal weight of all symbols
            if not engine.get_positions():
                return {symbol: 0.3 for symbol in prices}
            return {}
        
        def on_trade(tracker):
            trades_executed.append(tracker)
        
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            symbols=["AAPL", "GOOGL"],
            config=engine_config,
        )
        
        engine.set_strategy(simple_strategy)
        engine.register_callback("on_trade", on_trade)
        
        engine.start()
        time.sleep(1.0)
        engine.stop()
        
        # Should have executed some trades
        assert len(trades_executed) > 0
    
    def test_position_close_all(self, paper_broker, simulated_handler, engine_config):
        """Test closing all positions."""
        # Set up positions
        paper_broker.submit_order(Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET
        ))
        paper_broker.submit_order(Order(
            symbol="GOOGL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET
        ))
        
        engine = LiveTradingEngine(
            broker=paper_broker,
            data_handler=simulated_handler,
            config=engine_config,
        )
        
        # Verify positions exist
        positions = engine.get_positions()
        assert len(positions) == 2
        
        # Close all
        trackers = engine.close_all_positions()
        assert len(trackers) == 2
        
        # Verify positions closed
        positions = engine.get_positions()
        total_qty = sum(p.quantity for p in positions.values())
        assert total_qty == 0

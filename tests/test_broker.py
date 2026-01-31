"""Tests for the broker integration module (Phase 14).

This module contains comprehensive tests for:
- Broker data models (Order, Position, Fill, AccountInfo)
- PaperBroker implementation
- Alpaca broker integration (when available)
"""

import pytest
from datetime import datetime, timedelta
from typing import List

from jsf.broker import (
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
    # Base classes
    Broker,
    BrokerError,
    OrderError,
    InsufficientFundsError,
    # Implementations
    PaperBroker,
    AlpacaBroker,
    ALPACA_AVAILABLE,
)


# =============================================================================
# Model Tests
# =============================================================================

class TestOrderModel:
    """Tests for Order data model."""
    
    def test_order_creation_market(self):
        """Test creating a market order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0.0
        assert order.is_active is True
        assert order.is_filled is False
        assert order.remaining_quantity == 100
    
    def test_order_creation_limit(self):
        """Test creating a limit order."""
        order = Order(
            symbol="GOOGL",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=2800.0,
        )
        
        assert order.limit_price == 2800.0
        assert order.order_type == OrderType.LIMIT
    
    def test_order_creation_stop(self):
        """Test creating a stop order."""
        order = Order(
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=25,
            order_type=OrderType.STOP,
            stop_price=350.0,
        )
        
        assert order.stop_price == 350.0
        assert order.order_type == OrderType.STOP
    
    def test_order_validation_quantity(self):
        """Test order validation rejects invalid quantity."""
        with pytest.raises(ValueError, match="quantity must be positive"):
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=0,
            )
    
    def test_order_validation_limit_price(self):
        """Test limit order requires limit price."""
        with pytest.raises(ValueError, match="requires positive limit_price"):
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.LIMIT,
            )
    
    def test_order_validation_stop_price(self):
        """Test stop order requires stop price."""
        with pytest.raises(ValueError, match="requires positive stop_price"):
            Order(
                symbol="AAPL",
                side=OrderSide.SELL,
                quantity=100,
                order_type=OrderType.STOP,
            )
    
    def test_order_to_dict(self):
        """Test order serialization."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        
        data = order.to_dict()
        
        assert data["symbol"] == "AAPL"
        assert data["side"] == "buy"
        assert data["quantity"] == 100
        assert data["order_type"] == "market"
    
    def test_order_from_dict(self):
        """Test order deserialization."""
        data = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
            "order_type": "limit",
            "limit_price": 150.0,
        }
        
        order = Order.from_dict(data)
        
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 150.0


class TestPositionModel:
    """Tests for Position data model."""
    
    def test_position_creation(self):
        """Test creating a position."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_cost=150.0,
            current_price=160.0,
        )
        
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.avg_cost == 150.0
        assert position.current_price == 160.0
        assert position.is_long is True
        assert position.is_short is False
        assert position.side == PositionSide.LONG
    
    def test_position_pnl_calculation(self):
        """Test P&L calculations."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_cost=150.0,
            current_price=160.0,
        )
        
        # Unrealized P&L = (160 - 150) * 100 = 1000
        assert position.unrealized_pnl == 1000.0
        assert position.market_value == 16000.0
        assert position.cost_basis == 15000.0
    
    def test_position_short(self):
        """Test short position."""
        position = Position(
            symbol="TSLA",
            quantity=-50,
            avg_cost=800.0,
            current_price=750.0,
        )
        
        assert position.is_short is True
        assert position.side == PositionSide.SHORT
        # Short P&L = (750 - 800) * (-50) = 2500 profit
        assert position.unrealized_pnl == 2500.0
    
    def test_position_update_price(self):
        """Test updating position price."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_cost=150.0,
        )
        
        position.update_price(170.0)
        
        assert position.current_price == 170.0
        assert position.unrealized_pnl == 2000.0
        assert position.market_value == 17000.0


class TestFillModel:
    """Tests for Fill data model."""
    
    def test_fill_creation(self):
        """Test creating a fill."""
        fill = Fill(
            order_id="ord_123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
            commission=1.0,
        )
        
        assert fill.order_id == "ord_123"
        assert fill.quantity == 100
        assert fill.price == 150.0
        assert fill.notional_value == 15000.0
        assert fill.total_cost == 15001.0  # Including commission
    
    def test_fill_validation(self):
        """Test fill validation."""
        with pytest.raises(ValueError, match="Fill quantity must be positive"):
            Fill(
                order_id="ord_123",
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=0,
                price=150.0,
                timestamp=datetime.now(),
            )


class TestAccountInfoModel:
    """Tests for AccountInfo data model."""
    
    def test_account_creation(self):
        """Test creating account info."""
        account = AccountInfo(
            account_id="test_123",
            cash=50000.0,
            portfolio_value=100000.0,
        )
        
        assert account.account_id == "test_123"
        assert account.cash == 50000.0
        assert account.portfolio_value == 100000.0
        assert account.equity == 100000.0
    
    def test_account_margin(self):
        """Test margin account calculations."""
        account = AccountInfo(
            account_id="margin_123",
            cash=50000.0,
            portfolio_value=100000.0,
            account_type="margin",
        )
        
        # Margin account gets 2x buying power
        assert account.buying_power == 100000.0


# =============================================================================
# Paper Broker Tests
# =============================================================================

class TestPaperBrokerCreation:
    """Tests for PaperBroker initialization."""
    
    def test_broker_creation(self):
        """Test creating a paper broker."""
        broker = PaperBroker(initial_cash=100000)
        
        assert broker.initial_cash == 100000
        assert broker.is_connected is False
    
    def test_broker_with_costs(self):
        """Test broker with transaction costs."""
        broker = PaperBroker(
            initial_cash=100000,
            commission=1.0,
            commission_per_share=0.005,
            slippage=0.001,
        )
        
        assert broker.commission == 1.0
        assert broker.commission_per_share == 0.005
        assert broker.slippage == 0.001
    
    def test_broker_validation(self):
        """Test broker validation."""
        with pytest.raises(ValueError, match="initial_cash must be positive"):
            PaperBroker(initial_cash=0)
        
        with pytest.raises(ValueError, match="commission must be non-negative"):
            PaperBroker(initial_cash=100000, commission=-1)
        
        with pytest.raises(ValueError, match="slippage must be between"):
            PaperBroker(initial_cash=100000, slippage=1.5)


class TestPaperBrokerConnection:
    """Tests for PaperBroker connection management."""
    
    def test_connect_disconnect(self):
        """Test connecting and disconnecting."""
        broker = PaperBroker()
        
        assert broker.is_connected is False
        
        result = broker.connect()
        assert result is True
        assert broker.is_connected is True
        
        result = broker.disconnect()
        assert result is True
        assert broker.is_connected is False
    
    def test_context_manager(self):
        """Test using broker as context manager."""
        with PaperBroker() as broker:
            assert broker.is_connected is True
        
        assert broker.is_connected is False


class TestPaperBrokerPrices:
    """Tests for PaperBroker price management."""
    
    def test_set_get_price(self):
        """Test setting and getting prices."""
        broker = PaperBroker()
        broker.connect()
        
        broker.set_price("AAPL", 150.0)
        broker.set_price("GOOGL", 2800.0)
        
        assert broker.get_price("AAPL") == 150.0
        assert broker.get_price("GOOGL") == 2800.0
        assert broker.get_price("MSFT") is None
    
    def test_set_prices_batch(self):
        """Test setting multiple prices."""
        broker = PaperBroker()
        broker.connect()
        
        broker.set_prices({
            "AAPL": 150.0,
            "GOOGL": 2800.0,
            "MSFT": 350.0,
        })
        
        assert broker.get_price("AAPL") == 150.0
        assert broker.get_price("GOOGL") == 2800.0
        assert broker.get_price("MSFT") == 350.0
    
    def test_invalid_price(self):
        """Test setting invalid price."""
        broker = PaperBroker()
        broker.connect()
        
        with pytest.raises(ValueError, match="Price must be positive"):
            broker.set_price("AAPL", -10.0)


class TestPaperBrokerOrders:
    """Tests for PaperBroker order execution."""
    
    @pytest.fixture
    def broker(self):
        """Create a connected paper broker with prices."""
        broker = PaperBroker(initial_cash=100000)
        broker.connect()
        broker.set_price("AAPL", 150.0)
        broker.set_price("GOOGL", 2800.0)
        return broker
    
    def test_market_buy_order(self, broker):
        """Test submitting a market buy order."""
        result = broker.buy("AAPL", 100)
        
        assert result.success is True
        assert result.order.is_filled is True
        assert result.order.filled_quantity == 100
        assert result.order.avg_fill_price == 150.0  # No slippage by default
    
    def test_market_sell_order(self, broker):
        """Test submitting a market sell order."""
        # First buy
        broker.buy("AAPL", 100)
        
        # Then sell
        result = broker.sell("AAPL", 50)
        
        assert result.success is True
        assert result.order.is_filled is True
        assert result.order.filled_quantity == 50
    
    def test_order_insufficient_funds(self, broker):
        """Test order rejection due to insufficient funds."""
        # Try to buy more than we can afford
        result = broker.buy("GOOGL", 100)  # Would cost $280,000
        
        assert result.success is False
        assert result.error_code == "INSUFFICIENT_FUNDS"
    
    def test_order_insufficient_shares(self, broker):
        """Test sell order rejection due to insufficient shares."""
        # Try to sell without owning
        result = broker.sell("AAPL", 100)
        
        assert result.success is False
        assert result.error_code == "INSUFFICIENT_SHARES"
    
    def test_order_no_price(self, broker):
        """Test order rejection when no price set."""
        result = broker.buy("MSFT", 100)  # No price set for MSFT
        
        assert result.success is False
        assert result.error_code == "NO_PRICE"
    
    def test_limit_buy_order_fills(self, broker):
        """Test limit buy order that fills."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=160.0,  # Above current price, should fill
        )
        
        result = broker.submit_order(order)
        
        assert result.success is True
        assert result.order.is_filled is True
    
    def test_limit_buy_order_pending(self, broker):
        """Test limit buy order that doesn't fill."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=140.0,  # Below current price
        )
        
        result = broker.submit_order(order)
        
        assert result.success is True
        # Order accepted but not filled (limit not reached)
        assert "pending" in result.message.lower() or result.order.status == OrderStatus.ACCEPTED
    
    def test_order_with_slippage(self):
        """Test order execution with slippage."""
        broker = PaperBroker(initial_cash=100000, slippage=0.001)
        broker.connect()
        broker.set_price("AAPL", 150.0)
        
        result = broker.buy("AAPL", 100)
        
        # Should pay more due to slippage
        expected_price = 150.0 * 1.001  # 150.15
        assert result.order.avg_fill_price == pytest.approx(expected_price, rel=1e-6)
    
    def test_order_with_commission(self):
        """Test order execution with commission."""
        broker = PaperBroker(
            initial_cash=100000,
            commission=5.0,
            commission_per_share=0.01,
        )
        broker.connect()
        broker.set_price("AAPL", 150.0)
        
        broker.buy("AAPL", 100)
        
        # Cost: 150 * 100 + 5 + 0.01 * 100 = 15006
        account = broker.get_account()
        assert account.cash == pytest.approx(100000 - 15006, rel=1e-6)


class TestPaperBrokerPositions:
    """Tests for PaperBroker position management."""
    
    @pytest.fixture
    def broker(self):
        """Create a connected paper broker with positions."""
        broker = PaperBroker(initial_cash=100000)
        broker.connect()
        broker.set_price("AAPL", 150.0)
        broker.set_price("GOOGL", 2800.0)
        broker.buy("AAPL", 100)
        broker.buy("GOOGL", 10)
        return broker
    
    def test_get_position(self, broker):
        """Test getting a position."""
        position = broker.get_position("AAPL")
        
        assert position is not None
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.avg_cost == 150.0
    
    def test_get_positions(self, broker):
        """Test getting all positions."""
        positions = broker.get_positions()
        
        assert len(positions) == 2
        symbols = [p.symbol for p in positions]
        assert "AAPL" in symbols
        assert "GOOGL" in symbols
    
    def test_position_not_found(self, broker):
        """Test getting non-existent position."""
        position = broker.get_position("MSFT")
        assert position is None
    
    def test_close_position(self, broker):
        """Test closing a position."""
        result = broker.close_position("AAPL")
        
        assert result.success is True
        
        position = broker.get_position("AAPL")
        assert position is None or position.quantity == 0
    
    def test_close_all_positions(self, broker):
        """Test closing all positions."""
        results = broker.close_all_positions()
        
        assert "AAPL" in results
        assert "GOOGL" in results
        assert all(r.success for r in results.values())
        
        positions = broker.get_positions()
        assert len(positions) == 0


class TestPaperBrokerAccount:
    """Tests for PaperBroker account management."""
    
    def test_initial_account(self):
        """Test initial account state."""
        broker = PaperBroker(initial_cash=100000)
        broker.connect()
        
        account = broker.get_account()
        
        assert account.cash == 100000
        assert account.portfolio_value == 100000
        assert account.equity == 100000
        assert account.account_type == "paper"
    
    def test_account_after_trades(self):
        """Test account state after trades."""
        broker = PaperBroker(initial_cash=100000)
        broker.connect()
        broker.set_price("AAPL", 150.0)
        
        broker.buy("AAPL", 100)  # Spend $15,000
        
        account = broker.get_account()
        
        assert account.cash == 85000
        assert account.portfolio_value == 100000  # Value unchanged
    
    def test_account_with_unrealized_gain(self):
        """Test account with unrealized gains."""
        broker = PaperBroker(initial_cash=100000)
        broker.connect()
        broker.set_price("AAPL", 150.0)
        
        broker.buy("AAPL", 100)
        broker.set_price("AAPL", 160.0)  # Price goes up
        
        account = broker.get_account()
        
        # Portfolio value should increase
        assert account.portfolio_value == 85000 + 16000  # Cash + position value
    
    def test_cash_property(self):
        """Test cash property shortcut."""
        broker = PaperBroker(initial_cash=100000)
        broker.connect()
        
        assert broker.cash == 100000
    
    def test_portfolio_value_property(self):
        """Test portfolio_value property shortcut."""
        broker = PaperBroker(initial_cash=100000)
        broker.connect()
        
        assert broker.portfolio_value == 100000


class TestPaperBrokerHistory:
    """Tests for PaperBroker order and fill history."""
    
    @pytest.fixture
    def broker(self):
        """Create broker with some trades."""
        broker = PaperBroker(initial_cash=100000)
        broker.connect()
        broker.set_price("AAPL", 150.0)
        broker.buy("AAPL", 100)
        broker.buy("AAPL", 50)
        broker.sell("AAPL", 25)
        return broker
    
    def test_get_orders(self, broker):
        """Test getting order history."""
        orders = broker.get_orders()
        
        assert len(orders) == 3
    
    def test_get_orders_by_status(self, broker):
        """Test filtering orders by status."""
        filled_orders = broker.get_orders(status=OrderStatus.FILLED)
        
        assert len(filled_orders) == 3
        assert all(o.status == OrderStatus.FILLED for o in filled_orders)
    
    def test_get_orders_by_symbol(self, broker):
        """Test filtering orders by symbol."""
        orders = broker.get_orders(symbol="AAPL")
        
        assert len(orders) == 3
        assert all(o.symbol == "AAPL" for o in orders)
    
    def test_get_fills(self, broker):
        """Test getting fill history."""
        fills = broker.get_fills()
        
        assert len(fills) == 3
    
    def test_get_fills_by_symbol(self, broker):
        """Test filtering fills by symbol."""
        fills = broker.get_fills(symbol="AAPL")
        
        assert len(fills) == 3


class TestPaperBrokerReset:
    """Tests for PaperBroker reset functionality."""
    
    def test_reset(self):
        """Test resetting broker state."""
        broker = PaperBroker(initial_cash=100000)
        broker.connect()
        broker.set_price("AAPL", 150.0)
        broker.buy("AAPL", 100)
        
        broker.reset()
        
        assert broker.cash == 100000
        assert len(broker.get_positions()) == 0
        assert len(broker.get_orders()) == 0
        assert len(broker.get_fills()) == 0


class TestPaperBrokerSummary:
    """Tests for PaperBroker summary functionality."""
    
    def test_get_summary(self):
        """Test getting broker summary."""
        broker = PaperBroker(initial_cash=100000)
        broker.connect()
        broker.set_price("AAPL", 150.0)
        broker.buy("AAPL", 100)
        
        summary = broker.get_summary()
        
        assert "initial_cash" in summary
        assert "current_cash" in summary
        assert "portfolio_value" in summary
        assert "total_return" in summary
        assert "num_positions" in summary
        assert "num_orders" in summary
        assert "num_fills" in summary
        
        assert summary["initial_cash"] == 100000
        assert summary["num_positions"] == 1
        assert summary["num_orders"] == 1


class TestPaperBrokerCallbacks:
    """Tests for PaperBroker event callbacks."""
    
    def test_on_fill_callback(self):
        """Test fill callback is called."""
        broker = PaperBroker(initial_cash=100000)
        broker.connect()
        broker.set_price("AAPL", 150.0)
        
        fills_received = []
        broker.on_fill(lambda fill: fills_received.append(fill))
        
        broker.buy("AAPL", 100)
        
        assert len(fills_received) == 1
        assert fills_received[0].symbol == "AAPL"
    
    def test_on_order_update_callback(self):
        """Test order update callback is called."""
        broker = PaperBroker(initial_cash=100000)
        broker.connect()
        broker.set_price("AAPL", 150.0)
        
        orders_updated = []
        broker.on_order_update(lambda order: orders_updated.append(order))
        
        broker.buy("AAPL", 100)
        
        assert len(orders_updated) >= 1


# =============================================================================
# Alpaca Broker Tests (Optional)
# =============================================================================

class TestAlpacaBroker:
    """Tests for AlpacaBroker (skipped if alpaca-trade-api not installed)."""
    
    @pytest.mark.skipif(not ALPACA_AVAILABLE, reason="alpaca-trade-api not installed")
    def test_alpaca_broker_creation(self):
        """Test creating Alpaca broker (without connecting)."""
        # Note: This will fail without valid credentials, but tests the creation
        with pytest.raises(ValueError, match="API credentials required"):
            AlpacaBroker()  # No credentials
    
    @pytest.mark.skipif(not ALPACA_AVAILABLE, reason="alpaca-trade-api not installed")
    def test_alpaca_broker_with_env_vars(self, monkeypatch):
        """Test Alpaca broker reads env vars."""
        monkeypatch.setenv("APCA_API_KEY_ID", "test_key")
        monkeypatch.setenv("APCA_API_SECRET_KEY", "test_secret")
        
        broker = AlpacaBroker(paper=True)
        
        assert broker.api_key == "test_key"
        assert broker.api_secret == "test_secret"
        assert broker.paper is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestBrokerIntegration:
    """Integration tests for broker module."""
    
    def test_full_trading_workflow(self):
        """Test a complete trading workflow."""
        broker = PaperBroker(
            initial_cash=100000,
            commission=1.0,
            slippage=0.001,
        )
        broker.connect()
        
        # Set initial prices
        broker.set_price("AAPL", 150.0)
        broker.set_price("GOOGL", 2800.0)
        
        # Buy some stocks
        result1 = broker.buy("AAPL", 100)
        assert result1.success is True
        
        result2 = broker.buy("GOOGL", 10)
        assert result2.success is True
        
        # Check positions
        positions = broker.get_positions()
        assert len(positions) == 2
        
        # Update prices (market moves)
        broker.set_price("AAPL", 160.0)  # +6.67%
        broker.set_price("GOOGL", 2700.0)  # -3.57%
        
        # Check P&L
        aapl_pos = broker.get_position("AAPL")
        assert aapl_pos.unrealized_pnl > 0  # Should be profitable
        
        googl_pos = broker.get_position("GOOGL")
        assert googl_pos.unrealized_pnl < 0  # Should be at a loss
        
        # Sell AAPL for profit
        result3 = broker.sell("AAPL", 100)
        assert result3.success is True
        
        # Close remaining position
        result4 = broker.close_position("GOOGL")
        assert result4.success is True
        
        # Check final state
        account = broker.get_account()
        positions = broker.get_positions()
        
        assert len(positions) == 0
        
        # Print summary
        broker.print_summary()
    
    def test_multiple_trades_same_symbol(self):
        """Test averaging into a position."""
        broker = PaperBroker(initial_cash=100000)
        broker.connect()
        broker.set_price("AAPL", 150.0)
        
        # Buy in tranches
        broker.buy("AAPL", 100)  # 100 @ $150
        broker.set_price("AAPL", 145.0)
        broker.buy("AAPL", 100)  # 100 @ $145
        broker.set_price("AAPL", 155.0)
        broker.buy("AAPL", 100)  # 100 @ $155
        
        # Check position
        position = broker.get_position("AAPL")
        assert position.quantity == 300
        
        # Average cost should be (150*100 + 145*100 + 155*100) / 300 = 150
        assert position.avg_cost == pytest.approx(150.0, rel=1e-6)


# =============================================================================
# Phase 14 Specific Tests
# =============================================================================

class TestPhase14BrokerIntegration:
    """Tests verifying Phase 14 broker integration requirements."""
    
    def test_broker_module_exports(self):
        """Test that broker module exports all required components."""
        from jsf import broker
        
        # Check enums
        assert hasattr(broker, 'OrderSide')
        assert hasattr(broker, 'OrderType')
        assert hasattr(broker, 'OrderStatus')
        
        # Check data classes
        assert hasattr(broker, 'Order')
        assert hasattr(broker, 'Position')
        assert hasattr(broker, 'Fill')
        assert hasattr(broker, 'AccountInfo')
        
        # Check base class
        assert hasattr(broker, 'Broker')
        
        # Check implementations
        assert hasattr(broker, 'PaperBroker')
    
    def test_paper_broker_matches_abc(self):
        """Test PaperBroker implements Broker ABC correctly."""
        broker = PaperBroker()
        
        # Check all abstract methods are implemented
        assert hasattr(broker, 'connect')
        assert hasattr(broker, 'disconnect')
        assert hasattr(broker, 'submit_order')
        assert hasattr(broker, 'cancel_order')
        assert hasattr(broker, 'modify_order')
        assert hasattr(broker, 'get_order')
        assert hasattr(broker, 'get_orders')
        assert hasattr(broker, 'get_position')
        assert hasattr(broker, 'get_positions')
        assert hasattr(broker, 'get_account')
        assert hasattr(broker, 'get_trades')
        assert hasattr(broker, 'get_fills')
    
    def test_broker_convenience_methods(self):
        """Test broker convenience methods work."""
        broker = PaperBroker(initial_cash=100000)
        broker.connect()
        broker.set_price("AAPL", 150.0)
        
        # Test buy() convenience method
        result = broker.buy("AAPL", 10)
        assert result.success is True
        
        # Test sell() convenience method
        result = broker.sell("AAPL", 5)
        assert result.success is True
        
        # Test close_position() convenience method
        result = broker.close_position("AAPL")
        assert result.success is True
    
    def test_order_lifecycle(self):
        """Test complete order lifecycle."""
        broker = PaperBroker(initial_cash=100000)
        broker.connect()
        broker.set_price("AAPL", 150.0)
        
        # Create order
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        
        # Initial state
        assert order.status == OrderStatus.PENDING
        assert order.is_active is True
        
        # Submit order
        result = broker.submit_order(order)
        
        # Final state
        assert result.order.status == OrderStatus.FILLED
        assert result.order.is_filled is True
        assert result.order.filled_quantity == 100
        assert result.order.avg_fill_price is not None

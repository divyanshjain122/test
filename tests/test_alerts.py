"""
Tests for alert system.

This module tests the alert infrastructure including models,
alerters, and the alert manager.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from jsf.alerts import (
    Alert,
    AlertType,
    AlertSeverity,
    AlertChannel,
    AlertError,
    BaseAlerter,
    ConsoleAlerter,
    AlertManager,
    create_default_manager,
    create_order_alert,
    create_position_alert,
    create_risk_alert,
    create_strategy_alert,
    create_system_alert,
)
from jsf.broker.models import (
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Position,
    PositionSide,
)


# =============================================================================
# Alert Model Tests
# =============================================================================

class TestAlert:
    """Test Alert dataclass."""
    
    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert(
            title="Test Alert",
            message="This is a test",
            severity=AlertSeverity.INFO,
            alert_type=AlertType.CUSTOM,
        )
        
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test"
        assert alert.severity == AlertSeverity.INFO
        assert alert.alert_type == AlertType.CUSTOM
        assert isinstance(alert.timestamp, datetime)
        assert alert.metadata == {}
    
    def test_alert_with_metadata(self):
        """Test alert with metadata."""
        metadata = {"symbol": "AAPL", "price": 150.0}
        alert = Alert(
            title="Order Filled",
            message="Buy order executed",
            metadata=metadata,
        )
        
        assert alert.metadata == metadata
    
    def test_alert_string_conversion(self):
        """Test converting strings to enums."""
        alert = Alert(
            title="Test",
            message="Test message",
            severity="warning",
            alert_type="order_filled",
        )
        
        assert alert.severity == AlertSeverity.WARNING
        assert alert.alert_type == AlertType.ORDER_FILLED
    
    def test_alert_format_message(self):
        """Test formatting alert as message."""
        alert = Alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.WARNING,
            metadata={"key": "value"},
        )
        
        formatted = alert.format_message()
        
        assert "⚠️" in formatted
        assert "Test Alert" in formatted
        assert "Test message" in formatted
        assert "key: value" in formatted
    
    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        alert = Alert(
            title="Test",
            message="Message",
            severity=AlertSeverity.ERROR,
            alert_type=AlertType.SYSTEM_ERROR,
        )
        
        alert_dict = alert.to_dict()
        
        assert alert_dict["title"] == "Test"
        assert alert_dict["message"] == "Message"
        assert alert_dict["severity"] == "error"
        assert alert_dict["alert_type"] == "system_error"
    
    def test_alert_validation(self):
        """Test alert validation."""
        with pytest.raises(ValueError):
            Alert(title="", message="test")
        
        with pytest.raises(ValueError):
            Alert(title="test", message="")


# =============================================================================
# Console Alerter Tests
# =============================================================================

class TestConsoleAlerter:
    """Test ConsoleAlerter."""
    
    def test_console_alerter_creation(self):
        """Test creating console alerter."""
        alerter = ConsoleAlerter()
        
        assert alerter.name == "console"
        assert alerter.enabled is True
        assert alerter.min_severity == AlertSeverity.DEBUG
    
    def test_console_send(self, capsys):
        """Test sending alert to console."""
        alerter = ConsoleAlerter(use_colors=False)
        
        alert = Alert(
            title="Test",
            message="Test message",
            severity=AlertSeverity.INFO,
        )
        
        result = alerter.send(alert)
        
        assert result is True
        
        captured = capsys.readouterr()
        assert "Test" in captured.out
        assert "Test message" in captured.out
    
    def test_console_severity_filtering(self, capsys):
        """Test severity filtering."""
        alerter = ConsoleAlerter(
            min_severity=AlertSeverity.WARNING,
            use_colors=False,
        )
        
        # INFO alert should not be displayed
        info_alert = Alert(
            title="Info",
            message="Info message",
            severity=AlertSeverity.INFO,
        )
        result = alerter.send(info_alert)
        assert result is False
        
        # WARNING alert should be displayed
        warning_alert = Alert(
            title="Warning",
            message="Warning message",
            severity=AlertSeverity.WARNING,
        )
        result = alerter.send(warning_alert)
        assert result is True
        
        captured = capsys.readouterr()
        assert "Info" not in captured.out
        assert "Warning" in captured.out


# =============================================================================
# Telegram Alerter Tests
# =============================================================================

class TestTelegramAlerter:
    """Test TelegramAlerter."""
    
    @patch('jsf.alerts.telegram.TELEGRAM_AVAILABLE', True)
    @patch('jsf.alerts.telegram.Bot')
    def test_telegram_creation(self, mock_bot):
        """Test creating Telegram alerter."""
        from jsf.alerts.telegram import TelegramAlerter
        
        alerter = TelegramAlerter(
            bot_token="test_token",
            chat_id="test_chat",
        )
        
        assert alerter.bot_token == "test_token"
        assert alerter.chat_id == "test_chat"
        assert alerter.name == "telegram"
    
    @patch('jsf.alerts.telegram.TELEGRAM_AVAILABLE', True)
    @patch('jsf.alerts.telegram.Bot')
    def test_telegram_connect(self, mock_bot_class):
        """Test connecting to Telegram."""
        from jsf.alerts.telegram import TelegramAlerter
        
        mock_bot = MagicMock()
        mock_bot.get_me.return_value = MagicMock(username="test_bot")
        mock_bot_class.return_value = mock_bot
        
        alerter = TelegramAlerter(
            bot_token="test_token",
            chat_id="test_chat",
        )
        
        result = alerter.connect()
        
        assert result is True
        assert alerter.is_connected
        mock_bot.get_me.assert_called_once()
    
    @patch('jsf.alerts.telegram.TELEGRAM_AVAILABLE', True)
    @patch('jsf.alerts.telegram.Bot')
    def test_telegram_send(self, mock_bot_class):
        """Test sending alert via Telegram."""
        from jsf.alerts.telegram import TelegramAlerter
        
        mock_bot = MagicMock()
        mock_bot.get_me.return_value = MagicMock(username="test_bot")
        mock_bot_class.return_value = mock_bot
        
        alerter = TelegramAlerter(
            bot_token="test_token",
            chat_id="test_chat",
        )
        
        alert = Alert(
            title="Test",
            message="Test message",
            severity=AlertSeverity.INFO,
        )
        
        result = alerter.send(alert)
        
        assert result is True
        mock_bot.send_message.assert_called_once()


# =============================================================================
# Alert Manager Tests
# =============================================================================

class TestAlertManager:
    """Test AlertManager."""
    
    def test_manager_creation(self):
        """Test creating alert manager."""
        manager = AlertManager()
        
        assert len(manager.alerters) == 0
        assert manager.async_mode is False
    
    def test_add_alerter(self):
        """Test adding alerters."""
        manager = AlertManager()
        alerter = ConsoleAlerter()
        
        manager.add_alerter(alerter)
        
        assert len(manager.alerters) == 1
        assert manager.alerters[0] == alerter
    
    def test_remove_alerter(self):
        """Test removing alerters."""
        manager = AlertManager()
        alerter = ConsoleAlerter(name="test_console")
        
        manager.add_alerter(alerter)
        assert len(manager.alerters) == 1
        
        removed = manager.remove_alerter("test_console")
        assert removed is True
        assert len(manager.alerters) == 0
    
    def test_get_alerter(self):
        """Test getting alerter by name."""
        manager = AlertManager()
        alerter = ConsoleAlerter(name="test_console")
        
        manager.add_alerter(alerter)
        
        found = manager.get_alerter("test_console")
        assert found == alerter
        
        not_found = manager.get_alerter("nonexistent")
        assert not_found is None
    
    def test_send_alert(self, capsys):
        """Test sending alert through manager."""
        manager = AlertManager()
        manager.add_alerter(ConsoleAlerter(use_colors=False))
        
        result = manager.send_alert(
            title="Test",
            message="Test message",
            severity=AlertSeverity.INFO,
        )
        
        assert result is True
        assert manager.stats["total_sent"] == 1
        
        captured = capsys.readouterr()
        assert "Test" in captured.out
    
    def test_alert_history(self):
        """Test alert history tracking."""
        manager = AlertManager()
        manager.add_alerter(ConsoleAlerter())
        
        manager.send_alert("Test 1", "Message 1")
        manager.send_alert("Test 2", "Message 2")
        
        assert len(manager.alert_history) == 2
        assert manager.alert_history[0].title == "Test 1"
        assert manager.alert_history[1].title == "Test 2"
    
    def test_alert_stats(self):
        """Test alert statistics."""
        manager = AlertManager()
        manager.add_alerter(ConsoleAlerter())
        
        manager.send_alert("Test", "Message", severity=AlertSeverity.INFO)
        manager.send_alert("Test", "Message", severity=AlertSeverity.WARNING)
        
        stats = manager.get_stats()
        
        assert stats["total_sent"] == 2
        assert stats["by_severity"]["info"] == 1
        assert stats["by_severity"]["warning"] == 1
    
    def test_callback_system(self):
        """Test callback system."""
        manager = AlertManager()
        manager.add_alerter(ConsoleAlerter())
        
        callback_called = []
        
        def callback(alert):
            callback_called.append(alert)
        
        manager.add_callback(callback)
        manager.send_alert("Test", "Message")
        
        assert len(callback_called) == 1
        assert callback_called[0].title == "Test"
    
    def test_context_manager(self):
        """Test using manager as context manager."""
        with AlertManager() as manager:
            manager.add_alerter(ConsoleAlerter())
            manager.send_alert("Test", "Message")
        
        # Should shutdown cleanly


# =============================================================================
# Integration Helper Tests
# =============================================================================

class TestIntegrationHelpers:
    """Test alert integration helpers."""
    
    def test_create_order_alert_submitted(self):
        """Test creating order submitted alert."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            order_id="test_123",
            status=OrderStatus.SUBMITTED,
        )
        
        alert = create_order_alert(order, OrderStatus.SUBMITTED)
        
        assert alert.title == "Order Submitted"
        assert "AAPL" in alert.message
        assert "BUY" in alert.message
        assert alert.severity == AlertSeverity.INFO
        assert alert.alert_type == AlertType.ORDER_SUBMITTED
        assert alert.metadata["Symbol"] == "AAPL"
    
    def test_create_order_alert_filled(self):
        """Test creating order filled alert."""
        order = Order(
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=300.0,
            order_id="test_456",
            status=OrderStatus.FILLED,
            filled_quantity=50,
            avg_fill_price=301.0,
        )
        
        alert = create_order_alert(order, OrderStatus.FILLED)
        
        assert alert.title == "Order Filled"
        assert "MSFT" in alert.message
        assert "SELL" in alert.message
        assert alert.severity == AlertSeverity.INFO
        assert alert.alert_type == AlertType.ORDER_FILLED
    
    def test_create_position_alert_opened(self):
        """Test creating position opened alert."""
        position = Position(
            symbol="GOOGL",
            side=PositionSide.LONG,
            quantity=25,
            avg_cost=2800.0,
            current_price=2850.0,
        )
        
        alert = create_position_alert(position, "opened")
        
        assert alert.title == "Position Opened"
        assert "GOOGL" in alert.message
        assert alert.alert_type == AlertType.TRADE_OPENED
        assert alert.metadata["Symbol"] == "GOOGL"
    
    def test_create_risk_alert(self):
        """Test creating risk alert."""
        alert = create_risk_alert(
            alert_type=AlertType.DRAWDOWN_THRESHOLD,
            title="Drawdown Alert",
            message="Portfolio down 5%",
            severity=AlertSeverity.WARNING,
            drawdown=-0.05,
        )
        
        assert alert.title == "Drawdown Alert"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.alert_type == AlertType.DRAWDOWN_THRESHOLD
        assert alert.metadata["drawdown"] == -0.05
    
    def test_create_strategy_alert(self):
        """Test creating strategy alert."""
        alert = create_strategy_alert(
            strategy_name="Momentum",
            event="started",
            message="Strategy initialized",
        )
        
        assert "Momentum" in alert.title
        assert alert.alert_type == AlertType.STRATEGY_STARTED
        assert alert.metadata["Strategy"] == "Momentum"
    
    def test_create_system_alert(self):
        """Test creating system alert."""
        alert = create_system_alert(
            title="Connection Error",
            message="Lost connection to broker",
            severity=AlertSeverity.CRITICAL,
        )
        
        assert alert.title == "Connection Error"
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.alert_type == AlertType.SYSTEM_ERROR


# =============================================================================
# Create Default Manager Tests
# =============================================================================

class TestCreateDefaultManager:
    """Test create_default_manager helper."""
    
    def test_create_console_only(self):
        """Test creating manager with console only."""
        manager = create_default_manager(include_console=True)
        
        assert len(manager.alerters) == 1
        assert isinstance(manager.alerters[0], ConsoleAlerter)
    
    def test_create_with_telegram(self):
        """Test creating manager with Telegram."""
        # Without telegram installed, should still create console
        manager = create_default_manager(
            telegram_token="test",
            telegram_chat_id="123",
            include_console=True,
        )
        
        # Should have at least console
        assert len(manager.alerters) >= 1


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

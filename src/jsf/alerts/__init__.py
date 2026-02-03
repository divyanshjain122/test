"""
Alert system for real-time notifications.

This module provides a flexible alerting framework for trading systems,
supporting multiple notification channels including Telegram, email, and console.

Key Components:
- Alert models and severity levels
- Base alerter interface
- Telegram integration
- Alert manager for routing and delivery
- Built-in alert types for trading events

Example:
    >>> from jsf.alerts import AlertManager, TelegramAlerter, Alert, AlertSeverity
    >>> 
    >>> # Setup Telegram alerter
    >>> telegram = TelegramAlerter(bot_token="YOUR_TOKEN", chat_id="YOUR_CHAT_ID")
    >>> manager = AlertManager()
    >>> manager.add_alerter(telegram)
    >>> 
    >>> # Send alert
    >>> alert = Alert(
    ...     title="Trade Executed",
    ...     message="Bought 100 shares of AAPL at $150.00",
    ...     severity=AlertSeverity.INFO
    ... )
    >>> manager.send(alert)
"""

from jsf.alerts.base import (
    Alert,
    AlertType,
    AlertSeverity,
    AlertChannel,
    BaseAlerter,
    AlertError,
)

from jsf.alerts.manager import AlertManager, create_default_manager
from jsf.alerts.console import ConsoleAlerter
from jsf.alerts.integration import (
    create_order_alert,
    create_position_alert,
    create_risk_alert,
    create_strategy_alert,
    create_system_alert,
)
from jsf.alerts.factory import (
    create_alert_manager_from_config,
    create_simple_alert_manager,
)

# Conditional imports
try:
    from jsf.alerts.telegram import TelegramAlerter, TelegramChannelAlerter
except ImportError:
    TelegramAlerter = None
    TelegramChannelAlerter = None

__all__ = [
    # Models
    "Alert",
    "AlertType",
    "AlertSeverity",
    "AlertChannel",
    # Base classes
    "BaseAlerter",
    "AlertError",
    # Manager
    "AlertManager",
    "create_default_manager",
    # Alerters
    "ConsoleAlerter",
    "TelegramAlerter",
    "TelegramChannelAlerter",
    # Factory (recommended for production)
    "create_alert_manager_from_config",
    "create_simple_alert_manager",
    # Integration helpers
    "create_order_alert",
    "create_position_alert",
    "create_risk_alert",
    "create_strategy_alert",
    "create_system_alert",
]

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

from jsf.alerts.manager import AlertManager

# Conditional imports
try:
    from jsf.alerts.telegram import TelegramAlerter
except ImportError:
    TelegramAlerter = None

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
    # Alerters
    "TelegramAlerter",
]

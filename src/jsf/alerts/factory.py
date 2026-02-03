"""
Factory functions for creating pre-configured alert managers.

Simplifies alert system setup by providing ready-to-use configurations.
"""

import logging
from typing import Optional, List

from jsf.alerts.base import AlertSeverity
from jsf.alerts.console import ConsoleAlerter
from jsf.alerts.manager import AlertManager

logger = logging.getLogger(__name__)

# Telegram import is optional
try:
    from jsf.alerts.telegram import TelegramAlerter
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    TelegramAlerter = None


def create_alert_manager_from_config(
    config=None,
    include_console: bool = True,
    console_min_severity: Optional[AlertSeverity] = None,
) -> AlertManager:
    """
    Create AlertManager using centralized configuration.
    
    Automatically loads configuration from .env file or environment variables.
    This is the recommended way to set up alerts for production use.
    
    Args:
        config: Optional AppConfig object. If None, loads from environment.
        include_console: Whether to include console alerter
        console_min_severity: Minimum severity for console alerts
        
    Returns:
        AlertManager: Configured alert manager
        
    Example:
        >>> # Simple setup - just works if .env is configured
        >>> from jsf.alerts import create_alert_manager_from_config
        >>> manager = create_alert_manager_from_config()
        >>> 
        >>> # Send alerts
        >>> from jsf.alerts import Alert, AlertSeverity
        >>> manager.send(Alert(
        ...     title="Trade Executed",
        ...     message="Bought 100 shares of AAPL at $150.00",
        ...     severity=AlertSeverity.INFO
        ... ))
    """
    from jsf.settings import get_config
    
    if config is None:
        config = get_config()
    
    manager = AlertManager()
    
    # Add console alerter
    if include_console and config.enable_console_alerts:
        min_sev = console_min_severity or AlertSeverity[config.min_alert_severity]
        console = ConsoleAlerter(
            name="console",
            min_severity=min_sev,
        )
        manager.add_alerter(console)
        logger.info(f"Added console alerter (min severity: {min_sev.value})")
    
    # Add Telegram alerter if configured
    if config.enable_telegram_alerts and config.telegram.is_configured:
        if not TELEGRAM_AVAILABLE:
            logger.warning(
                "Telegram alerts enabled but python-telegram-bot not installed. "
                "Install with: pip install python-telegram-bot"
            )
        else:
            # Create alerters for each chat ID
            for i, chat_id in enumerate(config.telegram.chat_ids):
                name = f"telegram_{i+1}" if len(config.telegram.chat_ids) > 1 else "telegram"
                telegram = TelegramAlerter(
                    bot_token=config.telegram.bot_token,
                    chat_id=chat_id,
                    name=name,
                    min_severity=AlertSeverity[config.min_alert_severity],
                    parse_mode=config.telegram.parse_mode,
                )
                # Connect the alerter
                if telegram.connect():
                    manager.add_alerter(telegram)
                    logger.info(f"Added Telegram alerter: {name} (chat_id: {chat_id[:8]}...)")
                else:
                    logger.warning(f"Failed to connect Telegram alerter: {name}")
            
            # Add channel alerter if configured
            if config.telegram.channel_id:
                channel = TelegramAlerter(
                    bot_token=config.telegram.bot_token,
                    chat_id=config.telegram.channel_id,
                    name="telegram_channel",
                    min_severity=AlertSeverity.WARNING,  # Only send important alerts to channel
                    parse_mode=config.telegram.parse_mode,
                )
                if channel.connect():
                    manager.add_alerter(channel)
                    logger.info("Added Telegram channel alerter")
                else:
                    logger.warning("Failed to connect Telegram channel alerter")
    
    elif config.enable_telegram_alerts and not config.telegram.is_configured:
        logger.warning(
            "Telegram alerts enabled but not configured. "
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS in .env file."
        )
    
    logger.info(f"Alert manager created with {len(manager.alerters)} alerter(s)")
    return manager


def create_simple_alert_manager(
    telegram_token: Optional[str] = None,
    telegram_chat_ids: Optional[List[str]] = None,
    include_console: bool = True,
    min_severity: AlertSeverity = AlertSeverity.INFO,
) -> AlertManager:
    """
    Create AlertManager with simple direct configuration.
    
    Useful for quick setup or testing without .env file.
    For production, prefer create_alert_manager_from_config().
    
    Args:
        telegram_token: Telegram bot token
        telegram_chat_ids: List of Telegram chat IDs
        include_console: Whether to include console alerter
        min_severity: Minimum alert severity
        
    Returns:
        AlertManager: Configured alert manager
        
    Example:
        >>> # Quick setup for testing
        >>> manager = create_simple_alert_manager(
        ...     telegram_token="123456:ABC-DEF",
        ...     telegram_chat_ids=["123456789"],
        ...     include_console=True
        ... )
    """
    manager = AlertManager()
    
    # Add console alerter
    if include_console:
        console = ConsoleAlerter(
            name="console",
            min_severity=min_severity,
        )
        manager.add_alerter(console)
    
    # Add Telegram alerters
    if telegram_token and telegram_chat_ids:
        if not TELEGRAM_AVAILABLE:
            logger.warning(
                "python-telegram-bot not installed. "
                "Install with: pip install python-telegram-bot"
            )
        else:
            for i, chat_id in enumerate(telegram_chat_ids):
                name = f"telegram_{i+1}" if len(telegram_chat_ids) > 1 else "telegram"
                telegram = TelegramAlerter(
                    bot_token=telegram_token,
                    chat_id=chat_id,
                    name=name,
                    min_severity=min_severity,
                )
                if telegram.connect():
                    manager.add_alerter(telegram)
    
    return manager


__all__ = [
    "create_alert_manager_from_config",
    "create_simple_alert_manager",
]

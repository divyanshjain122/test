"""
Telegram alerter for sending notifications via Telegram bot.

This module provides Telegram integration for real-time trading alerts.
Requires the 'python-telegram-bot' package.
"""

from typing import Optional, List
import logging

from jsf.alerts.base import BaseAlerter, Alert, AlertSeverity, AlertError

logger = logging.getLogger(__name__)

# Try to import telegram
try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Bot = None
    TelegramError = Exception


class TelegramAlerter(BaseAlerter):
    """
    Send alerts via Telegram bot.
    
    Requires:
        - Telegram bot token (from @BotFather)
        - Chat ID (user or channel to send to)
        - python-telegram-bot package
    
    Args:
        bot_token: Telegram bot API token
        chat_id: Telegram chat ID to send messages to
        name: Alerter name
        enabled: Whether alerter is enabled
        min_severity: Minimum severity to send
        parse_mode: Message parse mode ('Markdown', 'HTML', or None)
        
    Example:
        >>> # Get token from @BotFather on Telegram
        >>> # Get chat_id by messaging @userinfobot
        >>> alerter = TelegramAlerter(
        ...     bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        ...     chat_id="123456789"
        ... )
        >>> alerter.send(alert)
        
    Setup Instructions:
        1. Message @BotFather on Telegram
        2. Create new bot with /newbot command
        3. Copy the bot token
        4. Message @userinfobot to get your chat_id
        5. Start a chat with your bot (required for it to message you)
    """
    
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        name: str = "telegram",
        enabled: bool = True,
        min_severity: Optional[AlertSeverity] = None,
        parse_mode: str = "Markdown",
    ):
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot is required for Telegram alerts. "
                "Install with: pip install python-telegram-bot"
            )
        
        super().__init__(name=name, enabled=enabled)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.min_severity = min_severity or AlertSeverity.INFO
        self.parse_mode = parse_mode
        self.bot: Optional[Bot] = None
        
        if not bot_token:
            raise ValueError("bot_token is required for TelegramAlerter")
        if not chat_id:
            raise ValueError("chat_id is required for TelegramAlerter")
    
    def connect(self) -> bool:
        """Initialize Telegram bot."""
        try:
            import asyncio
            self.bot = Bot(token=self.bot_token)
            
            # Test connection (handle async)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't test in running loop, just assume it works
                    logger.info(f"Telegram bot initialized (token: ...{self.bot_token[-8:]})")
                    self._connected = True
                    return True
                else:
                    bot_info = loop.run_until_complete(self.bot.get_me())
                    logger.info(f"Connected to Telegram bot: {bot_info.username}")
                    self._connected = True
                    return True
            except RuntimeError:
                # No event loop, create one
                bot_info = asyncio.run(self.bot.get_me())
                logger.info(f"Connected to Telegram bot: {bot_info.username}")
                self._connected = True
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to Telegram: {e}")
            self._connected = False
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Telegram."""
        self.bot = None
        self._connected = False
        return True
    
    def send(self, alert: Alert) -> bool:
        """
        Send alert via Telegram.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False
        
        # Check minimum severity
        severity_order = {
            AlertSeverity.DEBUG: 0,
            AlertSeverity.INFO: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.ERROR: 3,
            AlertSeverity.CRITICAL: 4,
        }
        
        if severity_order[alert.severity] < severity_order[self.min_severity]:
            return False
        
        # Connect if not connected
        if not self._connected:
            if not self.connect():
                return False
        
        try:
            import asyncio
            # Format message for Telegram
            message = self._format_telegram_message(alert)
            
            # Send message (handle async)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule coroutine in existing loop
                    asyncio.create_task(
                        self.bot.send_message(
                            chat_id=self.chat_id,
                            text=message,
                            parse_mode=self.parse_mode,
                        )
                    )
                else:
                    loop.run_until_complete(
                        self.bot.send_message(
                            chat_id=self.chat_id,
                            text=message,
                            parse_mode=self.parse_mode,
                        )
                    )
            except RuntimeError:
                # No event loop, create one
                asyncio.run(
                    self.bot.send_message(
                        chat_id=self.chat_id,
                        text=message,
                        parse_mode=self.parse_mode,
                    )
                )
            
            logger.debug(f"Sent Telegram alert: {alert.title}")
            return True
            
        except TelegramError as e:
            logger.error(f"Telegram error sending alert: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
    
    def _format_telegram_message(self, alert: Alert) -> str:
        """
        Format alert for Telegram.
        
        Args:
            alert: Alert to format
            
        Returns:
            Formatted message string
        """
        # Emoji for severity
        severity_emoji = {
            AlertSeverity.DEBUG: "🔍",
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨",
        }
        
        emoji = severity_emoji.get(alert.severity, "📢")
        
        # Build message
        parts = [
            f"{emoji} *{alert.title}*",
            "",
            alert.message,
        ]
        
        # Add metadata if present
        if alert.metadata:
            parts.append("")
            parts.append("*Details:*")
            for key, value in alert.metadata.items():
                # Format value
                if isinstance(value, float):
                    if abs(value) >= 1000:
                        value_str = f"${value:,.2f}" if 'price' in key.lower() or 'value' in key.lower() else f"{value:,.2f}"
                    else:
                        value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                
                parts.append(f"• {key}: {value_str}")
        
        # Add timestamp
        parts.append("")
        parts.append(f"_Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_")
        
        return "\n".join(parts)
    
    def test(self) -> bool:
        """
        Send a test message to verify setup.
        
        Returns:
            True if test successful
        """
        test_alert = Alert(
            title="🎉 Telegram Alert Test",
            message="Your Telegram alerter is configured correctly and working!",
            severity=AlertSeverity.INFO,
            metadata={
                "Bot Name": self.name,
                "Chat ID": self.chat_id,
                "Status": "Operational ✅",
            }
        )
        
        result = self.send(test_alert)
        
        if result:
            logger.info("Telegram test alert sent successfully")
        else:
            logger.error("Telegram test alert failed")
        
        return result


class TelegramChannelAlerter(TelegramAlerter):
    """
    Send alerts to a Telegram channel.
    
    Similar to TelegramAlerter but optimized for channels.
    Use channel username (e.g., '@mychannel') or channel ID as chat_id.
    
    Note: Bot must be added as administrator to the channel.
    
    Example:
        >>> alerter = TelegramChannelAlerter(
        ...     bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        ...     chat_id="@my_trading_alerts"
        ... )
    """
    
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        name: str = "telegram_channel",
        enabled: bool = True,
        min_severity: Optional[AlertSeverity] = None,
    ):
        super().__init__(
            bot_token=bot_token,
            chat_id=chat_id,
            name=name,
            enabled=enabled,
            min_severity=min_severity,
            parse_mode="Markdown",
        )

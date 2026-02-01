"""
Base alert models and abstract alerter interface.

This module defines the core data structures and interfaces for the alert system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    DEBUG = "debug"         # Detailed diagnostic information
    INFO = "info"           # General informational messages
    WARNING = "warning"     # Warning messages for potential issues
    ERROR = "error"         # Error messages for failures
    CRITICAL = "critical"   # Critical alerts requiring immediate attention


class AlertType(Enum):
    """Types of trading alerts."""
    
    # Trading events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    
    # Risk events
    DRAWDOWN_THRESHOLD = "drawdown_threshold"
    POSITION_LIMIT = "position_limit"
    LOSS_LIMIT = "loss_limit"
    VOLATILITY_SPIKE = "volatility_spike"
    
    # Strategy events
    SIGNAL_GENERATED = "signal_generated"
    REBALANCE_TRIGGERED = "rebalance_triggered"
    STRATEGY_STOPPED = "strategy_stopped"
    STRATEGY_STARTED = "strategy_started"
    
    # System events
    CONNECTION_LOST = "connection_lost"
    CONNECTION_RESTORED = "connection_restored"
    DATA_ERROR = "data_error"
    SYSTEM_ERROR = "system_error"
    
    # Performance milestones
    PROFIT_TARGET = "profit_target"
    NEW_EQUITY_HIGH = "new_equity_high"
    
    # Custom
    CUSTOM = "custom"


class AlertChannel(Enum):
    """Alert delivery channels."""
    
    CONSOLE = "console"         # Print to console
    TELEGRAM = "telegram"       # Telegram bot
    EMAIL = "email"             # Email notification
    SMS = "sms"                 # SMS text message
    WEBHOOK = "webhook"         # HTTP webhook
    SLACK = "slack"             # Slack channel
    DISCORD = "discord"         # Discord channel


class AlertError(Exception):
    """Base exception for alert system errors."""
    pass


@dataclass
class Alert:
    """
    Represents a trading alert.
    
    Attributes:
        title: Alert title/subject
        message: Detailed alert message
        severity: Alert severity level
        alert_type: Type of alert
        timestamp: When the alert was created
        metadata: Additional alert data (positions, prices, etc.)
        channels: Specific channels to send to (None = all)
    """
    
    title: str
    message: str
    severity: AlertSeverity = AlertSeverity.INFO
    alert_type: AlertType = AlertType.CUSTOM
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    channels: Optional[List[AlertChannel]] = None
    
    def __post_init__(self):
        """Validate alert data."""
        if not self.title:
            raise ValueError("Alert title cannot be empty")
        if not self.message:
            raise ValueError("Alert message cannot be empty")
        
        # Convert strings to enums if needed
        if isinstance(self.severity, str):
            self.severity = AlertSeverity(self.severity)
        if isinstance(self.alert_type, str):
            self.alert_type = AlertType(self.alert_type)
        if self.channels and isinstance(self.channels[0], str):
            self.channels = [AlertChannel(ch) for ch in self.channels]
    
    def format_message(self, include_metadata: bool = True) -> str:
        """
        Format alert as a text message.
        
        Args:
            include_metadata: Whether to include metadata in message
            
        Returns:
            Formatted message string
        """
        severity_emoji = {
            AlertSeverity.DEBUG: "🔍",
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨",
        }
        
        emoji = severity_emoji.get(self.severity, "📢")
        
        msg_parts = [
            f"{emoji} **{self.title}**",
            f"",
            f"{self.message}",
        ]
        
        if include_metadata and self.metadata:
            msg_parts.append("")
            msg_parts.append("**Details:**")
            for key, value in self.metadata.items():
                msg_parts.append(f"• {key}: {value}")
        
        msg_parts.append("")
        msg_parts.append(f"_Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_")
        
        return "\n".join(msg_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "alert_type": self.alert_type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "channels": [ch.value for ch in self.channels] if self.channels else None,
        }


class BaseAlerter(ABC):
    """
    Abstract base class for alert delivery channels.
    
    All alerter implementations must inherit from this class and
    implement the send() method.
    """
    
    def __init__(self, name: str, enabled: bool = True):
        """
        Initialize alerter.
        
        Args:
            name: Alerter name
            enabled: Whether alerter is enabled
        """
        self.name = name
        self.enabled = enabled
        self._connected = False
    
    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """
        Send an alert through this channel.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent successfully, False otherwise
            
        Raises:
            AlertError: If sending fails critically
        """
        pass
    
    def connect(self) -> bool:
        """
        Connect to the alert service.
        
        Returns:
            True if connected successfully
        """
        self._connected = True
        return True
    
    def disconnect(self) -> bool:
        """
        Disconnect from the alert service.
        
        Returns:
            True if disconnected successfully
        """
        self._connected = False
        return True
    
    @property
    def is_connected(self) -> bool:
        """Check if alerter is connected."""
        return self._connected
    
    def test(self) -> bool:
        """
        Send a test alert to verify configuration.
        
        Returns:
            True if test successful
        """
        test_alert = Alert(
            title="Test Alert",
            message=f"This is a test alert from {self.name}",
            severity=AlertSeverity.INFO,
            alert_type=AlertType.CUSTOM,
        )
        return self.send(test_alert)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.enabled})"

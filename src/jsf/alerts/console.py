"""
Console alerter for printing alerts to stdout/stderr.

Simple alerter that prints alerts to the console with color coding.
"""

import sys
from typing import Optional
from datetime import datetime

from jsf.alerts.base import BaseAlerter, Alert, AlertSeverity
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class ConsoleAlerter(BaseAlerter):
    """
    Print alerts to console with color coding.
    
    Args:
        name: Alerter name
        enabled: Whether alerter is enabled
        use_colors: Whether to use ANSI color codes
        min_severity: Minimum severity to display
        
    Example:
        >>> alerter = ConsoleAlerter(min_severity=AlertSeverity.WARNING)
        >>> alerter.send(alert)
    """
    
    def __init__(
        self,
        name: str = "console",
        enabled: bool = True,
        use_colors: bool = True,
        min_severity: Optional[AlertSeverity] = None,
    ):
        super().__init__(name=name, enabled=enabled)
        self.use_colors = use_colors
        self.min_severity = min_severity or AlertSeverity.DEBUG
        
        # ANSI color codes
        self.colors = {
            AlertSeverity.DEBUG: "\033[37m",      # White
            AlertSeverity.INFO: "\033[36m",       # Cyan
            AlertSeverity.WARNING: "\033[33m",    # Yellow
            AlertSeverity.ERROR: "\033[31m",      # Red
            AlertSeverity.CRITICAL: "\033[1;31m", # Bold Red
        }
        self.reset = "\033[0m"
    
    def send(self, alert: Alert) -> bool:
        """Send alert to console."""
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
        
        try:
            # Format message
            timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            header = f"[{timestamp}] [{alert.severity.value.upper()}] {alert.title}"
            
            # Add color if enabled
            if self.use_colors:
                color = self.colors.get(alert.severity, "")
                header = f"{color}{header}{self.reset}"
            
            # Print to appropriate stream
            stream = sys.stderr if alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL] else sys.stdout
            
            print(header, file=stream)
            print(f"  {alert.message}", file=stream)
            
            if alert.metadata:
                print("  Metadata:", file=stream)
                for key, value in alert.metadata.items():
                    print(f"    {key}: {value}", file=stream)
            
            print("", file=stream)  # Blank line
            stream.flush()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send console alert: {e}")
            return False

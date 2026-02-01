"""
Alert manager for routing and delivering alerts.

This module provides centralized alert management with support for
multiple delivery channels and filtering.
"""

from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
import threading
import queue
import logging

from jsf.alerts.base import (
    BaseAlerter,
    Alert,
    AlertSeverity,
    AlertType,
    AlertChannel,
    AlertError,
)

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Centralized alert management system.
    
    Manages multiple alerters and routes alerts to appropriate channels
    based on severity, type, and configuration.
    
    Args:
        alerters: List of alerter instances
        async_mode: Whether to send alerts asynchronously
        queue_size: Size of async alert queue
        
    Example:
        >>> from jsf.alerts import AlertManager, TelegramAlerter, ConsoleAlerter
        >>> 
        >>> manager = AlertManager()
        >>> manager.add_alerter(ConsoleAlerter())
        >>> manager.add_alerter(TelegramAlerter(bot_token="...", chat_id="..."))
        >>> 
        >>> # Send alert
        >>> manager.send_alert(
        ...     title="Order Filled",
        ...     message="Bought 100 shares of AAPL at $150.00",
        ...     severity=AlertSeverity.INFO,
        ...     alert_type=AlertType.ORDER_FILLED
        ... )
    """
    
    def __init__(
        self,
        alerters: Optional[List[BaseAlerter]] = None,
        async_mode: bool = False,
        queue_size: int = 1000,
    ):
        self.alerters: List[BaseAlerter] = alerters or []
        self.async_mode = async_mode
        self.queue_size = queue_size
        
        # Alert history
        self.alert_history: List[Alert] = []
        self.max_history = 1000
        
        # Statistics
        self.stats = {
            "total_sent": 0,
            "total_failed": 0,
            "by_severity": {sev: 0 for sev in AlertSeverity},
            "by_type": {typ: 0 for typ in AlertType},
        }
        
        # Async processing
        self._alert_queue: Optional[queue.Queue] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Callbacks
        self._callbacks: List[Callable[[Alert], None]] = []
        
        if self.async_mode:
            self._start_async_worker()
    
    def add_alerter(self, alerter: BaseAlerter) -> None:
        """
        Add an alerter to the manager.
        
        Args:
            alerter: Alerter instance to add
        """
        if not isinstance(alerter, BaseAlerter):
            raise TypeError("alerter must be an instance of BaseAlerter")
        
        self.alerters.append(alerter)
        logger.info(f"Added alerter: {alerter.name}")
    
    def remove_alerter(self, name: str) -> bool:
        """
        Remove an alerter by name.
        
        Args:
            name: Name of alerter to remove
            
        Returns:
            True if removed, False if not found
        """
        initial_count = len(self.alerters)
        self.alerters = [a for a in self.alerters if a.name != name]
        removed = len(self.alerters) < initial_count
        
        if removed:
            logger.info(f"Removed alerter: {name}")
        
        return removed
    
    def get_alerter(self, name: str) -> Optional[BaseAlerter]:
        """
        Get alerter by name.
        
        Args:
            name: Alerter name
            
        Returns:
            Alerter instance or None if not found
        """
        for alerter in self.alerters:
            if alerter.name == name:
                return alerter
        return None
    
    def send(self, alert: Alert) -> bool:
        """
        Send an alert through all configured alerters.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent to at least one alerter successfully
        """
        if self.async_mode:
            return self._send_async(alert)
        else:
            return self._send_sync(alert)
    
    def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        alert_type: AlertType = AlertType.CUSTOM,
        metadata: Optional[Dict[str, Any]] = None,
        channels: Optional[List[AlertChannel]] = None,
    ) -> bool:
        """
        Create and send an alert.
        
        Convenience method for creating and sending alerts in one call.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            alert_type: Type of alert
            metadata: Additional metadata
            channels: Specific channels to send to
            
        Returns:
            True if sent successfully
        """
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            alert_type=alert_type,
            metadata=metadata or {},
            channels=channels,
        )
        return self.send(alert)
    
    def _send_sync(self, alert: Alert) -> bool:
        """Send alert synchronously."""
        if not self.alerters:
            logger.warning("No alerters configured")
            return False
        
        # Add to history
        self._add_to_history(alert)
        
        # Update stats
        self._update_stats(alert)
        
        # Trigger callbacks
        self._trigger_callbacks(alert)
        
        # Send to alerters
        success_count = 0
        for alerter in self.alerters:
            if not alerter.enabled:
                continue
            
            try:
                if alerter.send(alert):
                    success_count += 1
            except Exception as e:
                logger.error(f"Error sending alert via {alerter.name}: {e}")
                self.stats["total_failed"] += 1
        
        if success_count > 0:
            self.stats["total_sent"] += 1
            return True
        else:
            return False
    
    def _send_async(self, alert: Alert) -> bool:
        """Send alert asynchronously via queue."""
        if self._alert_queue is None:
            logger.error("Async mode not initialized")
            return False
        
        try:
            self._alert_queue.put_nowait(alert)
            return True
        except queue.Full:
            logger.error("Alert queue is full, dropping alert")
            return False
    
    def _start_async_worker(self) -> None:
        """Start async worker thread."""
        self._alert_queue = queue.Queue(maxsize=self.queue_size)
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._async_worker,
            daemon=True,
            name="AlertWorker",
        )
        self._worker_thread.start()
        logger.info("Started async alert worker")
    
    def _async_worker(self) -> None:
        """Async worker that processes alert queue."""
        while not self._stop_event.is_set():
            try:
                alert = self._alert_queue.get(timeout=0.1)
                self._send_sync(alert)
                self._alert_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in async alert worker: {e}")
    
    def _add_to_history(self, alert: Alert) -> None:
        """Add alert to history."""
        self.alert_history.append(alert)
        
        # Trim history if needed
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
    
    def _update_stats(self, alert: Alert) -> None:
        """Update statistics."""
        self.stats["by_severity"][alert.severity] += 1
        self.stats["by_type"][alert.alert_type] += 1
    
    def _trigger_callbacks(self, alert: Alert) -> None:
        """Trigger registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        Add a callback function to be called when alerts are sent.
        
        Args:
            callback: Function that takes an Alert and returns None
        """
        self._callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get alert statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_sent": self.stats["total_sent"],
            "total_failed": self.stats["total_failed"],
            "success_rate": (
                self.stats["total_sent"] / (self.stats["total_sent"] + self.stats["total_failed"])
                if (self.stats["total_sent"] + self.stats["total_failed"]) > 0
                else 0.0
            ),
            "by_severity": {k.value: v for k, v in self.stats["by_severity"].items()},
            "by_type": {k.value: v for k, v in self.stats["by_type"].items()},
            "alerters_count": len(self.alerters),
            "history_size": len(self.alert_history),
        }
    
    def test_all(self) -> Dict[str, bool]:
        """
        Test all configured alerters.
        
        Returns:
            Dictionary mapping alerter names to test results
        """
        results = {}
        for alerter in self.alerters:
            logger.info(f"Testing alerter: {alerter.name}")
            results[alerter.name] = alerter.test()
        return results
    
    def shutdown(self) -> None:
        """Shutdown the alert manager."""
        if self.async_mode and self._worker_thread:
            logger.info("Shutting down alert manager...")
            self._stop_event.set()
            
            # Wait for queue to empty
            if self._alert_queue:
                self._alert_queue.join()
            
            # Wait for worker thread
            if self._worker_thread:
                self._worker_thread.join(timeout=5.0)
            
            logger.info("Alert manager shut down")
        
        # Disconnect all alerters
        for alerter in self.alerters:
            if alerter.is_connected:
                alerter.disconnect()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    def __repr__(self) -> str:
        return (
            f"AlertManager(alerters={len(self.alerters)}, "
            f"async_mode={self.async_mode}, "
            f"total_sent={self.stats['total_sent']})"
        )


def create_default_manager(
    telegram_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
    include_console: bool = True,
    async_mode: bool = False,
) -> AlertManager:
    """
    Create a default alert manager with common alerters.
    
    Args:
        telegram_token: Telegram bot token (optional)
        telegram_chat_id: Telegram chat ID (optional)
        include_console: Whether to include console alerter
        async_mode: Whether to use async mode
        
    Returns:
        Configured AlertManager instance
        
    Example:
        >>> manager = create_default_manager(
        ...     telegram_token="123456:ABC...",
        ...     telegram_chat_id="123456789",
        ...     include_console=True
        ... )
    """
    from jsf.alerts.console import ConsoleAlerter
    
    manager = AlertManager(async_mode=async_mode)
    
    if include_console:
        manager.add_alerter(ConsoleAlerter(min_severity=AlertSeverity.INFO))
    
    if telegram_token and telegram_chat_id:
        try:
            from jsf.alerts.telegram import TelegramAlerter
            telegram = TelegramAlerter(
                bot_token=telegram_token,
                chat_id=telegram_chat_id,
                min_severity=AlertSeverity.WARNING,
            )
            manager.add_alerter(telegram)
        except ImportError:
            logger.warning("python-telegram-bot not installed, skipping Telegram alerter")
    
    return manager

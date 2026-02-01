"""
Alert integration helpers for broker and live trading.

This module provides helper functions to integrate alerts with
trading systems, brokers, and engines.
"""

from typing import Optional
from decimal import Decimal

from jsf.alerts import Alert, AlertType, AlertSeverity, AlertManager
from jsf.broker.models import Order, OrderStatus, Fill, Position


def create_order_alert(
    order: Order,
    status: OrderStatus,
    message: Optional[str] = None,
) -> Alert:
    """
    Create an alert for order events.
    
    Args:
        order: Order object
        status: Order status
        message: Custom message (optional)
        
    Returns:
        Alert object
    """
    # Determine alert type and severity
    alert_type_map = {
        OrderStatus.SUBMITTED: AlertType.ORDER_SUBMITTED,
        OrderStatus.FILLED: AlertType.ORDER_FILLED,
        OrderStatus.CANCELLED: AlertType.ORDER_CANCELLED,
        OrderStatus.REJECTED: AlertType.ORDER_REJECTED,
    }
    
    severity_map = {
        OrderStatus.SUBMITTED: AlertSeverity.INFO,
        OrderStatus.FILLED: AlertSeverity.INFO,
        OrderStatus.CANCELLED: AlertSeverity.WARNING,
        OrderStatus.REJECTED: AlertSeverity.ERROR,
    }
    
    alert_type = alert_type_map.get(status, AlertType.CUSTOM)
    severity = severity_map.get(status, AlertSeverity.INFO)
    
    # Create title
    title_map = {
        OrderStatus.SUBMITTED: "Order Submitted",
        OrderStatus.FILLED: "Order Filled",
        OrderStatus.CANCELLED: "Order Cancelled",
        OrderStatus.REJECTED: "Order Rejected",
    }
    title = title_map.get(status, "Order Update")
    
    # Create message
    if message is None:
        side_str = order.side.value.upper()
        
        if status == OrderStatus.FILLED:
            price_str = f"${order.avg_fill_price:.2f}" if order.avg_fill_price else "market"
            message = f"{side_str} {order.filled_quantity:.0f} shares of {order.symbol} @ {price_str}"
        elif status == OrderStatus.SUBMITTED:
            if order.order_type.value == "market":
                message = f"{side_str} {order.quantity:.0f} shares of {order.symbol} (MARKET)"
            elif order.order_type.value == "limit":
                message = f"{side_str} {order.quantity:.0f} shares of {order.symbol} @ ${order.limit_price:.2f}"
            else:
                message = f"{side_str} {order.quantity:.0f} shares of {order.symbol}"
        else:
            message = f"{side_str} order for {order.quantity:.0f} shares of {order.symbol}"
    
    # Create metadata
    metadata = {
        "Symbol": order.symbol,
        "Side": order.side.value,
        "Quantity": f"{order.quantity:.0f}",
        "Order Type": order.order_type.value,
        "Order ID": order.order_id or "N/A",
    }
    
    if order.limit_price:
        metadata["Limit Price"] = f"${order.limit_price:.2f}"
    
    if order.avg_fill_price and status == OrderStatus.FILLED:
        metadata["Fill Price"] = f"${order.avg_fill_price:.2f}"
        total_value = order.filled_quantity * order.avg_fill_price
        metadata["Total Value"] = f"${total_value:,.2f}"
    
    return Alert(
        title=title,
        message=message,
        severity=severity,
        alert_type=alert_type,
        metadata=metadata,
    )


def create_position_alert(
    position: Position,
    action: str,
    message: Optional[str] = None,
) -> Alert:
    """
    Create an alert for position events.
    
    Args:
        position: Position object
        action: Action type ('opened', 'closed', 'updated')
        message: Custom message (optional)
        
    Returns:
        Alert object
    """
    action_titles = {
        "opened": "Position Opened",
        "closed": "Position Closed",
        "updated": "Position Updated",
    }
    
    title = action_titles.get(action, "Position Update")
    
    if message is None:
        if action == "opened":
            message = f"Opened {position.side.value} position in {position.symbol}: {position.quantity:.0f} shares @ ${position.avg_cost:.2f}"
        elif action == "closed":
            pnl_str = f"+${position.realized_pnl:.2f}" if position.realized_pnl >= 0 else f"-${abs(position.realized_pnl):.2f}"
            message = f"Closed {position.side.value} position in {position.symbol}. P&L: {pnl_str}"
        else:
            message = f"Updated {position.side.value} position in {position.symbol}: {position.quantity:.0f} shares"
    
    metadata = {
        "Symbol": position.symbol,
        "Side": position.side.value,
        "Quantity": f"{position.quantity:.0f}",
        "Avg Cost": f"${position.avg_cost:.2f}",
        "Market Value": f"${position.market_value:.2f}",
        "Unrealized P&L": f"${position.unrealized_pnl:.2f}",
    }
    
    if position.realized_pnl != 0:
        metadata["Realized P&L"] = f"${position.realized_pnl:.2f}"
    
    severity = AlertSeverity.INFO
    alert_type = AlertType.TRADE_OPENED if action == "opened" else AlertType.TRADE_CLOSED
    
    return Alert(
        title=title,
        message=message,
        severity=severity,
        alert_type=alert_type,
        metadata=metadata,
    )


def create_risk_alert(
    alert_type: AlertType,
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.WARNING,
    **metadata,
) -> Alert:
    """
    Create a risk management alert.
    
    Args:
        alert_type: Type of alert
        title: Alert title
        message: Alert message
        severity: Alert severity
        **metadata: Additional metadata
        
    Returns:
        Alert object
    """
    return Alert(
        title=title,
        message=message,
        severity=severity,
        alert_type=alert_type,
        metadata=metadata,
    )


def create_strategy_alert(
    strategy_name: str,
    event: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.INFO,
    **metadata,
) -> Alert:
    """
    Create a strategy event alert.
    
    Args:
        strategy_name: Name of the strategy
        event: Event type ('started', 'stopped', 'signal', 'rebalance')
        message: Alert message
        severity: Alert severity
        **metadata: Additional metadata
        
    Returns:
        Alert object
    """
    event_types = {
        "started": AlertType.STRATEGY_STARTED,
        "stopped": AlertType.STRATEGY_STOPPED,
        "signal": AlertType.SIGNAL_GENERATED,
        "rebalance": AlertType.REBALANCE_TRIGGERED,
    }
    
    event_titles = {
        "started": "Strategy Started",
        "stopped": "Strategy Stopped",
        "signal": "Signal Generated",
        "rebalance": "Rebalance Triggered",
    }
    
    alert_type = event_types.get(event, AlertType.CUSTOM)
    title = f"{strategy_name}: {event_titles.get(event, 'Event')}"
    
    metadata["Strategy"] = strategy_name
    
    return Alert(
        title=title,
        message=message,
        severity=severity,
        alert_type=alert_type,
        metadata=metadata,
    )


def create_system_alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.ERROR,
    **metadata,
) -> Alert:
    """
    Create a system/error alert.
    
    Args:
        title: Alert title
        message: Alert message
        severity: Alert severity
        **metadata: Additional metadata
        
    Returns:
        Alert object
    """
    return Alert(
        title=title,
        message=message,
        severity=severity,
        alert_type=AlertType.SYSTEM_ERROR,
        metadata=metadata,
    )

"""Dashboard Data Models

Core data structures for the monitoring dashboard.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import pandas as pd


class DashboardPage(Enum):
    """Available dashboard pages."""
    OVERVIEW = "overview"
    PORTFOLIO = "portfolio"
    PNL = "pnl"
    TRADES = "trades"
    RISK = "risk"
    SETTINGS = "settings"


class RefreshRate(Enum):
    """Dashboard refresh rate options."""
    MANUAL = 0
    FAST = 1  # 1 second
    NORMAL = 5  # 5 seconds
    SLOW = 15  # 15 seconds
    MINUTE = 60  # 1 minute


@dataclass
class DashboardConfig:
    """Configuration for dashboard behavior.
    
    Attributes:
        refresh_rate: Auto-refresh interval in seconds
        theme: Dashboard theme ('light' or 'dark')
        show_notifications: Whether to show trade notifications
        max_trade_history: Maximum number of trades to display
        chart_height: Default chart height in pixels
        decimal_places: Decimal places for currency display
        date_format: Date format string for display
        timezone: Timezone for time display
    """
    refresh_rate: RefreshRate = RefreshRate.NORMAL
    theme: str = "light"
    show_notifications: bool = True
    max_trade_history: int = 100
    chart_height: int = 400
    decimal_places: int = 2
    date_format: str = "%Y-%m-%d %H:%M:%S"
    timezone: str = "UTC"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "refresh_rate": self.refresh_rate.value,
            "theme": self.theme,
            "show_notifications": self.show_notifications,
            "max_trade_history": self.max_trade_history,
            "chart_height": self.chart_height,
            "decimal_places": self.decimal_places,
            "date_format": self.date_format,
            "timezone": self.timezone,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DashboardConfig":
        """Create config from dictionary."""
        refresh_value = data.get("refresh_rate", 5)
        refresh_rate = RefreshRate(refresh_value) if isinstance(refresh_value, int) else RefreshRate.NORMAL
        
        return cls(
            refresh_rate=refresh_rate,
            theme=data.get("theme", "light"),
            show_notifications=data.get("show_notifications", True),
            max_trade_history=data.get("max_trade_history", 100),
            chart_height=data.get("chart_height", 400),
            decimal_places=data.get("decimal_places", 2),
            date_format=data.get("date_format", "%Y-%m-%d %H:%M:%S"),
            timezone=data.get("timezone", "UTC"),
        )


@dataclass
class PositionSnapshot:
    """Snapshot of a single position.
    
    Attributes:
        symbol: Ticker symbol
        quantity: Number of shares
        avg_cost: Average cost per share
        current_price: Current market price
        market_value: Current market value (quantity * current_price)
        unrealized_pnl: Unrealized profit/loss
        unrealized_pnl_pct: Unrealized P&L as percentage
        weight: Position weight in portfolio
        side: 'long' or 'short'
    """
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight: float
    side: str = "long"
    
    @classmethod
    def from_position(cls, position: Any, current_price: float, total_equity: float) -> "PositionSnapshot":
        """Create snapshot from broker Position object."""
        market_value = position.quantity * current_price
        cost_basis = position.quantity * position.avg_cost
        unrealized_pnl = market_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis != 0 else 0.0
        weight = (market_value / total_equity * 100) if total_equity != 0 else 0.0
        side = "long" if position.quantity > 0 else "short"
        
        return cls(
            symbol=position.symbol,
            quantity=abs(position.quantity),
            avg_cost=position.avg_cost,
            current_price=current_price,
            market_value=abs(market_value),
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            weight=abs(weight),
            side=side,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "Symbol": self.symbol,
            "Qty": self.quantity,
            "Avg Cost": self.avg_cost,
            "Price": self.current_price,
            "Market Value": self.market_value,
            "P&L": self.unrealized_pnl,
            "P&L %": self.unrealized_pnl_pct,
            "Weight %": self.weight,
            "Side": self.side,
        }


@dataclass
class PortfolioSnapshot:
    """Point-in-time snapshot of entire portfolio.
    
    Attributes:
        timestamp: When snapshot was taken
        cash: Available cash
        portfolio_value: Total value of positions
        equity: Total equity (cash + portfolio_value)
        positions: List of position snapshots
        num_positions: Number of open positions
        daily_pnl: P&L for current day
        daily_return: Return for current day as percentage
        total_pnl: Total P&L since inception
        total_return: Total return since inception as percentage
    """
    timestamp: datetime
    cash: float
    portfolio_value: float
    equity: float
    positions: List[PositionSnapshot]
    num_positions: int
    daily_pnl: float = 0.0
    daily_return: float = 0.0
    total_pnl: float = 0.0
    total_return: float = 0.0
    
    @property
    def cash_weight(self) -> float:
        """Cash as percentage of equity."""
        return (self.cash / self.equity * 100) if self.equity != 0 else 100.0
    
    @property
    def invested_weight(self) -> float:
        """Invested amount as percentage of equity."""
        return 100.0 - self.cash_weight
    
    def get_allocation_data(self) -> pd.DataFrame:
        """Get allocation data for charts."""
        data = []
        for pos in self.positions:
            data.append({
                "Asset": pos.symbol,
                "Value": pos.market_value,
                "Weight": pos.weight,
            })
        # Add cash
        data.append({
            "Asset": "Cash",
            "Value": self.cash,
            "Weight": self.cash_weight,
        })
        return pd.DataFrame(data)
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Get summary metrics as dictionary."""
        return {
            "Timestamp": self.timestamp,
            "Cash": self.cash,
            "Portfolio Value": self.portfolio_value,
            "Equity": self.equity,
            "Positions": self.num_positions,
            "Daily P&L": self.daily_pnl,
            "Daily Return %": self.daily_return,
            "Total P&L": self.total_pnl,
            "Total Return %": self.total_return,
        }


@dataclass
class TradeRecord:
    """Record of a completed trade for display.
    
    Attributes:
        trade_id: Unique trade identifier
        order_id: Associated order ID
        timestamp: When trade was executed
        symbol: Ticker symbol
        side: 'buy' or 'sell'
        quantity: Number of shares traded
        price: Execution price
        value: Total trade value
        commission: Commission paid
        pnl: Realized P&L (for closing trades)
        notes: Optional notes about the trade
    """
    trade_id: str
    order_id: str
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    value: float
    commission: float = 0.0
    pnl: Optional[float] = None
    notes: str = ""
    
    @classmethod
    def from_fill(cls, fill: Any, trade_id: str = None) -> "TradeRecord":
        """Create record from broker Fill object."""
        import uuid
        
        return cls(
            trade_id=trade_id or str(uuid.uuid4())[:8],
            order_id=fill.order_id,
            timestamp=fill.timestamp,
            symbol=fill.symbol,
            side=fill.side.value if hasattr(fill.side, 'value') else str(fill.side),
            quantity=fill.quantity,
            price=fill.price,
            value=fill.quantity * fill.price,
            commission=getattr(fill, 'commission', 0.0),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "Time": self.timestamp,
            "Symbol": self.symbol,
            "Side": self.side.upper(),
            "Qty": self.quantity,
            "Price": self.price,
            "Value": self.value,
            "Commission": self.commission,
            "P&L": self.pnl if self.pnl is not None else "-",
        }


@dataclass 
class RiskMetrics:
    """Calculated risk metrics for display.
    
    Attributes:
        timestamp: When metrics were calculated
        max_drawdown: Maximum drawdown percentage
        current_drawdown: Current drawdown percentage
        var_95: Value at Risk at 95% confidence
        var_99: Value at Risk at 99% confidence
        volatility: Annualized volatility
        beta: Portfolio beta to benchmark
        correlation: Correlation to benchmark
        sharpe_ratio: Sharpe ratio (annualized)
        sortino_ratio: Sortino ratio (annualized)
        calmar_ratio: Calmar ratio
        max_position_size: Largest position as % of equity
        gross_exposure: Total gross exposure
        net_exposure: Net exposure (long - short)
        sector_concentration: Optional sector concentration data
    """
    timestamp: datetime
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    volatility: float = 0.0
    beta: Optional[float] = None
    correlation: Optional[float] = None
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_position_size: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    sector_concentration: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "Max Drawdown %": self.max_drawdown,
            "Current Drawdown %": self.current_drawdown,
            "VaR 95%": self.var_95,
            "VaR 99%": self.var_99,
            "Volatility (Ann.)": self.volatility,
            "Sharpe Ratio": self.sharpe_ratio,
            "Sortino Ratio": self.sortino_ratio,
            "Calmar Ratio": self.calmar_ratio,
            "Max Position %": self.max_position_size,
            "Gross Exposure %": self.gross_exposure,
            "Net Exposure %": self.net_exposure,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics over various time periods.
    
    Attributes:
        timestamp: When metrics were calculated
        return_1d: 1-day return
        return_1w: 1-week return
        return_1m: 1-month return
        return_3m: 3-month return
        return_ytd: Year-to-date return
        return_1y: 1-year return
        return_total: Total return since inception
        best_day: Best single day return
        worst_day: Worst single day return
        win_rate: Percentage of winning days
        profit_factor: Gross profit / gross loss
        avg_win: Average winning trade
        avg_loss: Average losing trade
        largest_win: Largest single win
        largest_loss: Largest single loss
    """
    timestamp: datetime
    return_1d: float = 0.0
    return_1w: float = 0.0
    return_1m: float = 0.0
    return_3m: float = 0.0
    return_ytd: float = 0.0
    return_1y: float = 0.0
    return_total: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    def get_return_series(self) -> Dict[str, float]:
        """Get returns as a series for display."""
        return {
            "1D": self.return_1d,
            "1W": self.return_1w,
            "1M": self.return_1m,
            "3M": self.return_3m,
            "YTD": self.return_ytd,
            "1Y": self.return_1y,
            "Total": self.return_total,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "1D Return %": self.return_1d,
            "1W Return %": self.return_1w,
            "1M Return %": self.return_1m,
            "3M Return %": self.return_3m,
            "YTD Return %": self.return_ytd,
            "1Y Return %": self.return_1y,
            "Total Return %": self.return_total,
            "Best Day %": self.best_day,
            "Worst Day %": self.worst_day,
            "Win Rate %": self.win_rate,
            "Profit Factor": self.profit_factor,
        }


@dataclass
class DashboardState:
    """Current state of the dashboard session.
    
    Maintains all data needed for dashboard display including
    current snapshot, historical data, and computed metrics.
    
    Attributes:
        config: Dashboard configuration
        current_page: Currently active page
        is_connected: Whether connected to broker/engine
        last_update: Timestamp of last data update
        initial_capital: Starting capital
        current_snapshot: Most recent portfolio snapshot
        trade_history: List of recent trades
        risk_metrics: Current risk metrics
        performance_metrics: Current performance metrics
        equity_history: Historical equity values
        alerts: Active alerts/notifications
    """
    config: DashboardConfig = field(default_factory=DashboardConfig)
    current_page: DashboardPage = DashboardPage.OVERVIEW
    is_connected: bool = False
    last_update: Optional[datetime] = None
    initial_capital: float = 100000.0
    current_snapshot: Optional[PortfolioSnapshot] = None
    trade_history: List[TradeRecord] = field(default_factory=list)
    risk_metrics: Optional[RiskMetrics] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    equity_history: List[tuple] = field(default_factory=list)  # [(timestamp, equity), ...]
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_alert(self, message: str, level: str = "info"):
        """Add an alert notification."""
        self.alerts.append({
            "timestamp": datetime.now(),
            "message": message,
            "level": level,  # info, warning, error, success
        })
        # Keep only last 50 alerts
        if len(self.alerts) > 50:
            self.alerts = self.alerts[-50:]
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()
    
    def update_equity_history(self, timestamp: datetime, equity: float):
        """Add equity data point to history."""
        self.equity_history.append((timestamp, equity))
    
    def get_equity_series(self) -> pd.Series:
        """Get equity history as pandas Series."""
        if not self.equity_history:
            return pd.Series(dtype=float)
        
        timestamps, values = zip(*self.equity_history)
        return pd.Series(values, index=pd.DatetimeIndex(timestamps), name="Equity")
    
    def get_returns_series(self) -> pd.Series:
        """Get returns history as pandas Series."""
        equity = self.get_equity_series()
        if len(equity) < 2:
            return pd.Series(dtype=float)
        return equity.pct_change().dropna()


# Type aliases for convenience
PositionList = List[PositionSnapshot]
TradeList = List[TradeRecord]
EquityHistory = List[tuple]

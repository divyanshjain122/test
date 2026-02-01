"""Tests for Dashboard Module

Tests for models, collectors, and metrics calculations.
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock

from jsf.dashboard.models import (
    DashboardConfig,
    DashboardState,
    DashboardPage,
    RefreshRate,
    PositionSnapshot,
    PortfolioSnapshot,
    TradeRecord,
    RiskMetrics,
    PerformanceMetrics,
)
from jsf.dashboard.collectors import (
    SnapshotHistory,
    DataCollector,
    MockDataCollector,
)
from jsf.dashboard.metrics import (
    MetricsCalculator,
    calculate_returns,
    calculate_drawdown,
    calculate_volatility,
    calculate_sharpe,
    calculate_sortino,
    calculate_var,
    calculate_win_rate,
    calculate_profit_factor,
)


# =============================================================================
# Model Tests
# =============================================================================

class TestDashboardConfig:
    """Tests for DashboardConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DashboardConfig()
        
        assert config.refresh_rate == RefreshRate.NORMAL
        assert config.theme == "light"
        assert config.show_notifications is True
        assert config.max_trade_history == 100
        assert config.chart_height == 400
        assert config.decimal_places == 2
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = DashboardConfig(
            refresh_rate=RefreshRate.FAST,
            theme="dark",
        )
        
        d = config.to_dict()
        
        assert d["refresh_rate"] == 1
        assert d["theme"] == "dark"
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        d = {
            "refresh_rate": 15,
            "theme": "dark",
            "max_trade_history": 500,
        }
        
        config = DashboardConfig.from_dict(d)
        
        assert config.refresh_rate == RefreshRate.SLOW
        assert config.theme == "dark"
        assert config.max_trade_history == 500


class TestPositionSnapshot:
    """Tests for PositionSnapshot."""
    
    def test_position_snapshot_creation(self):
        """Test creating a position snapshot."""
        pos = PositionSnapshot(
            symbol="AAPL",
            quantity=100,
            avg_cost=150.0,
            current_price=160.0,
            market_value=16000.0,
            unrealized_pnl=1000.0,
            unrealized_pnl_pct=6.67,
            weight=10.0,
        )
        
        assert pos.symbol == "AAPL"
        assert pos.quantity == 100
        assert pos.unrealized_pnl == 1000.0
        assert pos.side == "long"
    
    def test_position_snapshot_from_position(self):
        """Test creating snapshot from broker Position."""
        # Mock position object
        mock_pos = Mock()
        mock_pos.symbol = "GOOGL"
        mock_pos.quantity = 50
        mock_pos.avg_cost = 140.0
        
        snapshot = PositionSnapshot.from_position(
            mock_pos,
            current_price=150.0,
            total_equity=100000.0,
        )
        
        assert snapshot.symbol == "GOOGL"
        assert snapshot.quantity == 50
        assert snapshot.current_price == 150.0
        assert snapshot.market_value == 7500.0
        assert snapshot.unrealized_pnl == 500.0  # (150-140) * 50
    
    def test_position_to_dict(self):
        """Test converting position to dictionary."""
        pos = PositionSnapshot(
            symbol="MSFT",
            quantity=25,
            avg_cost=380.0,
            current_price=400.0,
            market_value=10000.0,
            unrealized_pnl=500.0,
            unrealized_pnl_pct=5.26,
            weight=5.0,
        )
        
        d = pos.to_dict()
        
        assert d["Symbol"] == "MSFT"
        assert d["Qty"] == 25
        assert d["P&L"] == 500.0


class TestPortfolioSnapshot:
    """Tests for PortfolioSnapshot."""
    
    def test_portfolio_snapshot_creation(self):
        """Test creating a portfolio snapshot."""
        positions = [
            PositionSnapshot(
                symbol="AAPL",
                quantity=100,
                avg_cost=150.0,
                current_price=160.0,
                market_value=16000.0,
                unrealized_pnl=1000.0,
                unrealized_pnl_pct=6.67,
                weight=16.0,
            )
        ]
        
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            cash=34000.0,
            portfolio_value=66000.0,
            equity=100000.0,
            positions=positions,
            num_positions=1,
            daily_pnl=500.0,
            daily_return=0.5,
        )
        
        assert snapshot.equity == 100000.0
        assert snapshot.num_positions == 1
        assert snapshot.cash_weight == 34.0
        assert snapshot.invested_weight == 66.0
    
    def test_portfolio_allocation_data(self):
        """Test getting allocation data for charts."""
        positions = [
            PositionSnapshot("AAPL", 100, 150, 160, 16000, 1000, 6.67, 16.0),
            PositionSnapshot("GOOGL", 50, 140, 150, 7500, 500, 7.14, 7.5),
        ]
        
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            cash=76500.0,
            portfolio_value=23500.0,
            equity=100000.0,
            positions=positions,
            num_positions=2,
        )
        
        allocation = snapshot.get_allocation_data()
        
        assert len(allocation) == 3  # 2 positions + cash
        assert "Cash" in allocation["Asset"].values


class TestTradeRecord:
    """Tests for TradeRecord."""
    
    def test_trade_record_creation(self):
        """Test creating a trade record."""
        trade = TradeRecord(
            trade_id="T001",
            order_id="O001",
            timestamp=datetime.now(),
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
            value=15000.0,
            commission=5.0,
        )
        
        assert trade.symbol == "AAPL"
        assert trade.value == 15000.0
        assert trade.pnl is None
    
    def test_trade_to_dict(self):
        """Test converting trade to dictionary."""
        trade = TradeRecord(
            trade_id="T002",
            order_id="O002",
            timestamp=datetime.now(),
            symbol="GOOGL",
            side="sell",
            quantity=50,
            price=150.0,
            value=7500.0,
            pnl=250.0,
        )
        
        d = trade.to_dict()
        
        assert d["Symbol"] == "GOOGL"
        assert d["Side"] == "SELL"
        assert d["P&L"] == 250.0


class TestRiskMetrics:
    """Tests for RiskMetrics."""
    
    def test_risk_metrics_creation(self):
        """Test creating risk metrics."""
        metrics = RiskMetrics(
            timestamp=datetime.now(),
            max_drawdown=5.0,
            current_drawdown=2.0,
            var_95=1.5,
            volatility=15.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
        )
        
        assert metrics.max_drawdown == 5.0
        assert metrics.sharpe_ratio == 1.5
    
    def test_risk_metrics_to_dict(self):
        """Test converting to dictionary."""
        metrics = RiskMetrics(
            timestamp=datetime.now(),
            max_drawdown=10.0,
            sharpe_ratio=1.2,
        )
        
        d = metrics.to_dict()
        
        assert "Max Drawdown %" in d
        assert d["Sharpe Ratio"] == 1.2


class TestDashboardState:
    """Tests for DashboardState."""
    
    def test_state_creation(self):
        """Test creating dashboard state."""
        state = DashboardState()
        
        assert state.current_page == DashboardPage.OVERVIEW
        assert state.is_connected is False
        assert len(state.alerts) == 0
    
    def test_add_alert(self):
        """Test adding alerts."""
        state = DashboardState()
        
        state.add_alert("Test message", "info")
        state.add_alert("Warning message", "warning")
        
        assert len(state.alerts) == 2
        assert state.alerts[0]["message"] == "Test message"
        assert state.alerts[1]["level"] == "warning"
    
    def test_clear_alerts(self):
        """Test clearing alerts."""
        state = DashboardState()
        state.add_alert("Test", "info")
        
        state.clear_alerts()
        
        assert len(state.alerts) == 0
    
    def test_equity_history(self):
        """Test equity history management."""
        state = DashboardState()
        
        now = datetime.now()
        state.update_equity_history(now, 100000.0)
        state.update_equity_history(now + timedelta(seconds=1), 100500.0)
        
        series = state.get_equity_series()
        
        assert len(series) == 2
        assert series.iloc[0] == 100000.0
        assert series.iloc[1] == 100500.0


# =============================================================================
# Collector Tests
# =============================================================================

class TestSnapshotHistory:
    """Tests for SnapshotHistory."""
    
    def test_snapshot_history_creation(self):
        """Test creating snapshot history."""
        history = SnapshotHistory(max_snapshots=100)
        
        assert len(history) == 0
        assert history.max_snapshots == 100
    
    def test_add_snapshot(self):
        """Test adding snapshots."""
        history = SnapshotHistory()
        
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            cash=50000.0,
            portfolio_value=50000.0,
            equity=100000.0,
            positions=[],
            num_positions=0,
        )
        
        history.add(snapshot)
        
        assert len(history) == 1
    
    def test_get_equity_series(self):
        """Test getting equity series."""
        history = SnapshotHistory()
        
        now = datetime.now()
        for i in range(5):
            snapshot = PortfolioSnapshot(
                timestamp=now + timedelta(seconds=i),
                cash=50000.0,
                portfolio_value=50000.0 + i * 100,
                equity=100000.0 + i * 100,
                positions=[],
                num_positions=0,
            )
            history.add(snapshot)
        
        series = history.get_equity_series()
        
        assert len(series) == 5
        assert series.iloc[0] == 100000.0
        assert series.iloc[4] == 100400.0
    
    def test_get_latest(self):
        """Test getting latest snapshots."""
        history = SnapshotHistory()
        
        now = datetime.now()
        for i in range(10):
            snapshot = PortfolioSnapshot(
                timestamp=now + timedelta(seconds=i),
                cash=50000.0,
                portfolio_value=50000.0,
                equity=100000.0 + i,
                positions=[],
                num_positions=0,
            )
            history.add(snapshot)
        
        latest = history.get_latest(3)
        
        assert len(latest) == 3
        assert latest[-1].equity == 100009.0
    
    def test_clear(self):
        """Test clearing history."""
        history = SnapshotHistory()
        
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            cash=50000.0,
            portfolio_value=50000.0,
            equity=100000.0,
            positions=[],
            num_positions=0,
        )
        history.add(snapshot)
        
        history.clear()
        
        assert len(history) == 0


class TestMockDataCollector:
    """Tests for MockDataCollector."""
    
    def test_mock_collector_creation(self):
        """Test creating mock collector."""
        collector = MockDataCollector(
            initial_capital=100000.0,
            symbols=["AAPL", "GOOGL"],
        )
        
        assert collector.initial_capital == 100000.0
        assert len(collector.symbols) == 2
    
    def test_collect_snapshot(self):
        """Test collecting mock snapshot."""
        collector = MockDataCollector(initial_capital=50000.0)
        
        snapshot = collector.collect_snapshot()
        
        assert snapshot is not None
        assert snapshot.equity > 0
        assert len(snapshot.positions) > 0
    
    def test_multiple_snapshots(self):
        """Test collecting multiple snapshots."""
        collector = MockDataCollector(initial_capital=100000.0)
        
        for _ in range(10):
            collector.collect_snapshot()
        
        assert len(collector.history) == 10
        
        series = collector.history.get_equity_series()
        assert len(series) == 10


# =============================================================================
# Metrics Tests
# =============================================================================

class TestCalculateReturns:
    """Tests for calculate_returns."""
    
    def test_simple_returns(self):
        """Test calculating simple returns."""
        prices = pd.Series([100, 102, 101, 105, 103])
        
        returns = calculate_returns(prices)
        
        assert len(returns) == 4
        assert abs(returns.iloc[0] - 0.02) < 0.001  # 2% return
    
    def test_empty_series(self):
        """Test with empty series."""
        prices = pd.Series(dtype=float)
        
        returns = calculate_returns(prices)
        
        assert len(returns) == 0


class TestCalculateDrawdown:
    """Tests for calculate_drawdown."""
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        equity = pd.Series([100, 105, 102, 108, 100, 110])
        
        dd_series, max_dd, current_dd = calculate_drawdown(equity)
        
        assert max_dd > 0
        assert len(dd_series) == len(equity)
    
    def test_no_drawdown(self):
        """Test with monotonically increasing equity."""
        equity = pd.Series([100, 101, 102, 103, 104])
        
        dd_series, max_dd, current_dd = calculate_drawdown(equity)
        
        assert max_dd == 0
        assert current_dd == 0
    
    def test_empty_series(self):
        """Test with empty series."""
        equity = pd.Series(dtype=float)
        
        dd_series, max_dd, current_dd = calculate_drawdown(equity)
        
        assert max_dd == 0
        assert current_dd == 0


class TestCalculateVolatility:
    """Tests for calculate_volatility."""
    
    def test_volatility_calculation(self):
        """Test volatility calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 100))
        
        vol = calculate_volatility(returns, annualize=True)
        
        assert vol > 0
    
    def test_zero_volatility(self):
        """Test with constant returns."""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])
        
        vol = calculate_volatility(returns, annualize=False)
        
        assert vol == 0.0


class TestCalculateSharpe:
    """Tests for calculate_sharpe."""
    
    def test_sharpe_calculation(self):
        """Test Sharpe ratio calculation."""
        np.random.seed(42)
        # Generate returns with positive mean
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        
        sharpe = calculate_sharpe(returns, risk_free_rate=0.02)
        
        # Should be a reasonable value
        assert isinstance(sharpe, float)
    
    def test_zero_volatility_sharpe(self):
        """Test Sharpe with zero volatility."""
        returns = pd.Series([0.001, 0.001, 0.001])
        
        sharpe = calculate_sharpe(returns)
        
        # With constant returns and zero std, Sharpe can be undefined
        # Our implementation returns 0 when std is 0
        assert isinstance(sharpe, float)
    
    def test_empty_returns(self):
        """Test with empty returns."""
        returns = pd.Series(dtype=float)
        
        sharpe = calculate_sharpe(returns)
        
        assert sharpe == 0.0


class TestCalculateSortino:
    """Tests for calculate_sortino."""
    
    def test_sortino_calculation(self):
        """Test Sortino ratio calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))
        
        sortino = calculate_sortino(returns)
        
        assert isinstance(sortino, float)
    
    def test_no_downside(self):
        """Test with no downside returns."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.015])
        
        sortino = calculate_sortino(returns)
        
        # Should be infinity or very high
        assert sortino == float('inf') or sortino > 10


class TestCalculateVar:
    """Tests for calculate_var."""
    
    def test_var_95(self):
        """Test VaR at 95% confidence."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        
        var = calculate_var(returns, confidence=0.95)
        
        assert var > 0
    
    def test_var_99(self):
        """Test VaR at 99% confidence."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        
        var_99 = calculate_var(returns, confidence=0.99)
        var_95 = calculate_var(returns, confidence=0.95)
        
        # 99% VaR should be higher than 95%
        assert var_99 > var_95


class TestCalculateWinRate:
    """Tests for calculate_win_rate."""
    
    def test_win_rate(self):
        """Test win rate calculation."""
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015, 0.005])
        
        win_rate = calculate_win_rate(returns)
        
        # 4 wins out of 6 = 66.67%
        assert abs(win_rate - 66.67) < 0.1
    
    def test_all_wins(self):
        """Test with all winning days."""
        returns = pd.Series([0.01, 0.02, 0.015])
        
        win_rate = calculate_win_rate(returns)
        
        assert win_rate == 100.0


class TestCalculateProfitFactor:
    """Tests for calculate_profit_factor."""
    
    def test_profit_factor(self):
        """Test profit factor calculation."""
        returns = pd.Series([0.02, -0.01, 0.03, -0.02])
        
        pf = calculate_profit_factor(returns)
        
        # Gains: 0.05, Losses: 0.03 => PF = 1.67
        assert abs(pf - 1.67) < 0.1
    
    def test_no_losses(self):
        """Test with no losses."""
        returns = pd.Series([0.01, 0.02, 0.015])
        
        pf = calculate_profit_factor(returns)
        
        assert pf == float('inf')


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""
    
    def test_calculator_creation(self):
        """Test creating calculator."""
        calculator = MetricsCalculator(risk_free_rate=0.03)
        
        assert calculator.risk_free_rate == 0.03
    
    def test_calculate_risk_metrics(self):
        """Test calculating comprehensive risk metrics."""
        calculator = MetricsCalculator()
        
        # Generate sample equity curve
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        equity = pd.Series(
            100000 * (1 + np.random.normal(0.0005, 0.01, 100)).cumprod(),
            index=dates
        )
        
        metrics = calculator.calculate_risk_metrics(equity)
        
        assert isinstance(metrics, RiskMetrics)
        assert metrics.max_drawdown >= 0
        assert isinstance(metrics.sharpe_ratio, float)
    
    def test_calculate_performance_metrics(self):
        """Test calculating performance metrics."""
        calculator = MetricsCalculator()
        
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        equity = pd.Series(
            100000 * (1 + np.random.normal(0.001, 0.01, 100)).cumprod(),
            index=dates
        )
        
        perf = calculator.calculate_performance_metrics(equity, initial_capital=100000)
        
        assert isinstance(perf, PerformanceMetrics)
        assert isinstance(perf.return_total, float)
    
    def test_rolling_metrics(self):
        """Test rolling metrics calculation."""
        calculator = MetricsCalculator()
        
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        equity = pd.Series(
            100000 * (1 + np.random.normal(0.001, 0.01, 100)).cumprod(),
            index=dates
        )
        
        rolling = calculator.get_rolling_metrics(equity, window=20)
        
        assert not rolling.empty
        assert 'Volatility' in rolling.columns
        assert 'Sharpe' in rolling.columns
    
    def test_monthly_returns(self):
        """Test monthly returns matrix."""
        calculator = MetricsCalculator()
        
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=365, freq='D')
        equity = pd.Series(
            100000 * (1 + np.random.normal(0.001, 0.01, 365)).cumprod(),
            index=dates
        )
        
        monthly = calculator.get_monthly_returns(equity)
        
        assert not monthly.empty
        # Should have month columns
        assert 'Jan' in monthly.columns or 'Feb' in monthly.columns


# =============================================================================
# Integration Tests
# =============================================================================

class TestCollectorIntegration:
    """Integration tests for collector with metrics."""
    
    def test_collector_metrics_pipeline(self):
        """Test full pipeline from collector to metrics."""
        # Create mock collector
        collector = MockDataCollector(initial_capital=100000.0)
        
        # Collect some snapshots
        for _ in range(50):
            collector.collect_snapshot()
        
        # Get equity series
        equity = collector.history.get_equity_series()
        
        # Calculate metrics
        calculator = MetricsCalculator()
        risk_metrics = calculator.calculate_risk_metrics(equity)
        perf_metrics = calculator.calculate_performance_metrics(equity, 100000.0)
        
        assert risk_metrics is not None
        assert perf_metrics is not None
        assert len(equity) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

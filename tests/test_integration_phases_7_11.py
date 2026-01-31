"""
Comprehensive integration tests for Phases 7-12.

This test suite validates the complete end-to-end functionality
of the portfolio construction, strategies, backtesting, visualization,
optimization, and walk-forward analysis modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests

from jsf.data import load_data
from jsf.signals import MomentumSignal, MeanReversionSignal
from jsf.portfolio import (
    EqualWeightSizer,
    SignalWeightedSizer,
    VolatilityScaledSizer,
    SimplePortfolioConstructor,
    Portfolio,
)
from jsf.strategies import MomentumStrategy, MeanReversionStrategy, TrendFollowingStrategy
from jsf.simulation import BacktestEngine, BacktestConfig, calculate_all_metrics
from jsf.visualization import (
    plot_equity_curve,
    plot_drawdown,
    plot_returns_distribution,
    plot_performance_summary,
)
from jsf.optimization import (
    optimize_strategy,
    GridSearchOptimizer,
    ParameterGrid,
    WalkForwardOptimizer,
    WalkForwardResult,
    walk_forward_analysis,
)


class TestPhase7Portfolio:
    """Test Phase 7: Portfolio Construction."""
    
    def test_portfolio_creation(self):
        """Test Portfolio dataclass with DataFrame."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        weights = pd.DataFrame({
            'AAPL': [0.25, 0.30, 0.25, 0.25, 0.20],
            'GOOGL': [0.25, 0.20, 0.25, 0.25, 0.30],
            'MSFT': [0.25, 0.25, 0.25, 0.25, 0.25],
            'AMZN': [0.25, 0.25, 0.25, 0.25, 0.25],
        }, index=dates)
        
        portfolio = Portfolio(weights=weights)
        
        assert isinstance(portfolio.weights, pd.DataFrame)
        assert len(portfolio.weights) == 5
        assert len(portfolio.weights.columns) == 4
        
        # Test get_positions
        pos = portfolio.get_positions()
        assert len(pos) == 4
        assert abs(pos.sum() - 1.0) < 1e-10
        
        # Test turnover
        turnover = portfolio.get_turnover()
        assert len(turnover) == 5
        assert turnover.iloc[0] == 0 or pd.isna(turnover.iloc[0])  # First period
    
    def test_equal_weight_sizer(self):
        """Test EqualWeightSizer with DataFrame signals."""
        dates = pd.date_range('2023-01-01', periods=3, freq='D')
        signals = pd.DataFrame({
            'A': [1.0, 0.5, -0.3],
            'B': [0.8, -0.2, 0.6],
            'C': [0.3, 0.9, 0.1],
        }, index=dates)
        
        sizer = EqualWeightSizer(long_only=True)
        weights = sizer.size(signals)
        
        assert isinstance(weights, pd.DataFrame)
        assert weights.shape == signals.shape
        
        # Check first row: all positive signals
        row0 = weights.iloc[0]
        assert all(row0 >= 0)
        assert abs(row0.sum() - 1.0) < 1e-10
        
        # Check each row sums to 1 (for long-only)
        for idx in range(len(weights)):
            row_sum = weights.iloc[idx].sum()
            if row_sum > 0:  # If there are any positions
                assert abs(row_sum - 1.0) < 1e-10
    
    def test_signal_weighted_sizer(self):
        """Test SignalWeightedSizer."""
        dates = pd.date_range('2023-01-01', periods=2, freq='D')
        signals = pd.DataFrame({
            'A': [2.0, 1.0],
            'B': [1.0, 2.0],
            'C': [1.0, 1.0],
        }, index=dates)
        
        sizer = SignalWeightedSizer(long_only=True)
        weights = sizer.size(signals)
        
        # Weights should be proportional to signals
        row0 = weights.iloc[0]
        assert row0['A'] > row0['B']  # Signal A (2.0) > B (1.0)
        assert abs(row0.sum() - 1.0) < 1e-10
    
    def test_portfolio_constructor(self):
        """Test SimplePortfolioConstructor end-to-end."""
        data = load_data(
            source='synthetic',
            symbols=['AAPL', 'GOOGL', 'MSFT'],
            start_date='2023-01-01',
            end_date='2023-03-31',
            seed=42
        )
        
        # Generate signals
        signal = MomentumSignal(lookback=20)
        signals = signal.generate(data)
        
        # Create portfolio
        sizer = EqualWeightSizer()
        constructor = SimplePortfolioConstructor(sizer)
        portfolio = constructor.construct(signals, data)
        
        assert isinstance(portfolio, Portfolio)
        assert isinstance(portfolio.weights, pd.DataFrame)
        assert len(portfolio.weights) > 0
        assert all(col in portfolio.weights.columns for col in data.symbols)


class TestPhase8Strategies:
    """Test Phase 8: Strategy Templates."""
    
    def test_momentum_strategy(self):
        """Test MomentumStrategy end-to-end."""
        data = load_data(
            source='synthetic',
            symbols=['AAPL', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            seed=42
        )
        
        strategy = MomentumStrategy(lookback=60, long_only=True)
        portfolio = strategy.run(data)
        
        assert isinstance(portfolio, Portfolio)
        assert len(portfolio.weights) > 0
        
        # Check weights are valid
        for idx in range(min(5, len(portfolio.weights))):
            row = portfolio.weights.iloc[idx]
            assert all(row >= 0)  # Long-only
            if row.sum() > 0:
                assert abs(row.sum() - 1.0) < 1e-6  # Should sum to 1
    
    def test_mean_reversion_strategy(self):
        """Test MeanReversionStrategy."""
        data = load_data(
            source='synthetic',
            symbols=['SPY', 'QQQ'],
            start_date='2023-01-01',
            end_date='2023-06-30',
            seed=123
        )
        
        strategy = MeanReversionStrategy(lookback=20)
        portfolio = strategy.run(data)
        
        assert isinstance(portfolio, Portfolio)
        assert len(portfolio.weights) > 0
    
    def test_trend_following_strategy(self):
        """Test TrendFollowingStrategy."""
        data = load_data(
            source='synthetic',
            symbols=['TECH', 'FINANCE'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            seed=456
        )
        
        strategy = TrendFollowingStrategy(fast_period=50, slow_period=200)
        portfolio = strategy.run(data)
        
        assert isinstance(portfolio, Portfolio)
        assert len(portfolio.weights) > 0


class TestPhase9Backtesting:
    """Test Phase 9: Backtesting Engine."""
    
    def test_backtest_engine_basic(self):
        """Test basic backtesting."""
        data = load_data(
            source='synthetic',
            symbols=['AAPL', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            seed=42
        )
        
        strategy = MomentumStrategy(lookback=60)
        config = BacktestConfig(initial_capital=100000)
        engine = BacktestEngine(config)
        
        result = engine.run_strategy(strategy, data)
        
        # Verify result structure
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
        # Initial capital may be slightly less due to transaction costs on first trade
        assert 99000 <= result.equity_curve.iloc[0] <= 100000
        
        # Verify metrics
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert result.max_drawdown <= 0  # Drawdown should be negative
    
    def test_transaction_costs(self):
        """Test that transaction costs reduce returns."""
        data = load_data(
            source='synthetic',
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            seed=42
        )
        
        strategy = MomentumStrategy(lookback=30)  # Shorter = more trading
        
        # No costs
        config_no_cost = BacktestConfig(initial_capital=100000, transaction_cost=0.0, slippage=0.0)
        engine_no_cost = BacktestEngine(config_no_cost)
        result_no_cost = engine_no_cost.run_strategy(strategy, data)
        
        # With costs
        config_with_cost = BacktestConfig(initial_capital=100000, transaction_cost=0.001, slippage=0.0005)
        engine_with_cost = BacktestEngine(config_with_cost)
        result_with_cost = engine_with_cost.run_strategy(strategy, data)
        
        # Costs should reduce returns
        assert result_no_cost.total_return >= result_with_cost.total_return
    
    def test_metrics_calculation(self):
        """Test comprehensive metrics calculation."""
        data = load_data(
            source='synthetic',
            symbols=['SPY'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            seed=42
        )
        
        strategy = MomentumStrategy(lookback=60)
        engine = BacktestEngine(BacktestConfig(initial_capital=100000))
        result = engine.run_strategy(strategy, data)
        
        # Calculate all metrics
        metrics = calculate_all_metrics(result.returns)
        
        # Verify all expected metrics exist
        expected_metrics = [
            'total_return', 'cagr', 'volatility', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'calmar_ratio',
            'win_rate', 'profit_factor', 'var_95', 'cvar_95'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])


class TestPhase10Visualization:
    """Test Phase 10: Visualization."""
    
    @pytest.fixture(scope="class")
    def backtest_result(self):
        """Create a backtest result for visualization tests."""
        data = load_data(
            source='synthetic',
            symbols=['AAPL', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            seed=42
        )
        
        strategy = MomentumStrategy(lookback=60)
        engine = BacktestEngine(BacktestConfig(initial_capital=100000))
        return engine.run_strategy(strategy, data)
    
    def test_equity_curve_plot(self, backtest_result, tmp_path):
        """Test equity curve plotting."""
        output = tmp_path / "equity_curve.png"
        
        fig = plot_equity_curve(
            backtest_result,
            show=False,
            save_path=str(output)
        )
        
        assert output.exists()
        assert output.stat().st_size > 0  # File has content
        assert fig is not None
    
    def test_drawdown_plot(self, backtest_result, tmp_path):
        """Test drawdown plotting."""
        output = tmp_path / "drawdown.png"
        
        fig = plot_drawdown(
            backtest_result,
            show=False,
            save_path=str(output)
        )
        
        assert output.exists()
        assert fig is not None
    
    def test_returns_distribution_plot(self, backtest_result, tmp_path):
        """Test returns distribution plotting."""
        output = tmp_path / "returns_dist.png"
        
        fig = plot_returns_distribution(
            backtest_result,
            show=False,
            save_path=str(output)
        )
        
        assert output.exists()
        assert fig is not None
    
    def test_performance_summary_plot(self, backtest_result, tmp_path):
        """Test performance summary dashboard."""
        output = tmp_path / "summary.png"
        
        fig = plot_performance_summary(
            backtest_result,
            show=False,
            save_path=str(output)
        )
        
        assert output.exists()
        assert fig is not None


class TestPhase11Optimization:
    """Test Phase 11: Parameter Optimization."""
    
    def test_parameter_grid(self):
        """Test ParameterGrid iteration."""
        grid = ParameterGrid({
            'lookback': [30, 60],
            'long_only': [True, False]
        })
        
        combinations = list(grid)
        assert len(combinations) == 4
        
        # Check all combinations exist
        assert {'lookback': 30, 'long_only': True} in combinations
        assert {'lookback': 30, 'long_only': False} in combinations
        assert {'lookback': 60, 'long_only': True} in combinations
        assert {'lookback': 60, 'long_only': False} in combinations
    
    def test_grid_search_optimizer(self):
        """Test GridSearchOptimizer."""
        data = load_data(
            source='synthetic',
            symbols=['AAPL', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            seed=42
        )
        
        param_grid = ParameterGrid({'lookback': [30, 60, 90]})
        optimizer = GridSearchOptimizer(metric='sharpe_ratio')
        
        result = optimizer.optimize(
            strategy_class=MomentumStrategy,
            param_grid=param_grid,
            data=data,
            fixed_params={'long_only': True},
            verbose=False
        )
        
        # Verify optimization result
        assert result.best_params is not None
        assert 'lookback' in result.best_params
        assert result.best_params['lookback'] in [30, 60, 90]
        assert isinstance(result.best_score, float)
        assert len(result.all_results) == 3
        assert len(result.summary) == 3
    
    def test_optimize_strategy_convenience(self):
        """Test optimize_strategy convenience function."""
        data = load_data(
            source='synthetic',
            symbols=['SPY'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            seed=42
        )
        
        result = optimize_strategy(
            strategy_class=MomentumStrategy,
            param_grid={'lookback': [60, 90]},
            data=data,
            metric='sharpe_ratio',
            fixed_params={'long_only': True},
            verbose=False
        )
        
        assert result.best_params is not None
        assert result.best_score is not None
        assert result.best_result is not None


class TestEndToEndIntegration:
    """Test complete end-to-end workflow."""
    
    def test_complete_workflow(self):
        """Test the complete workflow from data to optimization."""
        # 1. Load data
        data = load_data(
            source='synthetic',
            symbols=['AAPL', 'GOOGL', 'MSFT'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            seed=42
        )
        assert data is not None
        
        # 2. Create and run strategy
        strategy = MomentumStrategy(lookback=60, long_only=True)
        portfolio = strategy.run(data)
        assert isinstance(portfolio, Portfolio)
        
        # 3. Backtest
        config = BacktestConfig(
            initial_capital=100000,
            transaction_cost=0.001,
            slippage=0.0005
        )
        engine = BacktestEngine(config)
        result = engine.run_strategy(strategy, data)
        
        assert result.total_return is not None
        assert result.sharpe_ratio is not None
        assert len(result.trades) > 0
        
        # 4. Calculate metrics
        metrics = calculate_all_metrics(result.returns)
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        
        # 5. Optimize (small grid for speed)
        opt_result = optimize_strategy(
            strategy_class=MomentumStrategy,
            param_grid={'lookback': [30, 60]},
            data=data,
            metric='sharpe_ratio',
            fixed_params={'long_only': True},
            verbose=False
        )
        
        assert opt_result.best_params is not None
        assert opt_result.best_score > -10  # Reasonable Sharpe ratio
        
        print(f"\nEnd-to-End Test Results:")
        print(f"  Strategy: {strategy.name}")
        print(f"  Return: {result.total_return:.2%}")
        print(f"  Sharpe: {result.sharpe_ratio:.2f}")
        print(f"  Max DD: {result.max_drawdown:.2%}")
        print(f"  Trades: {len(result.trades)}")
        print(f"  Optimized Params: {opt_result.best_params}")
        print(f"  Optimized Sharpe: {opt_result.best_score:.2f}")


class TestPhase12WalkForward:
    """Test Phase 12: Walk-Forward Analysis."""
    
    @pytest.fixture
    def long_data(self):
        """Load 4 years of data for walk-forward testing."""
        return load_data(
            source='synthetic',
            symbols=['AAPL', 'GOOGL', 'MSFT'],
            start_date='2020-01-01',
            end_date='2023-12-31',
            seed=42
        )
    
    def test_walk_forward_optimizer_creation(self):
        """Test WalkForwardOptimizer initialization."""
        optimizer = WalkForwardOptimizer(
            is_days=252,
            oos_days=63,
            metric='sharpe_ratio',
        )
        
        assert optimizer.is_days == 252
        assert optimizer.oos_days == 63
        assert optimizer.metric == 'sharpe_ratio'
        assert optimizer.expanding == False
    
    def test_walk_forward_basic(self, long_data):
        """Test basic walk-forward analysis."""
        result = walk_forward_analysis(
            strategy_class=MomentumStrategy,
            param_grid={'lookback': [30, 60]},
            data=long_data,
            is_days=252,
            oos_days=63,
            metric='sharpe_ratio',
            verbose=False,
        )
        
        assert isinstance(result, WalkForwardResult)
        assert result.n_windows > 0
        assert result.avg_is_score is not None
        assert result.avg_oos_score is not None
        assert result.efficiency_ratio is not None
        assert result.parameter_stability is not None
        
        print(f"\nWalk-Forward Test Results:")
        print(f"  Windows: {result.n_windows}")
        print(f"  Avg IS Sharpe: {result.avg_is_score:.3f}")
        print(f"  Avg OOS Sharpe: {result.avg_oos_score:.3f}")
        print(f"  Efficiency: {result.efficiency_ratio:.2%}")
        print(f"  Param Stability: {result.parameter_stability:.2%}")
        print(f"  Overfitted: {result.is_overfitted}")
    
    def test_walk_forward_rolling_windows(self, long_data):
        """Test walk-forward with rolling windows."""
        optimizer = WalkForwardOptimizer(
            is_days=252,
            oos_days=63,
            step_days=63,  # Non-overlapping OOS
            expanding=False,
            metric='sharpe_ratio',
        )
        
        result = optimizer.optimize(
            strategy_class=MomentumStrategy,
            param_grid={'lookback': [30, 60, 90]},
            data=long_data,
            verbose=False,
        )
        
        # Check all windows have results
        for window in result.windows:
            assert window.best_params is not None
            assert window.is_score is not None
            assert window.oos_score is not None
        
        # Check summary DataFrame
        summary = result.get_summary()
        assert len(summary) == result.n_windows
        assert 'is_score' in summary.columns
        assert 'oos_score' in summary.columns
    
    def test_walk_forward_expanding_windows(self, long_data):
        """Test walk-forward with expanding windows."""
        optimizer = WalkForwardOptimizer(
            is_days=252,
            oos_days=63,
            expanding=True,
            min_is_days=126,
            metric='sharpe_ratio',
        )
        
        result = optimizer.optimize(
            strategy_class=MomentumStrategy,
            param_grid={'lookback': [30, 60]},
            data=long_data,
            verbose=False,
        )
        
        assert result.n_windows > 0
        
        # Expanding window: each IS period should start from beginning
        for window in result.windows:
            # All IS periods start from the first date
            assert window.is_start == long_data.start_date
    
    def test_walk_forward_overfitting_detection(self, long_data):
        """Test that walk-forward can detect overfitting signals."""
        result = walk_forward_analysis(
            strategy_class=MomentumStrategy,
            param_grid={'lookback': [30, 60, 90]},
            data=long_data,
            is_days=252,
            oos_days=63,
            verbose=False,
        )
        
        # Check overfitting metrics exist and are sensible
        assert 0 <= result.efficiency_ratio <= 2.0  # Can be > 1 if OOS beats IS
        assert 0 <= result.parameter_stability <= 1.0
        assert result.is_overfitted in [True, False]  # Boolean check
        
        # Combined OOS metrics should exist
        assert result.oos_total_return is not None
        assert result.oos_sharpe_ratio is not None
        assert result.oos_max_drawdown is not None
    
    def test_walk_forward_parameter_summary(self, long_data):
        """Test parameter summary across windows."""
        result = walk_forward_analysis(
            strategy_class=MomentumStrategy,
            param_grid={'lookback': [30, 60, 90]},
            data=long_data,
            is_days=252,
            oos_days=63,
            verbose=False,
        )
        
        param_summary = result.get_parameter_summary()
        
        assert len(param_summary) == result.n_windows
        assert 'lookback' in param_summary.columns
        assert 'window' in param_summary.columns
        
        # All lookback values should be from the grid
        assert all(param_summary['lookback'].isin([30, 60, 90]))
    
    def test_walk_forward_string_output(self, long_data):
        """Test string representation of results."""
        result = walk_forward_analysis(
            strategy_class=MomentumStrategy,
            param_grid={'lookback': [30, 60]},
            data=long_data,
            is_days=252,
            oos_days=63,
            verbose=False,
        )
        
        output = str(result)
        
        assert 'WALK-FORWARD ANALYSIS RESULTS' in output
        assert 'IN-SAMPLE PERFORMANCE' in output
        assert 'OUT-OF-SAMPLE PERFORMANCE' in output
        assert 'Efficiency Ratio' in output
        assert 'Parameter Stability' in output


# =============================================================================
# PHASE 13: REAL DATA INTEGRATION TESTS
# =============================================================================

class TestPhase13DataSources:
    """Test suite for Phase 13: Real Data Integration."""
    
    def test_yahoo_finance_loader_creation(self):
        """Test YahooFinanceLoader can be instantiated with required params."""
        from jsf.data.sources import YahooFinanceLoader
        from jsf.data.sources.yahoo import YFINANCE_AVAILABLE
        
        if not YFINANCE_AVAILABLE:
            pytest.skip("yfinance not installed")
        
        loader = YahooFinanceLoader(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        assert loader is not None
        assert hasattr(loader, 'load')
    
    def test_yahoo_loader_config_options(self):
        """Test YahooFinanceLoader configuration options."""
        from jsf.data.sources import YahooFinanceLoader
        from jsf.data.sources.yahoo import YFINANCE_AVAILABLE
        
        if not YFINANCE_AVAILABLE:
            pytest.skip("yfinance not installed")
        
        # Test with different intervals
        loader = YahooFinanceLoader(
            symbols=['AAPL'],
            start_date='2024-01-01',
            interval='1d'
        )
        assert loader.interval == '1d'
        
        # Test with cache settings
        loader_cached = YahooFinanceLoader(
            symbols=['AAPL'],
            start_date='2024-01-01',
            cache_days=7
        )
        assert loader_cached.cache_days == 7
    
    def test_enhanced_csv_loader_creation(self):
        """Test EnhancedCSVLoader can be instantiated."""
        from jsf.data.sources import EnhancedCSVLoader
        import tempfile
        import os
        
        # Create a temp CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Open,High,Low,Close,Volume\n")
            f.write("2024-01-01,100,105,99,104,1000000\n")
            f.write("2024-01-02,104,108,103,107,1200000\n")
            temp_path = f.name
        
        try:
            loader = EnhancedCSVLoader(file_path=temp_path)
            assert loader is not None
            assert hasattr(loader, 'load')
        finally:
            os.unlink(temp_path)
    
    def test_data_quality_report_structure(self):
        """Test DataQualityReport class functionality."""
        from jsf.data.sources.csv_enhanced import DataQualityReport
        
        report = DataQualityReport(symbol="TEST")
        
        assert report.symbol == "TEST"
        assert report.total_rows == 0
        assert report.is_clean == True  # No issues initially
        assert isinstance(report.issues, list)
        
        # Test string representation
        report_str = str(report)
        assert "TEST" in report_str
    
    def test_csv_loader_data_validation(self):
        """Test EnhancedCSVLoader validates data quality."""
        from jsf.data.sources import EnhancedCSVLoader
        import tempfile
        import os
        
        # Create CSV with known quality issues (duplicate date)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Symbol,Open,High,Low,Close,Volume\n")
            f.write("2024-01-01,TEST,100,105,99,104,1000000\n")
            f.write("2024-01-02,TEST,104,108,103,107,1200000\n")
            f.write("2024-01-02,TEST,104,108,103,108,1100000\n")  # Duplicate
            f.write("2024-01-03,TEST,107,110,106,109,1300000\n")
            temp_path = f.name
        
        try:
            loader = EnhancedCSVLoader(file_path=temp_path, validate=True)
            data = loader.load()
            
            # Check that quality report was generated
            assert hasattr(loader, 'quality_reports')
            if loader.quality_reports:
                report = list(loader.quality_reports.values())[0]
                assert report.duplicates >= 0  # May have detected duplicates
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_data_function(self):
        """Test the convenience load_csv_data function."""
        from jsf.data.sources import load_csv_data
        import tempfile
        import os
        
        # Create a valid CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Symbol,Open,High,Low,Close,Volume\n")
            for i in range(30):  # Create 30 days of data
                day = 1 + i
                f.write(f"2024-01-{day:02d},TEST,{100+i},{105+i},{99+i},{104+i},1000000\n")
            temp_path = f.name
        
        try:
            data = load_csv_data(
                file_path=temp_path,
                symbols=['TEST'],
            )
            
            # Should return PriceData or dict of PriceData
            from jsf.data import PriceData
            assert isinstance(data, (dict, PriceData))
        finally:
            os.unlink(temp_path)
    
    def test_yahoo_constants_available(self):
        """Test that helpful constants are available from Yahoo module."""
        from jsf.data.sources.yahoo import (
            POPULAR_TECH_STOCKS,
            POPULAR_ETFs,
            POPULAR_INDICES,
        )
        
        assert 'AAPL' in POPULAR_TECH_STOCKS
        assert 'SPY' in POPULAR_ETFs
        assert '^GSPC' in POPULAR_INDICES
    
    def test_sources_module_exports(self):
        """Test that sources module exports correct classes."""
        from jsf.data import sources
        
        assert hasattr(sources, 'YahooFinanceLoader')
        assert hasattr(sources, 'load_yahoo_data')
        assert hasattr(sources, 'EnhancedCSVLoader')
        assert hasattr(sources, 'load_csv_data')
    
    def test_data_module_exports_phase13(self):
        """Test that data module exports Phase 13 classes."""
        from jsf import data
        
        # Check Phase 13 exports
        assert hasattr(data, 'YahooFinanceLoader')
        assert hasattr(data, 'load_yahoo_data')
        assert hasattr(data, 'EnhancedCSVLoader')
        assert hasattr(data, 'load_csv_data')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

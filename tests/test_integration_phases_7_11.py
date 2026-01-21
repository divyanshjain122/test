"""
Comprehensive integration tests for Phases 7-11.

This test suite validates the complete end-to-end functionality
of the portfolio construction, strategies, backtesting, visualization,
and optimization modules.
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
from jsf.optimization import optimize_strategy, GridSearchOptimizer, ParameterGrid


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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

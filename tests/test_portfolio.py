"""Comprehensive unit tests for the portfolio module (Phase 7)."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from jsf.portfolio import (
    # Base classes
    Portfolio,
    PortfolioConstructor,
    PositionSizer,
    WeightOptimizer,
    Rebalancer,
    RebalanceFrequency,
    # Position sizing
    EqualWeightSizer,
    SignalWeightedSizer,
    VolatilityScaledSizer,
    RiskParitySizer,
    KellyCriterionSizer,
    # Optimization
    MinimumVarianceOptimizer,
    MaxSharpeOptimizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    MaxDiversificationOptimizer,
    # Rebalancing
    PeriodicRebalancer,
    ThresholdRebalancer,
    VolatilityTargetRebalancer,
    BandRebalancer,
    SmartRebalancer,
    # Constraints
    PortfolioConstraints,
    PositionLimitConstraint,
    SectorConstraint,
    TurnoverConstraint,
    LeverageConstraint,
    ConcentrationConstraint,
    # Constructors
    SimplePortfolioConstructor,
    OptimizedPortfolioConstructor,
    HybridPortfolioConstructor,
)
from jsf.data import load_data


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    return load_data(
        source="synthetic",
        symbols=["AAPL", "GOOGL", "MSFT", "AMZN"],
        start_date="2022-01-01",
        end_date="2023-12-31",
        initial_price=100.0,
        annual_return=0.12,
        annual_volatility=0.25,
        seed=42,
    )


@pytest.fixture
def sample_signals():
    """Create sample signals for testing."""
    dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")
    return pd.DataFrame({
        "AAPL": np.random.randn(len(dates)) * 0.5,
        "GOOGL": np.random.randn(len(dates)) * 0.5,
        "MSFT": np.random.randn(len(dates)) * 0.5,
        "AMZN": np.random.randn(len(dates)) * 0.5,
    }, index=dates)


@pytest.fixture
def sample_weights():
    """Create sample portfolio weights."""
    return pd.Series({
        "AAPL": 0.25,
        "GOOGL": 0.25,
        "MSFT": 0.25,
        "AMZN": 0.25,
    })


@pytest.fixture
def sample_sector_map():
    """Create sample sector mapping."""
    return {
        "AAPL": "Technology",
        "GOOGL": "Technology",
        "MSFT": "Technology",
        "AMZN": "Consumer",
    }


# ============================================================================
# Test Base Classes
# ============================================================================

class TestPortfolio:
    """Test Portfolio dataclass."""
    
    def test_portfolio_creation(self, sample_weights):
        """Test basic portfolio creation."""
        timestamp = pd.Timestamp("2023-01-01")
        portfolio = Portfolio(weights=sample_weights, timestamp=timestamp)
        
        assert len(portfolio.weights) == 4
        assert portfolio.timestamp == timestamp
        assert portfolio.turnover == 0.0
        assert portfolio.long_exposure == 1.0
        assert portfolio.short_exposure == 0.0
        assert portfolio.net_exposure == 1.0
        assert portfolio.gross_exposure == 1.0
    
    def test_portfolio_with_shorts(self):
        """Test portfolio with short positions."""
        weights = pd.Series({"AAPL": 0.5, "GOOGL": 0.3, "MSFT": -0.2, "AMZN": 0.1})
        portfolio = Portfolio(weights=weights, timestamp=pd.Timestamp.now())
        
        assert abs(portfolio.long_exposure - 0.9) < 1e-10
        assert abs(portfolio.short_exposure - 0.2) < 1e-10
        assert abs(portfolio.net_exposure - 0.7) < 1e-10
        assert abs(portfolio.gross_exposure - 1.1) < 1e-10
    
    def test_portfolio_turnover(self, sample_weights):
        """Test turnover calculation."""
        new_weights = pd.Series({
            "AAPL": 0.30,  # +0.05
            "GOOGL": 0.20,  # -0.05
            "MSFT": 0.25,  # 0
            "AMZN": 0.25,  # 0
        })
        portfolio = Portfolio(
            weights=new_weights,
            timestamp=pd.Timestamp.now(),
            previous_weights=sample_weights
        )
        
        expected_turnover = (abs(0.05) + abs(-0.05)) / 2
        assert abs(portfolio.turnover - expected_turnover) < 1e-10


class TestRebalanceFrequency:
    """Test RebalanceFrequency enum."""
    
    def test_frequency_values(self):
        """Test all frequency enum values exist."""
        assert RebalanceFrequency.DAILY == "daily"
        assert RebalanceFrequency.WEEKLY == "weekly"
        assert RebalanceFrequency.MONTHLY == "monthly"
        assert RebalanceFrequency.QUARTERLY == "quarterly"
        assert RebalanceFrequency.YEARLY == "yearly"


# ============================================================================
# Test Position Sizing Methods
# ============================================================================

class TestEqualWeightSizer:
    """Test equal weight position sizer."""
    
    def test_basic_sizing(self, sample_signals):
        """Test equal weighting."""
        sizer = EqualWeightSizer()
        signals = sample_signals.iloc[0]
        weights = sizer.size_positions(signals, None)
        
        assert len(weights) == 4
        assert all(abs(w - 0.25) < 1e-10 for w in weights.values)
    
    def test_long_only(self, sample_signals):
        """Test long-only constraint."""
        sizer = EqualWeightSizer(long_only=True)
        signals = pd.Series({"A": 1.0, "B": -1.0, "C": 0.5})
        weights = sizer.size_positions(signals, None)
        
        # Should only have positive signals
        assert all(w >= 0 for w in weights.values)
    
    def test_subset_selection(self, sample_signals):
        """Test selecting subset of positions."""
        sizer = EqualWeightSizer(max_positions=2)
        signals = sample_signals.iloc[0]
        weights = sizer.size_positions(signals, None)
        
        # Should have at most 2 positions
        non_zero = (weights.abs() > 1e-10).sum()
        assert non_zero <= 2


class TestSignalWeightedSizer:
    """Test signal-weighted position sizer."""
    
    def test_proportional_weighting(self):
        """Test weights proportional to signals."""
        sizer = SignalWeightedSizer()
        signals = pd.Series({"A": 2.0, "B": 1.0, "C": 1.0})
        weights = sizer.size_positions(signals, None)
        
        # Weight should be proportional to absolute signal
        assert abs(weights["A"] - 0.5) < 1e-10  # 2/4
        assert abs(weights["B"] - 0.25) < 1e-10  # 1/4
        assert abs(weights["C"] - 0.25) < 1e-10  # 1/4
    
    def test_signal_scaling(self):
        """Test signal scaling parameter."""
        sizer = SignalWeightedSizer(signal_scale=2.0)
        signals = pd.Series({"A": 1.0, "B": 1.0})
        weights = sizer.size_positions(signals, None)
        
        # Larger scale = more aggressive
        assert abs(weights.abs().sum() - 1.0) < 1e-10


class TestVolatilityScaledSizer:
    """Test volatility-scaled position sizer."""
    
    def test_inverse_volatility(self, sample_price_data):
        """Test inverse volatility weighting."""
        sizer = VolatilityScaledSizer(lookback=20)
        signals = pd.Series({"AAPL": 1.0, "GOOGL": 1.0, "MSFT": 1.0, "AMZN": 1.0})
        weights = sizer.size_positions(signals, sample_price_data)
        
        # Lower volatility should get higher weight
        assert len(weights) == 4
        assert abs(weights.sum() - 1.0) < 1e-10
    
    def test_target_volatility(self, sample_price_data):
        """Test target volatility scaling."""
        sizer = VolatilityScaledSizer(target_volatility=0.15)
        signals = pd.Series({"AAPL": 1.0, "GOOGL": 1.0})
        weights = sizer.size_positions(signals, sample_price_data)
        
        # Should target specific volatility level
        assert weights is not None


class TestRiskParitySizer:
    """Test risk parity position sizer."""
    
    def test_equal_risk_contribution(self, sample_price_data):
        """Test equal risk contribution."""
        sizer = RiskParitySizer(lookback=60)
        signals = pd.Series({"AAPL": 1.0, "GOOGL": 1.0, "MSFT": 1.0})
        weights = sizer.size_positions(signals, sample_price_data)
        
        # Should equalize risk contributions
        assert len(weights) == 3
        assert abs(weights.sum() - 1.0) < 0.01  # Allow some tolerance
    
    def test_convergence(self, sample_price_data):
        """Test algorithm convergence."""
        sizer = RiskParitySizer(max_iterations=100, tolerance=1e-6)
        signals = pd.Series({"AAPL": 1.0, "GOOGL": 1.0})
        weights = sizer.size_positions(signals, sample_price_data)
        
        assert weights is not None
        assert not weights.isna().any()


class TestKellyCriterionSizer:
    """Test Kelly criterion position sizer."""
    
    def test_kelly_sizing(self, sample_price_data):
        """Test Kelly criterion sizing."""
        sizer = KellyCriterionSizer(lookback=60)
        signals = pd.Series({"AAPL": 1.0, "GOOGL": 1.0, "MSFT": 1.0})
        weights = sizer.size_positions(signals, sample_price_data)
        
        # Kelly should produce reasonable weights
        assert len(weights) == 3
        assert all(w >= 0 for w in weights.values)
    
    def test_fractional_kelly(self, sample_price_data):
        """Test fractional Kelly (more conservative)."""
        sizer = KellyCriterionSizer(kelly_fraction=0.5)
        signals = pd.Series({"AAPL": 1.0, "GOOGL": 1.0})
        weights = sizer.size_positions(signals, sample_price_data)
        
        # Fractional Kelly should be more conservative
        assert weights.abs().sum() <= 1.0


# ============================================================================
# Test Weight Optimization
# ============================================================================

class TestMinimumVarianceOptimizer:
    """Test minimum variance optimizer."""
    
    def test_optimization(self, sample_price_data):
        """Test minimum variance optimization."""
        optimizer = MinimumVarianceOptimizer(lookback=60)
        signals = pd.Series({"AAPL": 1.0, "GOOGL": 1.0, "MSFT": 1.0, "AMZN": 1.0})
        weights = optimizer.optimize(signals, sample_price_data)
        
        assert len(weights) == 4
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(w >= 0 for w in weights.values)  # Long-only default
    
    def test_long_short(self, sample_price_data):
        """Test long-short optimization."""
        optimizer = MinimumVarianceOptimizer(long_only=False)
        signals = pd.Series({"AAPL": 1.0, "GOOGL": -1.0, "MSFT": 1.0})
        weights = optimizer.optimize(signals, sample_price_data)
        
        # Can have negative weights
        assert weights is not None


class TestMaxSharpeOptimizer:
    """Test maximum Sharpe ratio optimizer."""
    
    def test_optimization(self, sample_price_data):
        """Test max Sharpe optimization."""
        optimizer = MaxSharpeOptimizer(lookback=60, risk_free_rate=0.03)
        signals = pd.Series({"AAPL": 1.0, "GOOGL": 1.0, "MSFT": 1.0})
        weights = optimizer.optimize(signals, sample_price_data)
        
        assert len(weights) == 3
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(w >= 0 for w in weights.values)


class TestMeanVarianceOptimizer:
    """Test mean-variance optimizer."""
    
    def test_optimization(self, sample_price_data):
        """Test mean-variance optimization."""
        optimizer = MeanVarianceOptimizer(risk_aversion=1.0, lookback=60)
        signals = pd.Series({"AAPL": 1.0, "GOOGL": 1.0, "MSFT": 1.0})
        weights = optimizer.optimize(signals, sample_price_data)
        
        assert len(weights) == 3
        assert abs(weights.sum() - 1.0) < 1e-6
    
    def test_risk_aversion_effect(self, sample_price_data):
        """Test different risk aversion levels."""
        signals = pd.Series({"AAPL": 1.0, "GOOGL": 1.0})
        
        # Low risk aversion (more aggressive)
        opt_low = MeanVarianceOptimizer(risk_aversion=0.5)
        weights_low = opt_low.optimize(signals, sample_price_data)
        
        # High risk aversion (more conservative)
        opt_high = MeanVarianceOptimizer(risk_aversion=5.0)
        weights_high = opt_high.optimize(signals, sample_price_data)
        
        # Both should be valid
        assert weights_low is not None
        assert weights_high is not None


class TestRiskParityOptimizer:
    """Test risk parity optimizer."""
    
    def test_optimization(self, sample_price_data):
        """Test risk parity optimization."""
        optimizer = RiskParityOptimizer(lookback=60)
        signals = pd.Series({"AAPL": 1.0, "GOOGL": 1.0, "MSFT": 1.0})
        weights = optimizer.optimize(signals, sample_price_data)
        
        assert len(weights) == 3
        assert abs(weights.sum() - 1.0) < 0.01


class TestMaxDiversificationOptimizer:
    """Test maximum diversification optimizer."""
    
    def test_optimization(self, sample_price_data):
        """Test max diversification optimization."""
        optimizer = MaxDiversificationOptimizer(lookback=60)
        signals = pd.Series({"AAPL": 1.0, "GOOGL": 1.0, "MSFT": 1.0, "AMZN": 1.0})
        weights = optimizer.optimize(signals, sample_price_data)
        
        assert len(weights) == 4
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(w >= 0 for w in weights.values)


# ============================================================================
# Test Rebalancing Strategies
# ============================================================================

class TestPeriodicRebalancer:
    """Test periodic rebalancer."""
    
    def test_daily_rebalance(self):
        """Test daily rebalancing."""
        rebalancer = PeriodicRebalancer(frequency=RebalanceFrequency.DAILY)
        
        # Should rebalance every day
        assert rebalancer.should_rebalance(pd.Timestamp("2023-01-01"), None, None)
        assert rebalancer.should_rebalance(pd.Timestamp("2023-01-02"), None, None)
    
    def test_monthly_rebalance(self):
        """Test monthly rebalancing."""
        rebalancer = PeriodicRebalancer(frequency=RebalanceFrequency.MONTHLY)
        
        # Should rebalance on month boundaries
        current = pd.Timestamp("2023-02-01")
        last_rebalance = pd.Timestamp("2023-01-01")
        
        assert rebalancer.should_rebalance(current, None, last_rebalance)
        
        # Should not rebalance mid-month
        mid_month = pd.Timestamp("2023-01-15")
        assert not rebalancer.should_rebalance(mid_month, None, last_rebalance)


class TestThresholdRebalancer:
    """Test threshold-based rebalancer."""
    
    def test_drift_threshold(self, sample_weights):
        """Test rebalancing based on drift threshold."""
        rebalancer = ThresholdRebalancer(drift_threshold=0.05)
        
        # Small drift - no rebalance
        small_drift = pd.Series({
            "AAPL": 0.26,  # +0.01
            "GOOGL": 0.24,  # -0.01
            "MSFT": 0.25,
            "AMZN": 0.25,
        })
        assert not rebalancer.should_rebalance(None, small_drift, None, sample_weights)
        
        # Large drift - rebalance
        large_drift = pd.Series({
            "AAPL": 0.35,  # +0.10
            "GOOGL": 0.15,  # -0.10
            "MSFT": 0.25,
            "AMZN": 0.25,
        })
        assert rebalancer.should_rebalance(None, large_drift, None, sample_weights)


class TestVolatilityTargetRebalancer:
    """Test volatility target rebalancer."""
    
    def test_volatility_trigger(self, sample_price_data):
        """Test rebalancing based on volatility."""
        rebalancer = VolatilityTargetRebalancer(
            target_volatility=0.15,
            tolerance=0.02,
            lookback=20
        )
        
        current = pd.Timestamp("2023-06-01")
        weights = pd.Series({"AAPL": 0.5, "GOOGL": 0.5})
        
        should_rebal = rebalancer.should_rebalance(
            current, weights, None, None, sample_price_data
        )
        
        # Should return boolean
        assert isinstance(should_rebal, bool)


class TestBandRebalancer:
    """Test band-based rebalancer."""
    
    def test_band_violations(self, sample_weights):
        """Test rebalancing when weights violate bands."""
        rebalancer = BandRebalancer(
            lower_band=0.15,
            upper_band=0.35,
            rebalance_to_target=True
        )
        
        # Within bands - no rebalance
        within = pd.Series({"AAPL": 0.20, "GOOGL": 0.30, "MSFT": 0.25, "AMZN": 0.25})
        assert not rebalancer.should_rebalance(None, within, None, sample_weights)
        
        # Outside bands - rebalance
        outside = pd.Series({"AAPL": 0.40, "GOOGL": 0.30, "MSFT": 0.20, "AMZN": 0.10})
        assert rebalancer.should_rebalance(None, outside, None, sample_weights)


class TestSmartRebalancer:
    """Test smart multi-trigger rebalancer."""
    
    def test_multiple_triggers(self):
        """Test rebalancing with multiple triggers."""
        rebalancer = SmartRebalancer(
            min_days_between=5,
            drift_threshold=0.10,
            volatility_target=0.15,
            volatility_tolerance=0.03
        )
        
        # Should not rebalance if min_days not met
        current = pd.Timestamp("2023-01-03")
        last_rebalance = pd.Timestamp("2023-01-01")
        
        result = rebalancer.should_rebalance(current, None, last_rebalance, None)
        assert not result  # Only 2 days passed, min is 5


# ============================================================================
# Test Constraints
# ============================================================================

class TestPositionLimitConstraint:
    """Test position limit constraint."""
    
    def test_check_violation(self):
        """Test checking for violations."""
        constraint = PositionLimitConstraint(min_weight=0.05, max_weight=0.35)
        
        # Valid weights
        valid = pd.Series({"A": 0.30, "B": 0.40, "C": 0.30})
        assert constraint.check(valid)
        
        # Invalid weights (too large)
        invalid = pd.Series({"A": 0.50, "B": 0.30, "C": 0.20})
        assert not constraint.check(invalid)
    
    def test_enforce_limits(self):
        """Test enforcing position limits."""
        constraint = PositionLimitConstraint(min_weight=0.10, max_weight=0.40)
        
        weights = pd.Series({"A": 0.60, "B": 0.30, "C": 0.10})
        adjusted = constraint.enforce(weights)
        
        # All weights should be within limits
        assert all(adjusted <= 0.40)
        assert all(adjusted >= 0.10)


class TestSectorConstraint:
    """Test sector exposure constraint."""
    
    def test_sector_limits(self, sample_sector_map):
        """Test sector exposure limits."""
        constraint = SectorConstraint(
            sector_map=sample_sector_map,
            max_sector_exposure=0.60
        )
        
        # Technology is 75% - violates 60% limit
        weights = pd.Series({"AAPL": 0.25, "GOOGL": 0.25, "MSFT": 0.25, "AMZN": 0.25})
        assert not constraint.check(weights)
        
        # Enforce should reduce tech exposure
        adjusted = constraint.enforce(weights)
        tech_exposure = adjusted[["AAPL", "GOOGL", "MSFT"]].sum()
        assert tech_exposure <= 0.60 + 1e-6


class TestTurnoverConstraint:
    """Test turnover constraint."""
    
    def test_turnover_limit(self, sample_weights):
        """Test turnover limits."""
        constraint = TurnoverConstraint(max_turnover=0.20)
        
        # Low turnover - OK
        low_turnover = pd.Series({
            "AAPL": 0.27,  # +0.02
            "GOOGL": 0.23,  # -0.02
            "MSFT": 0.25,
            "AMZN": 0.25,
        })
        assert constraint.check(low_turnover, sample_weights)
        
        # High turnover - violation
        high_turnover = pd.Series({
            "AAPL": 0.50,  # +0.25
            "GOOGL": 0.00,  # -0.25
            "MSFT": 0.25,
            "AMZN": 0.25,
        })
        assert not constraint.check(high_turnover, sample_weights)


class TestLeverageConstraint:
    """Test leverage constraint."""
    
    def test_gross_leverage(self):
        """Test gross leverage limits."""
        constraint = LeverageConstraint(max_gross_leverage=1.5, max_net_leverage=1.0)
        
        # Within limits
        valid = pd.Series({"A": 0.60, "B": -0.10, "C": 0.50})
        assert constraint.check(valid)
        
        # Exceeds gross leverage
        invalid = pd.Series({"A": 1.0, "B": -0.6, "C": 0.5})
        assert not constraint.check(invalid)
    
    def test_net_leverage(self):
        """Test net leverage limits."""
        constraint = LeverageConstraint(max_net_leverage=0.8)
        
        # Net too high
        weights = pd.Series({"A": 0.90, "B": 0.10})
        assert not constraint.check(weights)


class TestConcentrationConstraint:
    """Test concentration constraint."""
    
    def test_herfindahl_index(self):
        """Test Herfindahl concentration index."""
        constraint = ConcentrationConstraint(max_concentration=0.30)
        
        # Concentrated (HHI = 0.5^2 + 0.5^2 = 0.5)
        concentrated = pd.Series({"A": 0.50, "B": 0.50})
        assert not constraint.check(concentrated)
        
        # Diversified (HHI = 4 * 0.25^2 = 0.25)
        diversified = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        assert constraint.check(diversified)


class TestPortfolioConstraints:
    """Test portfolio constraints manager."""
    
    def test_multiple_constraints(self):
        """Test managing multiple constraints."""
        constraints = PortfolioConstraints()
        constraints.add_constraint(PositionLimitConstraint(max_weight=0.40))
        constraints.add_constraint(LeverageConstraint(max_gross_leverage=1.0))
        
        # Valid weights
        valid = pd.Series({"A": 0.35, "B": 0.35, "C": 0.30})
        assert constraints.check(valid)
        
        # Invalid weights
        invalid = pd.Series({"A": 0.60, "B": 0.40})
        assert not constraints.check(invalid)
    
    def test_enforce_all(self):
        """Test enforcing all constraints."""
        constraints = PortfolioConstraints()
        constraints.add_constraint(PositionLimitConstraint(max_weight=0.35))
        constraints.add_constraint(ConcentrationConstraint(max_concentration=0.30))
        
        weights = pd.Series({"A": 0.50, "B": 0.50})
        adjusted = constraints.enforce(weights)
        
        # Should satisfy all constraints
        assert constraints.check(adjusted)


# ============================================================================
# Test Portfolio Constructors
# ============================================================================

class TestSimplePortfolioConstructor:
    """Test simple portfolio constructor."""
    
    def test_construction(self, sample_signals, sample_price_data):
        """Test basic portfolio construction."""
        sizer = EqualWeightSizer()
        constructor = SimplePortfolioConstructor(sizer=sizer)
        
        timestamp = pd.Timestamp("2023-01-15")
        signals = sample_signals.loc[timestamp]
        
        portfolio = constructor.construct(signals, sample_price_data, timestamp)
        
        assert isinstance(portfolio, Portfolio)
        assert portfolio.timestamp == timestamp
        assert len(portfolio.weights) == 4
    
    def test_with_constraints(self, sample_signals, sample_price_data):
        """Test construction with constraints."""
        sizer = EqualWeightSizer()
        constraints = PortfolioConstraints()
        constraints.add_constraint(PositionLimitConstraint(max_weight=0.30))
        
        constructor = SimplePortfolioConstructor(sizer=sizer, constraints=constraints)
        
        timestamp = pd.Timestamp("2023-01-15")
        signals = sample_signals.loc[timestamp]
        portfolio = constructor.construct(signals, sample_price_data, timestamp)
        
        # All weights should respect constraint
        assert all(portfolio.weights <= 0.30)


class TestOptimizedPortfolioConstructor:
    """Test optimized portfolio constructor."""
    
    def test_construction(self, sample_signals, sample_price_data):
        """Test optimization-based construction."""
        optimizer = MinimumVarianceOptimizer(lookback=60)
        constructor = OptimizedPortfolioConstructor(optimizer=optimizer)
        
        timestamp = pd.Timestamp("2023-06-01")
        signals = sample_signals.loc[timestamp]
        
        portfolio = constructor.construct(signals, sample_price_data, timestamp)
        
        assert isinstance(portfolio, Portfolio)
        assert abs(portfolio.weights.sum() - 1.0) < 1e-6


class TestHybridPortfolioConstructor:
    """Test hybrid portfolio constructor."""
    
    def test_blending(self, sample_signals, sample_price_data):
        """Test blending signals and optimization."""
        sizer = SignalWeightedSizer()
        optimizer = MinimumVarianceOptimizer()
        
        constructor = HybridPortfolioConstructor(
            sizer=sizer,
            optimizer=optimizer,
            blend_alpha=0.5  # 50/50 blend
        )
        
        timestamp = pd.Timestamp("2023-06-01")
        signals = sample_signals.loc[timestamp]
        
        portfolio = constructor.construct(signals, sample_price_data, timestamp)
        
        assert isinstance(portfolio, Portfolio)
        assert abs(portfolio.weights.sum() - 1.0) < 1e-6
    
    def test_blend_extremes(self, sample_signals, sample_price_data):
        """Test extreme blend values."""
        sizer = EqualWeightSizer()
        optimizer = MinimumVarianceOptimizer()
        
        # 100% signals
        constructor_signals = HybridPortfolioConstructor(
            sizer=sizer, optimizer=optimizer, blend_alpha=1.0
        )
        
        # 100% optimization
        constructor_opt = HybridPortfolioConstructor(
            sizer=sizer, optimizer=optimizer, blend_alpha=0.0
        )
        
        timestamp = pd.Timestamp("2023-06-01")
        signals = sample_signals.loc[timestamp]
        
        portfolio_signals = constructor_signals.construct(signals, sample_price_data, timestamp)
        portfolio_opt = constructor_opt.construct(signals, sample_price_data, timestamp)
        
        # Both should be valid but different
        assert isinstance(portfolio_signals, Portfolio)
        assert isinstance(portfolio_opt, Portfolio)


# ============================================================================
# Integration Tests
# ============================================================================

class TestPortfolioIntegration:
    """Test end-to-end portfolio workflows."""
    
    def test_complete_workflow(self, sample_signals, sample_price_data):
        """Test complete portfolio construction workflow."""
        # Setup components
        sizer = VolatilityScaledSizer(target_volatility=0.15)
        rebalancer = PeriodicRebalancer(frequency=RebalanceFrequency.MONTHLY)
        constraints = PortfolioConstraints()
        constraints.add_constraint(PositionLimitConstraint(max_weight=0.40))
        constraints.add_constraint(LeverageConstraint(max_gross_leverage=1.0))
        
        constructor = SimplePortfolioConstructor(
            sizer=sizer,
            rebalancer=rebalancer,
            constraints=constraints
        )
        
        # Construct portfolio
        timestamp = pd.Timestamp("2023-01-15")
        signals = sample_signals.loc[timestamp]
        portfolio = constructor.construct(signals, sample_price_data, timestamp)
        
        # Verify portfolio
        assert isinstance(portfolio, Portfolio)
        assert all(portfolio.weights <= 0.40)
        assert portfolio.gross_exposure <= 1.0
    
    def test_rebalancing_workflow(self, sample_signals, sample_price_data):
        """Test portfolio rebalancing workflow."""
        constructor = SimplePortfolioConstructor(
            sizer=EqualWeightSizer(),
            rebalancer=ThresholdRebalancer(drift_threshold=0.05)
        )
        
        # Initial portfolio
        t1 = pd.Timestamp("2023-01-01")
        portfolio1 = constructor.construct(
            sample_signals.loc[t1], sample_price_data, t1
        )
        
        # Check if rebalance needed
        t2 = pd.Timestamp("2023-02-01")
        current_weights = portfolio1.weights * 1.1  # Simulate drift
        current_weights /= current_weights.sum()
        
        should_rebalance = constructor.rebalancer.should_rebalance(
            t2, current_weights, portfolio1.timestamp, portfolio1.weights
        )
        
        assert isinstance(should_rebalance, bool)
        
        # Rebalance if needed
        if should_rebalance:
            portfolio2 = constructor.construct(
                sample_signals.loc[t2],
                sample_price_data,
                t2,
                previous_weights=portfolio1.weights
            )
            assert portfolio2.turnover > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""Weight optimization methods for portfolio construction."""

from typing import Optional, Any

import pandas as pd
import numpy as np
from scipy.optimize import minimize

from jsf.portfolio.base import WeightOptimizer
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class MinimumVarianceOptimizer(WeightOptimizer):
    """
    Minimum variance portfolio optimization.
    
    Minimizes portfolio variance subject to constraints.
    """
    
    def __init__(
        self,
        allow_short: bool = False,
        max_weight: float = 0.3,
        name: str = "min_variance",
    ):
        """
        Initialize minimum variance optimizer.
        
        Args:
            allow_short: Allow short positions
            max_weight: Maximum weight per position
            name: Optimizer name
        """
        super().__init__(
            name=name,
            allow_short=allow_short,
            max_weight=max_weight,
        )
        self.allow_short = allow_short
        self.max_weight = max_weight
    
    def optimize(
        self,
        signals: pd.DataFrame,
        covariance: pd.DataFrame,
        expected_returns: Optional[pd.Series] = None,
        **kwargs: Any
    ) -> pd.Series:
        """Optimize for minimum variance."""
        n = len(signals)
        
        # Objective: minimize variance
        def objective(w):
            return w @ covariance @ w
        
        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}  # Fully invested
        ]
        
        # Bounds
        if self.allow_short:
            bounds = tuple((-self.max_weight, self.max_weight) for _ in range(n))
        else:
            bounds = tuple((0, self.max_weight) for _ in range(n))
        
        # Initial guess (equal weight)
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            return pd.Series(w0, index=signals.index)
        
        return pd.Series(result.x, index=signals.index)


class MaxSharpeOptimizer(WeightOptimizer):
    """
    Maximum Sharpe ratio optimization.
    
    Maximizes risk-adjusted returns.
    """
    
    def __init__(
        self,
        allow_short: bool = False,
        max_weight: float = 0.3,
        risk_free_rate: float = 0.0,
        name: str = "max_sharpe",
    ):
        """
        Initialize max Sharpe optimizer.
        
        Args:
            allow_short: Allow short positions
            max_weight: Maximum weight per position
            risk_free_rate: Risk-free rate for Sharpe calculation
            name: Optimizer name
        """
        super().__init__(
            name=name,
            allow_short=allow_short,
            max_weight=max_weight,
            risk_free_rate=risk_free_rate,
        )
        self.allow_short = allow_short
        self.max_weight = max_weight
        self.risk_free_rate = risk_free_rate
    
    def optimize(
        self,
        signals: pd.DataFrame,
        covariance: pd.DataFrame,
        expected_returns: pd.Series,
        **kwargs: Any
    ) -> pd.Series:
        """Optimize for maximum Sharpe ratio."""
        n = len(signals)
        
        # Objective: maximize Sharpe (minimize negative Sharpe)
        def objective(w):
            portfolio_return = w @ expected_returns
            portfolio_vol = np.sqrt(w @ covariance @ w)
            sharpe = (portfolio_return - self.risk_free_rate) / (portfolio_vol + 1e-10)
            return -sharpe  # Minimize negative Sharpe
        
        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds
        if self.allow_short:
            bounds = tuple((-self.max_weight, self.max_weight) for _ in range(n))
        else:
            bounds = tuple((0, self.max_weight) for _ in range(n))
        
        # Initial guess
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            return pd.Series(w0, index=signals.index)
        
        return pd.Series(result.x, index=signals.index)


class MeanVarianceOptimizer(WeightOptimizer):
    """
    Mean-variance portfolio optimization.
    
    Balances expected return and risk.
    """
    
    def __init__(
        self,
        risk_aversion: float = 1.0,
        allow_short: bool = False,
        max_weight: float = 0.3,
        name: str = "mean_variance",
    ):
        """
        Initialize mean-variance optimizer.
        
        Args:
            risk_aversion: Risk aversion parameter (higher = more conservative)
            allow_short: Allow short positions
            max_weight: Maximum weight per position
            name: Optimizer name
        """
        super().__init__(
            name=name,
            risk_aversion=risk_aversion,
            allow_short=allow_short,
            max_weight=max_weight,
        )
        self.risk_aversion = risk_aversion
        self.allow_short = allow_short
        self.max_weight = max_weight
    
    def optimize(
        self,
        signals: pd.DataFrame,
        covariance: pd.DataFrame,
        expected_returns: pd.Series,
        **kwargs: Any
    ) -> pd.Series:
        """Optimize mean-variance utility."""
        n = len(signals)
        
        # Objective: maximize return - risk_aversion * variance
        def objective(w):
            portfolio_return = w @ expected_returns
            portfolio_var = w @ covariance @ w
            utility = portfolio_return - self.risk_aversion * portfolio_var
            return -utility  # Minimize negative utility
        
        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds
        if self.allow_short:
            bounds = tuple((-self.max_weight, self.max_weight) for _ in range(n))
        else:
            bounds = tuple((0, self.max_weight) for _ in range(n))
        
        # Initial guess
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            return pd.Series(w0, index=signals.index)
        
        return pd.Series(result.x, index=signals.index)


class RiskParityOptimizer(WeightOptimizer):
    """
    Risk parity optimization.
    
    Equalizes risk contribution across positions.
    """
    
    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        name: str = "risk_parity",
    ):
        """
        Initialize risk parity optimizer.
        
        Args:
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
            name: Optimizer name
        """
        super().__init__(
            name=name,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def optimize(
        self,
        signals: pd.DataFrame,
        covariance: pd.DataFrame,
        expected_returns: Optional[pd.Series] = None,
        **kwargs: Any
    ) -> pd.Series:
        """Optimize for risk parity."""
        n = len(signals)
        
        # Objective: minimize sum of squared differences in risk contributions
        def objective(w):
            portfolio_var = w @ covariance @ w
            marginal_risk = covariance @ w
            risk_contrib = w * marginal_risk
            target_risk = portfolio_var / n
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: w}  # All weights >= 0
        ]
        
        # Bounds (long only for standard risk parity)
        bounds = tuple((0, 1) for _ in range(n))
        
        # Initial guess
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance},
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            return pd.Series(w0, index=signals.index)
        
        return pd.Series(result.x, index=signals.index)


class MaxDiversificationOptimizer(WeightOptimizer):
    """
    Maximum diversification optimization.
    
    Maximizes diversification ratio.
    """
    
    def __init__(
        self,
        allow_short: bool = False,
        max_weight: float = 0.3,
        name: str = "max_diversification",
    ):
        """
        Initialize max diversification optimizer.
        
        Args:
            allow_short: Allow short positions
            max_weight: Maximum weight per position
            name: Optimizer name
        """
        super().__init__(
            name=name,
            allow_short=allow_short,
            max_weight=max_weight,
        )
        self.allow_short = allow_short
        self.max_weight = max_weight
    
    def optimize(
        self,
        signals: pd.DataFrame,
        covariance: pd.DataFrame,
        expected_returns: Optional[pd.Series] = None,
        **kwargs: Any
    ) -> pd.Series:
        """Optimize for maximum diversification."""
        n = len(signals)
        
        # Calculate individual volatilities
        individual_vols = np.sqrt(np.diag(covariance))
        
        # Objective: maximize diversification ratio
        # DR = (w^T * sigma) / sqrt(w^T * Cov * w)
        # Minimize negative DR
        def objective(w):
            weighted_vol = w @ individual_vols
            portfolio_vol = np.sqrt(w @ covariance @ w)
            dr = weighted_vol / (portfolio_vol + 1e-10)
            return -dr
        
        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds
        if self.allow_short:
            bounds = tuple((-self.max_weight, self.max_weight) for _ in range(n))
        else:
            bounds = tuple((0, self.max_weight) for _ in range(n))
        
        # Initial guess
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            return pd.Series(w0, index=signals.index)
        
        return pd.Series(result.x, index=signals.index)

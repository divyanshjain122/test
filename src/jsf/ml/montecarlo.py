"""Monte Carlo Simulation for Portfolio Risk Assessment.

This module provides Monte Carlo simulation capabilities for:
- Portfolio risk metrics (VaR, CVaR, Expected Shortfall)
- Drawdown probability estimation
- Return distribution analysis
- Scenario generation and stress testing
- Confidence intervals for performance metrics

Monte Carlo simulation generates thousands of possible future scenarios
based on historical return distributions, allowing robust risk assessment
without assuming normal distributions.

Example:
    >>> from jsf.ml.montecarlo import MonteCarloSimulator, RiskMetrics
    >>> 
    >>> # Initialize simulator with historical returns
    >>> simulator = MonteCarloSimulator(
    ...     returns=historical_returns,
    ...     n_simulations=10000,
    ...     time_horizon=252  # 1 year
    ... )
    >>> 
    >>> # Run simulation
    >>> results = simulator.run()
    >>> 
    >>> # Get risk metrics
    >>> print(f"VaR (95%): {results.var_95:.2%}")
    >>> print(f"CVaR (95%): {results.cvar_95:.2%}")
    >>> print(f"Max Drawdown (median): {results.max_drawdown_median:.2%}")
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from pathlib import Path

from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class ReturnModel(Enum):
    """Model for simulating returns."""
    HISTORICAL = "historical"  # Bootstrap from historical returns
    NORMAL = "normal"  # Normal distribution
    T_DISTRIBUTION = "t_distribution"  # Student's t (fat tails)
    GARCH = "garch"  # GARCH model for volatility clustering
    BLOCK_BOOTSTRAP = "block_bootstrap"  # Preserve autocorrelation


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    
    # Simulation parameters
    n_simulations: int = 10000
    time_horizon: int = 252  # Trading days
    
    # Return model
    return_model: ReturnModel = ReturnModel.HISTORICAL
    
    # Risk-free rate (annualized)
    risk_free_rate: float = 0.0
    
    # Block bootstrap parameters
    block_size: int = 20  # For block bootstrap
    
    # t-distribution parameters (auto-estimated if None)
    t_df: Optional[float] = None  # Degrees of freedom
    
    # GARCH parameters (auto-estimated if None)
    garch_p: int = 1
    garch_q: int = 1
    
    # Initial capital
    initial_capital: float = 100.0
    
    # Confidence levels
    confidence_levels: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.99])
    
    # Random seed
    random_state: Optional[int] = 42
    
    # Parallel execution
    n_jobs: int = 1


@dataclass
class RiskMetrics:
    """Container for computed risk metrics from simulation."""
    
    # Value at Risk (VaR) - maximum loss at confidence level
    var_90: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    
    # Conditional VaR / Expected Shortfall - average loss beyond VaR
    cvar_90: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    
    # Drawdown metrics
    max_drawdown_mean: float = 0.0
    max_drawdown_median: float = 0.0
    max_drawdown_95: float = 0.0  # 95th percentile
    max_drawdown_99: float = 0.0  # 99th percentile
    
    # Return metrics
    expected_return: float = 0.0
    return_std: float = 0.0
    return_skew: float = 0.0
    return_kurtosis: float = 0.0
    
    # Terminal wealth metrics
    terminal_wealth_mean: float = 0.0
    terminal_wealth_median: float = 0.0
    terminal_wealth_5th: float = 0.0  # 5th percentile (bad scenario)
    terminal_wealth_95th: float = 0.0  # 95th percentile (good scenario)
    
    # Probability metrics
    prob_loss: float = 0.0  # Probability of any loss
    prob_loss_10pct: float = 0.0  # Probability of >10% loss
    prob_loss_20pct: float = 0.0  # Probability of >20% loss
    prob_gain_10pct: float = 0.0  # Probability of >10% gain
    
    # Sharpe ratio distribution
    sharpe_mean: float = 0.0
    sharpe_median: float = 0.0
    sharpe_5th: float = 0.0
    sharpe_95th: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'var_90': self.var_90,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_90': self.cvar_90,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            'max_drawdown_mean': self.max_drawdown_mean,
            'max_drawdown_median': self.max_drawdown_median,
            'max_drawdown_95': self.max_drawdown_95,
            'max_drawdown_99': self.max_drawdown_99,
            'expected_return': self.expected_return,
            'return_std': self.return_std,
            'return_skew': self.return_skew,
            'return_kurtosis': self.return_kurtosis,
            'terminal_wealth_mean': self.terminal_wealth_mean,
            'terminal_wealth_median': self.terminal_wealth_median,
            'terminal_wealth_5th': self.terminal_wealth_5th,
            'terminal_wealth_95th': self.terminal_wealth_95th,
            'prob_loss': self.prob_loss,
            'prob_loss_10pct': self.prob_loss_10pct,
            'prob_loss_20pct': self.prob_loss_20pct,
            'prob_gain_10pct': self.prob_gain_10pct,
            'sharpe_mean': self.sharpe_mean,
            'sharpe_median': self.sharpe_median,
            'sharpe_5th': self.sharpe_5th,
            'sharpe_95th': self.sharpe_95th,
        }
    
    def summary(self) -> str:
        """Get formatted summary string."""
        return f"""
Monte Carlo Risk Metrics Summary
================================
VaR (95%):            {self.var_95:>10.2%}
CVaR (95%):           {self.cvar_95:>10.2%}
Max Drawdown (median):{self.max_drawdown_median:>10.2%}
Max Drawdown (95th):  {self.max_drawdown_95:>10.2%}

Expected Return:      {self.expected_return:>10.2%}
Return Std:           {self.return_std:>10.2%}
Sharpe (median):      {self.sharpe_median:>10.2f}

Probability of Loss:  {self.prob_loss:>10.2%}
Prob Loss > 10%:      {self.prob_loss_10pct:>10.2%}
Prob Gain > 10%:      {self.prob_gain_10pct:>10.2%}

Terminal Wealth:
  5th percentile:     {self.terminal_wealth_5th:>10.2f}
  Median:             {self.terminal_wealth_median:>10.2f}
  95th percentile:    {self.terminal_wealth_95th:>10.2f}
"""


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    
    # Simulated price paths (n_simulations, time_horizon + 1)
    price_paths: np.ndarray = None
    
    # Simulated return paths (n_simulations, time_horizon)
    return_paths: np.ndarray = None
    
    # Risk metrics
    metrics: RiskMetrics = None
    
    # Simulation configuration
    config: SimulationConfig = None
    
    # Per-simulation statistics
    terminal_values: np.ndarray = None
    max_drawdowns: np.ndarray = None
    total_returns: np.ndarray = None
    sharpe_ratios: np.ndarray = None
    
    def get_percentile_path(self, percentile: float) -> np.ndarray:
        """Get the price path at a given percentile.
        
        Args:
            percentile: Percentile (0-100)
            
        Returns:
            Price path at that percentile
        """
        return np.percentile(self.price_paths, percentile, axis=0)
    
    def get_confidence_band(
        self, 
        lower_pct: float = 5, 
        upper_pct: float = 95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get confidence band for price paths.
        
        Args:
            lower_pct: Lower percentile
            upper_pct: Upper percentile
            
        Returns:
            (median_path, lower_path, upper_path)
        """
        median = np.percentile(self.price_paths, 50, axis=0)
        lower = np.percentile(self.price_paths, lower_pct, axis=0)
        upper = np.percentile(self.price_paths, upper_pct, axis=0)
        
        return median, lower, upper
    
    def plot_paths(
        self, 
        n_paths: int = 100,
        show_percentiles: bool = True,
        figsize: Tuple[int, int] = (12, 6)
    ) -> Any:
        """Plot simulated price paths.
        
        Args:
            n_paths: Number of paths to plot
            show_percentiles: Show 5th, 50th, 95th percentile bands
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot sample paths
        n_to_plot = min(n_paths, self.price_paths.shape[0])
        indices = np.random.choice(
            self.price_paths.shape[0], 
            n_to_plot, 
            replace=False
        )
        
        for i in indices:
            ax.plot(
                self.price_paths[i], 
                alpha=0.1, 
                color='blue', 
                linewidth=0.5
            )
        
        if show_percentiles:
            median, lower, upper = self.get_confidence_band()
            x = np.arange(len(median))
            
            ax.plot(x, median, color='red', linewidth=2, label='Median')
            ax.fill_between(
                x, lower, upper, 
                alpha=0.3, 
                color='red', 
                label='5-95% CI'
            )
        
        ax.axhline(
            y=self.config.initial_capital, 
            color='black', 
            linestyle='--', 
            label='Initial'
        )
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Portfolio Value')
        ax.set_title('Monte Carlo Simulation - Price Paths')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_distribution(
        self, 
        metric: str = 'terminal_values',
        bins: int = 50,
        figsize: Tuple[int, int] = (10, 6)
    ) -> Any:
        """Plot distribution of a metric.
        
        Args:
            metric: 'terminal_values', 'total_returns', 'max_drawdowns', 'sharpe_ratios'
            bins: Number of histogram bins
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        data = getattr(self, metric)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.hist(data, bins=bins, density=True, alpha=0.7, color='steelblue')
        
        # Add percentile lines
        for pct, color in [(5, 'red'), (50, 'green'), (95, 'red')]:
            val = np.percentile(data, pct)
            ax.axvline(x=val, color=color, linestyle='--', 
                       label=f'{pct}th pct: {val:.2f}')
        
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


class MonteCarloSimulator:
    """Monte Carlo simulator for portfolio risk assessment.
    
    Generates thousands of possible future scenarios based on
    historical return characteristics, enabling robust risk estimation.
    
    Supports multiple return models:
    - Historical bootstrapping (non-parametric)
    - Normal distribution (Gaussian)
    - Student's t-distribution (fat tails)
    - Block bootstrap (preserves autocorrelation)
    - GARCH (volatility clustering)
    
    Example:
        >>> # From historical returns
        >>> simulator = MonteCarloSimulator(returns=daily_returns)
        >>> result = simulator.run()
        >>> print(result.metrics.summary())
        >>> 
        >>> # With custom configuration
        >>> config = SimulationConfig(
        ...     n_simulations=50000,
        ...     time_horizon=504,  # 2 years
        ...     return_model=ReturnModel.T_DISTRIBUTION
        ... )
        >>> simulator = MonteCarloSimulator(returns=returns, config=config)
        >>> result = simulator.run()
    """
    
    def __init__(
        self,
        returns: Optional[Union[pd.Series, np.ndarray]] = None,
        prices: Optional[Union[pd.Series, np.ndarray]] = None,
        config: Optional[SimulationConfig] = None,
        n_simulations: int = 10000,
        time_horizon: int = 252,
        return_model: Union[str, ReturnModel] = ReturnModel.HISTORICAL,
        **kwargs
    ):
        """Initialize Monte Carlo simulator.
        
        Args:
            returns: Historical returns series (preferred)
            prices: Historical price series (will convert to returns)
            config: SimulationConfig object
            n_simulations: Number of simulations to run
            time_horizon: Number of periods to simulate
            return_model: Model for generating returns
            **kwargs: Additional config parameters
        """
        # Build configuration
        if config is not None:
            self.config = config
        else:
            if isinstance(return_model, str):
                return_model = ReturnModel(return_model)
            
            self.config = SimulationConfig(
                n_simulations=n_simulations,
                time_horizon=time_horizon,
                return_model=return_model,
                **kwargs
            )
        
        # Process input data
        if returns is not None:
            self.returns = returns.values if hasattr(returns, 'values') else returns
        elif prices is not None:
            prices_arr = prices.values if hasattr(prices, 'values') else prices
            self.returns = np.diff(prices_arr) / prices_arr[:-1]
        else:
            raise ValueError("Either returns or prices must be provided")
        
        # Remove NaN values
        self.returns = self.returns[~np.isnan(self.returns)]
        
        # Estimate distribution parameters
        self._estimate_parameters()
        
        # Set random state
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)
        
        logger.info(
            f"MonteCarloSimulator initialized: {self.config.n_simulations} simulations, "
            f"{self.config.time_horizon} period horizon, model={self.config.return_model.value}"
        )
    
    def _estimate_parameters(self):
        """Estimate distribution parameters from historical returns."""
        self.mean = np.mean(self.returns)
        self.std = np.std(self.returns)
        self.skew = scipy_stats.skew(self.returns)
        self.kurtosis = scipy_stats.kurtosis(self.returns)
        
        # Estimate t-distribution degrees of freedom
        if self.config.t_df is None:
            # Use excess kurtosis to estimate df: kurtosis = 6 / (df - 4) for df > 4
            excess_kurt = self.kurtosis
            if excess_kurt > 0:
                # df = 6 / kurtosis + 4
                self.config.t_df = max(3, 6 / excess_kurt + 4)
            else:
                self.config.t_df = 30  # Approximately normal
        
        logger.debug(
            f"Estimated params: mean={self.mean:.6f}, std={self.std:.6f}, "
            f"skew={self.skew:.2f}, kurt={self.kurtosis:.2f}"
        )
    
    def _generate_returns_historical(self) -> np.ndarray:
        """Generate returns using bootstrap from historical data."""
        n_sims = self.config.n_simulations
        horizon = self.config.time_horizon
        
        # Random sampling with replacement
        indices = np.random.randint(0, len(self.returns), size=(n_sims, horizon))
        return self.returns[indices]
    
    def _generate_returns_normal(self) -> np.ndarray:
        """Generate returns from normal distribution."""
        n_sims = self.config.n_simulations
        horizon = self.config.time_horizon
        
        return np.random.normal(
            loc=self.mean,
            scale=self.std,
            size=(n_sims, horizon)
        )
    
    def _generate_returns_t_distribution(self) -> np.ndarray:
        """Generate returns from Student's t-distribution."""
        n_sims = self.config.n_simulations
        horizon = self.config.time_horizon
        df = self.config.t_df
        
        # Generate t-distributed random numbers, then scale
        t_samples = scipy_stats.t.rvs(df=df, size=(n_sims, horizon))
        
        # Scale to match historical mean and std
        # t-distribution has std = sqrt(df / (df - 2)) for df > 2
        t_std = np.sqrt(df / (df - 2)) if df > 2 else 1.0
        
        return self.mean + (t_samples / t_std) * self.std
    
    def _generate_returns_block_bootstrap(self) -> np.ndarray:
        """Generate returns using block bootstrap to preserve autocorrelation."""
        n_sims = self.config.n_simulations
        horizon = self.config.time_horizon
        block_size = self.config.block_size
        
        n_blocks = int(np.ceil(horizon / block_size))
        max_start = len(self.returns) - block_size
        
        if max_start < 0:
            logger.warning("Block size larger than history, falling back to simple bootstrap")
            return self._generate_returns_historical()
        
        simulated_returns = np.zeros((n_sims, horizon))
        
        for i in range(n_sims):
            blocks = []
            for _ in range(n_blocks):
                start = np.random.randint(0, max_start + 1)
                blocks.append(self.returns[start:start + block_size])
            
            path = np.concatenate(blocks)[:horizon]
            simulated_returns[i] = path
        
        return simulated_returns
    
    def _generate_returns_garch(self) -> np.ndarray:
        """Generate returns using GARCH model for volatility clustering.
        
        Uses a simple GARCH(1,1) model:
        r_t = mu + sigma_t * z_t
        sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
        """
        n_sims = self.config.n_simulations
        horizon = self.config.time_horizon
        
        # Estimate GARCH parameters using moment matching
        # These are simplified estimates; for production, use arch package
        omega = self.std ** 2 * 0.1  # Long-run variance contribution
        alpha = 0.1  # ARCH effect
        beta = 0.85  # GARCH persistence
        
        # Ensure stationarity: alpha + beta < 1
        if alpha + beta >= 1:
            beta = 0.9 - alpha
        
        simulated_returns = np.zeros((n_sims, horizon))
        
        # Initialize variance
        var_t = np.full(n_sims, self.std ** 2)
        
        for t in range(horizon):
            # Generate innovations
            z = np.random.standard_normal(n_sims)
            
            # Generate returns
            sigma_t = np.sqrt(var_t)
            simulated_returns[:, t] = self.mean + sigma_t * z
            
            # Update variance for next period
            var_t = omega + alpha * simulated_returns[:, t] ** 2 + beta * var_t
        
        return simulated_returns
    
    def _generate_returns(self) -> np.ndarray:
        """Generate simulated returns based on configured model."""
        model = self.config.return_model
        
        generators = {
            ReturnModel.HISTORICAL: self._generate_returns_historical,
            ReturnModel.NORMAL: self._generate_returns_normal,
            ReturnModel.T_DISTRIBUTION: self._generate_returns_t_distribution,
            ReturnModel.BLOCK_BOOTSTRAP: self._generate_returns_block_bootstrap,
            ReturnModel.GARCH: self._generate_returns_garch,
        }
        
        generator = generators.get(model)
        if generator is None:
            raise ValueError(f"Unknown return model: {model}")
        
        return generator()
    
    def _compute_price_paths(self, returns: np.ndarray) -> np.ndarray:
        """Convert returns to price paths."""
        initial = self.config.initial_capital
        
        # Cumulative returns: (1 + r1) * (1 + r2) * ...
        cumulative = np.cumprod(1 + returns, axis=1)
        
        # Prepend initial value
        prices = np.zeros((returns.shape[0], returns.shape[1] + 1))
        prices[:, 0] = initial
        prices[:, 1:] = initial * cumulative
        
        return prices
    
    def _compute_drawdowns(self, prices: np.ndarray) -> np.ndarray:
        """Compute maximum drawdown for each simulation path."""
        # Running maximum
        running_max = np.maximum.accumulate(prices, axis=1)
        
        # Drawdown at each point
        drawdowns = (running_max - prices) / running_max
        
        # Maximum drawdown per path
        max_drawdowns = np.max(drawdowns, axis=1)
        
        return max_drawdowns
    
    def _compute_sharpe_ratios(self, returns: np.ndarray) -> np.ndarray:
        """Compute Sharpe ratio for each simulation path."""
        # Daily risk-free rate
        rf_daily = self.config.risk_free_rate / 252
        
        # Excess returns
        excess_returns = returns - rf_daily
        
        # Sharpe ratio (annualized)
        mean_excess = np.mean(excess_returns, axis=1)
        std_returns = np.std(returns, axis=1)
        
        # Avoid division by zero
        std_returns[std_returns < 1e-10] = 1e-10
        
        sharpe = mean_excess / std_returns * np.sqrt(252)
        
        return sharpe
    
    def _compute_risk_metrics(
        self,
        price_paths: np.ndarray,
        return_paths: np.ndarray,
        max_drawdowns: np.ndarray,
        sharpe_ratios: np.ndarray
    ) -> RiskMetrics:
        """Compute comprehensive risk metrics from simulation results."""
        initial = self.config.initial_capital
        
        # Terminal values
        terminal_values = price_paths[:, -1]
        total_returns = (terminal_values - initial) / initial
        
        metrics = RiskMetrics()
        
        # VaR (loss at confidence level)
        # VaR is typically reported as a positive number representing potential loss
        metrics.var_90 = -np.percentile(total_returns, 10)
        metrics.var_95 = -np.percentile(total_returns, 5)
        metrics.var_99 = -np.percentile(total_returns, 1)
        
        # CVaR / Expected Shortfall (average loss beyond VaR)
        metrics.cvar_90 = -np.mean(total_returns[total_returns <= np.percentile(total_returns, 10)])
        metrics.cvar_95 = -np.mean(total_returns[total_returns <= np.percentile(total_returns, 5)])
        metrics.cvar_99 = -np.mean(total_returns[total_returns <= np.percentile(total_returns, 1)])
        
        # Drawdown metrics
        metrics.max_drawdown_mean = np.mean(max_drawdowns)
        metrics.max_drawdown_median = np.median(max_drawdowns)
        metrics.max_drawdown_95 = np.percentile(max_drawdowns, 95)
        metrics.max_drawdown_99 = np.percentile(max_drawdowns, 99)
        
        # Return metrics
        metrics.expected_return = np.mean(total_returns)
        metrics.return_std = np.std(total_returns)
        metrics.return_skew = scipy_stats.skew(total_returns)
        metrics.return_kurtosis = scipy_stats.kurtosis(total_returns)
        
        # Terminal wealth metrics
        metrics.terminal_wealth_mean = np.mean(terminal_values)
        metrics.terminal_wealth_median = np.median(terminal_values)
        metrics.terminal_wealth_5th = np.percentile(terminal_values, 5)
        metrics.terminal_wealth_95th = np.percentile(terminal_values, 95)
        
        # Probability metrics
        metrics.prob_loss = np.mean(total_returns < 0)
        metrics.prob_loss_10pct = np.mean(total_returns < -0.10)
        metrics.prob_loss_20pct = np.mean(total_returns < -0.20)
        metrics.prob_gain_10pct = np.mean(total_returns > 0.10)
        
        # Sharpe ratio metrics
        metrics.sharpe_mean = np.mean(sharpe_ratios)
        metrics.sharpe_median = np.median(sharpe_ratios)
        metrics.sharpe_5th = np.percentile(sharpe_ratios, 5)
        metrics.sharpe_95th = np.percentile(sharpe_ratios, 95)
        
        return metrics
    
    def run(self) -> SimulationResult:
        """Run Monte Carlo simulation.
        
        Returns:
            SimulationResult containing paths, metrics, and statistics
        """
        logger.info("Running Monte Carlo simulation...")
        
        # Generate returns
        return_paths = self._generate_returns()
        
        # Convert to price paths
        price_paths = self._compute_price_paths(return_paths)
        
        # Compute per-path statistics
        terminal_values = price_paths[:, -1]
        total_returns = (terminal_values - self.config.initial_capital) / self.config.initial_capital
        max_drawdowns = self._compute_drawdowns(price_paths)
        sharpe_ratios = self._compute_sharpe_ratios(return_paths)
        
        # Compute risk metrics
        metrics = self._compute_risk_metrics(
            price_paths, return_paths, max_drawdowns, sharpe_ratios
        )
        
        result = SimulationResult(
            price_paths=price_paths,
            return_paths=return_paths,
            metrics=metrics,
            config=self.config,
            terminal_values=terminal_values,
            max_drawdowns=max_drawdowns,
            total_returns=total_returns,
            sharpe_ratios=sharpe_ratios,
        )
        
        logger.info(
            f"Simulation complete. VaR(95%)={metrics.var_95:.2%}, "
            f"Expected Return={metrics.expected_return:.2%}"
        )
        
        return result
    
    def run_stress_test(
        self,
        scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, SimulationResult]:
        """Run simulation under different stress scenarios.
        
        Args:
            scenarios: Dict of scenario_name -> parameter adjustments
                Parameters can include: 'mean_shift', 'vol_mult', 'correlation_shock'
                
        Returns:
            Dict of scenario_name -> SimulationResult
            
        Example:
            >>> scenarios = {
            ...     'base': {},
            ...     'high_vol': {'vol_mult': 1.5},
            ...     'crash': {'mean_shift': -0.02, 'vol_mult': 2.0},
            ...     'low_vol_rally': {'mean_shift': 0.01, 'vol_mult': 0.5}
            ... }
            >>> results = simulator.run_stress_test(scenarios)
        """
        results = {}
        
        # Store original parameters
        original_mean = self.mean
        original_std = self.std
        
        for name, params in scenarios.items():
            logger.info(f"Running stress test scenario: {name}")
            
            # Apply parameter adjustments
            self.mean = original_mean + params.get('mean_shift', 0)
            self.std = original_std * params.get('vol_mult', 1.0)
            
            # Run simulation
            results[name] = self.run()
        
        # Restore original parameters
        self.mean = original_mean
        self.std = original_std
        
        return results
    
    def compute_var_at_horizons(
        self,
        horizons: List[int],
        confidence: float = 0.95
    ) -> Dict[int, float]:
        """Compute VaR at multiple time horizons.
        
        Args:
            horizons: List of time horizons (in days)
            confidence: Confidence level
            
        Returns:
            Dict of horizon -> VaR
        """
        results = {}
        original_horizon = self.config.time_horizon
        
        for horizon in horizons:
            self.config.time_horizon = horizon
            sim_result = self.run()
            
            percentile = (1 - confidence) * 100
            var = -np.percentile(sim_result.total_returns, percentile)
            results[horizon] = var
        
        self.config.time_horizon = original_horizon
        
        return results


class PortfolioMonteCarloSimulator:
    """Monte Carlo simulator for multi-asset portfolios.
    
    Handles correlation structure between assets and portfolio-level risk.
    
    Example:
        >>> # With correlated assets
        >>> simulator = PortfolioMonteCarloSimulator(
        ...     returns=returns_df,  # DataFrame with columns for each asset
        ...     weights={'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
        ... )
        >>> result = simulator.run()
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
        config: Optional[SimulationConfig] = None,
        **kwargs
    ):
        """Initialize portfolio simulator.
        
        Args:
            returns: DataFrame of asset returns (columns = assets)
            weights: Dict of asset -> weight (default: equal weight)
            config: SimulationConfig object
            **kwargs: Additional config parameters
        """
        self.returns = returns
        self.assets = list(returns.columns)
        
        # Default to equal weights
        if weights is None:
            n_assets = len(self.assets)
            self.weights = {asset: 1.0 / n_assets for asset in self.assets}
        else:
            self.weights = weights
            # Normalize weights
            total = sum(self.weights.values())
            self.weights = {k: v / total for k, v in self.weights.items()}
        
        # Configuration
        self.config = config or SimulationConfig(**kwargs)
        
        # Compute correlation matrix
        self.correlation_matrix = returns.corr().values
        self.means = returns.mean().values
        self.stds = returns.std().values
        
        # Weight vector
        self.weight_vector = np.array([self.weights.get(a, 0) for a in self.assets])
        
        # Cholesky decomposition for correlated sampling
        try:
            self.cholesky = np.linalg.cholesky(self.correlation_matrix)
        except np.linalg.LinAlgError:
            # Matrix not positive definite, use nearest positive definite
            logger.warning("Correlation matrix not positive definite, adjusting...")
            self.correlation_matrix = self._nearest_positive_definite(
                self.correlation_matrix
            )
            self.cholesky = np.linalg.cholesky(self.correlation_matrix)
        
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)
    
    def _nearest_positive_definite(self, A: np.ndarray) -> np.ndarray:
        """Find nearest positive definite matrix."""
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        
        if self._is_positive_definite(A3):
            return A3
        
        # Add small positive value to diagonal
        eps = 1e-8
        I = np.eye(A.shape[0])
        for _ in range(100):
            try:
                np.linalg.cholesky(A3 + eps * I)
                return A3 + eps * I
            except np.linalg.LinAlgError:
                eps *= 2
        
        return A3 + eps * I
    
    def _is_positive_definite(self, A: np.ndarray) -> bool:
        """Check if matrix is positive definite."""
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def _generate_correlated_returns(self) -> np.ndarray:
        """Generate correlated multi-asset returns."""
        n_sims = self.config.n_simulations
        horizon = self.config.time_horizon
        n_assets = len(self.assets)
        
        # Generate uncorrelated standard normal samples
        Z = np.random.standard_normal((n_sims, horizon, n_assets))
        
        # Apply Cholesky transformation to induce correlation
        # For each time step, transform the n_assets dimensional vector
        correlated = np.zeros_like(Z)
        for t in range(horizon):
            correlated[:, t, :] = Z[:, t, :] @ self.cholesky.T
        
        # Scale by mean and std
        returns = correlated * self.stds + self.means
        
        return returns
    
    def run(self) -> SimulationResult:
        """Run portfolio Monte Carlo simulation.
        
        Returns:
            SimulationResult for the portfolio
        """
        logger.info("Running portfolio Monte Carlo simulation...")
        
        # Generate correlated asset returns
        asset_returns = self._generate_correlated_returns()  # (n_sims, horizon, n_assets)
        
        # Compute portfolio returns as weighted sum
        portfolio_returns = np.einsum('ijk,k->ij', asset_returns, self.weight_vector)
        
        # Create single-asset simulator to compute metrics
        portfolio_simulator = MonteCarloSimulator(
            returns=portfolio_returns.flatten(),
            config=self.config
        )
        portfolio_simulator.returns = portfolio_returns.flatten()
        
        # Compute price paths and metrics
        price_paths = portfolio_simulator._compute_price_paths(portfolio_returns)
        terminal_values = price_paths[:, -1]
        total_returns = (terminal_values - self.config.initial_capital) / self.config.initial_capital
        max_drawdowns = portfolio_simulator._compute_drawdowns(price_paths)
        sharpe_ratios = portfolio_simulator._compute_sharpe_ratios(portfolio_returns)
        
        metrics = portfolio_simulator._compute_risk_metrics(
            price_paths, portfolio_returns, max_drawdowns, sharpe_ratios
        )
        
        result = SimulationResult(
            price_paths=price_paths,
            return_paths=portfolio_returns,
            metrics=metrics,
            config=self.config,
            terminal_values=terminal_values,
            max_drawdowns=max_drawdowns,
            total_returns=total_returns,
            sharpe_ratios=sharpe_ratios,
        )
        
        logger.info(
            f"Portfolio simulation complete. VaR(95%)={metrics.var_95:.2%}, "
            f"Expected Return={metrics.expected_return:.2%}"
        )
        
        return result
    
    def analyze_weights(
        self,
        weight_scenarios: List[Dict[str, float]]
    ) -> pd.DataFrame:
        """Analyze risk metrics under different weight scenarios.
        
        Args:
            weight_scenarios: List of weight dictionaries
            
        Returns:
            DataFrame comparing metrics across weight scenarios
        """
        results = []
        
        for i, weights in enumerate(weight_scenarios):
            self.weights = weights
            total = sum(weights.values())
            self.weights = {k: v / total for k, v in weights.items()}
            self.weight_vector = np.array([
                self.weights.get(a, 0) for a in self.assets
            ])
            
            sim_result = self.run()
            
            row = sim_result.metrics.to_dict()
            row['scenario'] = i
            row.update({f'weight_{k}': v for k, v in weights.items()})
            
            results.append(row)
        
        return pd.DataFrame(results)

"""Position sizing methods for portfolio construction."""

from typing import Optional, Any

import pandas as pd
import numpy as np

from jsf.portfolio.base import PositionSizer
from jsf.data import PriceData
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class EqualWeightSizer(PositionSizer):
    """
    Equal weight position sizing.
    
    Assigns equal weight to all positions with non-zero signals.
    """
    
    def __init__(
        self,
        long_only: bool = False,
        max_positions: Optional[int] = None,
        name: str = "equal_weight",
    ):
        """
        Initialize equal weight sizer.
        
        Args:
            long_only: If True, only take long positions
            max_positions: Maximum number of positions (None = unlimited)
            name: Sizer name
        """
        super().__init__(
            name=name,
            long_only=long_only,
            max_positions=max_positions,
        )
        self.long_only = long_only
        self.max_positions = max_positions
    
    def size(
        self,
        signals: pd.DataFrame,
        price_data: Optional[PriceData] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Calculate equal-weighted positions."""
        weights = signals.copy()
        
        # Assign equal weight
        result = pd.DataFrame(
            index=weights.index,
            columns=weights.columns,
            dtype=float,
        )
        
        for idx in weights.index:
            row = weights.loc[idx]
            active = row != 0
            
            if self.long_only:
                # Only consider non-negative signals
                active = active & (row > 0)
            
            # Apply max_positions limit if specified
            if self.max_positions is not None and active.sum() > self.max_positions:
                # Select top positions by absolute signal value
                top_positions = row.abs().nlargest(self.max_positions).index
                active = row.index.isin(top_positions)
                if self.long_only:
                    active = active & (row > 0)
            
            if active.sum() > 0:
                # Equal absolute weight (always positive)
                equal_weight = 1.0 / active.sum()
                result.loc[idx, active] = equal_weight
            else:
                result.loc[idx] = 0.0
        
        return result.fillna(0)


class SignalWeightedSizer(PositionSizer):
    """
    Signal-weighted position sizing.
    
    Weights positions proportionally to signal strength.
    """
    
    def __init__(
        self,
        normalize: bool = True,
        long_only: bool = False,
        signal_scale: float = 1.0,
        name: str = "signal_weighted",
    ):
        """
        Initialize signal-weighted sizer.
        
        Args:
            normalize: If True, normalize weights to sum to 1
            long_only: If True, only take long positions
            signal_scale: Scaling factor for signals
            name: Sizer name
        """
        super().__init__(
            name=name,
            normalize=normalize,
            long_only=long_only,
            signal_scale=signal_scale,
        )
        self.normalize = normalize
        self.long_only = long_only
        self.signal_scale = signal_scale
    
    def size(
        self,
        signals: pd.DataFrame,
        price_data: Optional[PriceData] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Calculate signal-weighted positions."""
        weights = signals.copy()
        
        if self.long_only:
            weights = weights.clip(lower=0)
        
        if self.normalize:
            # Normalize row-wise
            if self.long_only:
                row_sums = weights.sum(axis=1)
                weights = weights.div(row_sums, axis=0).fillna(0)
            else:
                # Normalize long and short separately
                long_weights = weights.clip(lower=0)
                short_weights = weights.clip(upper=0)
                
                long_sums = long_weights.sum(axis=1)
                short_sums = short_weights.abs().sum(axis=1)
                
                long_weights = long_weights.div(long_sums, axis=0).fillna(0) * 0.5
                short_weights = short_weights.div(short_sums, axis=0).fillna(0) * 0.5
                
                weights = long_weights + short_weights
        
        return weights


class VolatilityScaledSizer(PositionSizer):
    """
    Volatility-scaled position sizing.
    
    Scales positions inversely to volatility (risk parity approach).
    """
    
    def __init__(
        self,
        lookback: int = 60,
        target_volatility: float = 0.10,
        long_only: bool = True,
        name: str = "volatility_scaled",
    ):
        """
        Initialize volatility-scaled sizer.
        
        Args:
            lookback: Lookback period for volatility calculation
            target_volatility: Target portfolio volatility
            long_only: If True, only take long positions
            name: Sizer name
        """
        super().__init__(
            name=name,
            lookback=lookback,
            target_volatility=target_volatility,
            long_only=long_only,
        )
        self.lookback = lookback
        self.target_volatility = target_volatility
        self.long_only = long_only
    
    def size(
        self,
        signals: pd.DataFrame,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Calculate volatility-scaled positions."""
        # Calculate returns
        returns = price_data.get_returns(periods=1)
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=self.lookback).std()
        
        # Inverse volatility weights
        inv_vol = 1.0 / (volatility + 1e-10)
        
        # Apply signal direction
        if self.long_only:
            signal_direction = (signals > 0).astype(float)
        else:
            signal_direction = np.sign(signals)
        
        # Combine signal with inverse volatility
        weights = inv_vol * signal_direction * abs(signals)
        
        # Normalize to target volatility
        row_sums = weights.abs().sum(axis=1)
        weights = weights.div(row_sums, axis=0).fillna(0)
        
        return weights


class RiskParitySizer(PositionSizer):
    """
    Risk parity position sizing.
    
    Allocates capital such that each position contributes equal risk.
    """
    
    def __init__(
        self,
        lookback: int = 60,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        name: str = "risk_parity",
    ):
        """
        Initialize risk parity sizer.
        
        Args:
            lookback: Lookback for covariance calculation
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
            name: Sizer name
        """
        super().__init__(
            name=name,
            lookback=lookback,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        self.lookback = lookback
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def size(
        self,
        signals: pd.DataFrame,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Calculate risk parity positions."""
        returns = price_data.get_returns(periods=1)
        
        weights = pd.DataFrame(
            index=signals.index,
            columns=signals.columns,
            dtype=float,
        )
        
        for idx in signals.index:
            # Get active signals
            sig_row = signals.loc[idx]
            active = sig_row != 0
            
            if active.sum() < 2:
                # Not enough assets for risk parity
                weights.loc[idx] = 0.0
                continue
            
            # Get historical returns for active assets
            hist_start = max(0, signals.index.get_loc(idx) - self.lookback)
            hist_returns = returns.iloc[hist_start:signals.index.get_loc(idx), :]
            hist_returns = hist_returns.loc[:, active]
            
            if len(hist_returns) < 10:
                weights.loc[idx] = 0.0
                continue
            
            # Calculate covariance
            cov_matrix = hist_returns.cov()
            
            # Risk parity optimization (simplified iterative approach)
            n_assets = active.sum()
            w = np.ones(n_assets) / n_assets  # Start with equal weights
            
            for _ in range(self.max_iterations):
                w_old = w.copy()
                
                # Calculate risk contributions
                portfolio_var = w @ cov_matrix @ w
                marginal_risk = cov_matrix @ w
                risk_contrib = w * marginal_risk
                
                # Update weights inversely proportional to risk contribution
                w = w / (risk_contrib + 1e-10)
                w = w / w.sum()  # Normalize
                
                if np.linalg.norm(w - w_old) < self.tolerance:
                    break
            
            # Apply signal direction
            w = w * np.sign(sig_row[active].values)
            
            weights.loc[idx, active] = w
            weights.loc[idx, ~active] = 0.0
        
        return weights.fillna(0)


class KellyCriterionSizer(PositionSizer):
    """
    Kelly Criterion position sizing.
    
    Sizes positions based on expected return and variance (Kelly formula).
    """
    
    def __init__(
        self,
        lookback: int = 60,
        fraction: float = 0.5,
        kelly_fraction: float = None,  # Alias for fraction
        long_only: bool = True,
        name: str = "kelly",
    ):
        """
        Initialize Kelly criterion sizer.
        
        Args:
            lookback: Lookback for return/variance estimation
            fraction: Fraction of Kelly (0.5 = half Kelly)
            kelly_fraction: Alias for fraction parameter
            long_only: If True, only take long positions
            name: Sizer name
        """
        # Handle kelly_fraction alias
        if kelly_fraction is not None:
            fraction = kelly_fraction
        
        super().__init__(
            name=name,
            lookback=lookback,
            fraction=fraction,
            long_only=long_only,
        )
        self.lookback = lookback
        self.fraction = fraction
        self.long_only = long_only
    
    def size(
        self,
        signals: pd.DataFrame,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Calculate Kelly criterion positions."""
        returns = price_data.get_returns(periods=1)
        
        # Calculate rolling mean and variance
        expected_returns = returns.rolling(window=self.lookback).mean()
        variance = returns.rolling(window=self.lookback).var()
        
        # Kelly formula: f = mu / sigma^2
        kelly_weights = expected_returns / (variance + 1e-10)
        kelly_weights = kelly_weights * self.fraction  # Fractional Kelly
        
        # Apply signal direction and strength
        if self.long_only:
            signal_filter = (signals > 0).astype(float)
        else:
            signal_filter = np.sign(signals)
        
        weights = kelly_weights * signal_filter * abs(signals)
        
        # Clip extreme values
        weights = weights.clip(lower=-1.0, upper=1.0)
        
        # Normalize
        if self.long_only:
            row_sums = weights.clip(lower=0).sum(axis=1)
        else:
            row_sums = weights.abs().sum(axis=1)
        
        weights = weights.div(row_sums, axis=0).fillna(0)
        
        return weights

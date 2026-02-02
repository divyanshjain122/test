"""Rebalancing strategies for portfolio management."""

from typing import Optional, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from jsf.portfolio.base import Rebalancer, RebalanceFrequency
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class PeriodicRebalancer(Rebalancer):
    """
    Periodic rebalancing strategy.
    
    Rebalances portfolio at fixed intervals.
    """
    
    def __init__(
        self,
        frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY,
        custom_days: Optional[int] = None,
        name: str = "periodic",
    ):
        """
        Initialize periodic rebalancer.
        
        Args:
            frequency: Rebalancing frequency
            custom_days: Custom rebalance period in days (if frequency=CUSTOM)
            name: Rebalancer name
        """
        super().__init__(
            name=name,
            frequency=frequency.value,
            custom_days=custom_days,
        )
        self.frequency = frequency
        self.custom_days = custom_days
        self.last_rebalance: Optional[datetime] = None
    
    def should_rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        date: datetime,
        **kwargs: Any
    ) -> bool:
        """Determine if enough time has passed for rebalancing."""
        if self.last_rebalance is None:
            self.last_rebalance = date
            return True
        
        if self.frequency == RebalanceFrequency.DAILY:
            should = (date - self.last_rebalance).days >= 1
        
        elif self.frequency == RebalanceFrequency.WEEKLY:
            should = (date - self.last_rebalance).days >= 7
        
        elif self.frequency == RebalanceFrequency.MONTHLY:
            # Rebalance if month changed
            should = date.month != self.last_rebalance.month
        
        elif self.frequency == RebalanceFrequency.QUARTERLY:
            # Rebalance if quarter changed
            should = (date.month - 1) // 3 != (self.last_rebalance.month - 1) // 3
        
        elif self.frequency == RebalanceFrequency.ANNUALLY:
            should = date.year != self.last_rebalance.year
        
        elif self.frequency == RebalanceFrequency.CUSTOM:
            if self.custom_days is None:
                raise ValueError("custom_days must be specified for CUSTOM frequency")
            should = (date - self.last_rebalance).days >= self.custom_days
        
        else:
            should = False
        
        if should:
            self.last_rebalance = date
        
        return should
    
    def rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        **kwargs: Any
    ) -> pd.Series:
        """Rebalance to target weights."""
        return target_weights


class ThresholdRebalancer(Rebalancer):
    """
    Threshold-based rebalancing.
    
    Rebalances when weights drift beyond threshold.
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.05,
        name: str = "threshold",
    ):
        """
        Initialize threshold rebalancer.
        
        Args:
            drift_threshold: Maximum allowed drift from target
            name: Rebalancer name
        """
        super().__init__(name=name, drift_threshold=drift_threshold)
        self.drift_threshold = drift_threshold
    
    def should_rebalance(
        self,
        *args,
        **kwargs: Any
    ) -> bool:
        """
        Check if drift exceeds threshold.
        
        Supports multiple signatures for backward compatibility:
        - (current_weights, target_weights, date, **kwargs)
        - (date, current_weights, date2, target_weights)
        """
        # Parse flexible arguments
        if len(args) >= 4:
            # Legacy API: (date, current, date, target)
            current_weights = args[1]
            target_weights = args[3]
        elif len(args) == 3:
            # New API: (current, target, date)
            current_weights = args[0]
            target_weights = args[1]
        else:
            # Kwargs
            current_weights = kwargs.get('current_weights', args[0] if len(args) > 0 else None)
            target_weights = kwargs.get('target_weights', args[1] if len(args) > 1 else None)
        
        # Skip if either is None
        if current_weights is None or target_weights is None:
            return False
        
        drift = (current_weights - target_weights).abs().sum()
        return drift > self.drift_threshold
    
    def rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        **kwargs: Any
    ) -> pd.Series:
        """Rebalance to target weights."""
        return target_weights


class VolatilityTargetRebalancer(Rebalancer):
    """
    Volatility-target rebalancing.
    
    Rebalances when position drift exceeds threshold.
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        absolute: bool = False,
        name: str = "threshold",
    ):
        """
        Initialize threshold rebalancer.
        
        Args:
            threshold: Rebalancing threshold (0.05 = 5% drift)
            absolute: If True, use absolute drift; else relative
            name: Rebalancer name
        """
        super().__init__(
            name=name,
            threshold=threshold,
            absolute=absolute,
        )
        self.threshold = threshold
        self.absolute = absolute
    
    def should_rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        date: datetime,
        **kwargs: Any
    ) -> bool:
        """Check if any position has drifted beyond threshold."""
        # Align indices
        common_idx = current_weights.index.intersection(target_weights.index)
        current = current_weights.reindex(common_idx, fill_value=0)
        target = target_weights.reindex(common_idx, fill_value=0)
        
        if self.absolute:
            # Absolute drift
            drift = (current - target).abs()
            max_drift = drift.max()
        else:
            # Relative drift
            drift = ((current - target) / (target.abs() + 1e-10)).abs()
            max_drift = drift.max()
        
        return max_drift > self.threshold
    
    def rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        **kwargs: Any
    ) -> pd.Series:
        """Rebalance to target weights."""
        return target_weights


class VolatilityTargetRebalancer(Rebalancer):
    """
    Volatility-targeted rebalancing.
    
    Adjusts exposure to maintain target volatility.
    """
    
    def __init__(
        self,
        target_volatility: float = 0.10,
        lookback: int = 60,
        tolerance: float = 0.02,
        rebalance_frequency: int = 5,
        name: str = "volatility_target",
    ):
        """
        Initialize volatility target rebalancer.
        
        Args:
            target_volatility: Target annualized volatility
            lookback: Lookback period for volatility estimation
            tolerance: Tolerance band around target (e.g., 0.02 = 2%)
            rebalance_frequency: Check volatility every N days
            name: Rebalancer name
        """
        super().__init__(
            name=name,
            target_volatility=target_volatility,
            lookback=lookback,
            tolerance=tolerance,
            rebalance_frequency=rebalance_frequency,
        )
        self.target_volatility = target_volatility
        self.lookback = lookback
        self.tolerance = tolerance
        self.rebalance_frequency = rebalance_frequency
        self.last_rebalance: Optional[datetime] = None
        self.check_counter = 0
    
    def should_rebalance(
        self,
        *args,
        **kwargs: Any
    ) -> bool:
        """
        Check if volatility check is due.
        
        Supports multiple signatures for backward compatibility.
        """
        self.check_counter += 1
        
        if self.check_counter >= self.rebalance_frequency:
            self.check_counter = 0
            # Extract date if provided
            date = None
            if len(args) >= 3:
                date = args[2]
            if date is not None:
                self.last_rebalance = date
            return True
        
        return False
    
    def rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        **kwargs: Any
    ) -> pd.Series:
        """Adjust weights to target volatility."""
        # Get realized volatility if available
        realized_vol = kwargs.get("realized_volatility", None)
        
        if realized_vol is None:
            # No volatility data, return target weights
            return target_weights
        
        # Calculate scaling factor
        scale = self.target_volatility / (realized_vol + 1e-10)
        
        # Clip extreme scaling
        scale = np.clip(scale, 0.5, 2.0)
        
        # Scale target weights
        scaled_weights = target_weights * scale
        
        # Renormalize
        total = scaled_weights.abs().sum()
        if total > 0:
            scaled_weights = scaled_weights / total
        
        return scaled_weights


class BandRebalancer(Rebalancer):
    """
    Band-based rebalancing.
    
    Rebalances when positions drift outside allowed bands.
    """
    
    def __init__(
        self,
        lower_band: float = -0.05,
        upper_band: float = 0.05,
        partial_rebalance: bool = True,
        rebalance_to_target: bool = True,
        name: str = "band",
    ):
        """
        Initialize band rebalancer.
        
        Args:
            lower_band: Lower deviation band
            upper_band: Upper deviation band
            partial_rebalance: If True, only rebalance drifted positions
            rebalance_to_target: If True, rebalance to target; else to band edge
            name: Rebalancer name
        """
        super().__init__(
            name=name,
            lower_band=lower_band,
            upper_band=upper_band,
            partial_rebalance=partial_rebalance,
            rebalance_to_target=rebalance_to_target,
        )
        self.lower_band = lower_band
        self.upper_band = upper_band
        self.partial_rebalance = partial_rebalance
        self.rebalance_to_target = rebalance_to_target
    
    def should_rebalance(
        self,
        *args,
        **kwargs: Any
    ) -> bool:
        """
        Check if any position is outside bands.
        
        Supports multiple signatures for backward compatibility.
        """
        # Parse flexible arguments
        if len(args) >= 4:
            # Legacy API: (date, current, date, target)
            current_weights = args[1]
            target_weights = args[3]
        elif len(args) == 3:
            # New API: (current, target, date)
            current_weights = args[0]
            target_weights = args[1]
        else:
            current_weights = kwargs.get('current_weights', args[0] if len(args) > 0 else None)
            target_weights = kwargs.get('target_weights', args[1] if len(args) > 1 else None)
        
        if current_weights is None or target_weights is None:
            return False
        
        common_idx = current_weights.index.intersection(target_weights.index)
        current = current_weights.reindex(common_idx, fill_value=0)
        target = target_weights.reindex(common_idx, fill_value=0)
        
        # Check if any position is outside absolute weight bands
        outside_bands = (current < self.lower_band) | (current > self.upper_band)
        
        return outside_bands.any()
    
    def rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        **kwargs: Any
    ) -> pd.Series:
        """Rebalance positions outside bands."""
        if not self.partial_rebalance:
            # Full rebalance
            return target_weights
        
        # Partial rebalance: only adjust positions outside bands
        common_idx = current_weights.index.intersection(target_weights.index)
        current = current_weights.reindex(common_idx, fill_value=0)
        target = target_weights.reindex(common_idx, fill_value=0)
        
        deviation = current - target
        outside_bands = (deviation < self.lower_band) | (deviation > self.upper_band)
        
        # Create rebalanced weights
        rebalanced = current.copy()
        rebalanced[outside_bands] = target[outside_bands]
        
        # Renormalize
        total = rebalanced.abs().sum()
        if total > 0:
            rebalanced = rebalanced / total
        
        return rebalanced


class SmartRebalancer(Rebalancer):
    """
    Smart rebalancing combining multiple triggers.
    
    Uses time-based, threshold-based, and volatility-based triggers.
    """
    
    def __init__(
        self,
        min_days: int = 5,
        min_days_between: int = None,  # Alias for min_days
        max_days: int = 30,
        threshold: float = 0.10,
        drift_threshold: float = None,  # Alias for threshold
        vol_threshold: float = 0.20,
        volatility_target: float = None,  # Volatility target level
        volatility_tolerance: float = None,  # Alias for vol_threshold
        name: str = "smart",
    ):
        """
        Initialize smart rebalancer.
        
        Args:
            min_days: Minimum days between rebalances
            min_days_between: Alias for min_days
            max_days: Maximum days between rebalances
            threshold: Position drift threshold
            drift_threshold: Alias for threshold
            vol_threshold: Volatility change threshold
            volatility_target: Target volatility level
            volatility_tolerance: Alias for vol_threshold
            name: Rebalancer name
        """
        # Handle aliases
        if min_days_between is not None:
            min_days = min_days_between
        if drift_threshold is not None:
            threshold = drift_threshold
        if volatility_tolerance is not None:
            vol_threshold = volatility_tolerance
        
        super().__init__(
            name=name,
            min_days=min_days,
            max_days=max_days,
            threshold=threshold,
            vol_threshold=vol_threshold,
        )
        self.min_days = min_days
        self.min_days_between = min_days  # Store alias
        self.max_days = max_days
        self.threshold = threshold
        self.drift_threshold = threshold  # Store alias
        self.vol_threshold = vol_threshold
        self.volatility_target = volatility_target
        self.volatility_tolerance = vol_threshold  # Store alias
        self.last_rebalance: Optional[datetime] = None
    
    def should_rebalance(
        self,
        *args,
        **kwargs: Any
    ) -> bool:
        """
        Smart rebalancing logic.
        
        Supports multiple signatures for backward compatibility.
        """
        # Parse flexible arguments
        if len(args) >= 4:
            # Legacy API: (current_date, current_weights, last_rebalance, target_weights)
            date = args[0]
            current_weights = args[1]
            last_rebal = args[2]
            target_weights = args[3]
            if last_rebal is not None:
                self.last_rebalance = last_rebal
        elif len(args) == 3:
            # New API: (current, target, date)
            current_weights = args[0]
            target_weights = args[1]
            date = args[2]
        else:
            current_weights = kwargs.get('current_weights')
            target_weights = kwargs.get('target_weights')
            date = kwargs.get('date')
        
        if self.last_rebalance is None:
            if date is not None:
                self.last_rebalance = date
            return True
        
        if date is None:
            return False
        
        days_since = (date - self.last_rebalance).days
        
        # Never rebalance before min_days
        if days_since < self.min_days:
            return False
        
        # Always rebalance after max_days
        if days_since >= self.max_days:
            self.last_rebalance = date
            return True
        
        # Check drift threshold (only if we have weights)
        if current_weights is not None and target_weights is not None:
            common_idx = current_weights.index.intersection(target_weights.index)
            current = current_weights.reindex(common_idx, fill_value=0)
            target = target_weights.reindex(common_idx, fill_value=0)
            
            max_drift = (current - target).abs().max()
            
            if max_drift > self.threshold:
                self.last_rebalance = date
                return True
        
        # Check volatility change if provided
        realized_vol = kwargs.get("realized_volatility", None)
        previous_vol = kwargs.get("previous_volatility", None)
        
        if realized_vol is not None and previous_vol is not None:
            vol_change = abs(realized_vol - previous_vol) / (previous_vol + 1e-10)
            if vol_change > self.vol_threshold:
                self.last_rebalance = date
                return True
        
        return False
    
    def rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        **kwargs: Any
    ) -> pd.Series:
        """Rebalance to target weights."""
        return target_weights

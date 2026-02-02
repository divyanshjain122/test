"""Portfolio constraints for risk management."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class Constraint(ABC):
    """Base class for portfolio constraints."""
    
    def __init__(self, name: str = "constraint"):
        """Initialize constraint."""
        self.name = name
    
    @abstractmethod
    def check(self, weights: pd.Series, **kwargs: Any) -> bool:
        """
        Check if weights satisfy constraint.
        
        Args:
            weights: Portfolio weights
            **kwargs: Additional context
        
        Returns:
            True if constraint satisfied
        """
        pass
    
    @abstractmethod
    def enforce(self, weights: pd.Series, **kwargs: Any) -> pd.Series:
        """
        Enforce constraint on weights.
        
        Args:
            weights: Portfolio weights
            **kwargs: Additional context
        
        Returns:
            Adjusted weights satisfying constraint
        """
        pass


@dataclass
class PortfolioConstraints:
    """
    Collection of portfolio constraints.
    
    Manages multiple constraints and ensures all are satisfied.
    """
    constraints: List[Constraint] = field(default_factory=list)
    
    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint."""
        self.constraints.append(constraint)
        logger.info(f"Added constraint: {constraint.name}")
    
    def check(self, weights: pd.Series, **kwargs: Any) -> bool:
        """Alias for check_all."""
        return self.check_all(weights, **kwargs)
    
    def check_all(self, weights: pd.Series, **kwargs: Any) -> bool:
        """Check if all constraints are satisfied."""
        for constraint in self.constraints:
            if not constraint.check(weights, **kwargs):
                logger.warning(f"Constraint violated: {constraint.name}")
                return False
        return True
    
    def enforce_all(self, weights: pd.Series, **kwargs: Any) -> pd.Series:
        """Enforce all constraints sequentially."""
        result = weights.copy()
        
        for constraint in self.constraints:
            if not constraint.check(result, **kwargs):
                result = constraint.enforce(result, **kwargs)
                logger.debug(f"Enforced constraint: {constraint.name}")
        
        return result


class PositionLimitConstraint(Constraint):
    """
    Position size limits.
    
    Constrains individual position weights.
    """
    
    def __init__(
        self,
        min_weight: float = 0.0,
        max_weight: float = 0.3,
        apply_to_shorts: bool = True,
    ):
        """
        Initialize position limit constraint.
        
        Args:
            min_weight: Minimum position weight
            max_weight: Maximum position weight
            apply_to_shorts: Apply limits to short positions
        """
        super().__init__(name="position_limit")
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.apply_to_shorts = apply_to_shorts
    
    def check(self, weights: pd.Series, **kwargs: Any) -> bool:
        """Check if position limits satisfied."""
        if self.apply_to_shorts:
            # Check absolute values
            violations = weights.abs() > self.max_weight
            return not violations.any()
        else:
            # Only check long positions
            long_positions = weights[weights > 0]
            violations = long_positions > self.max_weight
            return not violations.any()
    
    def enforce(self, weights: pd.Series, **kwargs: Any) -> pd.Series:
        """Enforce position limits."""
        result = weights.copy()
        
        if self.apply_to_shorts:
            # Clip both long and short positions
            result = result.clip(lower=-self.max_weight, upper=self.max_weight)
        else:
            # Only clip long positions
            result[result > 0] = result[result > 0].clip(upper=self.max_weight)
        
        # Renormalize
        total = result.abs().sum()
        if total > 0:
            result = result / total
        
        return result


class SectorConstraint(Constraint):
    """
    Sector exposure constraints.
    
    Limits exposure to specific sectors.
    """
    
    def __init__(
        self,
        sector_map: Dict[str, str],
        max_sector_weight: float = 0.5,
        max_sector_exposure: float = None,  # Alias for max_sector_weight
        min_sector_weight: float = 0.0,
    ):
        """
        Initialize sector constraint.
        
        Args:
            sector_map: Mapping of symbol -> sector
            max_sector_weight: Maximum weight per sector
            max_sector_exposure: Alias for max_sector_weight
            min_sector_weight: Minimum weight per sector
        """
        super().__init__(name="sector")
        # Handle alias
        if max_sector_exposure is not None:
            max_sector_weight = max_sector_exposure
        
        self.sector_map = sector_map
        self.max_sector_weight = max_sector_weight
        self.min_sector_weight = min_sector_weight
    
    def check(self, weights: pd.Series, **kwargs: Any) -> bool:
        """Check if sector limits satisfied."""
        sector_weights = self._calculate_sector_weights(weights)
        
        return (
            (sector_weights >= self.min_sector_weight).all() and
            (sector_weights <= self.max_sector_weight).all()
        )
    
    def enforce(self, weights: pd.Series, **kwargs: Any) -> pd.Series:
        """Enforce sector limits."""
        result = weights.copy()
        sector_weights = self._calculate_sector_weights(result)
        
        # Scale down overweight sectors
        for sector, sector_weight in sector_weights.items():
            if sector_weight > self.max_sector_weight:
                # Get symbols in this sector
                sector_symbols = [
                    sym for sym, sec in self.sector_map.items()
                    if sec == sector and sym in result.index
                ]
                
                # Scale down proportionally
                scale = self.max_sector_weight / sector_weight
                result[sector_symbols] = result[sector_symbols] * scale
        
        # Renormalize
        total = result.abs().sum()
        if total > 0:
            result = result / total
        
        return result
    
    def _calculate_sector_weights(self, weights: pd.Series) -> pd.Series:
        """Calculate sector exposure."""
        sector_weights = {}
        
        for symbol, weight in weights.items():
            if symbol in self.sector_map:
                sector = self.sector_map[symbol]
                sector_weights[sector] = sector_weights.get(sector, 0) + abs(weight)
        
        return pd.Series(sector_weights)


class TurnoverConstraint(Constraint):
    """
    Turnover constraint.
    
    Limits portfolio turnover from previous period.
    """
    
    def __init__(
        self,
        max_turnover: float = 0.5,
    ):
        """
        Initialize turnover constraint.
        
        Args:
            max_turnover: Maximum allowed turnover (0.5 = 50%)
        """
        super().__init__(name="turnover")
        self.max_turnover = max_turnover
    
    def check(self, weights: pd.Series, previous_weights: pd.Series = None, **kwargs: Any) -> bool:
        """Check if turnover is within limit."""
        # Get previous weights from kwargs if not provided directly
        if previous_weights is None:
            previous_weights = kwargs.get("previous_weights", None)
        
        if previous_weights is None:
            return True  # No previous weights to compare
        
        turnover = self._calculate_turnover(weights, previous_weights)
        return turnover <= self.max_turnover
    
    def enforce(self, weights: pd.Series, **kwargs: Any) -> pd.Series:
        """Enforce turnover limit."""
        previous_weights = kwargs.get("previous_weights", None)
        
        if previous_weights is None:
            return weights
        
        # Align indices
        common_idx = weights.index.union(previous_weights.index)
        current = weights.reindex(common_idx, fill_value=0)
        previous = previous_weights.reindex(common_idx, fill_value=0)
        
        # Calculate current turnover
        turnover = self._calculate_turnover(current, previous)
        
        if turnover <= self.max_turnover:
            return weights
        
        # Scale changes to meet turnover constraint
        changes = current - previous
        scale = self.max_turnover / turnover
        
        adjusted = previous + changes * scale
        
        # Renormalize
        total = adjusted.abs().sum()
        if total > 0:
            adjusted = adjusted / total
        
        return adjusted
    
    def _calculate_turnover(
        self,
        current: pd.Series,
        previous: pd.Series
    ) -> float:
        """Calculate turnover between two weight vectors."""
        common_idx = current.index.union(previous.index)
        curr = current.reindex(common_idx, fill_value=0)
        prev = previous.reindex(common_idx, fill_value=0)
        
        return (curr - prev).abs().sum() / 2  # Divide by 2 to avoid double counting


class LeverageConstraint(Constraint):
    """
    Leverage constraint.
    
    Limits gross and/or net exposure.
    """
    
    def __init__(
        self,
        max_gross_leverage: float = 1.0,
        max_net_leverage: Optional[float] = None,
        min_net_leverage: Optional[float] = None,
    ):
        """
        Initialize leverage constraint.
        
        Args:
            max_gross_leverage: Maximum gross exposure
            max_net_leverage: Maximum net exposure (optional)
            min_net_leverage: Minimum net exposure (optional)
        """
        super().__init__(name="leverage")
        self.max_gross_leverage = max_gross_leverage
        self.max_net_leverage = max_net_leverage
        self.min_net_leverage = min_net_leverage
    
    def check(self, weights: pd.Series, **kwargs: Any) -> bool:
        """Check if leverage limits satisfied."""
        gross_leverage = weights.abs().sum()
        net_leverage = weights.sum()
        
        gross_ok = gross_leverage <= self.max_gross_leverage
        
        net_ok = True
        if self.max_net_leverage is not None:
            net_ok = net_ok and (net_leverage <= self.max_net_leverage)
        if self.min_net_leverage is not None:
            net_ok = net_ok and (net_leverage >= self.min_net_leverage)
        
        return gross_ok and net_ok
    
    def enforce(self, weights: pd.Series, **kwargs: Any) -> pd.Series:
        """Enforce leverage limits."""
        result = weights.copy()
        
        # Enforce gross leverage
        gross_leverage = result.abs().sum()
        if gross_leverage > self.max_gross_leverage:
            result = result * (self.max_gross_leverage / gross_leverage)
        
        # Enforce net leverage
        net_leverage = result.sum()
        
        if self.max_net_leverage is not None and net_leverage > self.max_net_leverage:
            # Scale down proportionally
            result = result * (self.max_net_leverage / net_leverage)
        
        if self.min_net_leverage is not None and net_leverage < self.min_net_leverage:
            # Scale up proportionally
            result = result * (self.min_net_leverage / net_leverage)
        
        return result


class ConcentrationConstraint(Constraint):
    """
    Concentration constraint.
    
    Limits concentration using Herfindahl index.
    """
    
    def __init__(
        self,
        max_herfindahl: float = 0.2,
        max_concentration: float = None,  # Alias for max_herfindahl
    ):
        """
        Initialize concentration constraint.
        
        Args:
            max_herfindahl: Maximum Herfindahl index
            max_concentration: Alias for max_herfindahl
        """
        # Handle alias
        if max_concentration is not None:
            max_herfindahl = max_concentration
        
        super().__init__(name="concentration")
        self.max_herfindahl = max_herfindahl
        self.max_concentration = max_herfindahl  # Store alias
    
    def check(self, weights: pd.Series, **kwargs: Any) -> bool:
        """Check if concentration limit satisfied."""
        herfindahl = (weights ** 2).sum()
        return herfindahl <= self.max_herfindahl
    
    def enforce(self, weights: pd.Series, **kwargs: Any) -> pd.Series:
        """Enforce concentration limit."""
        herfindahl = (weights ** 2).sum()
        
        if herfindahl <= self.max_herfindahl:
            return weights
        
        # Move towards equal weight to reduce concentration
        n = len(weights[weights != 0])
        equal_weight = 1.0 / n if n > 0 else 0
        
        # Blend with equal weight
        blend_factor = 0.5
        result = weights * (1 - blend_factor) + equal_weight * blend_factor
        
        # Renormalize
        total = result.abs().sum()
        if total > 0:
            result = result / total
        
        return result

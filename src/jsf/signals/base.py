"""Base classes and interfaces for signal generation."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

from jsf.utils.logging import get_logger
from jsf.data import PriceData

logger = get_logger(__name__)


class SignalType(Enum):
    """Types of trading signals."""
    
    TECHNICAL = "technical"
    STATISTICAL = "statistical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    COMPOSITE = "composite"


class SignalDirection(Enum):
    """Signal direction indicators."""
    
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass
class SignalMetadata:
    """Metadata for a signal."""
    
    name: str
    signal_type: SignalType
    description: str
    parameters: Dict[str, Any]
    lookback_period: int = 0
    requires_volume: bool = False
    requires_fundamentals: bool = False


class SignalError(Exception):
    """Exception raised for signal generation errors."""
    
    pass


class Signal(ABC):
    """
    Abstract base class for all trading signals.
    
    A signal generates trading indicators from price data and other inputs.
    Signals should be stateless and reproducible.
    """
    
    def __init__(
        self,
        name: str,
        signal_type: SignalType,
        description: str = "",
        **parameters: Any
    ):
        """
        Initialize signal.
        
        Args:
            name: Signal name
            signal_type: Type of signal
            description: Signal description
            **parameters: Signal-specific parameters
        """
        self.name = name
        self.signal_type = signal_type
        self.description = description
        self.parameters = parameters
        
        # Cache for computed signals
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_enabled = True
        
        logger.info(f"Initialized {self.__class__.__name__}: {self.name}")
    
    @abstractmethod
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Generate signal values.
        
        Args:
            price_data: Price data to generate signals from
            **kwargs: Additional data or parameters
        
        Returns:
            DataFrame with signal values (wide format with symbols as columns)
            Values typically in range [-1, 1] or [0, 1] depending on signal
        
        Raises:
            SignalError: If signal generation fails
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> SignalMetadata:
        """
        Get signal metadata.
        
        Returns:
            SignalMetadata object with signal information
        """
        pass
    
    def validate_data(self, price_data: PriceData) -> None:
        """
        Validate that price data meets signal requirements.
        
        Args:
            price_data: Price data to validate
        
        Raises:
            SignalError: If validation fails
        """
        metadata = self.get_metadata()
        
        # Check lookback period
        if len(price_data.data) < metadata.lookback_period:
            raise SignalError(
                f"Insufficient data: need {metadata.lookback_period} periods, "
                f"got {len(price_data.data)}"
            )
        
        # Check volume requirement
        if metadata.requires_volume:
            if "volume" not in price_data.data.columns:
                raise SignalError("Signal requires volume data")
    
    def enable_cache(self, enabled: bool = True) -> None:
        """Enable or disable signal caching."""
        self._cache_enabled = enabled
        if not enabled:
            self._cache.clear()
    
    def clear_cache(self) -> None:
        """Clear signal cache."""
        self._cache.clear()
    
    def _get_cache_key(self, price_data: PriceData, **kwargs: Any) -> str:
        """Generate cache key from inputs."""
        key_parts = [
            str(price_data.start_date),
            str(price_data.end_date),
            "_".join(sorted(price_data.symbols)),
        ]
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        return "|".join(key_parts)
    
    def __call__(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Callable interface for signal generation.
        
        Args:
            price_data: Price data
            **kwargs: Additional parameters
        
        Returns:
            Signal DataFrame
        """
        # Check cache
        if self._cache_enabled:
            cache_key = self._get_cache_key(price_data, **kwargs)
            if cache_key in self._cache:
                logger.debug(f"Using cached signal for {self.name}")
                return self._cache[cache_key].copy()
        
        # Validate data
        self.validate_data(price_data)
        
        # Generate signal
        signal = self.generate(price_data, **kwargs)
        
        # Validate output
        self._validate_output(signal, price_data)
        
        # Cache result
        if self._cache_enabled:
            self._cache[cache_key] = signal.copy()
        
        logger.info(
            f"Generated signal '{self.name}' for {len(price_data.symbols)} symbols, "
            f"{len(signal)} periods"
        )
        
        return signal
    
    def _validate_output(
        self,
        signal: pd.DataFrame,
        price_data: PriceData
    ) -> None:
        """
        Validate signal output.
        
        Args:
            signal: Generated signal DataFrame
            price_data: Original price data
        
        Raises:
            SignalError: If output is invalid
        """
        if not isinstance(signal, pd.DataFrame):
            raise SignalError("Signal must return a DataFrame")
        
        if signal.empty:
            raise SignalError("Signal returned empty DataFrame")
        
        # Check for infinite or NaN values (warn only)
        if signal.isnull().any().any():
            logger.warning(f"Signal '{self.name}' contains NaN values")
        
        if np.isinf(signal.values).any():
            raise SignalError(f"Signal '{self.name}' contains infinite values")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', type={self.signal_type})"


class CompositeSignal(Signal):
    """
    Combine multiple signals using various methods.
    
    Supports weighted averaging, voting, and custom combination functions.
    """
    
    def __init__(
        self,
        name: str,
        signals: List[Signal],
        weights: Optional[List[float]] = None,
        method: str = "average",
        description: str = "",
    ):
        """
        Initialize composite signal.
        
        Args:
            name: Signal name
            signals: List of signals to combine
            weights: Weights for each signal (only for 'weighted_average')
            method: Combination method ('average', 'weighted_average', 'vote', 'max', 'min')
            description: Signal description
        """
        super().__init__(
            name=name,
            signal_type=SignalType.COMPOSITE,
            description=description or f"Composite of {len(signals)} signals",
            method=method,
            n_signals=len(signals),
        )
        
        self.signals = signals
        self.weights = weights
        self.method = method
        
        # Validate weights
        if method == "weighted_average":
            if weights is None:
                raise ValueError("weights required for weighted_average method")
            if len(weights) != len(signals):
                raise ValueError("weights must match number of signals")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("weights must sum to 1.0")
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Generate composite signal.
        
        Args:
            price_data: Price data
            **kwargs: Additional parameters
        
        Returns:
            Combined signal DataFrame
        """
        # Generate all component signals
        component_signals = []
        for signal in self.signals:
            sig = signal(price_data, **kwargs)
            component_signals.append(sig)
        
        # Align all signals to same index/columns
        aligned_signals = self._align_signals(component_signals)
        
        # Combine signals
        if self.method == "average":
            combined = sum(aligned_signals) / len(aligned_signals)
        
        elif self.method == "weighted_average":
            combined = sum(
                sig * weight
                for sig, weight in zip(aligned_signals, self.weights)
            )
        
        elif self.method == "vote":
            # Majority vote based on sign
            stacked = np.stack([sig.values for sig in aligned_signals])
            combined = pd.DataFrame(
                np.sign(np.sum(np.sign(stacked), axis=0)),
                index=aligned_signals[0].index,
                columns=aligned_signals[0].columns,
            )
        
        elif self.method == "max":
            stacked = np.stack([sig.values for sig in aligned_signals])
            combined = pd.DataFrame(
                np.max(stacked, axis=0),
                index=aligned_signals[0].index,
                columns=aligned_signals[0].columns,
            )
        
        elif self.method == "min":
            stacked = np.stack([sig.values for sig in aligned_signals])
            combined = pd.DataFrame(
                np.min(stacked, axis=0),
                index=aligned_signals[0].index,
                columns=aligned_signals[0].columns,
            )
        
        else:
            raise ValueError(f"Unknown combination method: {self.method}")
        
        return combined
    
    def _align_signals(
        self,
        signals: List[pd.DataFrame]
    ) -> List[pd.DataFrame]:
        """Align all signals to common index and columns."""
        if not signals:
            return []
        
        # Get common index (intersection of all dates)
        common_index = signals[0].index
        for sig in signals[1:]:
            common_index = common_index.intersection(sig.index)
        
        # Get common columns (intersection of all symbols)
        common_columns = signals[0].columns
        for sig in signals[1:]:
            common_columns = common_columns.intersection(sig.columns)
        
        # Align all signals
        aligned = [
            sig.loc[common_index, common_columns]
            for sig in signals
        ]
        
        return aligned
    
    def get_metadata(self) -> SignalMetadata:
        """Get composite signal metadata."""
        # Aggregate requirements from component signals
        max_lookback = max(sig.get_metadata().lookback_period for sig in self.signals)
        requires_volume = any(sig.get_metadata().requires_volume for sig in self.signals)
        requires_fundamentals = any(
            sig.get_metadata().requires_fundamentals for sig in self.signals
        )
        
        return SignalMetadata(
            name=self.name,
            signal_type=self.signal_type,
            description=self.description,
            parameters=self.parameters,
            lookback_period=max_lookback,
            requires_volume=requires_volume,
            requires_fundamentals=requires_fundamentals,
        )

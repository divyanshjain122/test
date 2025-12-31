"""Advanced composite signal patterns and utilities.

This module provides sophisticated signal combination patterns including
rotation strategies, multi-timeframe analysis, and adaptive weighting.
"""

from typing import List, Optional, Dict, Callable, Any
from enum import Enum

import pandas as pd
import numpy as np

from jsf.signals.base import Signal, SignalType, SignalMetadata, CompositeSignal
from jsf.data import PriceData
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class RotationStrategy(Enum):
    """Rotation strategy types."""
    TOP_N = "top_n"
    THRESHOLD = "threshold"
    PERCENTILE = "percentile"


class RotationSignal(CompositeSignal):
    """
    Rotation signal for selecting top performers.
    
    Combines multiple signals and selects top N securities.
    """
    
    def __init__(
        self,
        signals: List[Signal],
        n_positions: int = 5,
        strategy: RotationStrategy = RotationStrategy.TOP_N,
        rebalance_frequency: int = 20,
        name: str = "rotation",
    ):
        """
        Initialize rotation signal.
        
        Args:
            signals: List of signals to combine
            n_positions: Number of positions to hold
            strategy: Rotation strategy type
            rebalance_frequency: Rebalance every N periods
            name: Signal name
        """
        # Equal weighting for constituent signals
        weights = [1.0 / len(signals)] * len(signals)
        
        super().__init__(
            signals=signals,
            weights=weights,
            name=name,
        )
        
        self.n_positions = n_positions
        self.strategy = strategy
        self.rebalance_frequency = rebalance_frequency
        
        self.parameters.update({
            "n_positions": n_positions,
            "strategy": strategy.value,
            "rebalance_frequency": rebalance_frequency,
        })
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate rotation signal."""
        # Get composite signal
        composite = super().generate(price_data, **kwargs)
        
        # Apply rotation logic
        rotated_signal = pd.DataFrame(
            0.0,
            index=composite.index,
            columns=composite.columns,
        )
        
        # Rebalance at specified frequency
        for i in range(0, len(composite), self.rebalance_frequency):
            period_end = min(i + self.rebalance_frequency, len(composite))
            
            # Get signal values at rebalance point
            rebalance_signals = composite.iloc[i]
            
            if self.strategy == RotationStrategy.TOP_N:
                # Select top N by signal strength
                top_n = rebalance_signals.nlargest(self.n_positions)
                selected = top_n.index
            
            elif self.strategy == RotationStrategy.THRESHOLD:
                # Select all above threshold
                selected = rebalance_signals[rebalance_signals > 0.5].index[:self.n_positions]
            
            elif self.strategy == RotationStrategy.PERCENTILE:
                # Select top percentile
                percentile = rebalance_signals.quantile(0.8)
                selected = rebalance_signals[rebalance_signals >= percentile].index[:self.n_positions]
            
            else:
                selected = []
            
            # Set signals for selected securities
            for symbol in selected:
                rotated_signal.loc[composite.index[i:period_end], symbol] = 1.0
        
        return rotated_signal


class MultiTimeframeSignal(Signal):
    """
    Multi-timeframe signal aggregation.
    
    Combines signals from different timeframes for confirmation.
    """
    
    def __init__(
        self,
        base_signal: Signal,
        timeframes: List[int],
        weights: Optional[List[float]] = None,
        name: str = "multi_timeframe",
    ):
        """
        Initialize multi-timeframe signal.
        
        Args:
            base_signal: Base signal to apply at different timeframes
            timeframes: List of timeframe periods
            weights: Weights for each timeframe (None = equal)
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.COMPOSITE,
            description=f"Multi-timeframe {base_signal.name}",
            base_signal=base_signal.name,
            timeframes=timeframes,
        )
        
        self.base_signal = base_signal
        self.timeframes = sorted(timeframes)
        self.weights = weights or [1.0 / len(timeframes)] * len(timeframes)
        
        if len(self.weights) != len(self.timeframes):
            raise ValueError("Number of weights must match number of timeframes")
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate multi-timeframe signal."""
        close_prices = price_data.get_close_prices()
        
        # Initialize aggregated signal
        aggregated = pd.DataFrame(
            0.0,
            index=close_prices.index,
            columns=close_prices.columns,
        )
        
        for timeframe, weight in zip(self.timeframes, self.weights):
            # Resample data to timeframe
            if timeframe == 1:
                tf_data = price_data
            else:
                # Simple resampling (every N periods)
                resampled_close = close_prices.iloc[::timeframe]
                tf_data = PriceData(
                    data={"close": resampled_close},
                    symbols=price_data.symbols,
                    start_date=price_data.start_date,
                    end_date=price_data.end_date,
                )
            
            # Generate signal at this timeframe
            tf_signal = self.base_signal.generate(tf_data, **kwargs)
            
            # Upsample back to original frequency
            if timeframe > 1:
                tf_signal = tf_signal.reindex(
                    close_prices.index,
                    method="ffill",
                )
            
            # Add weighted contribution
            aggregated += weight * tf_signal.fillna(0)
        
        return aggregated
    
    def get_metadata(self) -> SignalMetadata:
        """Get signal metadata."""
        return SignalMetadata(
            name=self.name,
            signal_type=self.signal_type,
            description=self.description,
            parameters=self.parameters,
            lookback_period=max(self.timeframes),
            requires_volume=self.base_signal.get_metadata().requires_volume,
        )


class AdaptiveWeightSignal(CompositeSignal):
    """
    Adaptive weight composite signal.
    
    Dynamically adjusts weights based on signal performance.
    """
    
    def __init__(
        self,
        signals: List[Signal],
        lookback: int = 60,
        weight_method: str = "sharpe",
        name: str = "adaptive_weight",
    ):
        """
        Initialize adaptive weight signal.
        
        Args:
            signals: List of signals to combine
            lookback: Lookback for performance evaluation
            weight_method: Method for weight calculation ('sharpe', 'information_ratio', 'win_rate')
            name: Signal name
        """
        # Initial equal weights
        initial_weights = [1.0 / len(signals)] * len(signals)
        
        super().__init__(
            signals=signals,
            weights=initial_weights,
            name=name,
        )
        
        self.lookback = lookback
        self.weight_method = weight_method
        
        self.parameters.update({
            "lookback": lookback,
            "weight_method": weight_method,
        })
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate adaptive weight signal."""
        returns = price_data.get_returns(periods=1)
        
        # Generate signals for all constituents
        individual_signals = []
        for signal in self.signals:
            sig = signal.generate(price_data, **kwargs)
            individual_signals.append(sig)
        
        # Initialize result
        result = pd.DataFrame(
            0.0,
            index=returns.index,
            columns=returns.columns,
        )
        
        # Calculate adaptive weights over time
        for i in range(self.lookback, len(returns)):
            # Get historical period
            hist_returns = returns.iloc[i - self.lookback:i]
            
            # Calculate performance for each signal
            performances = []
            for sig in individual_signals:
                hist_sig = sig.iloc[i - self.lookback:i - 1]  # Lag by 1
                
                # Calculate strategy returns
                strategy_returns = (hist_sig.shift(1) * hist_returns).mean(axis=1)
                
                if self.weight_method == "sharpe":
                    perf = strategy_returns.mean() / (strategy_returns.std() + 1e-10)
                elif self.weight_method == "information_ratio":
                    excess_returns = strategy_returns - hist_returns.mean(axis=1)
                    perf = excess_returns.mean() / (excess_returns.std() + 1e-10)
                elif self.weight_method == "win_rate":
                    perf = (strategy_returns > 0).mean()
                else:
                    perf = strategy_returns.mean()
                
                performances.append(max(perf, 0))  # No negative weights
            
            # Normalize to weights
            total_perf = sum(performances) + 1e-10
            current_weights = [p / total_perf for p in performances]
            
            # Apply weighted combination
            combined = pd.Series(0.0, index=returns.columns)
            for sig, weight in zip(individual_signals, current_weights):
                combined += weight * sig.iloc[i]
            
            result.iloc[i] = combined
        
        return result


class ThresholdFilterSignal(Signal):
    """
    Threshold-based signal filter.
    
    Filters a base signal through threshold conditions.
    """
    
    def __init__(
        self,
        base_signal: Signal,
        threshold: float = 0.5,
        mode: str = "absolute",
        name: str = "threshold_filter",
    ):
        """
        Initialize threshold filter.
        
        Args:
            base_signal: Base signal to filter
            threshold: Threshold value
            mode: 'absolute' or 'percentile'
            name: Signal name
        """
        super().__init__(
            name=name,
            signal_type=SignalType.COMPOSITE,
            description=f"Threshold filter for {base_signal.name}",
            base_signal=base_signal.name,
            threshold=threshold,
            mode=mode,
        )
        
        self.base_signal = base_signal
        self.threshold = threshold
        self.mode = mode
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate filtered signal."""
        base_sig = self.base_signal.generate(price_data, **kwargs)
        
        filtered = pd.DataFrame(
            0.0,
            index=base_sig.index,
            columns=base_sig.columns,
        )
        
        if self.mode == "absolute":
            # Simple threshold
            filtered[base_sig > self.threshold] = 1.0
            filtered[base_sig < -self.threshold] = -1.0
        
        elif self.mode == "percentile":
            # Cross-sectional percentile
            for idx in base_sig.index:
                row = base_sig.loc[idx]
                upper_threshold = row.quantile(self.threshold)
                lower_threshold = row.quantile(1 - self.threshold)
                
                filtered.loc[idx, row >= upper_threshold] = 1.0
                filtered.loc[idx, row <= lower_threshold] = -1.0
        
        return filtered
    
    def get_metadata(self) -> SignalMetadata:
        """Get signal metadata."""
        base_meta = self.base_signal.get_metadata()
        return SignalMetadata(
            name=self.name,
            signal_type=self.signal_type,
            description=self.description,
            parameters=self.parameters,
            lookback_period=base_meta.lookback_period,
            requires_volume=base_meta.requires_volume,
        )


class ConsensusSignal(CompositeSignal):
    """
    Consensus signal requiring agreement across signals.
    
    Only generates signal when multiple constituent signals agree.
    """
    
    def __init__(
        self,
        signals: List[Signal],
        consensus_threshold: float = 0.7,
        name: str = "consensus",
    ):
        """
        Initialize consensus signal.
        
        Args:
            signals: List of signals requiring consensus
            consensus_threshold: Fraction of signals that must agree
            name: Signal name
        """
        weights = [1.0 / len(signals)] * len(signals)
        
        super().__init__(
            signals=signals,
            weights=weights,
            name=name,
        )
        
        self.consensus_threshold = consensus_threshold
        self.parameters["consensus_threshold"] = consensus_threshold
    
    def generate(
        self,
        price_data: PriceData,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Generate consensus signal."""
        # Generate all constituent signals
        individual_signals = []
        for signal in self.signals:
            sig = signal.generate(price_data, **kwargs)
            individual_signals.append(sig)
        
        # Initialize result
        result = pd.DataFrame(
            0.0,
            index=individual_signals[0].index,
            columns=individual_signals[0].columns,
        )
        
        # Calculate consensus
        n_signals = len(individual_signals)
        required_agreement = int(np.ceil(n_signals * self.consensus_threshold))
        
        for col in result.columns:
            # Stack signals for this symbol
            stacked = pd.concat([sig[col] for sig in individual_signals], axis=1)
            
            # Count bullish/bearish agreements
            bullish_count = (stacked > 0).sum(axis=1)
            bearish_count = (stacked < 0).sum(axis=1)
            
            # Set signal based on consensus
            result.loc[bullish_count >= required_agreement, col] = 1.0
            result.loc[bearish_count >= required_agreement, col] = -1.0
        
        return result

"""Signal transformation and normalization utilities.

This module provides utilities for transforming, normalizing, and
ranking signals for use in portfolio construction.
"""

from typing import Optional, Callable, Union
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats

from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class NormalizationMethod(Enum):
    """Signal normalization methods."""
    ZSCORE = "zscore"
    MINMAX = "minmax"
    RANK = "rank"
    PERCENTILE = "percentile"
    TANH = "tanh"


class RankingMethod(Enum):
    """Signal ranking methods."""
    CROSS_SECTIONAL = "cross_sectional"
    TIME_SERIES = "time_series"
    HYBRID = "hybrid"


def normalize_signal(
    signal: pd.DataFrame,
    method: Union[str, NormalizationMethod] = NormalizationMethod.ZSCORE,
    axis: int = 1,
    **kwargs,
) -> pd.DataFrame:
    """
    Normalize signal values.
    
    Args:
        signal: Signal DataFrame to normalize
        method: Normalization method
        axis: 0 for time-series, 1 for cross-sectional
        **kwargs: Additional parameters for normalization
    
    Returns:
        Normalized signal DataFrame
    """
    if isinstance(method, str):
        method = NormalizationMethod(method)
    
    if method == NormalizationMethod.ZSCORE:
        # Z-score normalization
        mean = signal.mean(axis=axis, **kwargs)
        std = signal.std(axis=axis, **kwargs)
        
        if axis == 1:
            normalized = signal.sub(mean, axis=0).div(std + 1e-10, axis=0)
        else:
            normalized = (signal - mean) / (std + 1e-10)
    
    elif method == NormalizationMethod.MINMAX:
        # Min-max normalization to [-1, 1]
        min_val = signal.min(axis=axis, **kwargs)
        max_val = signal.max(axis=axis, **kwargs)
        
        if axis == 1:
            range_val = max_val - min_val
            normalized = signal.sub(min_val, axis=0).div(range_val + 1e-10, axis=0)
        else:
            normalized = (signal - min_val) / (max_val - min_val + 1e-10)
        
        # Scale to [-1, 1]
        normalized = 2 * normalized - 1
    
    elif method == NormalizationMethod.RANK:
        # Rank normalization
        if axis == 1:
            normalized = signal.rank(axis=1, pct=True)
        else:
            normalized = signal.rank(axis=0, pct=True)
        
        # Scale to [-1, 1]
        normalized = 2 * normalized - 1
    
    elif method == NormalizationMethod.PERCENTILE:
        # Percentile-based normalization
        if axis == 1:
            normalized = signal.rank(axis=1, pct=True)
        else:
            normalized = signal.rank(axis=0, pct=True)
    
    elif method == NormalizationMethod.TANH:
        # Tanh normalization (soft clipping)
        scale = kwargs.get("scale", 1.0)
        normalized = pd.DataFrame(
            np.tanh(signal.values / scale),
            index=signal.index,
            columns=signal.columns,
        )
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.fillna(0)


def rank_signal(
    signal: pd.DataFrame,
    method: Union[str, RankingMethod] = RankingMethod.CROSS_SECTIONAL,
    lookback: Optional[int] = None,
) -> pd.DataFrame:
    """
    Rank signal values.
    
    Args:
        signal: Signal DataFrame to rank
        method: Ranking method
        lookback: Lookback period for time-series ranking
    
    Returns:
        Ranked signal DataFrame with percentiles [0, 1]
    """
    if isinstance(method, str):
        method = RankingMethod(method)
    
    if method == RankingMethod.CROSS_SECTIONAL:
        # Rank across securities at each time point
        ranked = signal.rank(axis=1, pct=True)
    
    elif method == RankingMethod.TIME_SERIES:
        # Rank over time for each security
        if lookback is None:
            lookback = len(signal)
        
        ranked = pd.DataFrame(
            index=signal.index,
            columns=signal.columns,
        )
        
        for col in signal.columns:
            ranked[col] = signal[col].rolling(window=lookback).rank(pct=True)
    
    elif method == RankingMethod.HYBRID:
        # Combine cross-sectional and time-series
        cs_rank = signal.rank(axis=1, pct=True)
        
        if lookback is None:
            lookback = len(signal) // 2
        
        ts_rank = pd.DataFrame(
            index=signal.index,
            columns=signal.columns,
        )
        
        for col in signal.columns:
            ts_rank[col] = signal[col].rolling(window=lookback).rank(pct=True)
        
        # Average the two rankings
        ranked = (cs_rank + ts_rank) / 2
    
    else:
        raise ValueError(f"Unknown ranking method: {method}")
    
    return ranked.fillna(0.5)


def smooth_signal(
    signal: pd.DataFrame,
    method: str = "ema",
    window: int = 5,
    **kwargs,
) -> pd.DataFrame:
    """
    Smooth signal to reduce noise.
    
    Args:
        signal: Signal DataFrame to smooth
        method: Smoothing method ('sma', 'ema', 'gaussian')
        window: Smoothing window size
        **kwargs: Additional parameters
    
    Returns:
        Smoothed signal DataFrame
    """
    if method == "sma":
        # Simple moving average
        smoothed = signal.rolling(window=window).mean()
    
    elif method == "ema":
        # Exponential moving average
        span = kwargs.get("span", window)
        smoothed = signal.ewm(span=span, adjust=False).mean()
    
    elif method == "gaussian":
        # Gaussian smoothing
        from scipy.ndimage import gaussian_filter1d
        
        sigma = kwargs.get("sigma", window / 3)
        smoothed = pd.DataFrame(
            index=signal.index,
            columns=signal.columns,
        )
        
        for col in signal.columns:
            smoothed[col] = gaussian_filter1d(
                signal[col].fillna(0).values,
                sigma=sigma,
            )
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    return smoothed.fillna(signal)


def clip_signal(
    signal: pd.DataFrame,
    lower: float = -1.0,
    upper: float = 1.0,
) -> pd.DataFrame:
    """
    Clip signal values to a range.
    
    Args:
        signal: Signal DataFrame to clip
        lower: Lower bound
        upper: Upper bound
    
    Returns:
        Clipped signal DataFrame
    """
    return signal.clip(lower=lower, upper=upper)


def winsorize_signal(
    signal: pd.DataFrame,
    limits: tuple = (0.05, 0.05),
    axis: int = 1,
) -> pd.DataFrame:
    """
    Winsorize signal to handle outliers.
    
    Args:
        signal: Signal DataFrame to winsorize
        limits: (lower, upper) percentile limits
        axis: 0 for time-series, 1 for cross-sectional
    
    Returns:
        Winsorized signal DataFrame
    """
    winsorized = signal.copy()
    
    if axis == 1:
        # Cross-sectional winsorization
        for idx in signal.index:
            row = signal.loc[idx]
            lower_val = row.quantile(limits[0])
            upper_val = row.quantile(1 - limits[1])
            winsorized.loc[idx] = row.clip(lower=lower_val, upper=upper_val)
    
    else:
        # Time-series winsorization
        for col in signal.columns:
            series = signal[col]
            lower_val = series.quantile(limits[0])
            upper_val = series.quantile(1 - limits[1])
            winsorized[col] = series.clip(lower=lower_val, upper=upper_val)
    
    return winsorized


def demean_signal(
    signal: pd.DataFrame,
    axis: int = 1,
) -> pd.DataFrame:
    """
    Demean signal (subtract mean).
    
    Args:
        signal: Signal DataFrame to demean
        axis: 0 for time-series, 1 for cross-sectional
    
    Returns:
        Demeaned signal DataFrame
    """
    if axis == 1:
        # Cross-sectional demeaning
        mean = signal.mean(axis=1)
        demeaned = signal.sub(mean, axis=0)
    else:
        # Time-series demeaning
        demeaned = signal - signal.mean(axis=0)
    
    return demeaned


def neutralize_signal(
    signal: pd.DataFrame,
    factor: pd.DataFrame,
) -> pd.DataFrame:
    """
    Neutralize signal against a factor.
    
    Removes correlation with factor using regression.
    
    Args:
        signal: Signal DataFrame to neutralize
        factor: Factor DataFrame to neutralize against
    
    Returns:
        Neutralized signal DataFrame
    """
    neutralized = signal.copy()
    
    for col in signal.columns:
        if col not in factor.columns:
            continue
        
        # Get valid data
        valid_idx = signal[col].notna() & factor[col].notna()
        
        if valid_idx.sum() < 10:
            continue
        
        # Regression to get residuals
        y = signal.loc[valid_idx, col].values
        x = factor.loc[valid_idx, col].values
        
        # Add constant
        X = np.column_stack([np.ones_like(x), x])
        
        # Solve for beta
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta
            neutralized.loc[valid_idx, col] = residuals
        except np.linalg.LinAlgError:
            logger.warning(f"Failed to neutralize signal for {col}")
            continue
    
    return neutralized


def apply_decay(
    signal: pd.DataFrame,
    half_life: int = 10,
) -> pd.DataFrame:
    """
    Apply exponential decay to signal.
    
    Args:
        signal: Signal DataFrame
        half_life: Half-life for decay (in periods)
    
    Returns:
        Decayed signal DataFrame
    """
    decay_factor = np.exp(-np.log(2) / half_life)
    
    decayed = pd.DataFrame(
        index=signal.index,
        columns=signal.columns,
    )
    
    for col in signal.columns:
        values = signal[col].fillna(0).values
        decayed_values = np.zeros_like(values)
        
        for i in range(len(values)):
            if i == 0:
                decayed_values[i] = values[i]
            else:
                decayed_values[i] = values[i] + decay_factor * decayed_values[i - 1]
        
        decayed[col] = decayed_values
    
    return decayed


def combine_signals(
    signals: list[pd.DataFrame],
    method: str = "average",
    weights: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Combine multiple signals.
    
    Args:
        signals: List of signal DataFrames
        method: Combination method ('average', 'median', 'max', 'min')
        weights: Weights for averaging (None = equal weights)
    
    Returns:
        Combined signal DataFrame
    """
    if not signals:
        raise ValueError("No signals provided")
    
    if method == "average":
        if weights is None:
            weights = [1.0 / len(signals)] * len(signals)
        
        combined = sum(w * sig for w, sig in zip(weights, signals))
    
    elif method == "median":
        # Stack and take median
        stacked = pd.concat(signals, axis=0)
        combined = stacked.groupby(level=0).median()
    
    elif method == "max":
        stacked = pd.concat(signals, axis=0)
        combined = stacked.groupby(level=0).max()
    
    elif method == "min":
        stacked = pd.concat(signals, axis=0)
        combined = stacked.groupby(level=0).min()
    
    else:
        raise ValueError(f"Unknown combination method: {method}")
    
    return combined


def score_signals(
    signal: pd.DataFrame,
    method: str = "zscore",
    **kwargs,
) -> pd.DataFrame:
    """
    Convert signals to scores for portfolio construction.
    
    Args:
        signal: Signal DataFrame
        method: Scoring method ('zscore', 'percentile', 'exponential')
        **kwargs: Additional parameters
    
    Returns:
        Signal scores DataFrame
    """
    if method == "zscore":
        # Z-score based scoring
        scores = normalize_signal(signal, NormalizationMethod.ZSCORE, axis=1)
    
    elif method == "percentile":
        # Percentile-based scoring
        scores = rank_signal(signal, RankingMethod.CROSS_SECTIONAL)
    
    elif method == "exponential":
        # Exponential scoring
        scale = kwargs.get("scale", 1.0)
        scores = pd.DataFrame(
            np.exp(signal.values / scale),
            index=signal.index,
            columns=signal.columns,
        )
        # Normalize to sum to 1 per row
        row_sums = scores.sum(axis=1)
        scores = scores.div(row_sums, axis=0)
    
    else:
        raise ValueError(f"Unknown scoring method: {method}")
    
    return scores

"""Signal generation framework.

This module provides a comprehensive framework for generating trading signals
from various data sources using technical, statistical, and fundamental analysis.
"""

# Base classes
from jsf.signals.base import (
    Signal,
    SignalType,
    SignalDirection,
    SignalMetadata,
    SignalError,
    CompositeSignal,
)

# Technical indicators
from jsf.signals.technical import (
    MomentumSignal,
    MovingAverageCrossSignal,
    RSISignal,
    BollingerBandsSignal,
    MACDSignal,
    VolumeWeightedSignal,
)

# Statistical signals
from jsf.signals.statistical import (
    MeanReversionSignal,
    PairsSignal,
    TrendStrengthSignal,
    VolatilitySignal,
    CorrelationSignal,
)

# Fundamental signals
from jsf.signals.fundamental import (
    ValueSignal,
    QualitySignal,
    GrowthSignal,
    SizeSignal,
    DividendSignal,
)

# Sentiment signals
from jsf.signals.sentiment import (
    MarketRegimeSignal,
    BreadthSignal,
    RelativeStrengthSignal,
    NewHighLowSignal,
    VolumeShockSignal,
    SeasonalitySignal,
)

# Advanced composites
from jsf.signals.composites import (
    RotationSignal,
    MultiTimeframeSignal,
    AdaptiveWeightSignal,
    ThresholdFilterSignal,
    ConsensusSignal,
)

# Transformation utilities
from jsf.signals.transforms import (
    normalize_signal,
    rank_signal,
    smooth_signal,
    clip_signal,
    winsorize_signal,
    demean_signal,
    neutralize_signal,
    apply_decay,
    combine_signals,
    score_signals,
    NormalizationMethod,
    RankingMethod,
)

__all__ = [
    # Base
    "Signal",
    "SignalType",
    "SignalDirection",
    "SignalMetadata",
    "SignalError",
    "CompositeSignal",
    # Technical
    "MomentumSignal",
    "MovingAverageCrossSignal",
    "RSISignal",
    "BollingerBandsSignal",
    "MACDSignal",
    "VolumeWeightedSignal",
    # Statistical
    "MeanReversionSignal",
    "PairsSignal",
    "TrendStrengthSignal",
    "VolatilitySignal",
    "CorrelationSignal",
    # Fundamental
    "ValueSignal",
    "QualitySignal",
    "GrowthSignal",
    "SizeSignal",
    "DividendSignal",
    # Sentiment
    "MarketRegimeSignal",
    "BreadthSignal",
    "RelativeStrengthSignal",
    "NewHighLowSignal",
    "VolumeShockSignal",
    "SeasonalitySignal",
    # Advanced Composites
    "RotationSignal",
    "MultiTimeframeSignal",
    "AdaptiveWeightSignal",
    "ThresholdFilterSignal",
    "ConsensusSignal",
    # Transforms
    "normalize_signal",
    "rank_signal",
    "smooth_signal",
    "clip_signal",
    "winsorize_signal",
    "demean_signal",
    "neutralize_signal",
    "apply_decay",
    "combine_signals",
    "score_signals",
    "NormalizationMethod",
    "RankingMethod",
]

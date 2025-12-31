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
]

"""Feature extraction for ML models.

Extracts features from price data using existing JSF signals.
Supports technical, statistical, fundamental, and sentiment features.
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from jsf.data import PriceData
from jsf.signals import (
    # Technical signals
    MomentumSignal,
    MovingAverageCrossSignal,
    RSISignal,
    BollingerBandsSignal,
    MACDSignal,
    VolumeWeightedSignal,
    # Statistical signals
    MeanReversionSignal,
    VolatilitySignal,
    TrendStrengthSignal,
    CorrelationSignal,
    # Fundamental signals (price-based proxies)
    ValueSignal,
    QualitySignal,
    GrowthSignal,
    # Sentiment signals
    MarketRegimeSignal,
    BreadthSignal,
    RelativeStrengthSignal,
    VolumeShockSignal,
)
from jsf.signals.transforms import (
    normalize_signal,
    rank_signal,
    smooth_signal,
    NormalizationMethod,
    RankingMethod,
)
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


# Feature group definitions
FEATURE_GROUPS = {
    "momentum": {
        "signals": [
            ("momentum_20", MomentumSignal, {"lookback": 20}),
            ("momentum_60", MomentumSignal, {"lookback": 60}),
            ("momentum_120", MomentumSignal, {"lookback": 120}),
        ],
        "description": "Price momentum over various lookback periods",
    },
    "mean_reversion": {
        "signals": [
            ("mean_rev_20", MeanReversionSignal, {"lookback": 20}),
            ("mean_rev_60", MeanReversionSignal, {"lookback": 60}),
        ],
        "description": "Mean reversion (z-score) signals",
    },
    "technical": {
        "signals": [
            ("rsi_14", RSISignal, {"period": 14}),
            ("rsi_28", RSISignal, {"period": 28}),
            ("bb_20", BollingerBandsSignal, {"lookback": 20}),
            ("macd", MACDSignal, {}),
        ],
        "description": "Technical indicators (RSI, Bollinger, MACD)",
    },
    "volatility": {
        "signals": [
            ("vol_20", VolatilitySignal, {"lookback": 20}),
            ("vol_60", VolatilitySignal, {"lookback": 60}),
        ],
        "description": "Volatility-based features",
    },
    "trend": {
        "signals": [
            ("trend_20", TrendStrengthSignal, {"lookback": 20}),
            ("trend_60", TrendStrengthSignal, {"lookback": 60}),
            ("ma_cross_20_60", MovingAverageCrossSignal, {"fast_period": 20, "slow_period": 60}),
        ],
        "description": "Trend strength and MA crossover signals",
    },
    "volume": {
        "signals": [
            ("vol_weighted", VolumeWeightedSignal, {"lookback": 20}),
            ("vol_shock", VolumeShockSignal, {"lookback": 20}),
        ],
        "description": "Volume-based features",
    },
    "cross_sectional": {
        "signals": [
            ("rel_strength", RelativeStrengthSignal, {"lookback": 60}),
            ("breadth", BreadthSignal, {"lookback": 20}),
        ],
        "description": "Cross-sectional ranking features",
    },
    "fundamental": {
        "signals": [
            ("value", ValueSignal, {"lookback": 60}),
            ("quality", QualitySignal, {"lookback": 60}),
            ("growth", GrowthSignal, {"lookback": 60}),
        ],
        "description": "Fundamental factor proxies from price data",
    },
    "regime": {
        "signals": [
            ("market_regime", MarketRegimeSignal, {"lookback": 60}),
        ],
        "description": "Market regime detection",
    },
    "text_sentiment": {
        "signals": [],  # Text features handled separately
        "description": "NLP-based sentiment from financial text (news, social media)",
        "requires": "text_data",  # Flag that this needs text input
    },
}


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    
    # Feature groups to use
    feature_groups: List[str] = field(default_factory=lambda: [
        "momentum", "mean_reversion", "technical", "volatility", "trend"
    ])
    
    # Custom signals to add (name, signal_class, kwargs)
    custom_signals: List[tuple] = field(default_factory=list)
    
    # Lag periods for creating lagged features (e.g., [1, 5, 10])
    lag_periods: List[int] = field(default_factory=lambda: [1, 5])
    
    # Whether to normalize features
    normalize: bool = True
    normalization_method: str = "zscore"
    
    # Whether to rank features cross-sectionally
    rank: bool = True
    
    # Whether to add interaction features
    add_interactions: bool = False
    
    # Rolling window for feature calculation
    rolling_window: Optional[int] = None
    
    # Minimum required history for features
    min_history: int = 120


class FeatureExtractor:
    """Extract ML features from price data.
    
    Uses existing JSF signals as base features and adds:
    - Lagged features (avoid lookahead bias)
    - Normalized/ranked versions
    - Optional interaction features
    
    Example:
        >>> extractor = FeatureExtractor(
        ...     feature_groups=['momentum', 'volatility'],
        ...     lag_periods=[1, 5, 10]
        ... )
        >>> features = extractor.extract(price_data)
    """
    
    def __init__(
        self,
        feature_groups: Optional[List[str]] = None,
        custom_signals: Optional[List[tuple]] = None,
        lag_periods: Optional[List[int]] = None,
        normalize: bool = True,
        rank: bool = True,
        add_interactions: bool = False,
        config: Optional[FeatureConfig] = None,
    ):
        """Initialize feature extractor.
        
        Args:
            feature_groups: List of feature group names to use
            custom_signals: List of (name, signal_class, kwargs) tuples
            lag_periods: Periods to lag features (for avoiding lookahead)
            normalize: Whether to normalize features
            rank: Whether to add cross-sectional ranks
            add_interactions: Whether to add interaction features
            config: FeatureConfig object (overrides other params)
        """
        if config is not None:
            self.config = config
        else:
            self.config = FeatureConfig(
                feature_groups=feature_groups or ["momentum", "volatility", "trend"],
                custom_signals=custom_signals or [],
                lag_periods=lag_periods or [1],
                normalize=normalize,
                rank=rank,
                add_interactions=add_interactions,
            )
        
        # Build signal instances
        self._signals = self._build_signals()
        self._feature_names = []
        
        logger.info(
            f"FeatureExtractor initialized with {len(self._signals)} base signals, "
            f"lags={self.config.lag_periods}"
        )
    
    def _build_signals(self) -> Dict[str, Any]:
        """Build signal instances from config."""
        signals = {}
        
        for group_name in self.config.feature_groups:
            if group_name not in FEATURE_GROUPS:
                logger.warning(f"Unknown feature group: {group_name}")
                continue
            
            group = FEATURE_GROUPS[group_name]
            for name, signal_class, kwargs in group["signals"]:
                try:
                    signals[name] = signal_class(**kwargs)
                except Exception as e:
                    logger.warning(f"Failed to create signal {name}: {e}")
        
        # Add custom signals
        for name, signal_class, kwargs in self.config.custom_signals:
            try:
                signals[name] = signal_class(**kwargs)
            except Exception as e:
                logger.warning(f"Failed to create custom signal {name}: {e}")
        
        return signals
    
    def extract(
        self,
        price_data: PriceData,
        include_raw: bool = True,
        include_lagged: bool = True,
        include_normalized: bool = None,
        include_ranked: bool = None,
    ) -> pd.DataFrame:
        """Extract all features from price data.
        
        Args:
            price_data: PriceData object
            include_raw: Include raw signal values
            include_lagged: Include lagged versions
            include_normalized: Include normalized versions (uses config if None)
            include_ranked: Include cross-sectional ranks (uses config if None)
            
        Returns:
            DataFrame with MultiIndex (date, symbol) and feature columns
        """
        if include_normalized is None:
            include_normalized = self.config.normalize
        if include_ranked is None:
            include_ranked = self.config.rank
        
        all_features = {}
        
        # Generate base features from signals
        for name, signal in self._signals.items():
            try:
                # Generate signal (returns date x symbol DataFrame)
                raw_signal = signal.generate(price_data)
                
                if include_raw:
                    all_features[name] = raw_signal
                
                # Add lagged versions (shift by lag to use only past data)
                if include_lagged:
                    for lag in self.config.lag_periods:
                        all_features[f"{name}_lag{lag}"] = raw_signal.shift(lag)
                
                # Add normalized version
                if include_normalized:
                    normalized = normalize_signal(
                        raw_signal,
                        method=NormalizationMethod(self.config.normalization_method)
                    )
                    all_features[f"{name}_norm"] = normalized
                
                # Add cross-sectional rank
                if include_ranked:
                    ranked = rank_signal(raw_signal, RankingMethod.CROSS_SECTIONAL)
                    all_features[f"{name}_rank"] = ranked
                    
            except Exception as e:
                logger.warning(f"Failed to generate feature {name}: {e}")
                continue
        
        # Add price-derived features
        all_features.update(self._extract_price_features(price_data))
        
        # Add interaction features if requested
        if self.config.add_interactions:
            all_features.update(self._create_interactions(all_features))
        
        # Combine all features into single DataFrame
        features_df = self._combine_features(all_features, price_data)
        
        self._feature_names = features_df.columns.tolist()
        
        logger.info(f"Extracted {len(self._feature_names)} features")
        
        return features_df
    
    def _extract_price_features(self, price_data: PriceData) -> Dict[str, pd.DataFrame]:
        """Extract additional features from raw price data."""
        features = {}
        
        close = price_data.get_close_prices()
        returns = price_data.get_returns(periods=1)
        
        # Daily returns (lagged to avoid lookahead)
        features["returns_1d"] = returns.shift(1)
        features["returns_5d"] = close.pct_change(5).shift(1)
        features["returns_20d"] = close.pct_change(20).shift(1)
        
        # Volatility features
        features["vol_realized_20d"] = returns.rolling(20).std().shift(1) * np.sqrt(252)
        features["vol_realized_60d"] = returns.rolling(60).std().shift(1) * np.sqrt(252)
        
        # Range features - use get_field if available
        try:
            high = price_data.get_field("high")
            low = price_data.get_field("low")
            features["daily_range"] = ((high - low) / close).shift(1)
            features["avg_range_20d"] = features["daily_range"].rolling(20).mean()
        except (ValueError, KeyError):
            # High/low not available
            pass
        
        # Volume features if available
        try:
            volume = price_data.get_field("volume")
            vol_ma = volume.rolling(20).mean()
            features["volume_ratio"] = (volume / vol_ma).shift(1)
        except (ValueError, KeyError):
            pass
        
        return features
    
    def _create_interactions(
        self,
        features: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """Create interaction features between base features."""
        interactions = {}
        
        # Select key features for interactions
        key_features = ["momentum_20", "vol_20", "rsi_14", "mean_rev_20"]
        available = [f for f in key_features if f in features]
        
        for i, f1 in enumerate(available):
            for f2 in available[i+1:]:
                # Multiplicative interaction
                interactions[f"{f1}_x_{f2}"] = features[f1] * features[f2]
        
        return interactions
    
    def _combine_features(
        self,
        features: Dict[str, pd.DataFrame],
        price_data: PriceData,
    ) -> pd.DataFrame:
        """Combine feature DataFrames into single MultiIndex DataFrame."""
        # Stack each feature from (date x symbol) to MultiIndex
        stacked_features = {}
        
        for name, df in features.items():
            if df is None:
                continue
            stacked = df.stack()
            stacked.name = name
            stacked_features[name] = stacked
        
        # Combine into single DataFrame
        combined = pd.DataFrame(stacked_features)
        combined.index.names = ['date', 'symbol']
        
        return combined
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self._feature_names
    
    @property
    def n_features(self) -> int:
        """Get number of features."""
        return len(self._feature_names)
    
    def get_feature_importance(
        self,
        model: Any,
        top_n: int = 20,
    ) -> pd.Series:
        """Get feature importance from trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            top_n: Number of top features to return
            
        Returns:
            Series of feature importances
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model doesn't have feature_importances_ attribute")
        
        importances = pd.Series(
            model.feature_importances_,
            index=self._feature_names
        ).sort_values(ascending=False)
        
        return importances.head(top_n)
    
    def extract_text_features(
        self,
        text_data: pd.DataFrame,
        text_column: str = "text",
        date_column: str = "date",
        symbol_column: Optional[str] = "symbol",
        model: str = "simple",
    ) -> pd.DataFrame:
        """Extract sentiment features from text data.
        
        Integrates with jsf.ml.transformers.sentiment module to convert
        financial text (news, earnings calls, social media) into numerical
        features suitable for ML models.
        
        ⚠️ EDUCATIONAL PURPOSE ONLY ⚠️
        Sentiment analysis is ONE factor among many. It should NOT be used
        as a standalone trading signal.
        
        Args:
            text_data: DataFrame with text, date, and optional symbol columns
            text_column: Name of column containing text
            date_column: Name of column containing dates
            symbol_column: Name of column containing symbols (None for market-level)
            model: Sentiment model ('simple' or 'finbert')
            
        Returns:
            DataFrame with sentiment features indexed by (date, symbol) or date
            
        Example:
            >>> # News data
            >>> news_df = pd.DataFrame({
            ...     'date': ['2024-01-01', '2024-01-02'],
            ...     'symbol': ['AAPL', 'AAPL'],
            ...     'headline': ['Apple beats earnings', 'iPhone sales decline'],
            ... })
            >>> 
            >>> extractor = FeatureExtractor(feature_groups=['momentum', 'text_sentiment'])
            >>> text_features = extractor.extract_text_features(
            ...     news_df, text_column='headline', model='simple'
            ... )
        """
        try:
            from jsf.ml.transformers.sentiment import SimpleSentiment, FinBERTSentiment
        except ImportError:
            raise ImportError(
                "Text features require jsf.ml.transformers module. "
                "This should be available in your installation."
            )
        
        # Select model
        if model == "simple":
            analyzer = SimpleSentiment()
        elif model == "finbert":
            analyzer = FinBERTSentiment()
        else:
            raise ValueError(f"Unknown model: {model}. Use 'simple' or 'finbert'.")
        
        # Analyze sentiment
        texts = text_data[text_column].tolist()
        results = analyzer.analyze(texts)
        
        # Build features DataFrame
        features = {
            'text_sentiment_score': [r.score for r in results],
            'text_sentiment_confidence': [r.confidence for r in results],
            'text_sentiment_positive': [1 if r.label.value == 'positive' else 0 for r in results],
            'text_sentiment_negative': [1 if r.label.value == 'negative' else 0 for r in results],
        }
        
        features_df = pd.DataFrame(features)
        features_df[date_column] = pd.to_datetime(text_data[date_column])
        
        if symbol_column and symbol_column in text_data.columns:
            features_df[symbol_column] = text_data[symbol_column]
            # Set MultiIndex
            features_df = features_df.set_index([date_column, symbol_column])
        else:
            features_df = features_df.set_index(date_column)
        
        # Aggregate if multiple texts per date/symbol
        features_df = features_df.groupby(level=features_df.index.names).mean()
        
        logger.info(f"Extracted text features: {features_df.shape[0]} rows, {features_df.shape[1]} features")
        
        return features_df
    
    def merge_text_features(
        self,
        price_features: pd.DataFrame,
        text_features: pd.DataFrame,
        how: str = "left",
    ) -> pd.DataFrame:
        """Merge text features with price-based features.
        
        Args:
            price_features: Features from extract() method
            text_features: Features from extract_text_features() method
            how: Merge method ('left', 'inner', 'outer')
            
        Returns:
            Combined DataFrame with all features
        """
        # Align indices
        merged = price_features.join(text_features, how=how)
        
        # Fill missing text features with neutral sentiment
        text_cols = [c for c in merged.columns if c.startswith('text_sentiment')]
        for col in text_cols:
            if 'score' in col:
                merged[col] = merged[col].fillna(0.0)
            elif 'confidence' in col:
                merged[col] = merged[col].fillna(0.5)
            else:
                merged[col] = merged[col].fillna(0)
        
        return merged


def create_feature_extractor(
    preset: str = "default",
    **kwargs,
) -> FeatureExtractor:
    """Create feature extractor from preset.
    
    Args:
        preset: Preset name ('default', 'minimal', 'comprehensive')
        **kwargs: Additional arguments to pass to FeatureExtractor
        
    Returns:
        FeatureExtractor instance
    """
    presets = {
        "default": {
            "feature_groups": ["momentum", "mean_reversion", "technical", "volatility", "trend"],
            "lag_periods": [1, 5],
            "normalize": True,
            "rank": True,
        },
        "minimal": {
            "feature_groups": ["momentum", "volatility"],
            "lag_periods": [1],
            "normalize": True,
            "rank": False,
        },
        "comprehensive": {
            "feature_groups": list(FEATURE_GROUPS.keys()),
            "lag_periods": [1, 5, 10, 20],
            "normalize": True,
            "rank": True,
            "add_interactions": True,
        },
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    config = {**presets[preset], **kwargs}
    
    return FeatureExtractor(**config)

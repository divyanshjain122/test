"""ML-based trading strategy.

Integrates ML models with the JSF strategy framework.
Supports automatic retraining and walk-forward validation.
"""

from typing import Dict, List, Optional, Union, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings

import pandas as pd
import numpy as np

from jsf.data import PriceData
from jsf.strategies.base import Strategy
from jsf.portfolio import Portfolio, SimplePortfolioConstructor, EqualWeightSizer
from jsf.signals.transforms import normalize_signal, rank_signal, NormalizationMethod
from jsf.utils.logging import get_logger

from .features import FeatureExtractor, FeatureConfig, create_feature_extractor
from .models import MLModel, EnsembleModel, RandomForestModel, PredictionType
from .preprocessing import (
    create_target_variable,
    prepare_ml_data,
    split_train_test,
    handle_missing_features,
    MLDataset,
    TargetType,
)

logger = get_logger(__name__)


@dataclass
class MLStrategyConfig:
    """Configuration for ML strategy."""
    
    # Prediction settings
    prediction_type: str = "both"  # 'returns', 'direction', 'both'
    forward_periods: int = 1  # Periods ahead to predict
    direction_threshold: float = 0.0  # Threshold for direction classification
    
    # Signal generation
    signal_method: str = "combined"  # 'returns', 'direction', 'combined'
    signal_scale: float = 1.0  # Scale factor for signals
    
    # Retraining
    retrain_frequency: int = 63  # Days between retraining
    min_train_samples: int = 252  # Minimum samples for training
    warmup_period: int = 120  # Days of warmup (no trading)
    
    # Drift monitoring
    monitor_drift: bool = True
    drift_threshold: float = 0.2  # Trigger retrain if performance drops > 20%
    
    # Feature extraction
    feature_groups: List[str] = field(default_factory=lambda: [
        "momentum", "mean_reversion", "volatility", "trend"
    ])
    lag_periods: List[int] = field(default_factory=lambda: [1, 5])
    
    # Portfolio construction
    long_only: bool = True
    top_n: Optional[int] = None  # If set, only trade top N predictions


class MLStrategy(Strategy):
    """Machine Learning based trading strategy.
    
    Uses ML models to predict returns and/or direction, converting
    predictions to trading signals for portfolio construction.
    
    Features:
    - Supports regression (returns) and classification (direction)
    - Automatic retraining at configurable frequency
    - Walk-forward validation to avoid lookahead bias
    - Drift monitoring with automatic retrain trigger
    - Ensemble models with weighted voting
    
    Example:
        >>> from jsf.ml import MLStrategy, EnsembleModel, FeatureExtractor
        >>> 
        >>> # Create ensemble model
        >>> model = EnsembleModel(
        ...     models=['random_forest', 'xgboost'],
        ...     weights={'random_forest': 0.4, 'xgboost': 0.6}
        ... )
        >>> 
        >>> # Create feature extractor
        >>> extractor = FeatureExtractor(
        ...     feature_groups=['momentum', 'volatility'],
        ...     lag_periods=[1, 5]
        ... )
        >>> 
        >>> # Create strategy
        >>> strategy = MLStrategy(
        ...     name='ml_ensemble',
        ...     model=model,
        ...     feature_extractor=extractor,
        ...     retrain_frequency=63
        ... )
        >>> 
        >>> # Run strategy
        >>> portfolio = strategy.run(price_data)
    """
    
    def __init__(
        self,
        name: str = "ml_strategy",
        model: Optional[MLModel] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        portfolio_constructor: Optional[Any] = None,
        config: Optional[MLStrategyConfig] = None,
        # Shorthand params (override config)
        prediction_type: str = "both",
        retrain_frequency: int = 63,
        long_only: bool = True,
        top_n: Optional[int] = None,
        **kwargs
    ):
        """Initialize ML strategy.
        
        Args:
            name: Strategy name
            model: ML model (default: EnsembleModel)
            feature_extractor: Feature extractor (default: create with momentum/vol)
            portfolio_constructor: Portfolio constructor (default: SimplePortfolioConstructor)
            config: Full configuration object
            prediction_type: Type of prediction ('returns', 'direction', 'both')
            retrain_frequency: Days between model retraining
            long_only: Whether to only take long positions
            top_n: Only trade top N predictions
            **kwargs: Additional strategy parameters
        """
        # Initialize base strategy
        super().__init__(
            name=name,
            signals=[],  # ML strategy doesn't use traditional signals
            portfolio_constructor=portfolio_constructor or SimplePortfolioConstructor(
                sizer=EqualWeightSizer()
            ),
            **kwargs
        )
        
        # Configuration
        if config is not None:
            self.config = config
        else:
            self.config = MLStrategyConfig(
                prediction_type=prediction_type,
                retrain_frequency=retrain_frequency,
                long_only=long_only,
                top_n=top_n,
            )
        
        # Model and feature extractor
        # Map strategy prediction types to model prediction types
        model_pred_type = self._map_prediction_type(self.config.prediction_type)
        self.model = model or EnsembleModel(
            models=['random_forest', 'xgboost', 'lightgbm'],
            prediction_type=model_pred_type,
        )
        
        self.feature_extractor = feature_extractor or create_feature_extractor(
            preset="default",
            feature_groups=self.config.feature_groups,
            lag_periods=self.config.lag_periods,
        )
        
        # State
        self._last_train_date: Optional[pd.Timestamp] = None
        self._train_metrics: Dict[str, float] = {}
        self._prediction_history: List[Dict] = []
        self._is_trained = False
    
    @staticmethod
    def _map_prediction_type(strategy_type: str) -> str:
        """Map strategy prediction type to model prediction type.
        
        Strategy uses: 'returns', 'direction', 'both'
        Model uses: 'regression', 'classification', 'both'
        """
        mapping = {
            'returns': 'regression',
            'direction': 'classification',
            'both': 'both',
        }
        return mapping.get(strategy_type, 'both')
    
    def generate_signals(
        self,
        price_data: PriceData,
        **kwargs
    ) -> pd.DataFrame:
        """Generate trading signals using ML predictions.
        
        This method implements walk-forward training:
        1. Extract features from price data
        2. Train/retrain model if needed (using only past data)
        3. Generate predictions
        4. Convert predictions to signals
        
        Args:
            price_data: PriceData object
            **kwargs: Additional arguments
            
        Returns:
            Signal DataFrame (date x symbol) with values in [-1, 1]
        """
        logger.info(f"Generating ML signals for {len(price_data.symbols)} symbols")
        
        # Extract features
        features = self.feature_extractor.extract(price_data)
        
        # Create targets
        y_returns, y_direction = create_target_variable(
            price_data,
            target_type=TargetType(self.config.prediction_type),
            forward_periods=self.config.forward_periods,
            direction_threshold=self.config.direction_threshold,
        )
        
        # Get unique dates
        if isinstance(features.index, pd.MultiIndex):
            dates = features.index.get_level_values(0).unique().sort_values()
        else:
            dates = features.index.sort_values()
        
        # Initialize signals DataFrame
        symbols = price_data.symbols
        signals = pd.DataFrame(
            index=dates,
            columns=symbols,
            dtype=float
        )
        signals[:] = 0.0
        
        # Walk-forward signal generation
        warmup_end = dates[min(self.config.warmup_period, len(dates) - 1)]
        
        for i, date in enumerate(dates):
            if date <= warmup_end:
                continue
            
            # Check if retraining needed
            if self._should_retrain(date, dates[:i]):
                self._train_model(
                    features=features,
                    y_returns=y_returns,
                    y_direction=y_direction,
                    train_end_date=date,
                    dates=dates[:i],
                )
                self._last_train_date = date
            
            # Generate predictions if model is trained
            if self._is_trained:
                day_signals = self._predict_day(
                    features=features,
                    date=date,
                    symbols=symbols,
                )
                signals.loc[date] = day_signals
        
        # Apply constraints
        signals = self._apply_constraints(signals)
        
        logger.info(f"Generated signals with shape {signals.shape}")
        
        return signals
    
    def _should_retrain(
        self,
        current_date: pd.Timestamp,
        past_dates: pd.DatetimeIndex,
    ) -> bool:
        """Check if model should be retrained."""
        # First training
        if not self._is_trained:
            if len(past_dates) >= self.config.min_train_samples:
                return True
            return False
        
        # Periodic retraining
        if self._last_train_date is not None:
            days_since_train = len(past_dates[past_dates > self._last_train_date])
            if days_since_train >= self.config.retrain_frequency:
                return True
        
        # Drift-based retraining
        if self.config.monitor_drift and self._check_drift():
            logger.warning("Performance drift detected, triggering retrain")
            return True
        
        return False
    
    def _check_drift(self) -> bool:
        """Check if prediction performance has drifted."""
        if len(self._prediction_history) < 20:
            return False
        
        # Compare recent vs historical performance
        recent = self._prediction_history[-20:]
        historical = self._prediction_history[:-20]
        
        if not historical:
            return False
        
        recent_accuracy = np.mean([p.get('accuracy', 0.5) for p in recent])
        historical_accuracy = np.mean([p.get('accuracy', 0.5) for p in historical])
        
        if historical_accuracy > 0:
            drift = (historical_accuracy - recent_accuracy) / historical_accuracy
            return drift > self.config.drift_threshold
        
        return False
    
    def _train_model(
        self,
        features: pd.DataFrame,
        y_returns: Optional[pd.DataFrame],
        y_direction: Optional[pd.DataFrame],
        train_end_date: pd.Timestamp,
        dates: pd.DatetimeIndex,
    ):
        """Train the ML model on historical data."""
        logger.info(f"Training model with data up to {train_end_date}")
        
        # Get training data (all data before train_end_date)
        if isinstance(features.index, pd.MultiIndex):
            train_mask = features.index.get_level_values(0) < train_end_date
        else:
            train_mask = features.index < train_end_date
        
        X_train = features[train_mask]
        
        # Stack targets to match features
        y_ret_train = None
        y_dir_train = None
        
        if y_returns is not None:
            y_ret_stacked = y_returns.stack()
            y_ret_train = y_ret_stacked[y_ret_stacked.index.get_level_values(0) < train_end_date]
        
        if y_direction is not None:
            y_dir_stacked = y_direction.stack()
            y_dir_train = y_dir_stacked[y_dir_stacked.index.get_level_values(0) < train_end_date]
        
        # Align indices
        if y_ret_train is not None:
            common_idx = X_train.index.intersection(y_ret_train.index)
            X_train = X_train.loc[common_idx]
            y_ret_train = y_ret_train.loc[common_idx]
            if y_dir_train is not None:
                y_dir_train = y_dir_train.loc[common_idx]
        elif y_dir_train is not None:
            common_idx = X_train.index.intersection(y_dir_train.index)
            X_train = X_train.loc[common_idx]
            y_dir_train = y_dir_train.loc[common_idx]
        
        # Drop NaN values
        valid_mask = ~X_train.isna().any(axis=1)
        if y_ret_train is not None:
            valid_mask &= ~y_ret_train.isna()
        if y_dir_train is not None:
            valid_mask &= ~y_dir_train.isna()
        
        X_train = X_train[valid_mask]
        if y_ret_train is not None:
            y_ret_train = y_ret_train[valid_mask]
        if y_dir_train is not None:
            y_dir_train = y_dir_train[valid_mask]
        
        if len(X_train) < self.config.min_train_samples:
            logger.warning(f"Not enough samples for training: {len(X_train)}")
            return
        
        # Train model
        try:
            self.model.fit(
                X=X_train,
                y_returns=y_ret_train,
                y_direction=y_dir_train,
            )
            self._is_trained = True
            
            # Store metrics
            self._train_metrics = {
                'n_samples': len(X_train),
                'n_features': X_train.shape[1],
                'train_end_date': str(train_end_date),
            }
            
            logger.info(
                f"Model trained on {len(X_train)} samples, "
                f"{X_train.shape[1]} features"
            )
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise
    
    def _predict_day(
        self,
        features: pd.DataFrame,
        date: pd.Timestamp,
        symbols: List[str],
    ) -> pd.Series:
        """Generate predictions for a single day."""
        # Get features for this date
        if isinstance(features.index, pd.MultiIndex):
            day_mask = features.index.get_level_values(0) == date
            X_day = features[day_mask]
        else:
            X_day = features.loc[[date]]
        
        if len(X_day) == 0:
            return pd.Series(0.0, index=symbols)
        
        # Handle missing values
        X_day = X_day.fillna(0)
        
        # Get predictions
        try:
            predictions = self.model.predict(X_day)
        except Exception as e:
            logger.warning(f"Prediction failed for {date}: {e}")
            return pd.Series(0.0, index=symbols)
        
        # Convert predictions to signals
        signals = self._predictions_to_signals(predictions, X_day.index, symbols)
        
        return signals
    
    def _predictions_to_signals(
        self,
        predictions: Dict[str, np.ndarray],
        index: pd.MultiIndex,
        symbols: List[str],
    ) -> pd.Series:
        """Convert model predictions to trading signals."""
        signals = pd.Series(0.0, index=symbols)
        
        # Get symbol order from index
        if isinstance(index, pd.MultiIndex):
            pred_symbols = index.get_level_values(1).tolist()
        else:
            pred_symbols = symbols
        
        # Combine returns and direction predictions
        if self.config.signal_method == "returns" and 'returns' in predictions:
            raw_signals = predictions['returns']
            # Normalize to [-1, 1] using tanh
            raw_signals = np.tanh(raw_signals * 10)
            
        elif self.config.signal_method == "direction" and 'direction' in predictions:
            raw_signals = predictions['direction'].astype(float)
            
        elif self.config.signal_method == "combined":
            # Combine returns and direction
            ret_signal = np.zeros(len(pred_symbols))
            dir_signal = np.zeros(len(pred_symbols))
            
            if 'returns' in predictions:
                ret_signal = np.tanh(predictions['returns'] * 10)
            
            if 'direction' in predictions:
                dir_signal = predictions['direction'].astype(float)
            elif 'direction_score' in predictions:
                dir_signal = predictions['direction_score']
            
            # Weighted combination (returns + direction should agree)
            raw_signals = 0.5 * ret_signal + 0.5 * dir_signal
            
        else:
            # Default to returns if available
            if 'returns' in predictions:
                raw_signals = np.tanh(predictions['returns'] * 10)
            else:
                raw_signals = predictions.get('direction', np.zeros(len(pred_symbols))).astype(float)
        
        # Apply scale
        raw_signals = raw_signals * self.config.signal_scale
        
        # Map to symbols
        for i, symbol in enumerate(pred_symbols):
            if symbol in signals.index:
                signals[symbol] = raw_signals[i]
        
        return signals
    
    def _apply_constraints(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Apply trading constraints to signals."""
        # Long-only constraint
        if self.config.long_only:
            signals = signals.clip(lower=0)
        
        # Top-N constraint
        if self.config.top_n is not None and self.config.top_n > 0:
            # Keep only top N signals per day
            for date in signals.index:
                day_signals = signals.loc[date]
                top_mask = day_signals.rank(ascending=False) <= self.config.top_n
                signals.loc[date, ~top_mask] = 0
        
        # Normalize signals to [-1, 1]
        max_abs = signals.abs().max().max()
        if max_abs > 1:
            signals = signals / max_abs
        
        return signals
    
    def construct_portfolio(
        self,
        signals: pd.DataFrame,
        price_data: PriceData,
        **kwargs
    ) -> Portfolio:
        """Construct portfolio from ML signals.
        
        Args:
            signals: Signal DataFrame (date x symbol)
            price_data: PriceData object
            **kwargs: Additional arguments
            
        Returns:
            Portfolio object
        """
        return self.portfolio_constructor.construct(
            signals=signals,
            price_data=price_data,
            **kwargs
        )
    
    def get_feature_importance(self, top_n: int = 20) -> pd.Series:
        """Get feature importance from trained model."""
        if not self._is_trained:
            raise ValueError("Model not trained")
        
        return self.feature_extractor.get_feature_importance(
            model=self.model,
            top_n=top_n,
        )
    
    def get_train_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        return self._train_metrics.copy()
    
    def get_model_weights(self) -> Optional[Dict[str, float]]:
        """Get ensemble model weights if applicable."""
        if hasattr(self.model, 'get_model_weights'):
            return self.model.get_model_weights()
        return None

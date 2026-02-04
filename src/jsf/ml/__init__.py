"""Machine Learning Integration Module.

This module provides ML-based trading strategies for JSF-Core, including:
- Feature extraction from 20+ technical/fundamental/sentiment signals
- ML model wrappers (RandomForest, XGBoost, LightGBM)
- Ensemble strategies with weighted voting
- Walk-forward validation for time-series ML
- MLStrategy class for backtesting ML models

Example:
    >>> from jsf.ml import MLStrategy, FeatureExtractor, EnsembleModel
    >>> from jsf.data import load_data
    >>> 
    >>> # Load data
    >>> data = load_data(source='synthetic', symbols=['AAPL', 'GOOGL'])
    >>> 
    >>> # Create feature extractor
    >>> extractor = FeatureExtractor(
    ...     feature_groups=['momentum', 'volatility', 'trend'],
    ...     lookbacks=[20, 60],
    ...     lag_periods=[1, 5]
    ... )
    >>> 
    >>> # Create ensemble model
    >>> model = EnsembleModel(models=['random_forest', 'xgboost', 'lightgbm'])
    >>> 
    >>> # Create ML strategy
    >>> strategy = MLStrategy(
    ...     name='ml_ensemble',
    ...     model=model,
    ...     feature_extractor=extractor,
    ...     prediction_type='both'  # returns + direction
    ... )
"""

from .features import (
    FeatureExtractor,
    FeatureConfig,
    create_feature_extractor,
    FEATURE_GROUPS,
)

from .models import (
    MLModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    EnsembleModel,
    ModelConfig,
)

from .strategy import (
    MLStrategy,
    MLStrategyConfig,
)

from .validation import (
    WalkForwardMLValidator,
    MLValidationResult,
    validate_ml_strategy,
)

from .preprocessing import (
    prepare_ml_data,
    create_target_variable,
    split_train_test,
    handle_missing_features,
    MultiIndexConverter,
)

__all__ = [
    # Feature extraction
    "FeatureExtractor",
    "FeatureConfig",
    "create_feature_extractor",
    "FEATURE_GROUPS",
    # Models
    "MLModel",
    "RandomForestModel",
    "XGBoostModel",
    "LightGBMModel",
    "EnsembleModel",
    "ModelConfig",
    # Strategy
    "MLStrategy",
    "MLStrategyConfig",
    # Validation
    "WalkForwardMLValidator",
    "MLValidationResult",
    "validate_ml_strategy",
    # Preprocessing
    "prepare_ml_data",
    "create_target_variable",
    "split_train_test",
    "handle_missing_features",
    "MultiIndexConverter",
]

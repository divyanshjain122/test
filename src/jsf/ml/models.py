"""ML model wrappers for trading strategies.

Provides unified interface for scikit-learn, XGBoost, and LightGBM models.
Supports both regression (returns prediction) and classification (direction).
"""

from typing import Dict, List, Optional, Union, Any, Literal
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import warnings

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from jsf.utils.logging import get_logger

logger = get_logger(__name__)


class PredictionType(Enum):
    """Type of prediction."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    BOTH = "both"


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    
    # Model type
    model_type: str = "random_forest"
    prediction_type: str = "both"
    
    # Common hyperparameters
    n_estimators: int = 100
    max_depth: Optional[int] = 5
    min_samples_split: int = 100
    min_samples_leaf: int = 50
    
    # Regularization
    learning_rate: float = 0.1  # For boosting models
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Training
    random_state: int = 42
    n_jobs: int = -1
    early_stopping_rounds: Optional[int] = 10
    
    # Ensemble weights (for EnsembleModel)
    ensemble_weights: Optional[Dict[str, float]] = None


class MLModel(ABC):
    """Abstract base class for ML models.
    
    Provides unified interface for different ML libraries.
    Supports both regression and classification.
    """
    
    def __init__(
        self,
        prediction_type: Union[str, PredictionType] = PredictionType.BOTH,
        config: Optional[ModelConfig] = None,
        **kwargs
    ):
        """Initialize ML model.
        
        Args:
            prediction_type: Type of prediction ('regression', 'classification', 'both')
            config: ModelConfig object
            **kwargs: Model-specific parameters
        """
        if isinstance(prediction_type, str):
            prediction_type = PredictionType(prediction_type)
        
        self.prediction_type = prediction_type
        self.config = config or ModelConfig(**kwargs)
        
        self.regressor = None
        self.classifier = None
        self._is_fitted = False
        self._feature_names = None
    
    @abstractmethod
    def _create_regressor(self) -> Any:
        """Create the underlying regressor model."""
        pass
    
    @abstractmethod
    def _create_classifier(self) -> Any:
        """Create the underlying classifier model."""
        pass
    
    def fit(
        self,
        X: pd.DataFrame,
        y_returns: Optional[pd.Series] = None,
        y_direction: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        validation_data: Optional[tuple] = None,
    ) -> "MLModel":
        """Fit the model.
        
        Args:
            X: Feature matrix
            y_returns: Continuous target (for regression)
            y_direction: Categorical target (for classification)
            sample_weight: Sample weights
            validation_data: (X_val, y_val) for early stopping
            
        Returns:
            self
        """
        self._feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        
        X_arr = X.values if hasattr(X, 'values') else X
        
        # Fit regressor
        if self.prediction_type in [PredictionType.REGRESSION, PredictionType.BOTH]:
            if y_returns is None:
                raise ValueError("y_returns required for regression")
            
            y_arr = y_returns.values if hasattr(y_returns, 'values') else y_returns
            
            self.regressor = self._create_regressor()
            
            fit_params = {}
            if sample_weight is not None:
                fit_params['sample_weight'] = sample_weight
            
            # Handle early stopping for boosting models
            if validation_data is not None and hasattr(self.regressor, 'set_params'):
                self._fit_with_early_stopping(
                    self.regressor, X_arr, y_arr, validation_data, fit_params
                )
            else:
                self.regressor.fit(X_arr, y_arr, **fit_params)
            
            logger.info(f"Fitted regressor on {len(X)} samples")
        
        # Fit classifier
        if self.prediction_type in [PredictionType.CLASSIFICATION, PredictionType.BOTH]:
            if y_direction is None:
                raise ValueError("y_direction required for classification")
            
            y_arr = y_direction.values if hasattr(y_direction, 'values') else y_direction
            
            self.classifier = self._create_classifier()
            
            fit_params = {}
            if sample_weight is not None:
                fit_params['sample_weight'] = sample_weight
            
            self.classifier.fit(X_arr, y_arr, **fit_params)
            
            logger.info(f"Fitted classifier on {len(X)} samples")
        
        self._is_fitted = True
        return self
    
    def _fit_with_early_stopping(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple,
        fit_params: dict,
    ):
        """Fit with early stopping for boosting models."""
        X_val, y_val = validation_data
        X_val_arr = X_val.values if hasattr(X_val, 'values') else X_val
        y_val_arr = y_val.values if hasattr(y_val, 'values') else y_val
        
        # XGBoost and LightGBM have different early stopping APIs
        if hasattr(model, 'fit'):
            try:
                model.fit(
                    X, y,
                    eval_set=[(X_val_arr, y_val_arr)],
                    **fit_params
                )
            except TypeError:
                # Fall back to regular fit
                model.fit(X, y, **fit_params)
    
    def predict(
        self,
        X: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dict with 'returns' and/or 'direction' predictions
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_arr = X.values if hasattr(X, 'values') else X
        predictions = {}
        
        if self.regressor is not None:
            predictions['returns'] = self.regressor.predict(X_arr)
        
        if self.classifier is not None:
            predictions['direction'] = self.classifier.predict(X_arr)
            
            # Get probability if available
            if hasattr(self.classifier, 'predict_proba'):
                proba = self.classifier.predict_proba(X_arr)
                # Convert to confidence score (-1 to 1)
                if proba.shape[1] == 3:  # -1, 0, 1
                    predictions['direction_proba'] = proba[:, 2] - proba[:, 0]
                elif proba.shape[1] == 2:  # 0, 1
                    predictions['direction_proba'] = proba[:, 1] * 2 - 1
        
        return predictions
    
    def predict_returns(self, X: pd.DataFrame) -> np.ndarray:
        """Predict returns only."""
        preds = self.predict(X)
        if 'returns' not in preds:
            raise ValueError("Model not fitted for regression")
        return preds['returns']
    
    def predict_direction(self, X: pd.DataFrame) -> np.ndarray:
        """Predict direction only."""
        preds = self.predict(X)
        if 'direction' not in preds:
            raise ValueError("Model not fitted for classification")
        return preds['direction']
    
    @property
    def feature_importances_(self) -> np.ndarray:
        """Get feature importances."""
        if self.regressor is not None and hasattr(self.regressor, 'feature_importances_'):
            return self.regressor.feature_importances_
        if self.classifier is not None and hasattr(self.classifier, 'feature_importances_'):
            return self.classifier.feature_importances_
        raise ValueError("Model doesn't have feature importances")
    
    def save(self, path: Union[str, Path]):
        """Save model to disk."""
        path = Path(path)
        joblib.dump(self, path)
        logger.info(f"Saved model to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "MLModel":
        """Load model from disk."""
        path = Path(path)
        model = joblib.load(path)
        logger.info(f"Loaded model from {path}")
        return model


class RandomForestModel(MLModel):
    """Random Forest model wrapper."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_split: int = 100,
        min_samples_leaf: int = 50,
        prediction_type: str = "both",
        **kwargs
    ):
        config = ModelConfig(
            model_type="random_forest",
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            **kwargs
        )
        super().__init__(prediction_type=prediction_type, config=config)
    
    def _create_regressor(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor
        
        return RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
    
    def _create_classifier(self) -> Any:
        from sklearn.ensemble import RandomForestClassifier
        
        return RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )


class XGBoostModel(MLModel):
    """XGBoost model wrapper."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        prediction_type: str = "both",
        **kwargs
    ):
        config = ModelConfig(
            model_type="xgboost",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            **kwargs
        )
        super().__init__(prediction_type=prediction_type, config=config)
    
    def _create_regressor(self) -> Any:
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Install with: pip install xgboost")
        
        return xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=0,
        )
    
    def _create_classifier(self) -> Any:
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Install with: pip install xgboost")
        
        return xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=0,
        )


class LightGBMModel(MLModel):
    """LightGBM model wrapper."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        prediction_type: str = "both",
        **kwargs
    ):
        config = ModelConfig(
            model_type="lightgbm",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            **kwargs
        )
        super().__init__(prediction_type=prediction_type, config=config)
    
    def _create_regressor(self) -> Any:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed. Install with: pip install lightgbm")
        
        return lgb.LGBMRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=-1,
        )
    
    def _create_classifier(self) -> Any:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed. Install with: pip install lightgbm")
        
        return lgb.LGBMClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=-1,
        )


class EnsembleModel(MLModel):
    """Weighted ensemble of multiple ML models.
    
    Combines predictions from RandomForest, XGBoost, LightGBM, and Neural Network
    models using configurable weights. Weights can be updated based on
    out-of-sample performance.
    
    Supports tree-based models (RF, XGBoost, LightGBM) and neural networks
    (MLP, LSTM, GRU) for hybrid ensembles that capture different patterns.
    
    Example:
        >>> # Traditional ensemble
        >>> ensemble = EnsembleModel(
        ...     models=['random_forest', 'xgboost', 'lightgbm'],
        ...     weights={'random_forest': 0.3, 'xgboost': 0.4, 'lightgbm': 0.3}
        ... )
        >>> ensemble.fit(X_train, y_returns=y_train)
        >>> predictions = ensemble.predict(X_test)
        >>> 
        >>> # Hybrid ensemble with neural networks
        >>> hybrid = EnsembleModel(
        ...     models=['xgboost', 'mlp', 'lstm'],
        ...     weights={'xgboost': 0.4, 'mlp': 0.3, 'lstm': 0.3}
        ... )
    """
    
    # Base model classes (always available)
    MODEL_CLASSES = {
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
    }
    
    # Neural network models (loaded on demand)
    _NEURAL_MODEL_CLASSES = None
    
    @classmethod
    def _get_neural_models(cls) -> Dict[str, type]:
        """Load neural network model classes."""
        if cls._NEURAL_MODEL_CLASSES is None:
            from .neural import MLPModel, LSTMModel, GRUModel, TransformerModel
            cls._NEURAL_MODEL_CLASSES = {
                'mlp': MLPModel,
                'lstm': LSTMModel,
                'gru': GRUModel,
                'transformer': TransformerModel,
            }
            logger.info("Neural network models loaded successfully")
        return cls._NEURAL_MODEL_CLASSES
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of all available model types."""
        models = list(cls.MODEL_CLASSES.keys())
        models.extend(cls._get_neural_models().keys())
        return models
    
    @classmethod
    def get_model_class(cls, name: str) -> type:
        """Get model class by name.
        
        Args:
            name: Model name (e.g., 'xgboost', 'mlp', 'lstm')
            
        Returns:
            Model class
            
        Raises:
            ValueError: If model name is not recognized
        """
        if name in cls.MODEL_CLASSES:
            return cls.MODEL_CLASSES[name]
        
        neural_models = cls._get_neural_models()
        if name in neural_models:
            return neural_models[name]
        
        available = cls.get_available_models()
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    
    def __init__(
        self,
        models: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        prediction_type: str = "both",
        update_weights_on_validation: bool = True,
        **model_kwargs
    ):
        """Initialize ensemble model.
        
        Args:
            models: List of model names to include
            weights: Dict of model_name -> weight (must sum to 1)
            prediction_type: Type of prediction
            update_weights_on_validation: Whether to update weights based on OOS performance
            **model_kwargs: Arguments passed to all models
        """
        super().__init__(prediction_type=prediction_type)
        
        self.model_names = models or ['random_forest', 'xgboost', 'lightgbm']
        self.update_weights_on_validation = update_weights_on_validation
        self.model_kwargs = model_kwargs
        
        # Initialize weights
        if weights is None:
            # Default weights
            n_models = len(self.model_names)
            self.weights = {name: 1.0 / n_models for name in self.model_names}
        else:
            self.weights = weights
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        
        # Create model instances
        self._models: Dict[str, MLModel] = {}
        for name in self.model_names:
            # Use get_model_class to support both tree and neural models
            model_class = self.get_model_class(name)
            
            # Check if this is a neural model by looking at the module
            is_neural = 'neural' in model_class.__module__
            
            # Filter kwargs: neural models get their params, tree models get theirs
            # Neural-specific params that tree models don't need
            neural_params = {
                'hidden_layers', 'dropout_rate', 'learning_rate', 'epochs',
                'batch_size', 'optimizer', 'early_stopping_patience',
                'reduce_lr_patience', 'validation_split', 'sequence_length',
                'recurrent_units', 'recurrent_dropout', 'return_sequences',
                'bidirectional', 'use_mixed_precision', 'l1_reg', 'l2_reg',
                'batch_norm', 'output_activation'
            }
            
            if is_neural:
                filtered_kwargs = model_kwargs
            else:
                # Tree model - exclude neural-specific params
                filtered_kwargs = {
                    k: v for k, v in model_kwargs.items()
                    if k not in neural_params
                }
            
            self._models[name] = model_class(
                prediction_type=prediction_type,
                **filtered_kwargs
            )
        
        logger.info(f"EnsembleModel with {self.model_names}, weights={self.weights}")
    
    def _create_regressor(self) -> Any:
        """Not used for ensemble."""
        pass
    
    def _create_classifier(self) -> Any:
        """Not used for ensemble."""
        pass
    
    def fit(
        self,
        X: pd.DataFrame,
        y_returns: Optional[pd.Series] = None,
        y_direction: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        validation_data: Optional[tuple] = None,
    ) -> "EnsembleModel":
        """Fit all models in the ensemble.
        
        Args:
            X: Feature matrix
            y_returns: Continuous target
            y_direction: Classification target
            sample_weight: Sample weights
            validation_data: (X_val, y_ret_val, y_dir_val) for weight update
            
        Returns:
            self
        """
        self._feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        
        # Fit each model
        for name, model in self._models.items():
            try:
                model.fit(
                    X=X,
                    y_returns=y_returns,
                    y_direction=y_direction,
                    sample_weight=sample_weight,
                )
                logger.info(f"Fitted {name}")
            except Exception as e:
                logger.warning(f"Failed to fit {name}: {e}")
                # Set weight to 0 for failed models
                self.weights[name] = 0.0
        
        # Update weights based on validation performance
        if validation_data is not None and self.update_weights_on_validation:
            self._update_weights_from_validation(validation_data)
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        
        self._is_fitted = True
        return self
    
    def _update_weights_from_validation(self, validation_data: tuple):
        """Update model weights based on validation performance."""
        if len(validation_data) == 3:
            X_val, y_ret_val, y_dir_val = validation_data
        else:
            X_val, y_ret_val = validation_data
            y_dir_val = None
        
        performances = {}
        
        for name, model in self._models.items():
            if self.weights[name] == 0:
                continue
            
            try:
                preds = model.predict(X_val)
                
                # Use correlation as performance metric
                if 'returns' in preds and y_ret_val is not None:
                    y_arr = y_ret_val.values if hasattr(y_ret_val, 'values') else y_ret_val
                    corr = np.corrcoef(preds['returns'], y_arr)[0, 1]
                    if np.isnan(corr):
                        corr = 0
                    performances[name] = max(0, corr)  # Only positive correlations
                elif 'direction' in preds and y_dir_val is not None:
                    y_arr = y_dir_val.values if hasattr(y_dir_val, 'values') else y_dir_val
                    accuracy = np.mean(preds['direction'] == y_arr)
                    performances[name] = accuracy
                else:
                    performances[name] = self.weights[name]
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate {name}: {e}")
                performances[name] = 0
        
        # Update weights proportional to performance
        if performances:
            total_perf = sum(performances.values())
            if total_perf > 0:
                for name in performances:
                    self.weights[name] = performances[name] / total_perf
                logger.info(f"Updated weights from validation: {self.weights}")
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make weighted ensemble predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dict with 'returns' and/or 'direction' predictions
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Collect predictions from all models
        all_returns = []
        all_directions = []
        all_weights = []
        
        for name, model in self._models.items():
            if self.weights.get(name, 0) == 0:
                continue
            
            try:
                preds = model.predict(X)
                weight = self.weights[name]
                
                if 'returns' in preds:
                    all_returns.append((preds['returns'], weight))
                if 'direction' in preds:
                    all_directions.append((preds['direction'], weight))
                
                all_weights.append(weight)
                
            except Exception as e:
                logger.warning(f"Failed to get predictions from {name}: {e}")
        
        predictions = {}
        
        # Weighted average for returns
        if all_returns:
            weighted_sum = sum(pred * w for pred, w in all_returns)
            total_weight = sum(w for _, w in all_returns)
            predictions['returns'] = weighted_sum / total_weight
        
        # Weighted voting for direction
        if all_directions:
            # Use weighted voting
            weighted_sum = sum(pred * w for pred, w in all_directions)
            total_weight = sum(w for _, w in all_directions)
            avg_direction = weighted_sum / total_weight
            
            # Convert to discrete: -1, 0, 1
            predictions['direction'] = np.sign(avg_direction).astype(int)
            predictions['direction_score'] = avg_direction
        
        return predictions
    
    @property
    def feature_importances_(self) -> np.ndarray:
        """Get weighted average feature importances."""
        importances = []
        weights = []
        
        for name, model in self._models.items():
            if self.weights.get(name, 0) == 0:
                continue
            try:
                imp = model.feature_importances_
                importances.append(imp * self.weights[name])
                weights.append(self.weights[name])
            except Exception:
                pass
        
        if not importances:
            raise ValueError("No models have feature importances")
        
        return sum(importances) / sum(weights)
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.weights.copy()
    
    def set_model_weights(self, weights: Dict[str, float]):
        """Set model weights."""
        for name in weights:
            if name not in self._models:
                raise ValueError(f"Unknown model: {name}")
        
        self.weights = weights
        
        # Normalize
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

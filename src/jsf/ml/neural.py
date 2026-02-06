"""TensorFlow Neural Network Models for Trading Strategies.

This module provides neural network implementations using TensorFlow/Keras,
including MLP, LSTM, and GRU models that integrate seamlessly with the
jsf.ml pipeline.

These models extend the MLModel base class and support:
- Both regression (return prediction) and classification (direction)
- GPU acceleration (when available)
- Sequence handling for temporal models (LSTM/GRU)
- Walk-forward validation compatibility
- Ensemble integration

Example:
    >>> from jsf.ml.neural import MLPModel, LSTMModel, GRUModel
    >>> 
    >>> # Simple MLP for non-linear relationships
    >>> mlp = MLPModel(hidden_layers=[64, 32], dropout_rate=0.3)
    >>> mlp.fit(X_train, y_returns=y_train_ret, y_direction=y_train_dir)
    >>> 
    >>> # LSTM for temporal patterns
    >>> lstm = LSTMModel(sequence_length=20, units=[64, 32])
    >>> lstm.fit(X_train, y_returns=y_train_ret)

GPU Support:
    TensorFlow will automatically use GPU if available. To force CPU:
    >>> import os
    >>> os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from abc import abstractmethod
import warnings

import pandas as pd
import numpy as np
from pathlib import Path

from .models import MLModel, ModelConfig, PredictionType
from jsf.utils.logging import get_logger

logger = get_logger(__name__)


def _check_tensorflow():
    """Check if TensorFlow is installed and return it."""
    try:
        import tensorflow as tf
        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
        return tf
    except ImportError:
        raise ImportError(
            "TensorFlow is not installed. Install with: pip install tensorflow\n"
            "For GPU support: pip install tensorflow[and-cuda]"
        )


def configure_gpu_memory(memory_limit_mb: Optional[int] = None, allow_growth: bool = True):
    """Configure GPU memory settings for TensorFlow.
    
    Call this BEFORE creating any models to avoid OOM errors.
    
    Args:
        memory_limit_mb: Optional memory limit in MB per GPU. If None, uses growth.
        allow_growth: If True, allocate memory as needed (recommended).
        
    Example:
        >>> from jsf.ml.neural import configure_gpu_memory
        >>> configure_gpu_memory(allow_growth=True)  # Recommended
        >>> # Or limit to 4GB:
        >>> configure_gpu_memory(memory_limit_mb=4096)
        
    Note:
        This is for EDUCATIONAL PURPOSES. In production, carefully tune
        memory settings based on your hardware and model requirements.
    """
    tf = _check_tensorflow()
    
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logger.info("No GPU detected. Using CPU.")
        return
    
    try:
        for gpu in gpus:
            if memory_limit_mb is not None:
                # Set hard memory limit
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                )
                logger.info(f"GPU memory limited to {memory_limit_mb}MB: {gpu}")
            elif allow_growth:
                # Allow dynamic memory allocation
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU memory growth enabled: {gpu}")
    except RuntimeError as e:
        # GPU configuration must be done before initialization
        logger.warning(f"GPU configuration failed (must be set before model creation): {e}")


def get_gpu_info() -> Dict[str, Any]:
    """Get information about available GPUs.
    
    Returns:
        Dict with GPU information for debugging/logging.
        
    Example:
        >>> info = get_gpu_info()
        >>> print(f"GPUs available: {info['n_gpus']}")
    """
    tf = _check_tensorflow()
    
    gpus = tf.config.list_physical_devices('GPU')
    
    info = {
        'n_gpus': len(gpus),
        'gpus': [gpu.name for gpu in gpus],
        'cuda_available': len(gpus) > 0,
        'tensorflow_version': tf.__version__,
    }
    
    # Try to get memory info
    if gpus:
        try:
            for i, gpu in enumerate(gpus):
                # This requires GPU to be initialized
                info[f'gpu_{i}_memory'] = 'Available (check nvidia-smi for details)'
        except Exception:
            pass
    
    return info


@dataclass
class NeuralConfig(ModelConfig):
    """Configuration for Neural Network models."""
    
    # Architecture
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    activation: str = "relu"
    output_activation: str = "linear"  # For regression
    
    # Regularization
    dropout_rate: float = 0.2
    l1_reg: float = 0.0
    l2_reg: float = 0.001
    batch_norm: bool = True
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    validation_split: float = 0.2
    
    # For sequence models (LSTM/GRU)
    sequence_length: int = 20
    recurrent_units: List[int] = field(default_factory=lambda: [64, 32])
    recurrent_dropout: float = 0.1
    return_sequences: bool = False
    bidirectional: bool = False
    
    # GPU/Performance
    use_mixed_precision: bool = False  # Enable for faster GPU training


class NeuralModel(MLModel):
    """Base class for TensorFlow Neural Network models.
    
    Provides common infrastructure for building, training, and predicting
    with neural networks. Subclasses implement specific architectures.
    """
    
    def __init__(
        self,
        prediction_type: Union[str, PredictionType] = PredictionType.BOTH,
        config: Optional[NeuralConfig] = None,
        **kwargs
    ):
        """Initialize neural network model.
        
        Args:
            prediction_type: 'regression', 'classification', or 'both'
            config: NeuralConfig object
            **kwargs: Override config parameters
        """
        # Build config from kwargs if not provided
        if config is None:
            config = NeuralConfig(**kwargs)
        
        super().__init__(prediction_type=prediction_type, config=config)
        
        self.tf = _check_tensorflow()
        self._scaler_X = None
        self._scaler_y = None
        self._n_classes = None
        self._history = None
        
        # Enable mixed precision if requested
        if self.config.use_mixed_precision:
            try:
                self.tf.keras.mixed_precision.set_global_policy('mixed_float16')
                logger.info("Enabled mixed precision training")
            except Exception as e:
                logger.warning(f"Could not enable mixed precision: {e}")
    
    def _build_regressor_model(self, input_shape: Tuple[int, ...]) -> Any:
        """Build the regressor network. Override in subclasses."""
        return self._build_dense_model(input_shape, output_units=1, is_classifier=False)
    
    def _build_classifier_model(
        self, 
        input_shape: Tuple[int, ...], 
        n_classes: int
    ) -> Any:
        """Build the classifier network. Override in subclasses."""
        return self._build_dense_model(
            input_shape, 
            output_units=n_classes, 
            is_classifier=True
        )
    
    def _build_dense_model(
        self,
        input_shape: Tuple[int, ...],
        output_units: int,
        is_classifier: bool = False,
    ) -> Any:
        """Build a dense (MLP) model architecture.
        
        Args:
            input_shape: Input feature shape
            output_units: Number of output units
            is_classifier: Whether this is for classification
            
        Returns:
            Compiled Keras model
        """
        tf = self.tf
        
        # Regularizers
        regularizer = None
        if self.config.l1_reg > 0 or self.config.l2_reg > 0:
            regularizer = tf.keras.regularizers.L1L2(
                l1=self.config.l1_reg, l2=self.config.l2_reg
            )
        
        # Build model
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        
        # Hidden layers
        for i, units in enumerate(self.config.hidden_layers):
            x = tf.keras.layers.Dense(
                units,
                activation=self.config.activation,
                kernel_regularizer=regularizer,
                name=f"dense_{i}"
            )(x)
            
            if self.config.batch_norm:
                x = tf.keras.layers.BatchNormalization(name=f"bn_{i}")(x)
            
            if self.config.dropout_rate > 0:
                x = tf.keras.layers.Dropout(
                    self.config.dropout_rate, 
                    name=f"dropout_{i}"
                )(x)
        
        # Output layer
        if is_classifier:
            if output_units == 2:
                outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
            else:
                outputs = tf.keras.layers.Dense(
                    output_units, activation='softmax', name='output'
                )(x)
        else:
            outputs = tf.keras.layers.Dense(
                output_units, 
                activation=self.config.output_activation,
                name='output'
            )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        optimizer = self._get_optimizer()
        
        if is_classifier:
            if output_units == 2:
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def _get_optimizer(self) -> Any:
        """Get optimizer based on config."""
        tf = self.tf
        
        optimizer_map = {
            'adam': tf.keras.optimizers.Adam,
            'sgd': tf.keras.optimizers.SGD,
            'rmsprop': tf.keras.optimizers.RMSprop,
            'adamw': tf.keras.optimizers.AdamW,
        }
        
        opt_class = optimizer_map.get(self.config.optimizer.lower(), tf.keras.optimizers.Adam)
        return opt_class(learning_rate=self.config.learning_rate)
    
    def _get_callbacks(self, validation_data: bool = True) -> List[Any]:
        """Get training callbacks."""
        tf = self.tf
        callbacks = []
        
        # Early stopping
        if validation_data and self.config.early_stopping_patience > 0:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=0
            ))
        
        # Reduce learning rate on plateau
        if validation_data and self.config.reduce_lr_patience > 0:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-7,
                verbose=0
            ))
        
        return callbacks
    
    def _preprocess_features(
        self, 
        X: pd.DataFrame, 
        fit: bool = False
    ) -> np.ndarray:
        """Preprocess features with standardization."""
        from sklearn.preprocessing import StandardScaler
        
        X_arr = X.values if hasattr(X, 'values') else X
        
        if fit:
            self._scaler_X = StandardScaler()
            X_scaled = self._scaler_X.fit_transform(X_arr)
        else:
            if self._scaler_X is None:
                raise ValueError("Scaler not fitted. Call fit() first.")
            X_scaled = self._scaler_X.transform(X_arr)
        
        # Handle NaN/Inf
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return X_scaled.astype(np.float32)
    
    def _preprocess_target(
        self, 
        y: pd.Series, 
        fit: bool = False,
        is_classification: bool = False
    ) -> np.ndarray:
        """Preprocess target variable."""
        y_arr = y.values if hasattr(y, 'values') else y
        
        if is_classification:
            # Convert to integer classes
            unique_classes = np.unique(y_arr[~np.isnan(y_arr)])
            self._n_classes = len(unique_classes)
            
            # Map to 0, 1, 2, ... indices
            if -1 in unique_classes:
                # Direction: -1, 0, 1 -> 0, 1, 2
                y_mapped = (y_arr + 1).astype(np.int32)
            else:
                y_mapped = y_arr.astype(np.int32)
            
            return y_mapped
        else:
            # Regression - standardize
            from sklearn.preprocessing import StandardScaler
            
            y_arr = y_arr.reshape(-1, 1)
            
            if fit:
                self._scaler_y = StandardScaler()
                y_scaled = self._scaler_y.fit_transform(y_arr)
            else:
                if self._scaler_y is None:
                    raise ValueError("Target scaler not fitted.")
                y_scaled = self._scaler_y.transform(y_arr)
            
            return y_scaled.flatten().astype(np.float32)
    
    def _create_regressor(self) -> Any:
        """Create regressor - deferred until fit() with known input shape."""
        return None
    
    def _create_classifier(self) -> Any:
        """Create classifier - deferred until fit() with known input shape."""
        return None
    
    def fit(
        self,
        X: pd.DataFrame,
        y_returns: Optional[pd.Series] = None,
        y_direction: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        validation_data: Optional[tuple] = None,
    ) -> "NeuralModel":
        """Fit the neural network model.
        
        Args:
            X: Feature matrix
            y_returns: Continuous target for regression
            y_direction: Categorical target for classification
            sample_weight: Sample weights
            validation_data: (X_val, y_val) for early stopping
            
        Returns:
            self
        """
        self._feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        
        # Preprocess features
        X_scaled = self._preprocess_features(X, fit=True)
        input_shape = (X_scaled.shape[1],)
        
        # Get callbacks
        use_validation = (
            validation_data is not None or 
            self.config.validation_split > 0
        )
        callbacks = self._get_callbacks(validation_data=use_validation)
        
        # Fit regressor
        if self.prediction_type in [PredictionType.REGRESSION, PredictionType.BOTH]:
            if y_returns is None:
                raise ValueError("y_returns required for regression")
            
            y_scaled = self._preprocess_target(y_returns, fit=True, is_classification=False)
            
            # Build model
            self.regressor = self._build_regressor_model(input_shape)
            
            # Prepare validation data
            val_data = None
            if validation_data is not None:
                X_val_scaled = self._preprocess_features(validation_data[0], fit=False)
                y_val_scaled = self._preprocess_target(
                    validation_data[1], fit=False, is_classification=False
                )
                val_data = (X_val_scaled, y_val_scaled)
            
            # Train
            fit_kwargs = {
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'callbacks': callbacks,
                'verbose': 0,
            }
            
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight
            
            if val_data is not None:
                fit_kwargs['validation_data'] = val_data
            elif self.config.validation_split > 0:
                fit_kwargs['validation_split'] = self.config.validation_split
            
            self._history = self.regressor.fit(X_scaled, y_scaled, **fit_kwargs)
            
            logger.info(
                f"Fitted regressor: epochs={len(self._history.history['loss'])}, "
                f"final_loss={self._history.history['loss'][-1]:.4f}"
            )
        
        # Fit classifier
        if self.prediction_type in [PredictionType.CLASSIFICATION, PredictionType.BOTH]:
            if y_direction is None:
                raise ValueError("y_direction required for classification")
            
            y_processed = self._preprocess_target(
                y_direction, fit=True, is_classification=True
            )
            
            # Build model
            self.classifier = self._build_classifier_model(input_shape, self._n_classes)
            
            # Prepare validation data
            val_data = None
            if validation_data is not None:
                X_val_scaled = self._preprocess_features(validation_data[0], fit=False)
                y_val_processed = self._preprocess_target(
                    validation_data[2] if len(validation_data) > 2 else validation_data[1],
                    fit=False, 
                    is_classification=True
                )
                val_data = (X_val_scaled, y_val_processed)
            
            # Train
            fit_kwargs = {
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'callbacks': callbacks,
                'verbose': 0,
            }
            
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight
            
            if val_data is not None:
                fit_kwargs['validation_data'] = val_data
            elif self.config.validation_split > 0:
                fit_kwargs['validation_split'] = self.config.validation_split
            
            history = self.classifier.fit(X_scaled, y_processed, **fit_kwargs)
            
            logger.info(
                f"Fitted classifier: epochs={len(history.history['loss'])}, "
                f"final_acc={history.history.get('accuracy', [0])[-1]:.4f}"
            )
        
        self._is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dict with 'returns' and/or 'direction' predictions
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self._preprocess_features(X, fit=False)
        predictions = {}
        
        if self.regressor is not None:
            pred_scaled = self.regressor.predict(X_scaled, verbose=0)
            # Inverse transform
            pred = self._scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
            predictions['returns'] = pred.flatten()
        
        if self.classifier is not None:
            pred_proba = self.classifier.predict(X_scaled, verbose=0)
            
            if self._n_classes == 2:
                # Binary classification
                predictions['direction'] = (pred_proba.flatten() > 0.5).astype(int)
                predictions['direction_proba'] = pred_proba.flatten() * 2 - 1
            else:
                # Multi-class: 0, 1, 2 -> -1, 0, 1
                pred_class = np.argmax(pred_proba, axis=1) - 1
                predictions['direction'] = pred_class
                # Confidence: max probability
                predictions['direction_proba'] = np.max(pred_proba, axis=1) * 2 - 1
        
        return predictions
    
    def save(self, path: Union[str, Path]):
        """Save model to disk."""
        import pickle
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save Keras models
        if self.regressor is not None:
            self.regressor.save(path / "regressor.keras")
        if self.classifier is not None:
            self.classifier.save(path / "classifier.keras")
        
        # Save scalers and config
        meta = {
            'config': self.config,
            'prediction_type': self.prediction_type,
            'scaler_X': self._scaler_X,
            'scaler_y': self._scaler_y,
            'n_classes': self._n_classes,
            'feature_names': self._feature_names,
            'is_fitted': self._is_fitted,
        }
        with open(path / "meta.pkl", 'wb') as f:
            pickle.dump(meta, f)
        
        logger.info(f"Saved model to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "NeuralModel":
        """Load model from disk."""
        import pickle
        path = Path(path)
        tf = _check_tensorflow()
        
        # Load metadata
        with open(path / "meta.pkl", 'rb') as f:
            meta = pickle.load(f)
        
        # Create instance with just prediction_type
        model = cls(prediction_type=meta['prediction_type'])
        
        # Restore config
        model.config = meta['config']
        
        # Load scalers
        model._scaler_X = meta['scaler_X']
        model._scaler_y = meta['scaler_y']
        model._n_classes = meta['n_classes']
        model._feature_names = meta['feature_names']
        model._is_fitted = meta['is_fitted']
        
        # Load Keras models
        if (path / "regressor.keras").exists():
            model.regressor = tf.keras.models.load_model(path / "regressor.keras")
        if (path / "classifier.keras").exists():
            model.classifier = tf.keras.models.load_model(path / "classifier.keras")
        
        logger.info(f"Loaded model from {path}")
        return model
    
    @property
    def feature_importances_(self) -> np.ndarray:
        """Get feature importances via gradient-based attribution.
        
        Uses integrated gradients approximation to estimate importance.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        # Use regressor if available, else classifier
        model = self.regressor if self.regressor is not None else self.classifier
        
        # Simple gradient-based importance
        # Create a baseline (zeros)
        n_features = len(self._feature_names) if self._feature_names else model.input_shape[1]
        
        # Use weights from first layer as proxy for importance
        first_layer_weights = None
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                first_layer_weights = np.abs(layer.kernel.numpy())
                break
        
        if first_layer_weights is None:
            # Fallback: uniform importance
            return np.ones(n_features) / n_features
        
        # Sum across output dimensions for importance
        importances = first_layer_weights.sum(axis=1)
        importances = importances / importances.sum()
        
        return importances
    
    def get_training_history(self) -> Optional[Dict[str, List[float]]]:
        """Get training history."""
        if self._history is None:
            return None
        return self._history.history


class MLPModel(NeuralModel):
    """Multi-Layer Perceptron (Feedforward Neural Network).
    
    Suitable for learning non-linear relationships between features
    and targets. Good baseline neural network for tabular data.
    
    Example:
        >>> mlp = MLPModel(
        ...     hidden_layers=[128, 64, 32],
        ...     dropout_rate=0.3,
        ...     epochs=50
        ... )
        >>> mlp.fit(X_train, y_returns=y_train)
        >>> predictions = mlp.predict(X_test)
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        prediction_type: str = "both",
        **kwargs
    ):
        """Initialize MLP model.
        
        Args:
            hidden_layers: List of hidden layer sizes (default: [64, 32])
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
            prediction_type: 'regression', 'classification', or 'both'
            **kwargs: Additional NeuralConfig parameters
        """
        config = NeuralConfig(
            hidden_layers=hidden_layers or [64, 32],
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )
        super().__init__(prediction_type=prediction_type, config=config)


class SequenceModel(NeuralModel):
    """Base class for sequence-based neural networks (LSTM, GRU).
    
    Handles conversion of tabular data to sequences for temporal modeling.
    """
    
    def __init__(
        self,
        sequence_length: int = 20,
        recurrent_units: List[int] = None,
        prediction_type: str = "both",
        config: Optional[NeuralConfig] = None,
        **kwargs
    ):
        """Initialize sequence model.
        
        Args:
            sequence_length: Number of time steps in each sequence
            recurrent_units: List of units for recurrent layers
            prediction_type: Type of prediction
            config: NeuralConfig object
            **kwargs: Additional config parameters
        """
        if config is None:
            config = NeuralConfig(
                sequence_length=sequence_length,
                recurrent_units=recurrent_units or [64, 32],
                **kwargs
            )
        super().__init__(prediction_type=prediction_type, config=config)
        
        self._sequence_cache = {}
    
    def _create_sequences(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert tabular data to sequences.
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Optional target array (n_samples,)
            
        Returns:
            X_seq: Sequence array (n_sequences, sequence_length, n_features)
            y_seq: Target array aligned with sequences (n_sequences,)
        """
        seq_len = self.config.sequence_length
        n_samples, n_features = X.shape
        
        if n_samples < seq_len:
            raise ValueError(
                f"Not enough samples ({n_samples}) for sequence length ({seq_len})"
            )
        
        n_sequences = n_samples - seq_len + 1
        
        # Create sequences using sliding window
        X_seq = np.zeros((n_sequences, seq_len, n_features), dtype=np.float32)
        for i in range(n_sequences):
            X_seq[i] = X[i:i + seq_len]
        
        if y is not None:
            # Target is aligned with the last element of each sequence
            y_seq = y[seq_len - 1:]
            return X_seq, y_seq
        
        return X_seq, None
    
    def _build_regressor_model(self, input_shape: Tuple[int, ...]) -> Any:
        """Build sequence regressor - override in subclasses."""
        raise NotImplementedError
    
    def _build_classifier_model(
        self, 
        input_shape: Tuple[int, ...], 
        n_classes: int
    ) -> Any:
        """Build sequence classifier - override in subclasses."""
        raise NotImplementedError
    
    def fit(
        self,
        X: pd.DataFrame,
        y_returns: Optional[pd.Series] = None,
        y_direction: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        validation_data: Optional[tuple] = None,
    ) -> "SequenceModel":
        """Fit the sequence model.
        
        Converts tabular data to sequences before training.
        """
        self._feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        
        # Preprocess features
        X_scaled = self._preprocess_features(X, fit=True)
        
        # Get callbacks
        use_validation = (
            validation_data is not None or 
            self.config.validation_split > 0
        )
        callbacks = self._get_callbacks(validation_data=use_validation)
        
        # Fit regressor
        if self.prediction_type in [PredictionType.REGRESSION, PredictionType.BOTH]:
            if y_returns is None:
                raise ValueError("y_returns required for regression")
            
            y_scaled = self._preprocess_target(y_returns, fit=True, is_classification=False)
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
            input_shape = (X_seq.shape[1], X_seq.shape[2])  # (seq_len, n_features)
            
            # Build model
            self.regressor = self._build_regressor_model(input_shape)
            
            # Handle sample weights for sequences
            seq_weights = None
            if sample_weight is not None:
                seq_weights = sample_weight[self.config.sequence_length - 1:]
            
            # Prepare validation data
            val_data = None
            if validation_data is not None:
                X_val_scaled = self._preprocess_features(validation_data[0], fit=False)
                y_val_scaled = self._preprocess_target(
                    validation_data[1], fit=False, is_classification=False
                )
                X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val_scaled)
                val_data = (X_val_seq, y_val_seq)
            
            # Train
            fit_kwargs = {
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'callbacks': callbacks,
                'verbose': 0,
            }
            
            if seq_weights is not None:
                fit_kwargs['sample_weight'] = seq_weights
            
            if val_data is not None:
                fit_kwargs['validation_data'] = val_data
            elif self.config.validation_split > 0:
                fit_kwargs['validation_split'] = self.config.validation_split
            
            self._history = self.regressor.fit(X_seq, y_seq, **fit_kwargs)
            
            logger.info(
                f"Fitted sequence regressor: epochs={len(self._history.history['loss'])}"
            )
        
        # Fit classifier
        if self.prediction_type in [PredictionType.CLASSIFICATION, PredictionType.BOTH]:
            if y_direction is None:
                raise ValueError("y_direction required for classification")
            
            y_processed = self._preprocess_target(
                y_direction, fit=True, is_classification=True
            )
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(X_scaled, y_processed)
            input_shape = (X_seq.shape[1], X_seq.shape[2])
            
            # Build model
            self.classifier = self._build_classifier_model(input_shape, self._n_classes)
            
            # Handle sample weights
            seq_weights = None
            if sample_weight is not None:
                seq_weights = sample_weight[self.config.sequence_length - 1:]
            
            # Prepare validation data
            val_data = None
            if validation_data is not None:
                X_val_scaled = self._preprocess_features(validation_data[0], fit=False)
                y_val_processed = self._preprocess_target(
                    validation_data[2] if len(validation_data) > 2 else validation_data[1],
                    fit=False, 
                    is_classification=True
                )
                X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val_processed)
                val_data = (X_val_seq, y_val_seq)
            
            # Train
            fit_kwargs = {
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'callbacks': callbacks,
                'verbose': 0,
            }
            
            if seq_weights is not None:
                fit_kwargs['sample_weight'] = seq_weights
            
            if val_data is not None:
                fit_kwargs['validation_data'] = val_data
            elif self.config.validation_split > 0:
                fit_kwargs['validation_split'] = self.config.validation_split
            
            history = self.classifier.fit(X_seq, y_seq, **fit_kwargs)
            
            logger.info(
                f"Fitted sequence classifier: epochs={len(history.history['loss'])}"
            )
        
        self._is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions on sequential data.
        
        Returns predictions only for indices where full sequence is available.
        First (sequence_length - 1) indices will have NaN.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self._preprocess_features(X, fit=False)
        X_seq, _ = self._create_sequences(X_scaled)
        
        predictions = {}
        n_pad = self.config.sequence_length - 1
        
        if self.regressor is not None:
            pred_scaled = self.regressor.predict(X_seq, verbose=0)
            pred = self._scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
            
            # Pad with NaN for first (seq_len - 1) indices
            full_pred = np.full(len(X), np.nan)
            full_pred[n_pad:] = pred.flatten()
            predictions['returns'] = full_pred
        
        if self.classifier is not None:
            pred_proba = self.classifier.predict(X_seq, verbose=0)
            
            if self._n_classes == 2:
                pred_class = (pred_proba.flatten() > 0.5).astype(int)
                pred_conf = pred_proba.flatten() * 2 - 1
            else:
                pred_class = np.argmax(pred_proba, axis=1) - 1
                pred_conf = np.max(pred_proba, axis=1) * 2 - 1
            
            # Pad
            full_dir = np.full(len(X), np.nan)
            full_conf = np.full(len(X), np.nan)
            full_dir[n_pad:] = pred_class
            full_conf[n_pad:] = pred_conf
            
            predictions['direction'] = full_dir
            predictions['direction_proba'] = full_conf
        
        return predictions


class LSTMModel(SequenceModel):
    """Long Short-Term Memory (LSTM) Neural Network.
    
    Captures long-range temporal dependencies in sequential data.
    Excellent for learning patterns across multiple time periods.
    
    Example:
        >>> lstm = LSTMModel(
        ...     sequence_length=30,
        ...     recurrent_units=[128, 64],
        ...     bidirectional=True
        ... )
        >>> lstm.fit(X_train, y_returns=y_train)
    """
    
    def __init__(
        self,
        sequence_length: int = 20,
        recurrent_units: List[int] = None,
        bidirectional: bool = False,
        recurrent_dropout: float = 0.1,
        prediction_type: str = "both",
        **kwargs
    ):
        """Initialize LSTM model.
        
        Args:
            sequence_length: Number of time steps in each sequence
            recurrent_units: List of LSTM units per layer
            bidirectional: Whether to use bidirectional LSTM
            recurrent_dropout: Dropout for recurrent connections
            prediction_type: Type of prediction
            **kwargs: Additional NeuralConfig parameters
        """
        config = NeuralConfig(
            sequence_length=sequence_length,
            recurrent_units=recurrent_units or [64, 32],
            bidirectional=bidirectional,
            recurrent_dropout=recurrent_dropout,
            **kwargs
        )
        super().__init__(
            sequence_length=sequence_length,
            recurrent_units=recurrent_units,
            prediction_type=prediction_type,
            config=config
        )
    
    def _build_recurrent_model(
        self,
        input_shape: Tuple[int, int],
        output_units: int,
        is_classifier: bool = False
    ) -> Any:
        """Build LSTM model architecture."""
        tf = self.tf
        
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        
        # LSTM layers
        recurrent_units = self.config.recurrent_units
        for i, units in enumerate(recurrent_units):
            return_sequences = (i < len(recurrent_units) - 1)
            
            lstm_layer = tf.keras.layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.recurrent_dropout,
                name=f"lstm_{i}"
            )
            
            if self.config.bidirectional:
                lstm_layer = tf.keras.layers.Bidirectional(
                    lstm_layer, name=f"bilstm_{i}"
                )
            
            x = lstm_layer(x)
            
            if self.config.batch_norm and return_sequences:
                x = tf.keras.layers.BatchNormalization(name=f"bn_{i}")(x)
        
        # Dense layers before output
        for i, units in enumerate(self.config.hidden_layers):
            x = tf.keras.layers.Dense(
                units, 
                activation=self.config.activation,
                name=f"dense_{i}"
            )(x)
            
            if self.config.dropout_rate > 0:
                x = tf.keras.layers.Dropout(
                    self.config.dropout_rate, 
                    name=f"dropout_{i}"
                )(x)
        
        # Output layer
        if is_classifier:
            if output_units == 2:
                outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
            else:
                outputs = tf.keras.layers.Dense(
                    output_units, activation='softmax', name='output'
                )(x)
        else:
            outputs = tf.keras.layers.Dense(
                output_units, 
                activation='linear',
                name='output'
            )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        optimizer = self._get_optimizer()
        
        if is_classifier:
            if output_units == 2:
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def _build_regressor_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build LSTM regressor."""
        return self._build_recurrent_model(input_shape, output_units=1, is_classifier=False)
    
    def _build_classifier_model(
        self, 
        input_shape: Tuple[int, int], 
        n_classes: int
    ) -> Any:
        """Build LSTM classifier."""
        return self._build_recurrent_model(input_shape, output_units=n_classes, is_classifier=True)


class GRUModel(SequenceModel):
    """Gated Recurrent Unit (GRU) Neural Network.
    
    Simpler alternative to LSTM with fewer parameters.
    Often performs comparably with faster training.
    
    Example:
        >>> gru = GRUModel(
        ...     sequence_length=20,
        ...     recurrent_units=[64, 32],
        ...     epochs=50
        ... )
        >>> gru.fit(X_train, y_returns=y_train)
    """
    
    def __init__(
        self,
        sequence_length: int = 20,
        recurrent_units: List[int] = None,
        bidirectional: bool = False,
        recurrent_dropout: float = 0.1,
        prediction_type: str = "both",
        **kwargs
    ):
        """Initialize GRU model.
        
        Args:
            sequence_length: Number of time steps in each sequence
            recurrent_units: List of GRU units per layer
            bidirectional: Whether to use bidirectional GRU
            recurrent_dropout: Dropout for recurrent connections
            prediction_type: Type of prediction
            **kwargs: Additional NeuralConfig parameters
        """
        config = NeuralConfig(
            sequence_length=sequence_length,
            recurrent_units=recurrent_units or [64, 32],
            bidirectional=bidirectional,
            recurrent_dropout=recurrent_dropout,
            **kwargs
        )
        super().__init__(
            sequence_length=sequence_length,
            recurrent_units=recurrent_units,
            prediction_type=prediction_type,
            config=config
        )
    
    def _build_recurrent_model(
        self,
        input_shape: Tuple[int, int],
        output_units: int,
        is_classifier: bool = False
    ) -> Any:
        """Build GRU model architecture."""
        tf = self.tf
        
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        
        # GRU layers
        recurrent_units = self.config.recurrent_units
        for i, units in enumerate(recurrent_units):
            return_sequences = (i < len(recurrent_units) - 1)
            
            gru_layer = tf.keras.layers.GRU(
                units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.recurrent_dropout,
                name=f"gru_{i}"
            )
            
            if self.config.bidirectional:
                gru_layer = tf.keras.layers.Bidirectional(
                    gru_layer, name=f"bigru_{i}"
                )
            
            x = gru_layer(x)
            
            if self.config.batch_norm and return_sequences:
                x = tf.keras.layers.BatchNormalization(name=f"bn_{i}")(x)
        
        # Dense layers
        for i, units in enumerate(self.config.hidden_layers):
            x = tf.keras.layers.Dense(
                units, 
                activation=self.config.activation,
                name=f"dense_{i}"
            )(x)
            
            if self.config.dropout_rate > 0:
                x = tf.keras.layers.Dropout(
                    self.config.dropout_rate, 
                    name=f"dropout_{i}"
                )(x)
        
        # Output layer
        if is_classifier:
            if output_units == 2:
                outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
            else:
                outputs = tf.keras.layers.Dense(
                    output_units, activation='softmax', name='output'
                )(x)
        else:
            outputs = tf.keras.layers.Dense(
                output_units, 
                activation='linear',
                name='output'
            )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        optimizer = self._get_optimizer()
        
        if is_classifier:
            if output_units == 2:
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def _build_regressor_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build GRU regressor."""
        return self._build_recurrent_model(input_shape, output_units=1, is_classifier=False)
    
    def _build_classifier_model(
        self, 
        input_shape: Tuple[int, int], 
        n_classes: int
    ) -> Any:
        """Build GRU classifier."""
        return self._build_recurrent_model(input_shape, output_units=n_classes, is_classifier=True)


class TransformerModel(NeuralModel):
    """Transformer-based Model for Sequential Trading Data.
    
    Uses self-attention mechanism to capture long-range dependencies.
    More powerful than LSTM/GRU for complex temporal patterns but
    requires more data and compute.
    
    Example:
        >>> transformer = TransformerModel(
        ...     sequence_length=50,
        ...     num_heads=4,
        ...     d_model=64,
        ...     num_layers=2
        ... )
        >>> transformer.fit(X_train, y_returns=y_train)
    
    Note: Optional - requires more data to train effectively.
    """
    
    def __init__(
        self,
        sequence_length: int = 30,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        prediction_type: str = "both",
        **kwargs
    ):
        """Initialize Transformer model.
        
        Args:
            sequence_length: Number of time steps
            d_model: Dimension of the model
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feedforward dimension
            prediction_type: Type of prediction
            **kwargs: Additional config parameters
        """
        config = NeuralConfig(
            sequence_length=sequence_length,
            hidden_layers=[ff_dim],
            **kwargs
        )
        super().__init__(prediction_type=prediction_type, config=config)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self._sequence_length = sequence_length
    
    def _create_sequences(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert tabular data to sequences."""
        seq_len = self._sequence_length
        n_samples, n_features = X.shape
        
        if n_samples < seq_len:
            raise ValueError(
                f"Not enough samples ({n_samples}) for sequence length ({seq_len})"
            )
        
        n_sequences = n_samples - seq_len + 1
        
        X_seq = np.zeros((n_sequences, seq_len, n_features), dtype=np.float32)
        for i in range(n_sequences):
            X_seq[i] = X[i:i + seq_len]
        
        if y is not None:
            y_seq = y[seq_len - 1:]
            return X_seq, y_seq
        
        return X_seq, None
    
    def _build_transformer_block(self, inputs: Any) -> Any:
        """Build a single transformer encoder block."""
        tf = self.tf
        
        # Multi-head attention
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.config.dropout_rate
        )(inputs, inputs)
        
        # Add & Norm
        x = tf.keras.layers.Add()([inputs, attn_output])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Feedforward
        ff = tf.keras.layers.Dense(self.ff_dim, activation='relu')(x)
        ff = tf.keras.layers.Dense(self.d_model)(ff)
        ff = tf.keras.layers.Dropout(self.config.dropout_rate)(ff)
        
        # Add & Norm
        x = tf.keras.layers.Add()([x, ff])
        x = tf.keras.layers.LayerNormalization()(x)
        
        return x
    
    def _build_transformer_model(
        self,
        input_shape: Tuple[int, int],
        output_units: int,
        is_classifier: bool = False
    ) -> Any:
        """Build Transformer model architecture."""
        tf = self.tf
        
        inputs = tf.keras.Input(shape=input_shape)
        
        # Project to d_model dimensions
        x = tf.keras.layers.Dense(self.d_model)(inputs)
        
        # Positional encoding (learned)
        positions = tf.keras.layers.Embedding(
            input_dim=input_shape[0],
            output_dim=self.d_model
        )(tf.range(input_shape[0]))
        x = x + positions
        
        # Transformer blocks
        for _ in range(self.num_layers):
            x = self._build_transformer_block(x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        for units in self.config.hidden_layers:
            x = tf.keras.layers.Dense(units, activation='relu')(x)
            x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        
        # Output
        if is_classifier:
            if output_units == 2:
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            else:
                outputs = tf.keras.layers.Dense(output_units, activation='softmax')(x)
        else:
            outputs = tf.keras.layers.Dense(output_units, activation='linear')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        optimizer = self._get_optimizer()
        
        if is_classifier:
            if output_units == 2:
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def _build_regressor_model(self, input_shape: Tuple[int, ...]) -> Any:
        """Build transformer regressor."""
        return self._build_transformer_model(input_shape, output_units=1, is_classifier=False)
    
    def _build_classifier_model(
        self, 
        input_shape: Tuple[int, ...], 
        n_classes: int
    ) -> Any:
        """Build transformer classifier."""
        return self._build_transformer_model(
            input_shape, output_units=n_classes, is_classifier=True
        )
    
    def fit(
        self,
        X: pd.DataFrame,
        y_returns: Optional[pd.Series] = None,
        y_direction: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        validation_data: Optional[tuple] = None,
    ) -> "TransformerModel":
        """Fit the transformer model with sequence handling."""
        self._feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        
        # Preprocess features
        X_scaled = self._preprocess_features(X, fit=True)
        
        # Get callbacks
        use_validation = (
            validation_data is not None or 
            self.config.validation_split > 0
        )
        callbacks = self._get_callbacks(validation_data=use_validation)
        
        # Fit regressor
        if self.prediction_type in [PredictionType.REGRESSION, PredictionType.BOTH]:
            if y_returns is None:
                raise ValueError("y_returns required for regression")
            
            y_scaled = self._preprocess_target(y_returns, fit=True, is_classification=False)
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
            input_shape = (X_seq.shape[1], X_seq.shape[2])
            
            # Build model
            self.regressor = self._build_regressor_model(input_shape)
            
            # Train
            fit_kwargs = {
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'callbacks': callbacks,
                'verbose': 0,
            }
            
            if self.config.validation_split > 0:
                fit_kwargs['validation_split'] = self.config.validation_split
            
            self._history = self.regressor.fit(X_seq, y_seq, **fit_kwargs)
            
            logger.info(
                f"Fitted transformer: epochs={len(self._history.history['loss'])}"
            )
        
        # Fit classifier
        if self.prediction_type in [PredictionType.CLASSIFICATION, PredictionType.BOTH]:
            if y_direction is None:
                raise ValueError("y_direction required for classification")
            
            y_processed = self._preprocess_target(
                y_direction, fit=True, is_classification=True
            )
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(X_scaled, y_processed)
            input_shape = (X_seq.shape[1], X_seq.shape[2])
            
            # Build model
            self.classifier = self._build_classifier_model(input_shape, self._n_classes)
            
            # Train
            fit_kwargs = {
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'callbacks': callbacks,
                'verbose': 0,
            }
            
            if self.config.validation_split > 0:
                fit_kwargs['validation_split'] = self.config.validation_split
            
            history = self.classifier.fit(X_seq, y_seq, **fit_kwargs)
            
            logger.info(
                f"Fitted transformer classifier: epochs={len(history.history['loss'])}"
            )
        
        self._is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions on sequential data."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self._preprocess_features(X, fit=False)
        X_seq, _ = self._create_sequences(X_scaled)
        
        predictions = {}
        n_pad = self._sequence_length - 1
        
        if self.regressor is not None:
            pred_scaled = self.regressor.predict(X_seq, verbose=0)
            pred = self._scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
            
            full_pred = np.full(len(X), np.nan)
            full_pred[n_pad:] = pred.flatten()
            predictions['returns'] = full_pred
        
        if self.classifier is not None:
            pred_proba = self.classifier.predict(X_seq, verbose=0)
            
            if self._n_classes == 2:
                pred_class = (pred_proba.flatten() > 0.5).astype(int)
                pred_conf = pred_proba.flatten() * 2 - 1
            else:
                pred_class = np.argmax(pred_proba, axis=1) - 1
                pred_conf = np.max(pred_proba, axis=1) * 2 - 1
            
            full_dir = np.full(len(X), np.nan)
            full_conf = np.full(len(X), np.nan)
            full_dir[n_pad:] = pred_class
            full_conf[n_pad:] = pred_conf
            
            predictions['direction'] = full_dir
            predictions['direction_proba'] = full_conf
        
        return predictions

"""Walk-forward validation for ML strategies.

Provides time-series cross-validation to avoid lookahead bias.
Computes efficiency ratio to detect overfitting.
"""

from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import numpy as np

from jsf.data import PriceData, load_data
from jsf.simulation import BacktestEngine, BacktestConfig, calculate_all_metrics
from jsf.utils.logging import get_logger

from .features import FeatureExtractor, create_feature_extractor
from .models import MLModel, EnsembleModel
from .strategy import MLStrategy, MLStrategyConfig
from .preprocessing import (
    create_target_variable,
    prepare_ml_data,
    split_train_test,
    MLDataset,
    TargetType,
)

logger = get_logger(__name__)


@dataclass
class MLValidationResult:
    """Results from ML walk-forward validation."""
    
    # Overall metrics
    n_windows: int = 0
    avg_is_return: float = 0.0
    avg_is_sharpe: float = 0.0
    avg_oos_return: float = 0.0
    avg_oos_sharpe: float = 0.0
    efficiency_ratio: float = 0.0
    
    # Per-window results
    window_results: List[Dict] = field(default_factory=list)
    
    # Model metrics
    avg_train_accuracy: float = 0.0
    avg_test_accuracy: float = 0.0
    avg_train_correlation: float = 0.0
    avg_test_correlation: float = 0.0
    
    # Feature importance
    feature_importance: Optional[pd.Series] = None
    
    # Stability metrics
    parameter_stability: float = 0.0
    weight_stability: float = 0.0  # For ensemble models
    
    # Overfitting detection
    is_overfitted: bool = False
    overfitting_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'n_windows': self.n_windows,
            'avg_is_return': self.avg_is_return,
            'avg_is_sharpe': self.avg_is_sharpe,
            'avg_oos_return': self.avg_oos_return,
            'avg_oos_sharpe': self.avg_oos_sharpe,
            'efficiency_ratio': self.efficiency_ratio,
            'avg_train_accuracy': self.avg_train_accuracy,
            'avg_test_accuracy': self.avg_test_accuracy,
            'is_overfitted': self.is_overfitted,
            'overfitting_score': self.overfitting_score,
        }
    
    def __repr__(self) -> str:
        return (
            f"MLValidationResult(\n"
            f"  windows={self.n_windows}, "
            f"OOS Sharpe={self.avg_oos_sharpe:.3f}, "
            f"efficiency={self.efficiency_ratio:.2%}\n"
            f"  overfitted={self.is_overfitted}, "
            f"score={self.overfitting_score:.2f}\n"
            f")"
        )


class WalkForwardMLValidator:
    """Walk-forward validation for ML trading strategies.
    
    Implements expanding or rolling window validation:
    - Train on in-sample period
    - Test on out-of-sample period
    - Move forward and repeat
    
    This ensures:
    - No lookahead bias (only uses past data for training)
    - Realistic performance estimation
    - Detection of overfitting via efficiency ratio
    
    Example:
        >>> validator = WalkForwardMLValidator(
        ...     is_days=252,  # 1 year training
        ...     oos_days=63,  # 3 months testing
        ...     window_type='rolling'
        ... )
        >>> result = validator.validate(strategy, price_data)
        >>> print(f"OOS Sharpe: {result.avg_oos_sharpe:.2f}")
        >>> print(f"Overfitted: {result.is_overfitted}")
    """
    
    def __init__(
        self,
        is_days: int = 252,
        oos_days: int = 63,
        window_type: str = "rolling",
        min_is_samples: int = 200,
        backtest_config: Optional[BacktestConfig] = None,
        efficiency_threshold: float = 0.5,
        verbose: bool = True,
    ):
        """Initialize validator.
        
        Args:
            is_days: In-sample (training) period in trading days
            oos_days: Out-of-sample (test) period in trading days
            window_type: 'rolling' or 'expanding'
            min_is_samples: Minimum samples required for training
            backtest_config: Configuration for backtesting
            efficiency_threshold: Threshold for overfitting detection
            verbose: Whether to print progress
        """
        self.is_days = is_days
        self.oos_days = oos_days
        self.window_type = window_type
        self.min_is_samples = min_is_samples
        self.backtest_config = backtest_config or BacktestConfig(
            initial_capital=100000,
            transaction_cost=0.001,
            slippage=0.0005,
        )
        self.efficiency_threshold = efficiency_threshold
        self.verbose = verbose
    
    def validate(
        self,
        strategy: MLStrategy,
        price_data: PriceData,
        metric: str = "sharpe_ratio",
    ) -> MLValidationResult:
        """Run walk-forward validation.
        
        Args:
            strategy: MLStrategy to validate
            price_data: Price data for backtesting
            metric: Primary metric for evaluation
            
        Returns:
            MLValidationResult with validation metrics
        """
        # Get dates
        dates = price_data.dates
        n_dates = len(dates)
        
        # Calculate number of windows
        total_period = self.is_days + self.oos_days
        n_windows = (n_dates - self.is_days) // self.oos_days
        
        if n_windows < 1:
            raise ValueError(
                f"Not enough data for walk-forward: need {total_period} days, "
                f"have {n_dates}"
            )
        
        logger.info(
            f"Running walk-forward validation: "
            f"{n_windows} windows, IS={self.is_days}d, OOS={self.oos_days}d"
        )
        
        window_results = []
        is_sharpes = []
        oos_sharpes = []
        is_returns = []
        oos_returns = []
        train_accuracies = []
        test_accuracies = []
        train_correlations = []
        test_correlations = []
        ensemble_weights_history = []
        
        for window_idx in range(n_windows):
            if self.verbose:
                print(f"  Window {window_idx + 1}/{n_windows}...", end=" ")
            
            try:
                window_result = self._run_window(
                    strategy=strategy,
                    price_data=price_data,
                    window_idx=window_idx,
                    dates=dates,
                )
                
                window_results.append(window_result)
                
                is_sharpes.append(window_result.get('is_sharpe', 0))
                oos_sharpes.append(window_result.get('oos_sharpe', 0))
                is_returns.append(window_result.get('is_return', 0))
                oos_returns.append(window_result.get('oos_return', 0))
                
                if 'train_accuracy' in window_result:
                    train_accuracies.append(window_result['train_accuracy'])
                if 'test_accuracy' in window_result:
                    test_accuracies.append(window_result['test_accuracy'])
                if 'train_correlation' in window_result:
                    train_correlations.append(window_result['train_correlation'])
                if 'test_correlation' in window_result:
                    test_correlations.append(window_result['test_correlation'])
                
                if 'ensemble_weights' in window_result:
                    ensemble_weights_history.append(window_result['ensemble_weights'])
                
                if self.verbose:
                    print(
                        f"IS Sharpe={window_result.get('is_sharpe', 0):.2f}, "
                        f"OOS Sharpe={window_result.get('oos_sharpe', 0):.2f}"
                    )
                    
            except Exception as e:
                logger.warning(f"Window {window_idx} failed: {e}")
                if self.verbose:
                    print(f"FAILED: {e}")
        
        if not window_results:
            raise ValueError("All validation windows failed")
        
        # Calculate aggregate metrics
        avg_is_sharpe = np.mean(is_sharpes) if is_sharpes else 0
        avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0
        avg_is_return = np.mean(is_returns) if is_returns else 0
        avg_oos_return = np.mean(oos_returns) if oos_returns else 0
        
        # Efficiency ratio (OOS/IS Sharpe)
        if avg_is_sharpe > 0:
            efficiency_ratio = avg_oos_sharpe / avg_is_sharpe
        else:
            efficiency_ratio = 0
        
        # Overfitting detection
        is_overfitted = efficiency_ratio < self.efficiency_threshold
        overfitting_score = 1 - efficiency_ratio if efficiency_ratio < 1 else 0
        
        # Weight stability (for ensemble models)
        weight_stability = 0.0
        if ensemble_weights_history and len(ensemble_weights_history) > 1:
            # Calculate consistency of weights across windows
            weight_df = pd.DataFrame(ensemble_weights_history)
            weight_stability = 1 - weight_df.std().mean()
        
        # Feature importance (from last window)
        feature_importance = None
        if hasattr(strategy, 'get_feature_importance') and strategy._is_trained:
            try:
                feature_importance = strategy.get_feature_importance()
            except Exception:
                pass
        
        result = MLValidationResult(
            n_windows=len(window_results),
            avg_is_return=avg_is_return,
            avg_is_sharpe=avg_is_sharpe,
            avg_oos_return=avg_oos_return,
            avg_oos_sharpe=avg_oos_sharpe,
            efficiency_ratio=efficiency_ratio,
            window_results=window_results,
            avg_train_accuracy=np.mean(train_accuracies) if train_accuracies else 0,
            avg_test_accuracy=np.mean(test_accuracies) if test_accuracies else 0,
            avg_train_correlation=np.mean(train_correlations) if train_correlations else 0,
            avg_test_correlation=np.mean(test_correlations) if test_correlations else 0,
            feature_importance=feature_importance,
            parameter_stability=weight_stability,
            weight_stability=weight_stability,
            is_overfitted=is_overfitted,
            overfitting_score=overfitting_score,
        )
        
        logger.info(
            f"Walk-forward complete: "
            f"OOS Sharpe={avg_oos_sharpe:.2f}, "
            f"Efficiency={efficiency_ratio:.2%}, "
            f"Overfitted={is_overfitted}"
        )
        
        return result
    
    def _run_window(
        self,
        strategy: MLStrategy,
        price_data: PriceData,
        window_idx: int,
        dates: pd.DatetimeIndex,
    ) -> Dict:
        """Run a single walk-forward window."""
        # Calculate window boundaries
        if self.window_type == "rolling":
            is_start_idx = window_idx * self.oos_days
        else:  # expanding
            is_start_idx = 0
        
        is_end_idx = window_idx * self.oos_days + self.is_days
        oos_start_idx = is_end_idx
        oos_end_idx = min(oos_start_idx + self.oos_days, len(dates))
        
        is_start = dates[is_start_idx]
        is_end = dates[is_end_idx - 1]
        oos_start = dates[oos_start_idx]
        oos_end = dates[oos_end_idx - 1]
        
        # Filter data for in-sample period
        is_data = price_data.filter_date_range(
            start_date=str(is_start.date()),
            end_date=str(is_end.date()),
        )
        
        # Filter data for out-of-sample period
        oos_data = price_data.filter_date_range(
            start_date=str(oos_start.date()),
            end_date=str(oos_end.date()),
        )
        
        # Create fresh strategy instance for this window
        window_strategy = self._clone_strategy(strategy)
        
        # Run backtest on in-sample
        engine = BacktestEngine(self.backtest_config)
        
        try:
            is_result = engine.run_strategy(window_strategy, is_data)
            is_sharpe = is_result.sharpe_ratio if is_result.sharpe_ratio is not None else 0
            is_return = is_result.total_return if is_result.total_return is not None else 0
        except Exception as e:
            logger.warning(f"IS backtest failed: {e}")
            is_sharpe = 0
            is_return = 0
        
        # Run backtest on out-of-sample (model already trained from IS)
        try:
            oos_result = engine.run_strategy(window_strategy, oos_data)
            oos_sharpe = oos_result.sharpe_ratio if oos_result.sharpe_ratio is not None else 0
            oos_return = oos_result.total_return if oos_result.total_return is not None else 0
        except Exception as e:
            logger.warning(f"OOS backtest failed: {e}")
            oos_sharpe = 0
            oos_return = 0
        
        result = {
            'window_idx': window_idx,
            'is_start': str(is_start.date()),
            'is_end': str(is_end.date()),
            'oos_start': str(oos_start.date()),
            'oos_end': str(oos_end.date()),
            'is_sharpe': is_sharpe,
            'is_return': is_return,
            'oos_sharpe': oos_sharpe,
            'oos_return': oos_return,
        }
        
        # Get ensemble weights if available
        if hasattr(window_strategy, 'get_model_weights'):
            weights = window_strategy.get_model_weights()
            if weights:
                result['ensemble_weights'] = weights
        
        return result
    
    def _clone_strategy(self, strategy: MLStrategy) -> MLStrategy:
        """Create a fresh copy of the strategy."""
        # Create new strategy with same config
        new_strategy = MLStrategy(
            name=strategy.name,
            model=self._clone_model(strategy.model),
            feature_extractor=strategy.feature_extractor,
            portfolio_constructor=strategy.portfolio_constructor,
            config=strategy.config,
        )
        return new_strategy
    
    def _clone_model(self, model: MLModel) -> MLModel:
        """Create a fresh copy of the model."""
        if isinstance(model, EnsembleModel):
            return EnsembleModel(
                models=model.model_names,
                weights=model.weights.copy(),
                prediction_type=model.prediction_type.value,
                **model.model_kwargs
            )
        else:
            return model.__class__(
                prediction_type=model.prediction_type.value,
                config=model.config,
            )


def validate_ml_strategy(
    strategy: MLStrategy,
    price_data: PriceData,
    is_days: int = 252,
    oos_days: int = 63,
    window_type: str = "rolling",
    metric: str = "sharpe_ratio",
    verbose: bool = True,
) -> MLValidationResult:
    """Convenience function for walk-forward validation.
    
    Args:
        strategy: MLStrategy to validate
        price_data: Price data for backtesting
        is_days: In-sample period (trading days)
        oos_days: Out-of-sample period (trading days)
        window_type: 'rolling' or 'expanding'
        metric: Metric for evaluation
        verbose: Whether to print progress
        
    Returns:
        MLValidationResult
        
    Example:
        >>> from jsf.ml import MLStrategy, validate_ml_strategy
        >>> 
        >>> strategy = MLStrategy(name='ml_test')
        >>> result = validate_ml_strategy(strategy, price_data)
        >>> 
        >>> print(f"OOS Sharpe: {result.avg_oos_sharpe:.2f}")
        >>> print(f"Overfitted: {result.is_overfitted}")
    """
    validator = WalkForwardMLValidator(
        is_days=is_days,
        oos_days=oos_days,
        window_type=window_type,
        verbose=verbose,
    )
    
    return validator.validate(strategy, price_data, metric)


def analyze_model_stability(
    strategy: MLStrategy,
    price_data: PriceData,
    n_runs: int = 5,
    sample_frac: float = 0.8,
) -> Dict[str, Any]:
    """Analyze model stability across multiple runs.
    
    Trains the model multiple times with different random samples
    to assess stability of predictions and feature importance.
    
    Args:
        strategy: MLStrategy to analyze
        price_data: Training data
        n_runs: Number of random runs
        sample_frac: Fraction of data to use per run
        
    Returns:
        Dict with stability metrics
    """
    feature_importances = []
    predictions_corrs = []
    ensemble_weights = []
    
    # Extract features once
    features = strategy.feature_extractor.extract(price_data)
    y_returns, y_direction = create_target_variable(price_data)
    
    # Stack for ML format
    y_ret_stacked = y_returns.stack() if y_returns is not None else None
    y_dir_stacked = y_direction.stack() if y_direction is not None else None
    
    # Align indices
    if y_ret_stacked is not None:
        common_idx = features.index.intersection(y_ret_stacked.index)
        features_aligned = features.loc[common_idx]
        y_ret_aligned = y_ret_stacked.loc[common_idx]
        y_dir_aligned = y_dir_stacked.loc[common_idx] if y_dir_stacked is not None else None
    else:
        features_aligned = features
        y_ret_aligned = None
        y_dir_aligned = y_dir_stacked
    
    # Drop NaN
    valid_mask = ~features_aligned.isna().any(axis=1)
    if y_ret_aligned is not None:
        valid_mask &= ~y_ret_aligned.isna()
    
    X = features_aligned[valid_mask]
    y_ret = y_ret_aligned[valid_mask] if y_ret_aligned is not None else None
    y_dir = y_dir_aligned[valid_mask] if y_dir_aligned is not None else None
    
    base_predictions = None
    
    for run in range(n_runs):
        # Random sample
        sample_idx = np.random.choice(
            len(X),
            size=int(len(X) * sample_frac),
            replace=False
        )
        
        X_sample = X.iloc[sample_idx]
        y_ret_sample = y_ret.iloc[sample_idx] if y_ret is not None else None
        y_dir_sample = y_dir.iloc[sample_idx] if y_dir is not None else None
        
        # Clone and train
        model_copy = strategy.model.__class__(
            prediction_type=strategy.model.prediction_type.value,
            config=strategy.model.config,
        )
        
        model_copy.fit(
            X=X_sample,
            y_returns=y_ret_sample,
            y_direction=y_dir_sample,
        )
        
        # Get predictions on full data
        preds = model_copy.predict(X)
        
        if 'returns' in preds:
            if base_predictions is None:
                base_predictions = preds['returns']
            else:
                corr = np.corrcoef(base_predictions, preds['returns'])[0, 1]
                predictions_corrs.append(corr)
        
        # Feature importance
        try:
            imp = model_copy.feature_importances_
            feature_importances.append(imp)
        except Exception:
            pass
        
        # Ensemble weights
        if hasattr(model_copy, 'get_model_weights'):
            weights = model_copy.get_model_weights()
            if weights:
                ensemble_weights.append(weights)
    
    # Calculate stability metrics
    results = {
        'n_runs': n_runs,
        'sample_frac': sample_frac,
    }
    
    if predictions_corrs:
        results['prediction_stability'] = np.mean(predictions_corrs)
        results['prediction_std'] = np.std(predictions_corrs)
    
    if feature_importances:
        imp_df = pd.DataFrame(feature_importances, columns=X.columns)
        results['feature_importance_mean'] = imp_df.mean()
        results['feature_importance_std'] = imp_df.std()
        results['top_stable_features'] = (
            imp_df.mean().sort_values(ascending=False).head(10)
        )
    
    if ensemble_weights:
        weight_df = pd.DataFrame(ensemble_weights)
        results['weight_stability'] = 1 - weight_df.std().mean()
        results['weight_mean'] = weight_df.mean().to_dict()
    
    return results

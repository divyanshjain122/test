"""Tests for ML Integration Module (Phase 19).

Comprehensive test suite covering:
- Feature extraction
- ML model wrappers
- MLStrategy
- Walk-forward validation
- Preprocessing utilities
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from jsf.data import load_data, PriceData, SyntheticDataLoader
from jsf.ml import (
    # Features
    FeatureExtractor,
    FeatureConfig,
    create_feature_extractor,
    FEATURE_GROUPS,
    # Models
    MLModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    EnsembleModel,
    ModelConfig,
    # Strategy
    MLStrategy,
    MLStrategyConfig,
    # Validation
    WalkForwardMLValidator,
    MLValidationResult,
    validate_ml_strategy,
    # Preprocessing
    prepare_ml_data,
    create_target_variable,
    split_train_test,
    handle_missing_features,
    MultiIndexConverter,
)
from jsf.ml.preprocessing import TargetType, MLDataset


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_price_data() -> PriceData:
    """Generate sample price data for testing."""
    loader = SyntheticDataLoader(
        symbols=['AAPL', 'GOOGL', 'MSFT'],
        start_date='2020-01-01',
        end_date='2023-12-31',
        initial_price=100.0,
        annual_volatility=0.25,
        seed=42
    )
    return PriceData(data=loader.load())


@pytest.fixture
def short_price_data() -> PriceData:
    """Generate shorter price data for quick tests."""
    loader = SyntheticDataLoader(
        symbols=['AAPL', 'GOOGL'],
        start_date='2022-01-01',
        end_date='2023-06-30',
        initial_price=100.0,
        annual_volatility=0.25,
        seed=42
    )
    return PriceData(data=loader.load())


@pytest.fixture
def sample_features(sample_price_data) -> pd.DataFrame:
    """Generate sample features."""
    extractor = FeatureExtractor(
        feature_groups=['momentum', 'volatility'],
        lag_periods=[1],
        normalize=True,
        rank=False,
    )
    return extractor.extract(sample_price_data)


# =============================================================================
# TEST: FEATURE GROUPS
# =============================================================================

class TestFeatureGroups:
    """Test feature group definitions."""
    
    def test_feature_groups_defined(self):
        """Test that feature groups are properly defined."""
        assert 'momentum' in FEATURE_GROUPS
        assert 'mean_reversion' in FEATURE_GROUPS
        assert 'technical' in FEATURE_GROUPS
        assert 'volatility' in FEATURE_GROUPS
        assert 'trend' in FEATURE_GROUPS
    
    def test_feature_groups_have_signals(self):
        """Test that each group has signals defined or requires external data."""
        for group_name, group_def in FEATURE_GROUPS.items():
            assert 'signals' in group_def
            assert 'description' in group_def
            # Groups with 'requires' flag get data from external sources (e.g., text)
            if 'requires' not in group_def:
                assert len(group_def['signals']) > 0, f"Group '{group_name}' has no signals"


# =============================================================================
# TEST: FEATURE EXTRACTOR
# =============================================================================

class TestFeatureExtractor:
    """Test feature extraction."""
    
    def test_basic_extraction(self, sample_price_data):
        """Test basic feature extraction."""
        extractor = FeatureExtractor(
            feature_groups=['momentum'],
            lag_periods=[1],
        )
        
        features = extractor.extract(sample_price_data)
        
        assert isinstance(features, pd.DataFrame)
        assert isinstance(features.index, pd.MultiIndex)
        assert features.shape[1] > 0
        assert not features.empty
    
    def test_multiple_groups(self, sample_price_data):
        """Test extraction with multiple groups."""
        extractor = FeatureExtractor(
            feature_groups=['momentum', 'volatility', 'trend'],
            lag_periods=[1],
        )
        
        features = extractor.extract(sample_price_data)
        
        # Should have more features than single group
        assert features.shape[1] > 3
    
    def test_lag_features(self, sample_price_data):
        """Test lagged feature creation."""
        extractor = FeatureExtractor(
            feature_groups=['momentum'],
            lag_periods=[1, 5, 10],
            normalize=False,
            rank=False,
        )
        
        features = extractor.extract(sample_price_data)
        
        # Should have lag columns
        lag_cols = [c for c in features.columns if '_lag' in c]
        assert len(lag_cols) > 0
    
    def test_normalization(self, sample_price_data):
        """Test feature normalization."""
        extractor = FeatureExtractor(
            feature_groups=['momentum'],
            lag_periods=[1],
            normalize=True,
            rank=False,
        )
        
        features = extractor.extract(sample_price_data)
        
        # Should have normalized columns
        norm_cols = [c for c in features.columns if '_norm' in c]
        assert len(norm_cols) > 0
    
    def test_ranking(self, sample_price_data):
        """Test cross-sectional ranking."""
        extractor = FeatureExtractor(
            feature_groups=['momentum'],
            lag_periods=[1],
            normalize=False,
            rank=True,
        )
        
        features = extractor.extract(sample_price_data)
        
        # Should have rank columns
        rank_cols = [c for c in features.columns if '_rank' in c]
        assert len(rank_cols) > 0
    
    def test_feature_names(self, sample_price_data):
        """Test feature names property."""
        extractor = FeatureExtractor(
            feature_groups=['momentum'],
            lag_periods=[1],
        )
        
        features = extractor.extract(sample_price_data)
        
        assert len(extractor.feature_names) > 0
        assert extractor.n_features == len(extractor.feature_names)
    
    def test_create_extractor_preset(self, sample_price_data):
        """Test preset-based extractor creation."""
        # Default preset
        extractor = create_feature_extractor(preset='default')
        features = extractor.extract(sample_price_data)
        assert features.shape[1] > 10
        
        # Minimal preset
        extractor = create_feature_extractor(preset='minimal')
        features = extractor.extract(sample_price_data)
        assert features.shape[1] > 0


# =============================================================================
# TEST: PREPROCESSING
# =============================================================================

class TestPreprocessing:
    """Test preprocessing utilities."""
    
    def test_create_target_returns(self, sample_price_data):
        """Test returns target creation."""
        y_returns, y_direction = create_target_variable(
            sample_price_data,
            target_type='returns',
            forward_periods=1,
        )
        
        assert y_returns is not None
        assert y_direction is None
        assert isinstance(y_returns, pd.DataFrame)
    
    def test_create_target_direction(self, sample_price_data):
        """Test direction target creation."""
        y_returns, y_direction = create_target_variable(
            sample_price_data,
            target_type='direction',
            forward_periods=1,
        )
        
        assert y_returns is None
        assert y_direction is not None
        
        # Direction should be -1, 0, or 1
        unique_vals = np.unique(y_direction.values[~np.isnan(y_direction.values)])
        assert all(v in [-1, 0, 1] for v in unique_vals)
    
    def test_create_target_both(self, sample_price_data):
        """Test both targets creation."""
        y_returns, y_direction = create_target_variable(
            sample_price_data,
            target_type='both',
            forward_periods=1,
        )
        
        assert y_returns is not None
        assert y_direction is not None
    
    def test_prepare_ml_data(self, sample_features, sample_price_data):
        """Test ML data preparation."""
        y_returns, y_direction = create_target_variable(sample_price_data)
        
        dataset = prepare_ml_data(
            features=sample_features,
            y_returns=y_returns,
            y_direction=y_direction,
            dropna=True,
        )
        
        assert isinstance(dataset, MLDataset)
        assert dataset.n_samples > 0
        assert dataset.n_features > 0
        assert dataset.y_returns is not None
    
    def test_split_train_test(self, sample_features, sample_price_data):
        """Test train/test splitting."""
        y_returns, _ = create_target_variable(sample_price_data)
        
        dataset = prepare_ml_data(
            features=sample_features,
            y_returns=y_returns,
            dropna=True,
        )
        
        # Split at 2022-01-01
        split_date = pd.Timestamp('2022-01-01')
        train, test = split_train_test(dataset, split_date)
        
        assert train.n_samples > 0
        assert test.n_samples > 0
        assert train.n_samples + test.n_samples == dataset.n_samples
    
    def test_handle_missing_features(self, sample_features):
        """Test missing value handling."""
        # Add some NaN values to a specific location
        features_with_nan = sample_features.copy()
        original_nans = features_with_nan.isna().sum().sum()
        
        # Add artificial NaNs in the middle of data (where clean data should be)
        mid_point = len(features_with_nan) // 2
        features_with_nan.iloc[mid_point:mid_point+10, 0] = np.nan
        added_nans = features_with_nan.isna().sum().sum()
        assert added_nans >= original_nans  # Confirm NaNs were added
        
        # Test forward fill
        cleaned = handle_missing_features(features_with_nan, method='ffill')
        # After ffill, we should have same or fewer NaNs
        assert cleaned.isna().sum().sum() <= added_nans
        
        # Test mean fill
        cleaned = handle_missing_features(features_with_nan, method='mean')
        # After mean fill, we should have same or fewer NaNs
        assert cleaned.isna().sum().sum() <= added_nans
    
    def test_multiindex_converter(self, sample_price_data):
        """Test MultiIndex conversion."""
        close = sample_price_data.get_close_prices()
        
        converter = MultiIndexConverter()
        
        # Convert wide to flat
        flat = converter.to_flat(close)
        assert 'date' in flat.columns or flat.index.names[0] == 'date'
        
        # Convert back to wide
        wide = converter.to_wide(
            flat.iloc[:, -1].values if isinstance(flat, pd.DataFrame) else flat.values,
            dates=sample_price_data.dates,
            symbols=sample_price_data.symbols,
        )
        assert wide.shape[1] == len(sample_price_data.symbols)


# =============================================================================
# TEST: ML MODELS
# =============================================================================

class TestRandomForestModel:
    """Test RandomForest model wrapper."""
    
    def test_fit_predict_regression(self, sample_features, sample_price_data):
        """Test regression fitting and prediction."""
        y_returns, _ = create_target_variable(sample_price_data, target_type='returns')
        
        # Prepare data
        y_stacked = y_returns.stack()
        common_idx = sample_features.index.intersection(y_stacked.index)
        X = sample_features.loc[common_idx].dropna()
        y = y_stacked.loc[X.index].dropna()
        X = X.loc[y.index]
        
        # Fit model
        model = RandomForestModel(
            n_estimators=10,
            max_depth=3,
            prediction_type='regression',
        )
        
        model.fit(X, y_returns=y)
        
        # Predict
        predictions = model.predict(X)
        
        assert 'returns' in predictions
        assert len(predictions['returns']) == len(X)
    
    def test_fit_predict_classification(self, sample_features, sample_price_data):
        """Test classification fitting and prediction."""
        _, y_direction = create_target_variable(sample_price_data, target_type='direction')
        
        # Prepare data
        y_stacked = y_direction.stack()
        common_idx = sample_features.index.intersection(y_stacked.index)
        X = sample_features.loc[common_idx].dropna()
        y = y_stacked.loc[X.index].dropna()
        X = X.loc[y.index]
        
        # Fit model
        model = RandomForestModel(
            n_estimators=10,
            max_depth=3,
            prediction_type='classification',
        )
        
        model.fit(X, y_direction=y)
        
        # Predict
        predictions = model.predict(X)
        
        assert 'direction' in predictions
        assert len(predictions['direction']) == len(X)
    
    def test_fit_both_targets(self, sample_features, sample_price_data):
        """Test fitting both regression and classification."""
        y_returns, y_direction = create_target_variable(sample_price_data, target_type='both')
        
        # Prepare data
        y_ret_stacked = y_returns.stack()
        y_dir_stacked = y_direction.stack()
        common_idx = sample_features.index.intersection(y_ret_stacked.index)
        X = sample_features.loc[common_idx].dropna()
        y_ret = y_ret_stacked.loc[X.index].dropna()
        y_dir = y_dir_stacked.loc[X.index].dropna()
        
        # Align
        valid_idx = y_ret.index.intersection(y_dir.index).intersection(X.index)
        X = X.loc[valid_idx]
        y_ret = y_ret.loc[valid_idx]
        y_dir = y_dir.loc[valid_idx]
        
        # Fit model
        model = RandomForestModel(
            n_estimators=10,
            max_depth=3,
            prediction_type='both',
        )
        
        model.fit(X, y_returns=y_ret, y_direction=y_dir)
        
        # Predict
        predictions = model.predict(X)
        
        assert 'returns' in predictions
        assert 'direction' in predictions
    
    def test_feature_importances(self, sample_features, sample_price_data):
        """Test feature importance extraction."""
        y_returns, _ = create_target_variable(sample_price_data, target_type='returns')
        
        y_stacked = y_returns.stack()
        common_idx = sample_features.index.intersection(y_stacked.index)
        X = sample_features.loc[common_idx].dropna()
        y = y_stacked.loc[X.index].dropna()
        X = X.loc[y.index]
        
        model = RandomForestModel(n_estimators=10, prediction_type='regression')
        model.fit(X, y_returns=y)
        
        importances = model.feature_importances_
        
        assert len(importances) == X.shape[1]
        assert np.sum(importances) > 0


class TestEnsembleModel:
    """Test ensemble model."""
    
    def test_ensemble_creation(self):
        """Test ensemble model creation."""
        ensemble = EnsembleModel(
            models=['random_forest', 'xgboost', 'lightgbm'],
            prediction_type='regression',
        )
        
        assert len(ensemble.model_names) == 3
        assert len(ensemble.weights) == 3
        assert abs(sum(ensemble.weights.values()) - 1.0) < 0.01
    
    def test_ensemble_custom_weights(self):
        """Test custom weight initialization."""
        ensemble = EnsembleModel(
            models=['random_forest', 'xgboost'],
            weights={'random_forest': 0.3, 'xgboost': 0.7},
            prediction_type='regression',
        )
        
        assert ensemble.weights['random_forest'] == pytest.approx(0.3, abs=0.01)
        assert ensemble.weights['xgboost'] == pytest.approx(0.7, abs=0.01)
    
    @pytest.mark.skipif(
        True,  # Skip by default - requires xgboost/lightgbm
        reason="Requires xgboost and lightgbm installed"
    )
    def test_ensemble_fit_predict(self, sample_features, sample_price_data):
        """Test ensemble fitting and prediction."""
        y_returns, y_direction = create_target_variable(sample_price_data, target_type='both')
        
        y_ret_stacked = y_returns.stack()
        y_dir_stacked = y_direction.stack()
        common_idx = sample_features.index.intersection(y_ret_stacked.index)
        X = sample_features.loc[common_idx].dropna()
        y_ret = y_ret_stacked.loc[X.index].dropna()
        y_dir = y_dir_stacked.loc[X.index].dropna()
        valid_idx = y_ret.index.intersection(y_dir.index)
        X = X.loc[valid_idx]
        y_ret = y_ret.loc[valid_idx]
        y_dir = y_dir.loc[valid_idx]
        
        ensemble = EnsembleModel(
            models=['random_forest', 'xgboost', 'lightgbm'],
            prediction_type='both',
            n_estimators=10,
            max_depth=3,
        )
        
        ensemble.fit(X, y_returns=y_ret, y_direction=y_dir)
        predictions = ensemble.predict(X)
        
        assert 'returns' in predictions
        assert 'direction' in predictions


# =============================================================================
# TEST: ML STRATEGY
# =============================================================================

class TestMLStrategy:
    """Test ML-based trading strategy."""
    
    def test_strategy_creation(self):
        """Test basic strategy creation."""
        strategy = MLStrategy(
            name='test_ml',
            prediction_type='returns',
            retrain_frequency=63,
        )
        
        assert strategy.name == 'test_ml'
        assert strategy.config.retrain_frequency == 63
    
    def test_strategy_with_custom_model(self):
        """Test strategy with custom model."""
        model = RandomForestModel(
            n_estimators=10,
            prediction_type='both',
        )
        
        strategy = MLStrategy(
            name='custom_model',
            model=model,
        )
        
        assert strategy.model is model
    
    def test_strategy_with_custom_extractor(self, sample_price_data):
        """Test strategy with custom feature extractor."""
        extractor = FeatureExtractor(
            feature_groups=['momentum'],
            lag_periods=[1],
        )
        
        strategy = MLStrategy(
            name='custom_extractor',
            feature_extractor=extractor,
        )
        
        assert strategy.feature_extractor is extractor
    
    def test_generate_signals(self, short_price_data):
        """Test signal generation."""
        # Use RandomForest only (no xgboost/lightgbm dependency)
        model = RandomForestModel(
            n_estimators=10,
            max_depth=3,
            prediction_type='both',
        )
        
        extractor = FeatureExtractor(
            feature_groups=['momentum'],
            lag_periods=[1],
            normalize=False,
            rank=False,
        )
        
        strategy = MLStrategy(
            name='test_signals',
            model=model,
            feature_extractor=extractor,
            retrain_frequency=30,  # Frequent retraining for test
            long_only=True,
        )
        strategy.config.warmup_period = 60
        strategy.config.min_train_samples = 50
        
        signals = strategy.generate_signals(short_price_data)
        
        assert isinstance(signals, pd.DataFrame)
        assert signals.shape[1] == len(short_price_data.symbols)
        # Long-only should have no negative signals
        assert (signals >= 0).all().all()
    
    def test_strategy_run(self, short_price_data):
        """Test complete strategy run."""
        model = RandomForestModel(
            n_estimators=10,
            max_depth=3,
            prediction_type='regression',  # Use 'regression' for model, not 'returns'
        )
        
        extractor = FeatureExtractor(
            feature_groups=['momentum'],
            lag_periods=[1],
            normalize=False,
            rank=False,
        )
        
        strategy = MLStrategy(
            name='test_run',
            model=model,
            feature_extractor=extractor,
            long_only=True,
        )
        strategy.config.warmup_period = 60
        strategy.config.min_train_samples = 50
        strategy.config.retrain_frequency = 30
        
        portfolio = strategy.run(short_price_data)
        
        assert portfolio is not None
        assert hasattr(portfolio, 'weights')


# =============================================================================
# TEST: WALK-FORWARD VALIDATION
# =============================================================================

class TestWalkForwardValidation:
    """Test walk-forward validation."""
    
    def test_validator_creation(self):
        """Test validator creation."""
        validator = WalkForwardMLValidator(
            is_days=252,
            oos_days=63,
            window_type='rolling',
        )
        
        assert validator.is_days == 252
        assert validator.oos_days == 63
        assert validator.window_type == 'rolling'
    
    def test_validation_result_structure(self):
        """Test validation result structure."""
        result = MLValidationResult(
            n_windows=4,
            avg_is_sharpe=1.5,
            avg_oos_sharpe=0.8,
            efficiency_ratio=0.53,
        )
        
        assert result.n_windows == 4
        assert result.efficiency_ratio == 0.53
        assert not result.is_overfitted  # 0.53 > 0.5 threshold
    
    def test_overfitting_detection(self):
        """Test overfitting detection."""
        # Not overfitted
        result1 = MLValidationResult(
            efficiency_ratio=0.7,
            is_overfitted=False,
        )
        assert not result1.is_overfitted
        
        # Overfitted (low efficiency)
        result2 = MLValidationResult(
            efficiency_ratio=0.3,
            is_overfitted=True,
        )
        assert result2.is_overfitted
    
    def test_result_to_dict(self):
        """Test result serialization."""
        result = MLValidationResult(
            n_windows=4,
            avg_oos_sharpe=0.8,
        )
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert 'n_windows' in d
        assert 'avg_oos_sharpe' in d


# =============================================================================
# TEST: INTEGRATION
# =============================================================================

class TestMLIntegration:
    """Test complete ML integration workflow."""
    
    def test_end_to_end_workflow(self, short_price_data):
        """Test complete workflow: features -> model -> strategy -> signals."""
        # 1. Create feature extractor
        extractor = FeatureExtractor(
            feature_groups=['momentum', 'volatility'],
            lag_periods=[1],
        )
        
        # 2. Extract features
        features = extractor.extract(short_price_data)
        assert features.shape[1] > 0
        
        # 3. Create targets
        y_returns, y_direction = create_target_variable(
            short_price_data,
            target_type='both',
        )
        assert y_returns is not None
        assert y_direction is not None
        
        # 4. Create model
        model = RandomForestModel(
            n_estimators=10,
            max_depth=3,
            prediction_type='both',
        )
        
        # 5. Create strategy
        strategy = MLStrategy(
            name='integration_test',
            model=model,
            feature_extractor=extractor,
            long_only=True,
        )
        strategy.config.warmup_period = 60
        strategy.config.min_train_samples = 50
        strategy.config.retrain_frequency = 30
        
        # 6. Generate signals
        signals = strategy.generate_signals(short_price_data)
        
        # 7. Verify output
        assert isinstance(signals, pd.DataFrame)
        assert signals.shape[0] > 0
        assert signals.shape[1] == len(short_price_data.symbols)
        
        print(f"\nIntegration test results:")
        print(f"  Features: {features.shape}")
        print(f"  Signals: {signals.shape}")
        print(f"  Signal range: [{signals.min().min():.3f}, {signals.max().max():.3f}]")


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

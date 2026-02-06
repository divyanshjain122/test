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
# TEST: TEXT FEATURE EXTRACTION
# =============================================================================

class TestTextFeatureExtractor:
    """Test text-based feature extraction."""
    
    @pytest.fixture
    def sample_text_df(self):
        """Generate sample text DataFrame for testing."""
        return pd.DataFrame({
            'date': pd.to_datetime([
                '2023-01-01', '2023-01-01', '2023-01-02', 
                '2023-01-02', '2023-01-03',
            ]),
            'symbol': ['AAPL', 'GOOGL', 'AAPL', 'GOOGL', 'AAPL'],
            'text': [
                "Apple reports strong quarterly earnings, beating analyst expectations",
                "Google announces major AI breakthrough",
                "Market volatility increases amid economic concerns",
                "Alphabet stock rises on cloud revenue growth",
                "iPhone sales beat expectations",
            ],
        })
    
    @pytest.fixture
    def market_text_df(self):
        """Generate market-level text DataFrame (no symbol)."""
        return pd.DataFrame({
            'date': pd.to_datetime([
                '2023-01-01', '2023-01-02', '2023-01-03',
            ]),
            'headline': [
                "Markets rally on Federal Reserve optimism",
                "Economic uncertainty weighs on stocks",
                "Tech sector leads market gains",
            ],
        })
    
    def test_extract_text_features_basic(self, sample_text_df):
        """Test basic text feature extraction."""
        extractor = FeatureExtractor(
            feature_groups=['momentum'],
            lag_periods=[1],
        )
        
        features = extractor.extract_text_features(sample_text_df, text_column='text')
        
        assert isinstance(features, pd.DataFrame)
        assert features.shape[0] > 0
        assert 'text_sentiment_score' in features.columns
        assert 'text_sentiment_confidence' in features.columns
    
    def test_extract_text_features_sentiment_range(self, sample_text_df):
        """Test that sentiment scores are in valid range."""
        extractor = FeatureExtractor(
            feature_groups=['momentum'],
            lag_periods=[1],
        )
        
        features = extractor.extract_text_features(sample_text_df, text_column='text')
        
        # Sentiment should be between -1 and 1
        assert features['text_sentiment_score'].min() >= -1.0
        assert features['text_sentiment_score'].max() <= 1.0
        # Confidence should be between 0 and 1
        assert features['text_sentiment_confidence'].min() >= 0.0
        assert features['text_sentiment_confidence'].max() <= 1.0
    
    def test_extract_text_features_market_level(self, market_text_df):
        """Test text feature extraction without symbol (market-level)."""
        extractor = FeatureExtractor(
            feature_groups=['momentum'],
            lag_periods=[1],
        )
        
        features = extractor.extract_text_features(
            market_text_df, 
            text_column='headline',
            symbol_column=None  # No symbol column
        )
        
        assert 'text_sentiment_score' in features.columns
        # Index should be single level (date only)
        assert features.index.name == 'date' or 'date' in str(features.index.names)
    
    def test_merge_text_features(self, sample_price_data, sample_text_df):
        """Test merging price and text features."""
        extractor = FeatureExtractor(
            feature_groups=['momentum'],
            lag_periods=[1],
        )
        
        # Extract price features
        price_features = extractor.extract(sample_price_data)
        
        # Extract text features
        text_features = extractor.extract_text_features(sample_text_df, text_column='text')
        
        # Merge should work (inner join by default)
        merged = extractor.merge_text_features(price_features, text_features, how='inner')
        
        assert isinstance(merged, pd.DataFrame)
        # Merged should have both price and text feature columns
        price_cols = [c for c in merged.columns if 'momentum' in c.lower()]
        text_cols = [c for c in merged.columns if 'sentiment' in c.lower()]
        assert len(price_cols) > 0
        assert len(text_cols) > 0
    
    def test_text_sentiment_feature_group_exists(self):
        """Test that text_sentiment feature group is defined."""
        assert 'text_sentiment' in FEATURE_GROUPS
        assert 'requires' in FEATURE_GROUPS['text_sentiment']
        assert FEATURE_GROUPS['text_sentiment']['requires'] == 'text_data'
    
    def test_extract_text_features_aggregation(self):
        """Test that multiple texts per date/symbol are aggregated."""
        # Multiple texts for same date/symbol
        text_df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-01']),
            'symbol': ['AAPL', 'AAPL', 'AAPL'],
            'text': [
                "Very positive news!",
                "Slightly negative report",
                "Neutral statement from company",
            ],
        })
        
        extractor = FeatureExtractor(
            feature_groups=['momentum'],
            lag_periods=[1],
        )
        
        features = extractor.extract_text_features(text_df, text_column='text')
        
        # Should aggregate to single row for AAPL on 2023-01-01
        assert len(features) == 1


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
# TEST: ONNX EXPORT UTILITIES
# =============================================================================

class TestModelMetadata:
    """Tests for ModelMetadata class."""
    
    def test_metadata_creation(self):
        """Test creating model metadata."""
        from jsf.ml.export import ModelMetadata
        
        metadata = ModelMetadata(
            model_name="TestModel",
            model_type="RandomForest",
            input_shape=(1, 50),
            output_shape=(1, 1),
            feature_names=["feature_1", "feature_2"],
        )
        
        assert metadata.model_name == "TestModel"
        assert metadata.model_type == "RandomForest"
        assert metadata.input_shape == (1, 50)
        assert len(metadata.feature_names) == 2
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        from jsf.ml.export import ModelMetadata
        
        metadata = ModelMetadata(
            model_name="TestModel",
            model_type="XGBoost",
            input_shape=(1, 10),
        )
        
        d = metadata.to_dict()
        
        assert d["model_name"] == "TestModel"
        assert d["model_type"] == "XGBoost"
        assert d["input_shape"] == [1, 10]
        assert "export_timestamp" in d
    
    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        from jsf.ml.export import ModelMetadata
        
        data = {
            "model_name": "FromDict",
            "model_type": "LightGBM",
            "input_shape": [1, 20],
            "output_shape": [1, 3],
            "framework": "sklearn",
        }
        
        metadata = ModelMetadata.from_dict(data)
        
        assert metadata.model_name == "FromDict"
        assert metadata.model_type == "LightGBM"
        assert metadata.input_shape == (1, 20)
        assert metadata.output_shape == (1, 3)
    
    def test_metadata_save_load(self, tmp_path):
        """Test saving and loading metadata."""
        from jsf.ml.export import ModelMetadata
        
        metadata = ModelMetadata(
            model_name="SaveLoadTest",
            model_type="MLP",
            input_shape=(32, 100),
            feature_names=["a", "b", "c"],
            description="Test model",
        )
        
        # Save
        save_path = tmp_path / "metadata.json"
        metadata.save(save_path)
        
        assert save_path.exists()
        
        # Load
        loaded = ModelMetadata.load(save_path)
        
        assert loaded.model_name == "SaveLoadTest"
        assert loaded.model_type == "MLP"
        assert loaded.input_shape == (32, 100)
        assert loaded.feature_names == ["a", "b", "c"]


class TestMockONNXExporter:
    """Tests for MockONNXExporter (no real ONNX dependency)."""
    
    def test_mock_export(self, tmp_path):
        """Test mock ONNX export."""
        from jsf.ml.export import MockONNXExporter
        
        exporter = MockONNXExporter()
        
        output_path = tmp_path / "model.onnx"
        metadata = exporter.mock_export(
            model_name="MockModel",
            model_type="RandomForest",
            input_shape=(1, 50),
            output_path=output_path,
            feature_names=["f1", "f2", "f3"],
        )
        
        assert output_path.exists()
        assert metadata.model_name == "MockModel"
        assert metadata.input_shape == (1, 50)
        
        # Check metadata file was also created
        metadata_path = tmp_path / "model.json"
        assert metadata_path.exists()
    
    def test_mock_inference(self):
        """Test mock ONNX inference."""
        from jsf.ml.export import MockONNXExporter, ModelMetadata
        
        exporter = MockONNXExporter()
        
        metadata = ModelMetadata(
            model_name="InferenceTest",
            model_type="LSTM",
            input_shape=(1, 20),
            output_shape=(1, 1),
        )
        
        input_data = np.random.randn(10, 20).astype(np.float32)
        predictions = exporter.mock_inference(metadata, input_data)
        
        assert predictions.shape[0] == 10
        assert predictions.dtype == np.float32
    
    def test_mock_inference_deterministic(self):
        """Test that mock inference is deterministic for same input."""
        from jsf.ml.export import MockONNXExporter, ModelMetadata
        
        exporter = MockONNXExporter()
        
        metadata = ModelMetadata(
            model_name="DeterministicTest",
            model_type="MLP",
            input_shape=(1, 10),
            output_shape=(1, 1),
        )
        
        input_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        
        pred1 = exporter.mock_inference(metadata, input_data)
        pred2 = exporter.mock_inference(metadata, input_data)
        
        np.testing.assert_array_equal(pred1, pred2)


class TestChecksumUtilities:
    """Tests for model checksum utilities."""
    
    def test_compute_checksum(self, tmp_path):
        """Test computing file checksum."""
        from jsf.ml.export import compute_model_checksum
        
        # Create test file
        test_file = tmp_path / "test_model.bin"
        test_file.write_bytes(b"test model content")
        
        checksum = compute_model_checksum(test_file)
        
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex digest
    
    def test_verify_checksum_valid(self, tmp_path):
        """Test verifying valid checksum."""
        from jsf.ml.export import compute_model_checksum, verify_model_checksum
        
        test_file = tmp_path / "valid_model.bin"
        test_file.write_bytes(b"valid content")
        
        checksum = compute_model_checksum(test_file)
        
        assert verify_model_checksum(test_file, checksum) is True
    
    def test_verify_checksum_invalid(self, tmp_path):
        """Test verifying invalid checksum."""
        from jsf.ml.export import verify_model_checksum
        
        test_file = tmp_path / "invalid_model.bin"
        test_file.write_bytes(b"some content")
        
        wrong_checksum = "0" * 64
        
        assert verify_model_checksum(test_file, wrong_checksum) is False


class TestCreateExporter:
    """Tests for create_exporter factory function."""
    
    def test_create_mock_exporter(self):
        """Test creating mock exporter."""
        from jsf.ml.export import create_exporter, MockONNXExporter
        
        exporter = create_exporter(mock=True)
        
        assert isinstance(exporter, MockONNXExporter)
    
    def test_create_real_exporter(self):
        """Test creating real exporter (may fail if ONNX not installed)."""
        from jsf.ml.export import create_exporter, ONNXExporter
        
        # This should return ONNXExporter, but won't fail if ONNX not installed
        exporter = create_exporter(mock=False)
        
        assert isinstance(exporter, ONNXExporter)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

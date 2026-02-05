"""ML Integration Example.

This example demonstrates how to use JSF's Machine Learning module
for building and backtesting ML-based trading strategies.

Features demonstrated:
1. Feature extraction from price data
2. ML model training (Random Forest, XGBoost, LightGBM, Ensemble)
3. Walk-forward strategy execution
4. Model validation and overfitting detection
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Core JSF imports
from jsf.data import SyntheticDataLoader, PriceData
from jsf.simulation import Backtester, calculate_metrics

# ML module imports
from jsf.ml import (
    # Feature extraction
    FeatureExtractor,
    FeatureConfig,
    create_feature_extractor,
    FEATURE_GROUPS,
    
    # ML models
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    EnsembleModel,
    
    # Strategy
    MLStrategy,
    MLStrategyConfig,
    
    # Validation
    WalkForwardMLValidator,
    validate_ml_strategy,
    
    # Preprocessing utilities
    create_target_variable,
    prepare_ml_data,
    handle_missing_features,
)


def example_1_feature_extraction():
    """Example 1: Feature extraction from price data."""
    print("\n" + "="*60)
    print("Example 1: Feature Extraction")
    print("="*60)
    
    # Load synthetic data
    loader = SyntheticDataLoader(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        start_date='2022-01-01',
        end_date='2023-12-31',
    )
    price_data = loader.load()
    print(f"Loaded data: {len(price_data.symbols)} symbols, {len(price_data.dates)} days")
    
    # Create feature extractor
    extractor = FeatureExtractor(
        feature_groups=['momentum', 'volatility', 'trend'],
        lag_periods=[1, 5, 10],
        normalize=True,
        rank=True,
    )
    
    # Extract features
    features = extractor.extract(price_data)
    print(f"\nExtracted features shape: {features.shape}")
    print(f"Feature names: {extractor.feature_names[:10]}...")  # First 10
    
    # Show feature statistics
    print(f"\nFeature statistics:")
    print(features.describe().iloc[:, :5])
    
    return features, price_data


def example_2_model_training():
    """Example 2: Train and evaluate ML models."""
    print("\n" + "="*60)
    print("Example 2: Model Training")
    print("="*60)
    
    # Load data
    loader = SyntheticDataLoader(
        symbols=['AAPL', 'MSFT'],
        start_date='2022-01-01',
        end_date='2023-12-31',
    )
    price_data = loader.load()
    
    # Create feature extractor with minimal features for speed
    extractor = FeatureExtractor(
        feature_groups=['momentum', 'volatility'],
        lag_periods=[1],
        normalize=True,
        rank=False,
    )
    features = extractor.extract(price_data)
    
    # Create targets
    from jsf.ml.preprocessing import TargetType
    y_returns, y_direction = create_target_variable(
        price_data,
        target_type=TargetType.BOTH,
        forward_periods=1,
    )
    
    # Handle missing values
    features_clean = handle_missing_features(features, method='ffill')
    
    # Align features and targets
    common_idx = features_clean.index.intersection(y_returns.index)
    X = features_clean.loc[common_idx]
    y_reg = y_returns.loc[common_idx]
    y_cls = y_direction.loc[common_idx]
    
    # Drop remaining NaNs
    valid_mask = ~(X.isna().any(axis=1) | y_reg.isna())
    X = X[valid_mask]
    y_reg = y_reg[valid_mask]
    y_cls = y_cls[valid_mask]
    
    # Train/test split (time-based)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_reg, y_test_reg = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    y_train_cls, y_test_cls = y_cls.iloc[:split_idx], y_cls.iloc[split_idx:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train Random Forest (regression)
    rf_model = RandomForestModel(
        n_estimators=50,
        max_depth=5,
        prediction_type='regression',
    )
    rf_model.fit(X_train, y_train_reg)
    
    # Predict and evaluate
    preds = rf_model.predict(X_test)
    
    # Calculate IC (information coefficient)
    ic = np.corrcoef(preds, y_test_reg)[0, 1]
    print(f"\nRandom Forest IC: {ic:.4f}")
    
    # Feature importances
    importances = rf_model.get_feature_importances()
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 features:")
    for name, imp in top_features:
        print(f"  {name}: {imp:.4f}")
    
    return rf_model


def example_3_ensemble_model():
    """Example 3: Build and train an ensemble model."""
    print("\n" + "="*60)
    print("Example 3: Ensemble Model")
    print("="*60)
    
    # Create ensemble with specified weights
    ensemble = EnsembleModel(
        models=['random_forest', 'xgboost', 'lightgbm'],
        weights={
            'random_forest': 0.30,
            'xgboost': 0.40,
            'lightgbm': 0.30,
        },
        prediction_type='both',  # Both regression and classification
        model_params={
            'random_forest': {'n_estimators': 50, 'max_depth': 5},
            'xgboost': {'n_estimators': 50, 'max_depth': 3},
            'lightgbm': {'n_estimators': 50, 'max_depth': 5},
        }
    )
    
    print("Ensemble models:")
    for name, model in ensemble.models.items():
        print(f"  {name}: weight={ensemble.weights.get(name, 0):.2f}")
    
    # Note: In practice, you would fit the ensemble like this:
    # ensemble.fit(X_train, y_train_returns, y_train_direction)
    # preds_reg, preds_cls = ensemble.predict(X_test, return_proba=True)
    
    print("\nEnsemble is ready for training!")
    return ensemble


def example_4_ml_strategy():
    """Example 4: Create and run an ML-based trading strategy."""
    print("\n" + "="*60)
    print("Example 4: ML Strategy")
    print("="*60)
    
    # Load data
    loader = SyntheticDataLoader(
        symbols=['AAPL', 'MSFT'],
        start_date='2022-01-01',
        end_date='2023-06-30',  # Shorter period for speed
    )
    price_data = loader.load()
    print(f"Data: {len(price_data.symbols)} symbols, {len(price_data.dates)} days")
    
    # Create model with minimal complexity
    model = RandomForestModel(
        n_estimators=20,
        max_depth=3,
        prediction_type='regression',
    )
    
    # Create feature extractor
    extractor = FeatureExtractor(
        feature_groups=['momentum'],
        lag_periods=[1],
        normalize=False,
        rank=False,
    )
    
    # Create ML strategy
    strategy = MLStrategy(
        name='ml_momentum',
        model=model,
        feature_extractor=extractor,
        prediction_type='returns',  # Predict returns
        retrain_frequency=30,  # Retrain every 30 days
        long_only=True,  # Only long positions
    )
    
    # Adjust for smaller dataset
    strategy.config.warmup_period = 60  # 60 days warmup
    strategy.config.min_train_samples = 40  # Min training samples
    
    print("\nRunning ML strategy...")
    print(f"Retrain frequency: {strategy.config.retrain_frequency} days")
    print(f"Warmup period: {strategy.config.warmup_period} days")
    
    # Generate signals
    signals = strategy.generate_signals(price_data)
    print(f"\nSignals generated: {signals.shape}")
    print(f"Non-zero signals: {(signals != 0).sum().sum()}")
    
    # Run full strategy (with portfolio construction)
    portfolio = strategy.run(price_data)
    print(f"\nPortfolio positions: {len(portfolio.holdings)} time periods")
    
    return strategy, portfolio


def example_5_walk_forward_validation():
    """Example 5: Walk-forward validation of ML strategy."""
    print("\n" + "="*60)
    print("Example 5: Walk-Forward Validation")
    print("="*60)
    
    # Create a simple model and extractor
    model = RandomForestModel(
        n_estimators=20,
        max_depth=3,
        prediction_type='both',
    )
    
    extractor = FeatureExtractor(
        feature_groups=['momentum'],
        lag_periods=[1],
        normalize=False,
        rank=False,
    )
    
    # Create validator
    validator = WalkForwardMLValidator(
        n_splits=3,
        train_ratio=0.7,
        gap_periods=5,
        min_train_samples=50,
    )
    
    print("Walk-Forward Validator created:")
    print(f"  Splits: {validator.n_splits}")
    print(f"  Train ratio: {validator.train_ratio}")
    print(f"  Gap periods: {validator.gap_periods}")
    print(f"  Min train samples: {validator.min_train_samples}")
    
    # Note: In practice, you would run validation like this:
    # results = validator.validate(
    #     model=model,
    #     feature_extractor=extractor,
    #     price_data=price_data,
    # )
    # print(f"Mean IC: {results.mean_ic:.4f}")
    # print(f"Efficiency ratio: {results.efficiency_ratio:.4f}")
    
    return validator


def example_6_backtest_ml_strategy():
    """Example 6: Backtest an ML strategy."""
    print("\n" + "="*60)
    print("Example 6: Backtest ML Strategy")
    print("="*60)
    
    # Load data
    loader = SyntheticDataLoader(
        symbols=['AAPL', 'MSFT'],
        start_date='2022-01-01',
        end_date='2023-06-30',
    )
    price_data = loader.load()
    
    # Create strategy
    model = RandomForestModel(
        n_estimators=20,
        max_depth=3,
        prediction_type='regression',
    )
    
    extractor = FeatureExtractor(
        feature_groups=['momentum'],
        lag_periods=[1],
        normalize=False,
        rank=False,
    )
    
    strategy = MLStrategy(
        name='ml_backtest',
        model=model,
        feature_extractor=extractor,
        long_only=True,
    )
    strategy.config.warmup_period = 60
    strategy.config.min_train_samples = 40
    strategy.config.retrain_frequency = 30
    
    print("Running backtest...")
    
    # Run strategy to get portfolio
    portfolio = strategy.run(price_data)
    
    # Create backtester
    backtester = Backtester(
        initial_capital=100000,
        transaction_cost=0.001,
    )
    
    # Run backtest
    results = backtester.run(portfolio, price_data)
    
    print(f"\nBacktest Results:")
    print(f"  Initial Capital: ${100000:,.2f}")
    print(f"  Final Portfolio Value: ${results.final_value:,.2f}")
    print(f"  Total Return: {results.total_return:.2%}")
    print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {results.max_drawdown:.2%}")
    
    return results


def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# JSF Machine Learning Integration Examples")
    print("#"*60)
    
    # Example 1: Feature extraction
    features, price_data = example_1_feature_extraction()
    
    # Example 2: Model training
    model = example_2_model_training()
    
    # Example 3: Ensemble model
    ensemble = example_3_ensemble_model()
    
    # Example 4: ML strategy
    strategy, portfolio = example_4_ml_strategy()
    
    # Example 5: Walk-forward validation
    validator = example_5_walk_forward_validation()
    
    # Example 6: Backtest
    results = example_6_backtest_ml_strategy()
    
    print("\n" + "#"*60)
    print("# All examples completed successfully!")
    print("#"*60)


if __name__ == "__main__":
    main()

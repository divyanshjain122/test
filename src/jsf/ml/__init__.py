"""Machine Learning Integration Module.

This module provides ML-based trading strategies for JSF-Core, including:
- Feature extraction from 20+ technical/fundamental/sentiment signals
- ML model wrappers (RandomForest, XGBoost, LightGBM)
- Neural network models (MLP, LSTM, GRU, Transformer) via TensorFlow
- Ensemble strategies with weighted voting (hybrid tree + neural)
- Walk-forward validation for time-series ML
- MLStrategy class for backtesting ML models
- Monte Carlo simulation for risk assessment
- Reinforcement learning agents (PPO) for trading policies

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
    >>> # Create ensemble model (now supports neural networks!)
    >>> model = EnsembleModel(
    ...     models=['xgboost', 'lightgbm', 'mlp', 'lstm'],
    ...     weights={'xgboost': 0.3, 'lightgbm': 0.3, 'mlp': 0.2, 'lstm': 0.2}
    ... )
    >>> 
    >>> # Create ML strategy
    >>> strategy = MLStrategy(
    ...     name='ml_ensemble',
    ...     model=model,
    ...     feature_extractor=extractor,
    ...     prediction_type='both'  # returns + direction
    ... )

Neural Networks Example:
    >>> from jsf.ml import LSTMModel, MLPModel, GRUModel
    >>> 
    >>> # LSTM for temporal patterns
    >>> lstm = LSTMModel(sequence_length=20, recurrent_units=[64, 32])
    >>> lstm.fit(X_train, y_returns=y_train)
    >>> 
    >>> # MLP for non-linear relationships
    >>> mlp = MLPModel(hidden_layers=[128, 64], dropout_rate=0.3)
    >>> mlp.fit(X_train, y_returns=y_train, y_direction=y_dir)

Monte Carlo Simulation Example:
    >>> from jsf.ml import MonteCarloSimulator
    >>> 
    >>> simulator = MonteCarloSimulator(returns=historical_returns,n_simulations=10000)
    >>> result = simulator.run()
    >>> print(f"VaR (95%): {result.metrics.var_95:.2%}")

Reinforcement Learning Example:
    >>> from jsf.ml import PPOAgent, TradingEnvironment
    >>> 
    >>> env = TradingEnvironment(prices=price_data, features=feature_data)
    >>> agent = PPOAgent(env)
    >>> agent.train(total_timesteps=100000)
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
    PredictionType,
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

# ONNX Export Utilities
from .export import (
    ONNXExporter,
    MockONNXExporter,
    ModelMetadata,
    create_exporter,
    compute_model_checksum,
    verify_model_checksum,
)

# Neural Networks (TensorFlow)
from .neural import (
    NeuralModel,
    NeuralConfig,
    MLPModel,
    LSTMModel,
    GRUModel,
    TransformerModel,
    SequenceModel,
    configure_gpu_memory,
    get_gpu_info,
)

# Monte Carlo Simulation
from .montecarlo import (
    MonteCarloSimulator,
    PortfolioMonteCarloSimulator,
    SimulationConfig,
    SimulationResult,
    RiskMetrics,
    ReturnModel,
)

# Reinforcement Learning (TensorFlow)
from .rl_agent import (
    PPOAgent,
    PPOConfig,
    TradingEnvironment,
    EnvironmentConfig,
    ActorCriticNetwork,
    RolloutBuffer,
    ActionSpace,
    RewardType,
    create_trading_agent,
)

__all__ = [
    # Feature extraction
    "FeatureExtractor",
    "FeatureConfig",
    "create_feature_extractor",
    "FEATURE_GROUPS",
    # Models - Tree-based
    "MLModel",
    "RandomForestModel",
    "XGBoostModel",
    "LightGBMModel",
    "EnsembleModel",
    "ModelConfig",
    "PredictionType",
    # Models - Neural Networks (TensorFlow)
    "NeuralModel",
    "NeuralConfig",
    "MLPModel",
    "LSTMModel",
    "GRUModel",
    "TransformerModel",
    "SequenceModel",
    "configure_gpu_memory",
    "get_gpu_info",
    # Monte Carlo Simulation
    "MonteCarloSimulator",
    "PortfolioMonteCarloSimulator",
    "SimulationConfig",
    "SimulationResult",
    "RiskMetrics",
    "ReturnModel",
    # Reinforcement Learning
    "PPOAgent",
    "PPOConfig",
    "TradingEnvironment",
    "EnvironmentConfig",
    "ActorCriticNetwork",
    "RolloutBuffer",
    "ActionSpace",
    "RewardType",
    "create_trading_agent",
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
    # ONNX Export
    "ONNXExporter",
    "MockONNXExporter",
    "ModelMetadata",
    "create_exporter",
    "compute_model_checksum",
    "verify_model_checksum",
]


def get_available_features() -> dict:
    """Get information about available ML features in this installation.
    
    Returns:
        Dict with feature availability information
    """
    return {
        'tree_models': True,
        'neural_networks': True,
        'monte_carlo': True,
        'reinforcement_learning': True,
        'available_models': EnsembleModel.get_available_models(),
    }

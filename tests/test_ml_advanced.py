"""Tests for Advanced ML Module (Neural Networks, Monte Carlo, RL).

Comprehensive test suite covering:
- Neural network models (MLP, LSTM, GRU, Transformer)
- Monte Carlo simulation for risk assessment
- Reinforcement Learning (PPO agent)
- Hybrid ensemble models
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Conditional TensorFlow import - skip neural tests if not installed
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    tf = None
    HAS_TENSORFLOW = False

requires_tensorflow = pytest.mark.skipif(
    not HAS_TENSORFLOW,
    reason="TensorFlow not installed"
)

# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_features():
    """Generate sample features for testing."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    # Generate features with some patterns
    X = np.random.randn(n_samples, n_features)
    
    # Add some autocorrelation
    for i in range(1, n_samples):
        X[i] = 0.7 * X[i-1] + 0.3 * X[i]
    
    columns = [f'feature_{i}' for i in range(n_features)]
    return pd.DataFrame(X, columns=columns)


@pytest.fixture
def sample_targets(sample_features):
    """Generate sample targets with some predictability."""
    np.random.seed(42)
    n_samples = len(sample_features)
    
    # Returns with some dependence on features
    feature_weights = np.random.randn(sample_features.shape[1]) * 0.1
    signal = sample_features.values @ feature_weights
    
    # Add noise
    returns = signal + np.random.randn(n_samples) * 0.02
    
    # Direction
    direction = np.sign(returns).astype(int)
    
    return pd.Series(returns, name='returns'), pd.Series(direction, name='direction')


@pytest.fixture
def sample_prices():
    """Generate sample price series."""
    np.random.seed(42)
    n_days = 500
    
    returns = np.random.randn(n_days) * 0.02
    prices = 100 * np.cumprod(1 + returns)
    
    return pd.Series(prices, name='price')


@pytest.fixture
def sample_returns(sample_prices):
    """Generate sample returns series."""
    returns = sample_prices.pct_change().dropna()
    return returns


# =============================================================================
# TEST: NEURAL NETWORK MODELS
# =============================================================================

@requires_tensorflow
class TestNeuralNetworks:
    """Tests for neural network models."""
    
    def test_import_neural_models(self):
        """Test that neural models can be imported."""
        from jsf.ml.neural import (
            MLPModel,
            LSTMModel,
            GRUModel,
            TransformerModel,
            NeuralConfig,
            NeuralModel,
        )
        
        assert MLPModel is not None
        assert LSTMModel is not None
        assert GRUModel is not None
        assert TransformerModel is not None
    
    def test_mlp_model_creation(self):
        """Test MLP model creation."""
        from jsf.ml.neural import MLPModel
        
        model = MLPModel(
            hidden_layers=[32, 16],
            dropout_rate=0.2,
            epochs=5,
            prediction_type='both'
        )
        
        assert model is not None
        assert model.config.hidden_layers == [32, 16]
        assert model.config.dropout_rate == 0.2
    
    def test_mlp_fit_predict(self, sample_features, sample_targets):
        """Test MLP model training and prediction."""
        from jsf.ml.neural import MLPModel
        
        y_returns, y_direction = sample_targets
        
        model = MLPModel(
            hidden_layers=[16, 8],
            epochs=3,  # Few epochs for speed
            batch_size=32,
            prediction_type='both'
        )
        
        # Fit model
        model.fit(sample_features, y_returns=y_returns, y_direction=y_direction)
        
        assert model._is_fitted
        
        # Predict
        predictions = model.predict(sample_features)
        
        assert 'returns' in predictions
        assert 'direction' in predictions
        assert len(predictions['returns']) == len(sample_features)
    
    def test_mlp_feature_importances(self, sample_features, sample_targets):
        """Test MLP feature importance extraction."""
        from jsf.ml.neural import MLPModel
        
        y_returns, _ = sample_targets
        
        model = MLPModel(
            hidden_layers=[16, 8],
            epochs=2,
            prediction_type='regression'
        )
        
        model.fit(sample_features, y_returns=y_returns)
        
        importances = model.feature_importances_
        
        assert len(importances) == sample_features.shape[1]
        assert np.sum(importances) > 0
    
    def test_lstm_model_creation(self):
        """Test LSTM model creation."""
        from jsf.ml.neural import LSTMModel
        
        model = LSTMModel(
            sequence_length=10,
            recurrent_units=[32, 16],
            bidirectional=False,
            epochs=5
        )
        
        assert model is not None
        assert model.config.sequence_length == 10
        assert model.config.recurrent_units == [32, 16]
    
    def test_lstm_fit_predict(self, sample_features, sample_targets):
        """Test LSTM model training and prediction."""
        from jsf.ml.neural import LSTMModel
        
        y_returns, y_direction = sample_targets
        
        model = LSTMModel(
            sequence_length=10,
            recurrent_units=[16],
            hidden_layers=[8],
            epochs=2,
            batch_size=32,
            prediction_type='both'
        )
        
        model.fit(sample_features, y_returns=y_returns, y_direction=y_direction)
        
        assert model._is_fitted
        
        predictions = model.predict(sample_features)
        
        assert 'returns' in predictions
        # First (seq_len - 1) values should be NaN
        assert np.isnan(predictions['returns'][:9]).all()
        assert not np.isnan(predictions['returns'][10:]).all()
    
    def test_gru_model_fit_predict(self, sample_features, sample_targets):
        """Test GRU model training and prediction."""
        from jsf.ml.neural import GRUModel
        
        y_returns, _ = sample_targets
        
        model = GRUModel(
            sequence_length=10,
            recurrent_units=[16],
            epochs=2,
            prediction_type='regression'
        )
        
        model.fit(sample_features, y_returns=y_returns)
        
        assert model._is_fitted
        
        predictions = model.predict(sample_features)
        assert 'returns' in predictions
    
    def test_transformer_model_fit_predict(self, sample_features, sample_targets):
        """Test Transformer model training and prediction."""
        from jsf.ml.neural import TransformerModel
        
        y_returns, _ = sample_targets
        
        model = TransformerModel(
            sequence_length=10,
            d_model=16,
            num_heads=2,
            num_layers=1,
            epochs=2,
            prediction_type='regression'
        )
        
        model.fit(sample_features, y_returns=y_returns)
        
        assert model._is_fitted
        
        predictions = model.predict(sample_features)
        assert 'returns' in predictions
    
    def test_neural_model_save_load(self, sample_features, sample_targets, tmp_path):
        """Test saving and loading neural models."""
        from jsf.ml.neural import MLPModel
        
        y_returns, _ = sample_targets
        
        model = MLPModel(
            hidden_layers=[16, 8],
            epochs=2,
            prediction_type='regression'
        )
        
        model.fit(sample_features, y_returns=y_returns)
        
        # Save
        save_path = tmp_path / "mlp_model"
        model.save(save_path)
        
        # Load
        loaded_model = MLPModel.load(save_path)
        
        # Compare predictions
        orig_preds = model.predict(sample_features)
        loaded_preds = loaded_model.predict(sample_features)
        
        np.testing.assert_array_almost_equal(
            orig_preds['returns'], 
            loaded_preds['returns'], 
            decimal=5
        )
    
    def test_training_history(self, sample_features, sample_targets):
        """Test getting training history."""
        from jsf.ml.neural import MLPModel
        
        y_returns, _ = sample_targets
        
        model = MLPModel(
            hidden_layers=[16],
            epochs=5,
            prediction_type='regression'
        )
        
        model.fit(sample_features, y_returns=y_returns)
        
        history = model.get_training_history()
        
        assert history is not None
        assert 'loss' in history
        assert len(history['loss']) <= 5


# =============================================================================
# TEST: MONTE CARLO SIMULATION
# =============================================================================

class TestMonteCarlo:
    """Tests for Monte Carlo simulation."""
    
    def test_import_montecarlo(self):
        """Test that Monte Carlo classes can be imported."""
        from jsf.ml.montecarlo import (
            MonteCarloSimulator,
            PortfolioMonteCarloSimulator,
            SimulationConfig,
            SimulationResult,
            RiskMetrics,
            ReturnModel,
        )
        
        assert MonteCarloSimulator is not None
        assert RiskMetrics is not None
    
    def test_simulator_creation(self, sample_returns):
        """Test Monte Carlo simulator creation."""
        from jsf.ml.montecarlo import MonteCarloSimulator
        
        simulator = MonteCarloSimulator(
            returns=sample_returns,
            n_simulations=1000,
            time_horizon=252
        )
        
        assert simulator is not None
        assert simulator.config.n_simulations == 1000
        assert simulator.config.time_horizon == 252
    
    def test_simulation_run(self, sample_returns):
        """Test running Monte Carlo simulation."""
        from jsf.ml.montecarlo import MonteCarloSimulator
        
        simulator = MonteCarloSimulator(
            returns=sample_returns,
            n_simulations=500,
            time_horizon=100,
            random_state=42
        )
        
        result = simulator.run()
        
        assert result is not None
        assert result.price_paths.shape == (500, 101)  # 100 steps + initial
        assert result.return_paths.shape == (500, 100)
        assert result.metrics is not None
    
    def test_risk_metrics(self, sample_returns):
        """Test risk metrics computation."""
        from jsf.ml.montecarlo import MonteCarloSimulator
        
        simulator = MonteCarloSimulator(
            returns=sample_returns,
            n_simulations=1000,
            time_horizon=100,
            random_state=42
        )
        
        result = simulator.run()
        metrics = result.metrics
        
        # Check VaR metrics
        assert 0 <= metrics.var_95 <= 1.0 or metrics.var_95 < 0  # Can be negative (gain)
        assert metrics.var_95 <= metrics.var_99  # 99% VaR should be >= 95% VaR
        
        # Check CVaR >= VaR
        assert metrics.cvar_95 >= metrics.var_95
        
        # Check drawdown metrics
        assert 0 <= metrics.max_drawdown_median <= 1.0
        assert metrics.max_drawdown_95 >= metrics.max_drawdown_median
        
        # Check probability metrics
        assert 0 <= metrics.prob_loss <= 1.0
    
    def test_risk_metrics_summary(self, sample_returns):
        """Test risk metrics summary string."""
        from jsf.ml.montecarlo import MonteCarloSimulator
        
        simulator = MonteCarloSimulator(
            returns=sample_returns,
            n_simulations=100,
            time_horizon=50
        )
        
        result = simulator.run()
        summary = result.metrics.summary()
        
        assert 'VaR' in summary
        assert 'CVaR' in summary
        assert 'Drawdown' in summary
    
    def test_historical_return_model(self, sample_returns):
        """Test historical bootstrap return model."""
        from jsf.ml.montecarlo import MonteCarloSimulator, ReturnModel
        
        simulator = MonteCarloSimulator(
            returns=sample_returns,
            n_simulations=100,
            time_horizon=50,
            return_model=ReturnModel.HISTORICAL
        )
        
        result = simulator.run()
        assert result.return_paths.shape == (100, 50)
    
    def test_normal_return_model(self, sample_returns):
        """Test normal distribution return model."""
        from jsf.ml.montecarlo import MonteCarloSimulator, ReturnModel
        
        simulator = MonteCarloSimulator(
            returns=sample_returns,
            n_simulations=100,
            time_horizon=50,
            return_model=ReturnModel.NORMAL
        )
        
        result = simulator.run()
        assert result.return_paths.shape == (100, 50)
    
    def test_t_distribution_return_model(self, sample_returns):
        """Test t-distribution return model."""
        from jsf.ml.montecarlo import MonteCarloSimulator, ReturnModel
        
        simulator = MonteCarloSimulator(
            returns=sample_returns,
            n_simulations=100,
            time_horizon=50,
            return_model=ReturnModel.T_DISTRIBUTION
        )
        
        result = simulator.run()
        assert result.return_paths.shape == (100, 50)
    
    def test_block_bootstrap_model(self, sample_returns):
        """Test block bootstrap return model."""
        from jsf.ml.montecarlo import MonteCarloSimulator, ReturnModel
        
        simulator = MonteCarloSimulator(
            returns=sample_returns,
            n_simulations=100,
            time_horizon=50,
            return_model=ReturnModel.BLOCK_BOOTSTRAP,
            block_size=10
        )
        
        result = simulator.run()
        assert result.return_paths.shape == (100, 50)
    
    def test_garch_return_model(self, sample_returns):
        """Test GARCH return model."""
        from jsf.ml.montecarlo import MonteCarloSimulator, ReturnModel
        
        simulator = MonteCarloSimulator(
            returns=sample_returns,
            n_simulations=100,
            time_horizon=50,
            return_model=ReturnModel.GARCH
        )
        
        result = simulator.run()
        assert result.return_paths.shape == (100, 50)
    
    def test_confidence_band(self, sample_returns):
        """Test confidence band computation."""
        from jsf.ml.montecarlo import MonteCarloSimulator
        
        simulator = MonteCarloSimulator(
            returns=sample_returns,
            n_simulations=100,
            time_horizon=50
        )
        
        result = simulator.run()
        median, lower, upper = result.get_confidence_band(lower_pct=5, upper_pct=95)
        
        assert len(median) == 51  # horizon + 1
        assert (upper >= median).all()
        assert (median >= lower).all()
    
    def test_stress_test(self, sample_returns):
        """Test stress testing scenarios."""
        from jsf.ml.montecarlo import MonteCarloSimulator
        
        simulator = MonteCarloSimulator(
            returns=sample_returns,
            n_simulations=100,
            time_horizon=50
        )
        
        scenarios = {
            'base': {},
            'high_vol': {'vol_mult': 1.5},
            'crash': {'mean_shift': -0.01, 'vol_mult': 2.0},
        }
        
        results = simulator.run_stress_test(scenarios)
        
        assert 'base' in results
        assert 'high_vol' in results
        assert 'crash' in results
        
        # High vol should have higher VaR
        assert results['high_vol'].metrics.var_95 > results['base'].metrics.var_95 * 0.8
    
    def test_portfolio_simulator(self):
        """Test portfolio Monte Carlo simulator."""
        from jsf.ml.montecarlo import PortfolioMonteCarloSimulator
        
        np.random.seed(42)
        n_days = 200
        
        # Create correlated returns
        returns_df = pd.DataFrame({
            'AAPL': np.random.randn(n_days) * 0.02,
            'GOOGL': np.random.randn(n_days) * 0.025,
            'MSFT': np.random.randn(n_days) * 0.018,
        })
        
        weights = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
        
        simulator = PortfolioMonteCarloSimulator(
            returns=returns_df,
            weights=weights,
            n_simulations=100,
            time_horizon=50
        )
        
        result = simulator.run()
        
        assert result is not None
        assert result.price_paths.shape[0] == 100


# =============================================================================
# TEST: REINFORCEMENT LEARNING
# =============================================================================

@requires_tensorflow
class TestReinforcementLearning:
    """Tests for reinforcement learning agents."""
    
    def test_import_rl_classes(self):
        """Test that RL classes can be imported."""
        from jsf.ml.rl_agent import (
            PPOAgent,
            PPOConfig,
            TradingEnvironment,
            EnvironmentConfig,
            ActionSpace,
            RewardType,
            create_trading_agent,
        )
        
        assert PPOAgent is not None
        assert TradingEnvironment is not None
    
    def test_trading_environment_creation(self, sample_prices, sample_features):
        """Test trading environment creation."""
        from jsf.ml.rl_agent import TradingEnvironment, EnvironmentConfig
        
        config = EnvironmentConfig(
            initial_capital=100000,
            transaction_cost=0.001
        )
        
        env = TradingEnvironment(
            prices=sample_prices,
            features=sample_features,
            config=config
        )
        
        assert env is not None
        assert env.observation_dim == sample_features.shape[1] + 3
    
    def test_environment_reset(self, sample_prices, sample_features):
        """Test environment reset."""
        from jsf.ml.rl_agent import TradingEnvironment
        
        env = TradingEnvironment(prices=sample_prices, features=sample_features)
        
        obs = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert len(obs) == env.observation_dim
    
    def test_environment_step(self, sample_prices, sample_features):
        """Test environment step."""
        from jsf.ml.rl_agent import TradingEnvironment
        
        env = TradingEnvironment(prices=sample_prices, features=sample_features)
        
        obs = env.reset()
        
        # Take action
        next_obs, reward, done, info = env.step(2)  # Buy
        
        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert 'portfolio_value' in info
    
    def test_environment_episode(self, sample_prices, sample_features):
        """Test running full episode."""
        from jsf.ml.rl_agent import TradingEnvironment, EnvironmentConfig
        
        config = EnvironmentConfig(episode_length=50)
        env = TradingEnvironment(
            prices=sample_prices, 
            features=sample_features,
            config=config
        )
        
        obs = env.reset()
        done = False
        steps = 0
        
        while not done:
            action = np.random.choice([0, 1, 2])
            obs, reward, done, info = env.step(action)
            steps += 1
        
        stats = env.get_episode_stats()
        
        assert 'total_return' in stats
        assert 'sharpe' in stats
        assert 'max_drawdown' in stats
        assert steps <= 50
    
    def test_ppo_agent_creation(self, sample_prices, sample_features):
        """Test PPO agent creation."""
        from jsf.ml.rl_agent import PPOAgent, TradingEnvironment, PPOConfig
        
        env = TradingEnvironment(prices=sample_prices, features=sample_features)
        
        config = PPOConfig(
            n_steps=64,
            batch_size=16,
            n_epochs=2
        )
        
        agent = PPOAgent(env, config=config)
        
        assert agent is not None
        assert agent.network is not None
    
    def test_ppo_get_action(self, sample_prices, sample_features):
        """Test PPO agent get action."""
        from jsf.ml.rl_agent import PPOAgent, TradingEnvironment
        
        env = TradingEnvironment(prices=sample_prices, features=sample_features)
        agent = PPOAgent(env, n_steps=32)
        
        obs = env.reset()
        action = agent.get_action(obs)
        
        assert action in [0, 1, 2]
    
    def test_ppo_collect_rollout(self, sample_prices, sample_features):
        """Test PPO rollout collection."""
        from jsf.ml.rl_agent import PPOAgent, TradingEnvironment
        
        env = TradingEnvironment(prices=sample_prices, features=sample_features)
        agent = PPOAgent(env, n_steps=32, batch_size=8)
        
        stats = agent.collect_rollout()
        
        assert isinstance(stats, dict)
        assert len(agent.buffer.observations) == 32
    
    def test_ppo_update(self, sample_prices, sample_features):
        """Test PPO policy update."""
        from jsf.ml.rl_agent import PPOAgent, TradingEnvironment
        
        env = TradingEnvironment(prices=sample_prices, features=sample_features)
        agent = PPOAgent(env, n_steps=32, batch_size=8, n_epochs=2)
        
        # Collect rollout
        agent.collect_rollout()
        
        # Update
        stats = agent.update()
        
        assert 'policy_loss' in stats
        assert 'value_loss' in stats
        assert 'entropy' in stats
    
    def test_ppo_short_training(self, sample_prices, sample_features):
        """Test short PPO training run."""
        from jsf.ml.rl_agent import PPOAgent, TradingEnvironment
        
        env = TradingEnvironment(prices=sample_prices, features=sample_features)
        agent = PPOAgent(env, n_steps=32, batch_size=16, n_epochs=2)
        
        # Train for 2 updates
        history = agent.train(total_timesteps=64, log_interval=1)
        
        assert len(history) == 2
        assert 'mean_episode_return' in history[0]
    
    def test_ppo_evaluate(self, sample_prices, sample_features):
        """Test PPO agent evaluation."""
        from jsf.ml.rl_agent import PPOAgent, TradingEnvironment
        
        env = TradingEnvironment(prices=sample_prices, features=sample_features)
        agent = PPOAgent(env, n_steps=32)
        
        # Evaluate without training
        episode_stats, episode_actions = agent.evaluate(n_episodes=2)
        
        assert len(episode_stats) == 2
        assert len(episode_actions) == 2
    
    def test_ppo_save_load(self, sample_prices, sample_features, tmp_path):
        """Test saving and loading PPO agent."""
        from jsf.ml.rl_agent import PPOAgent, TradingEnvironment
        
        env = TradingEnvironment(prices=sample_prices, features=sample_features)
        agent = PPOAgent(env, n_steps=32)
        
        # Do a short training
        agent.train(total_timesteps=32, log_interval=10)
        
        # Save
        save_path = tmp_path / "ppo_agent"
        agent.save(save_path)
        
        # Load
        loaded_agent = PPOAgent.load(save_path, env)
        
        # Compare actions
        obs = env.reset()
        orig_action = agent.get_action(obs)
        loaded_action = loaded_agent.get_action(obs)
        
        # Actions should be deterministic
        assert orig_action == loaded_action
    
    def test_create_trading_agent_helper(self, sample_prices, sample_features):
        """Test create_trading_agent helper function."""
        from jsf.ml.rl_agent import create_trading_agent
        
        agent, env = create_trading_agent(
            prices=sample_prices,
            features=sample_features,
            action_space='discrete',
            initial_capital=100000
        )
        
        assert agent is not None
        assert env is not None
    
    def test_continuous_action_space(self, sample_prices, sample_features):
        """Test continuous action space."""
        from jsf.ml.rl_agent import PPOAgent, TradingEnvironment, PPOConfig, ActionSpace
        
        env = TradingEnvironment(prices=sample_prices, features=sample_features)
        config = PPOConfig(
            action_space=ActionSpace.CONTINUOUS,
            n_steps=32,
            batch_size=8
        )
        
        agent = PPOAgent(env, config=config)
        
        obs = env.reset()
        action = agent.get_action(obs)
        
        # Continuous action should be in [-1, 1]
        assert -1 <= action <= 1


# =============================================================================
# TEST: HYBRID ENSEMBLE
# =============================================================================

@requires_tensorflow
class TestHybridEnsemble:
    """Tests for hybrid ensemble with tree and neural models."""
    
    def test_ensemble_with_neural_models(self, sample_features, sample_targets):
        """Test ensemble creation with neural models."""
        from jsf.ml import EnsembleModel
        
        y_returns, y_direction = sample_targets
        
        # Create hybrid ensemble
        ensemble = EnsembleModel(
            models=['random_forest', 'mlp'],
            weights={'random_forest': 0.6, 'mlp': 0.4},
            prediction_type='both',
            # Common kwargs
            n_estimators=10,  # For RF
            epochs=2,  # For MLP
        )
        
        # Fit
        ensemble.fit(sample_features, y_returns=y_returns, y_direction=y_direction)
        
        assert ensemble._is_fitted
        
        # Predict
        predictions = ensemble.predict(sample_features)
        
        assert 'returns' in predictions
        assert 'direction' in predictions
    
    def test_get_available_models(self):
        """Test getting list of available models."""
        from jsf.ml import EnsembleModel
        
        models = EnsembleModel.get_available_models()
        
        assert 'random_forest' in models
        assert 'xgboost' in models
        assert 'lightgbm' in models
        assert 'mlp' in models
        assert 'lstm' in models
        assert 'gru' in models
    
    def test_get_model_class(self):
        """Test getting model class by name."""
        from jsf.ml import EnsembleModel
        
        rf_class = EnsembleModel.get_model_class('random_forest')
        assert rf_class is not None
        
        mlp_class = EnsembleModel.get_model_class('mlp')
        assert mlp_class is not None
    
    def test_invalid_model_name(self):
        """Test error for invalid model name."""
        from jsf.ml import EnsembleModel
        
        with pytest.raises(ValueError, match="Unknown model"):
            EnsembleModel.get_model_class('invalid_model')


# =============================================================================
# TEST: MODULE AVAILABILITY
# =============================================================================

class TestModuleAvailability:
    """Tests for checking module availability."""
    
    def test_get_available_features(self):
        """Test get_available_features function."""
        from jsf.ml import get_available_features
        
        features = get_available_features()
        
        assert 'tree_models' in features
        assert 'neural_networks' in features
        assert 'monte_carlo' in features
        assert 'reinforcement_learning' in features
        assert features['tree_models'] is True
    
    def test_base_imports_always_work(self):
        """Test that base ML imports always work."""
        from jsf.ml import (
            MLModel,
            RandomForestModel,
            EnsembleModel,
            FeatureExtractor,
            MLStrategy,
        )
        
        assert MLModel is not None
        assert RandomForestModel is not None
        assert EnsembleModel is not None

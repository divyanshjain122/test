#!/usr/bin/env python
"""Advanced ML Pipeline Example.

This example demonstrates the full advanced ML pipeline including:
1. Neural Network Models (MLP, LSTM, GRU)
2. Hybrid Ensemble (trees + neural networks)
3. Monte Carlo Simulation for risk assessment
4. Reinforcement Learning (PPO) for trading

Requirements:
    pip install jsf-core[ml-full]
    # OR separately:
    pip install tensorflow scikit-learn xgboost lightgbm

Usage:
    python examples/advanced_ml_example.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

print("=" * 70)
print("JSF-Core Advanced ML Pipeline Example")
print("=" * 70)

# =============================================================================
# 1. GENERATE SAMPLE DATA
# =============================================================================

print("\n1. Generating synthetic market data...")

np.random.seed(42)

# Generate price series
n_days = 1000
dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')

# Simulate prices with trends and volatility clustering
returns = np.random.randn(n_days) * 0.02  # Base volatility
# Add volatility clustering
for i in range(1, n_days):
    returns[i] = returns[i] * (0.7 + 0.6 * abs(returns[i-1]) / 0.02)

prices = 100 * np.cumprod(1 + returns)
prices = pd.Series(prices, index=dates, name='price')
returns = pd.Series(returns, index=dates, name='returns')

print(f"   Generated {n_days} days of price data")
print(f"   Price range: ${prices.min():.2f} - ${prices.max():.2f}")
print(f"   Total return: {(prices.iloc[-1] / prices.iloc[0] - 1) * 100:.1f}%")

# Generate features
def create_features(prices, returns):
    """Create technical features."""
    features = pd.DataFrame(index=prices.index)
    
    # Returns at different lags
    for lag in [1, 5, 10, 20]:
        features[f'ret_{lag}d'] = returns.shift(lag)
    
    # Rolling statistics
    for window in [5, 10, 20, 50]:
        features[f'ma_{window}'] = prices.rolling(window).mean() / prices - 1
        features[f'vol_{window}'] = returns.rolling(window).std()
        features[f'mom_{window}'] = prices / prices.shift(window) - 1
    
    # RSI-like indicator
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    features['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # Normalize RSI to 0-1
    features['rsi'] = features['rsi'] / 100
    
    return features.dropna()

features = create_features(prices, returns)
print(f"   Created {len(features.columns)} features")

# Create targets
forward_returns = returns.shift(-1)  # Next day returns
forward_direction = np.sign(forward_returns).astype(int)

# Align everything
common_idx = features.index.intersection(forward_returns.dropna().index)
X = features.loc[common_idx]
y_returns = forward_returns.loc[common_idx]
y_direction = forward_direction.loc[common_idx]
prices_aligned = prices.loc[common_idx]

# Train/test split
split_idx = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_ret_train, y_ret_test = y_returns.iloc[:split_idx], y_returns.iloc[split_idx:]
y_dir_train, y_dir_test = y_direction.iloc[:split_idx], y_direction.iloc[split_idx:]
prices_test = prices_aligned.iloc[split_idx:]

print(f"   Train: {len(X_train)} samples, Test: {len(X_test)} samples")

# =============================================================================
# 2. NEURAL NETWORK MODELS
# =============================================================================

print("\n2. Training Neural Network Models...")

try:
    from jsf.ml import MLPModel, LSTMModel, GRUModel
    
    # 2a. MLP Model
    print("\n   2a. Training MLP (Multi-Layer Perceptron)...")
    mlp = MLPModel(
        hidden_layers=[64, 32, 16],
        dropout_rate=0.3,
        learning_rate=0.001,
        epochs=30,
        batch_size=32,
        prediction_type='both'
    )
    
    mlp.fit(
        X_train, 
        y_returns=y_ret_train, 
        y_direction=y_dir_train
    )
    
    mlp_preds = mlp.predict(X_test)
    mlp_corr = np.corrcoef(mlp_preds['returns'], y_ret_test)[0, 1]
    mlp_acc = np.mean(np.sign(mlp_preds['direction']) == y_dir_test)
    
    print(f"       Return prediction correlation: {mlp_corr:.4f}")
    print(f"       Direction accuracy: {mlp_acc:.2%}")
    
    # 2b. LSTM Model
    print("\n   2b. Training LSTM (Long Short-Term Memory)...")
    lstm = LSTMModel(
        sequence_length=20,
        recurrent_units=[32, 16],
        hidden_layers=[16],
        dropout_rate=0.2,
        epochs=20,
        batch_size=32,
        prediction_type='regression'
    )
    
    lstm.fit(X_train, y_returns=y_ret_train)
    
    lstm_preds = lstm.predict(X_test)
    lstm_returns = lstm_preds['returns']
    # Filter out NaN (first sequence_length-1 values)
    valid_idx = ~np.isnan(lstm_returns)
    lstm_corr = np.corrcoef(lstm_returns[valid_idx], y_ret_test.values[valid_idx])[0, 1]
    
    print(f"       Return prediction correlation: {lstm_corr:.4f}")
    print(f"       (LSTM uses sequences, first 19 predictions are NaN)")
    
    # 2c. GRU Model
    print("\n   2c. Training GRU (Gated Recurrent Unit)...")
    gru = GRUModel(
        sequence_length=15,
        recurrent_units=[32],
        hidden_layers=[16],
        epochs=15,
        prediction_type='regression'
    )
    
    gru.fit(X_train, y_returns=y_ret_train)
    
    gru_preds = gru.predict(X_test)
    gru_returns = gru_preds['returns']
    valid_idx = ~np.isnan(gru_returns)
    gru_corr = np.corrcoef(gru_returns[valid_idx], y_ret_test.values[valid_idx])[0, 1]
    
    print(f"       Return prediction correlation: {gru_corr:.4f}")
    
    neural_available = True
    
except ImportError as e:
    print(f"   [SKIPPED] Neural networks require TensorFlow: {e}")
    print("   Install with: pip install tensorflow")
    neural_available = False

# =============================================================================
# 3. HYBRID ENSEMBLE
# =============================================================================

print("\n3. Building Hybrid Ensemble (Trees + Neural Networks)...")

try:
    from jsf.ml import EnsembleModel, RandomForestModel
    
    # Show available models
    available_models = EnsembleModel.get_available_models()
    print(f"   Available models: {available_models}")
    
    if neural_available:
        # Hybrid ensemble with trees and neural networks
        print("\n   Creating hybrid ensemble: RF + XGBoost + MLP...")
        
        ensemble = EnsembleModel(
            models=['random_forest', 'xgboost', 'mlp'],
            weights={'random_forest': 0.35, 'xgboost': 0.35, 'mlp': 0.30},
            prediction_type='both',
            # Tree model kwargs
            n_estimators=50,
            max_depth=5,
            # Neural model kwargs
            hidden_layers=[32, 16],
            epochs=10,
        )
    else:
        # Tree-only ensemble
        print("\n   Creating tree ensemble: RF + XGBoost...")
        
        ensemble = EnsembleModel(
            models=['random_forest', 'xgboost'],
            weights={'random_forest': 0.5, 'xgboost': 0.5},
            prediction_type='both',
            n_estimators=50,
            max_depth=5,
        )
    
    ensemble.fit(
        X_train,
        y_returns=y_ret_train,
        y_direction=y_dir_train
    )
    
    ensemble_preds = ensemble.predict(X_test)
    
    ens_corr = np.corrcoef(ensemble_preds['returns'], y_ret_test)[0, 1]
    ens_acc = np.mean(np.sign(ensemble_preds['direction']) == y_dir_test)
    
    print(f"   Ensemble weights: {ensemble.get_model_weights()}")
    print(f"   Return prediction correlation: {ens_corr:.4f}")
    print(f"   Direction accuracy: {ens_acc:.2%}")
    
except Exception as e:
    print(f"   [ERROR] Ensemble creation failed: {e}")

# =============================================================================
# 4. MONTE CARLO SIMULATION
# =============================================================================

print("\n4. Running Monte Carlo Simulation for Risk Assessment...")

try:
    from jsf.ml import MonteCarloSimulator, ReturnModel, RiskMetrics
    
    # Use historical returns for simulation
    historical_returns = returns.iloc[:split_idx]
    
    # 4a. Basic Monte Carlo
    print("\n   4a. Historical Bootstrap Simulation...")
    
    simulator = MonteCarloSimulator(
        returns=historical_returns,
        n_simulations=5000,
        time_horizon=252,  # 1 year
        return_model=ReturnModel.HISTORICAL,
        initial_capital=100.0,
        random_state=42
    )
    
    result = simulator.run()
    metrics = result.metrics
    
    print(f"\n   Risk Metrics (1-year horizon):")
    print(f"   {'─' * 40}")
    print(f"   VaR (95%):              {metrics.var_95:>10.2%}")
    print(f"   CVaR (95%):             {metrics.cvar_95:>10.2%}")
    print(f"   Max Drawdown (median):  {metrics.max_drawdown_median:>10.2%}")
    print(f"   Max Drawdown (95th):    {metrics.max_drawdown_95:>10.2%}")
    print(f"   Expected Return:        {metrics.expected_return:>10.2%}")
    print(f"   Sharpe Ratio (median):  {metrics.sharpe_median:>10.2f}")
    print(f"   Probability of Loss:    {metrics.prob_loss:>10.2%}")
    print(f"   Prob Loss > 10%:        {metrics.prob_loss_10pct:>10.2%}")
    
    # 4b. Compare Return Models
    print("\n   4b. Comparing Return Models...")
    
    models_to_test = [
        (ReturnModel.HISTORICAL, "Historical Bootstrap"),
        (ReturnModel.NORMAL, "Normal Distribution"),
        (ReturnModel.T_DISTRIBUTION, "Student's t"),
        (ReturnModel.GARCH, "GARCH(1,1)"),
    ]
    
    model_results = {}
    for model, name in models_to_test:
        sim = MonteCarloSimulator(
            returns=historical_returns,
            n_simulations=2000,
            time_horizon=126,  # 6 months
            return_model=model,
            random_state=42
        )
        res = sim.run()
        model_results[name] = res.metrics
    
    print(f"\n   {'Model':<25} {'VaR(95%)':<12} {'CVaR(95%)':<12} {'Max DD':<12}")
    print(f"   {'─' * 60}")
    for name, m in model_results.items():
        print(f"   {name:<25} {m.var_95:>10.2%}   {m.cvar_95:>10.2%}   {m.max_drawdown_median:>10.2%}")
    
    # 4c. Stress Testing
    print("\n   4c. Running Stress Test Scenarios...")
    
    stress_scenarios = {
        'Base Case': {},
        'High Volatility': {'vol_mult': 1.5},
        'Market Crash': {'mean_shift': -0.001, 'vol_mult': 2.0},
        'Low Vol Rally': {'mean_shift': 0.001, 'vol_mult': 0.7},
    }
    
    stress_results = simulator.run_stress_test(stress_scenarios)
    
    print(f"\n   {'Scenario':<20} {'VaR(95%)':<12} {'Max DD(95th)':<12} {'Prob Loss':<12}")
    print(f"   {'─' * 56}")
    for name, res in stress_results.items():
        m = res.metrics
        print(f"   {name:<20} {m.var_95:>10.2%}   {m.max_drawdown_95:>10.2%}   {m.prob_loss:>10.2%}")
    
    montecarlo_available = True
    
except ImportError as e:
    print(f"   [SKIPPED] Monte Carlo requires scipy: {e}")
    montecarlo_available = False

# =============================================================================
# 5. REINFORCEMENT LEARNING (PPO)
# =============================================================================

print("\n5. Training Reinforcement Learning Agent (PPO)...")

try:
    from jsf.ml import PPOAgent, TradingEnvironment, PPOConfig, ActionSpace
    
    # Prepare data for RL
    rl_prices = prices.iloc[:split_idx].values
    rl_features = X_train.values
    
    print(f"   Environment: {len(rl_prices)} timesteps, {rl_features.shape[1]} features")
    
    # Create trading environment
    env = TradingEnvironment(
        prices=rl_prices,
        features=rl_features,
        initial_capital=100000,
        transaction_cost=0.001,
        episode_length=100  # Short episodes for demo
    )
    
    print(f"   Observation dim: {env.observation_dim}")
    
    # Create PPO agent
    config = PPOConfig(
        actor_layers=[64, 32],
        critic_layers=[64, 32],
        action_space=ActionSpace.DISCRETE,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.99,
    )
    
    agent = PPOAgent(env, config=config)
    
    # Train for a short time (demo only - increase for real use)
    print("\n   Training PPO agent (this may take a minute)...")
    print("   [For real use, train for 100k+ timesteps]")
    
    history = agent.train(
        total_timesteps=2048,  # Short for demo
        log_interval=1
    )
    
    # Evaluate on test data
    print("\n   Evaluating trained agent...")
    
    test_prices = prices.iloc[split_idx:split_idx+200].values
    test_features = X_test.iloc[:200].values
    
    test_env = TradingEnvironment(
        prices=test_prices,
        features=test_features,
        initial_capital=100000,
        transaction_cost=0.001,
        episode_length=199
    )
    
    episode_stats, episode_actions = agent.evaluate(test_env, n_episodes=3)
    
    print(f"\n   Evaluation Results (3 episodes):")
    print(f"   {'─' * 40}")
    for i, stats in enumerate(episode_stats):
        print(f"   Episode {i+1}: Return={stats['total_return']:>7.2%}, "
              f"Sharpe={stats['sharpe']:>6.2f}, MaxDD={stats['max_drawdown']:>6.2%}")
    
    avg_return = np.mean([s['total_return'] for s in episode_stats])
    avg_sharpe = np.mean([s['sharpe'] for s in episode_stats])
    
    print(f"\n   Average Return: {avg_return:.2%}")
    print(f"   Average Sharpe: {avg_sharpe:.2f}")
    
    # Show action distribution
    all_actions = []
    for actions in episode_actions:
        all_actions.extend(actions)
    action_counts = np.bincount(all_actions, minlength=3)
    print(f"\n   Action Distribution: Sell={action_counts[0]}, "
          f"Hold={action_counts[1]}, Buy={action_counts[2]}")
    
    rl_available = True
    
except ImportError as e:
    print(f"   [SKIPPED] RL requires TensorFlow: {e}")
    rl_available = False

# =============================================================================
# 6. SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: Advanced ML Pipeline Components")
print("=" * 70)

print(f"""
Component                  Status
{'─' * 50}
Neural Networks (MLP)      {'✅ Available' if neural_available else '❌ Requires TensorFlow'}
Neural Networks (LSTM)     {'✅ Available' if neural_available else '❌ Requires TensorFlow'}
Neural Networks (GRU)      {'✅ Available' if neural_available else '❌ Requires TensorFlow'}
Hybrid Ensemble            ✅ Available (trees always, NN with TensorFlow)
Monte Carlo Simulation     {'✅ Available' if montecarlo_available else '❌ Requires scipy'}
Reinforcement Learning     {'✅ Available' if rl_available else '❌ Requires TensorFlow'}

Installation Commands:
{'─' * 50}
Full ML pipeline:          pip install jsf-core[ml-full]
Tree models only:          pip install jsf-core[ml]
Neural networks:           pip install tensorflow
""")

if neural_available:
    print(f"""
Performance Summary:
{'─' * 50}
MLP Return Correlation:    {mlp_corr:.4f}
MLP Direction Accuracy:    {mlp_acc:.2%}
LSTM Return Correlation:   {lstm_corr:.4f}
Ensemble Return Corr:      {ens_corr:.4f}
Ensemble Direction Acc:    {ens_acc:.2%}
""")

if montecarlo_available:
    print(f"""
Risk Assessment (1-year horizon):
{'─' * 50}
VaR (95%):                 {metrics.var_95:.2%}
CVaR (95%):                {metrics.cvar_95:.2%}
Max Drawdown (median):     {metrics.max_drawdown_median:.2%}
Probability of Loss:       {metrics.prob_loss:.2%}
""")

print("\n" + "=" * 70)
print("Example complete! See source code for implementation details.")
print("=" * 70)

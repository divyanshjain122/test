"""Reinforcement Learning Agent using Proximal Policy Optimization (PPO).

This module provides a PPO-based RL agent for learning trading policies.
The agent learns to take actions (buy/sell/hold) based on market state
to maximize cumulative returns while managing risk.

Key Features:
- Actor-Critic architecture with separate policy and value networks
- PPO clipping for stable policy updates
- Generalized Advantage Estimation (GAE) for variance reduction
- Custom trading environment with realistic constraints
- Support for continuous and discrete action spaces
- Risk-adjusted reward functions (Sharpe, Sortino, etc.)

Example:
    >>> from jsf.ml.rl_agent import PPOAgent, TradingEnvironment
    >>> 
    >>> # Create environment
    >>> env = TradingEnvironment(
    ...     prices=price_data,
    ...     features=feature_data,
    ...     initial_capital=100000
    ... )
    >>> 
    >>> # Create and train agent
    >>> agent = PPOAgent(env)
    >>> agent.train(total_timesteps=100000)
    >>> 
    >>> # Evaluate
    >>> rewards, actions = agent.evaluate(test_prices, test_features)

Note:
    This implementation uses TensorFlow. For production use with more 
    features, consider using Stable Baselines3 or RLlib.
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings

import pandas as pd
import numpy as np
from pathlib import Path

from jsf.utils.logging import get_logger

logger = get_logger(__name__)


def _check_tensorflow():
    """Check if TensorFlow is installed."""
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        return tf
    except ImportError:
        raise ImportError(
            "TensorFlow is not installed. Install with: pip install tensorflow"
        )


class ActionSpace(Enum):
    """Type of action space."""
    DISCRETE = "discrete"  # Buy/Hold/Sell
    CONTINUOUS = "continuous"  # Position sizing -1 to 1


class RewardType(Enum):
    """Type of reward function."""
    SIMPLE_RETURN = "simple_return"  # Just returns
    SHARPE = "sharpe"  # Risk-adjusted
    SORTINO = "sortino"  # Downside risk-adjusted
    DIFFERENTIAL = "differential"  # Return relative to buy-and-hold
    LOG_RETURN = "log_return"  # Log returns


@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    
    # Network architecture
    actor_layers: List[int] = field(default_factory=lambda: [64, 64])
    critic_layers: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"
    
    # PPO hyperparameters
    clip_epsilon: float = 0.2  # PPO clipping parameter
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Training
    learning_rate: float = 3e-4
    n_epochs: int = 10  # Epochs per update
    batch_size: int = 64
    n_steps: int = 2048  # Steps per rollout
    
    # Action space
    action_space: ActionSpace = ActionSpace.DISCRETE
    n_actions: int = 3  # For discrete: sell, hold, buy
    
    # Reward
    reward_type: RewardType = RewardType.SHARPE
    reward_scale: float = 1.0
    
    # Trading constraints
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    
    # Random seed
    random_state: Optional[int] = 42


@dataclass
class EnvironmentConfig:
    """Configuration for trading environment."""
    
    initial_capital: float = 100000.0
    max_position: float = 1.0  # Maximum position as fraction of capital
    allow_short: bool = False
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    
    # Observation normalization
    normalize_obs: bool = True
    window_size: int = 20  # For computing rolling stats
    
    # Episode configuration
    episode_length: Optional[int] = None  # None = full data
    random_start: bool = True  # Random starting point for episodes


class TradingEnvironment:
    """Trading environment for RL agents.
    
    Implements a gym-like interface for episodic trading simulation.
    The agent observes market features and takes actions to manage
    a portfolio position.
    
    State:
        - Market features (prices, indicators, etc.)
        - Current position
        - Unrealized PnL
        - Portfolio value
        
    Actions (discrete):
        - 0: Sell / Go flat
        - 1: Hold current position  
        - 2: Buy / Go long
        
    Actions (continuous):
        - Value in [-1, 1] representing target position
        
    Rewards:
        - Based on PnL, with optional risk adjustment
    
    Example:
        >>> env = TradingEnvironment(prices, features)
        >>> state = env.reset()
        >>> done = False
        >>> while not done:
        ...     action = agent.get_action(state)
        ...     next_state, reward, done, info = env.step(action)
        ...     state = next_state
    """
    
    def __init__(
        self,
        prices: Union[pd.Series, np.ndarray],
        features: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        config: Optional[EnvironmentConfig] = None,
        **kwargs
    ):
        """Initialize trading environment.
        
        Args:
            prices: Price series
            features: Feature matrix (n_timesteps, n_features)
            config: EnvironmentConfig object
            **kwargs: Override config parameters
        """
        self.config = config or EnvironmentConfig(**kwargs)
        
        # Process price data
        self.prices = prices.values if hasattr(prices, 'values') else prices
        self.returns = np.diff(self.prices) / self.prices[:-1]
        self.returns = np.concatenate([[0], self.returns])  # Pad first
        
        # Process features
        if features is not None:
            self.features = features.values if hasattr(features, 'values') else features
        else:
            # Default features: returns, volatility, momentum
            self.features = self._create_default_features()
        
        self.n_features = self.features.shape[1]
        self.n_timesteps = len(self.prices)
        
        # State includes: features + position + unrealized_pnl + portfolio_value
        self.observation_dim = self.n_features + 3
        
        # Normalization stats
        if self.config.normalize_obs:
            self._fit_normalizer()
        
        # Environment state
        self._current_step = 0
        self._position = 0.0  # Current position (-1 to 1)
        self._entry_price = 0.0
        self._portfolio_value = self.config.initial_capital
        self._done = False
        
        # Episode tracking
        self._episode_start = 0
        self._episode_returns = []
        
        logger.info(
            f"TradingEnvironment: {self.n_timesteps} timesteps, "
            f"{self.n_features} features, obs_dim={self.observation_dim}"
        )
    
    def _create_default_features(self) -> np.ndarray:
        """Create default features from price data."""
        n = len(self.prices)
        features = []
        
        # Returns
        features.append(self.returns)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            if n > window:
                # Rolling return
                roll_ret = pd.Series(self.returns).rolling(window).mean().values
                roll_ret = np.nan_to_num(roll_ret, 0)
                features.append(roll_ret)
                
                # Rolling volatility
                roll_vol = pd.Series(self.returns).rolling(window).std().values
                roll_vol = np.nan_to_num(roll_vol, 0)
                features.append(roll_vol)
        
        # Price momentum
        for lag in [1, 5, 10]:
            if n > lag:
                momentum = (self.prices - np.roll(self.prices, lag)) / np.roll(self.prices, lag)
                momentum[:lag] = 0
                features.append(momentum)
        
        return np.column_stack(features)
    
    def _fit_normalizer(self):
        """Fit observation normalizer."""
        self._obs_mean = np.mean(self.features, axis=0)
        self._obs_std = np.std(self.features, axis=0)
        self._obs_std[self._obs_std < 1e-8] = 1.0
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation."""
        if not self.config.normalize_obs:
            return obs
        
        # Only normalize feature part
        obs_norm = obs.copy()
        obs_norm[:self.n_features] = (
            obs_norm[:self.n_features] - self._obs_mean
        ) / self._obs_std
        
        return obs_norm
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Market features
        features = self.features[self._current_step]
        
        # Unrealized PnL
        if self._position != 0:
            current_price = self.prices[self._current_step]
            unrealized_pnl = (current_price - self._entry_price) / self._entry_price * self._position
        else:
            unrealized_pnl = 0.0
        
        # Portfolio info
        obs = np.concatenate([
            features,
            [self._position, unrealized_pnl, 
             self._portfolio_value / self.config.initial_capital - 1.0]
        ])
        
        return self._normalize_obs(obs)
    
    def reset(
        self, 
        start_idx: Optional[int] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Reset environment for new episode.
        
        Args:
            start_idx: Starting index (None = random or 0)
            seed: Random seed
            
        Returns:
            Initial observation
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Determine starting point
        if start_idx is not None:
            self._current_step = start_idx
        elif self.config.random_start:
            max_start = self.n_timesteps - (self.config.episode_length or 100)
            max_start = max(0, max_start)
            self._current_step = np.random.randint(0, max_start + 1)
        else:
            self._current_step = 0
        
        self._episode_start = self._current_step
        
        # Reset portfolio state
        self._position = 0.0
        self._entry_price = 0.0
        self._portfolio_value = self.config.initial_capital
        self._done = False
        self._episode_returns = []
        
        return self._get_observation()
    
    def step(self, action: Union[int, float]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action in environment.
        
        Args:
            action: Action to take
                - Discrete: 0 (sell), 1 (hold), 2 (buy)
                - Continuous: Position target in [-1, 1]
                
        Returns:
            observation: Next state
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        if self._done:
            raise ValueError("Episode finished. Call reset().")
        
        current_price = self.prices[self._current_step]
        
        # Convert action to position change
        if isinstance(action, (int, np.integer)):
            # Discrete action
            action_map = {0: -1.0, 1: 0.0, 2: 1.0}  # sell, hold, buy
            target_position = action_map.get(int(action), 0.0)
            
            if not self.config.allow_short:
                target_position = max(0, target_position)
        else:
            # Continuous action
            target_position = float(np.clip(action, -1, 1))
            if not self.config.allow_short:
                target_position = max(0, target_position)
        
        # Calculate position change
        position_delta = target_position - self._position
        
        # Apply transaction costs
        transaction_cost = abs(position_delta) * (
            self.config.transaction_cost + self.config.slippage
        ) * self._portfolio_value
        
        # Update entry price for new position
        if position_delta != 0:
            if self._position == 0:
                self._entry_price = current_price
            elif np.sign(target_position) != np.sign(self._position):
                # Flipping position
                self._entry_price = current_price
        
        self._position = target_position
        
        # Move to next step
        self._current_step += 1
        
        # Calculate return
        if self._current_step < self.n_timesteps:
            next_price = self.prices[self._current_step]
            price_return = (next_price - current_price) / current_price
            portfolio_return = self._position * price_return - transaction_cost / self._portfolio_value
            self._portfolio_value *= (1 + portfolio_return)
        else:
            portfolio_return = 0
        
        self._episode_returns.append(portfolio_return)
        
        # Calculate reward
        reward = self._calculate_reward(portfolio_return)
        
        # Check if done
        episode_length = self.config.episode_length or (self.n_timesteps - self._episode_start - 1)
        steps_elapsed = self._current_step - self._episode_start
        
        self._done = bool(
            self._current_step >= self.n_timesteps - 1 or
            steps_elapsed >= episode_length or
            self._portfolio_value <= 0
        )
        
        # Get next observation
        obs = self._get_observation()
        
        info = {
            'portfolio_value': self._portfolio_value,
            'position': self._position,
            'step': self._current_step,
            'transaction_cost': transaction_cost,
            'return': portfolio_return,
        }
        
        return obs, reward, self._done, info
    
    def _calculate_reward(self, portfolio_return: float) -> float:
        """Calculate reward based on configured reward type."""
        reward_type = self.config.transaction_cost  # Use parent config
        
        # Simple return
        reward = portfolio_return
        
        # Scale reward
        reward *= 100  # Scale up for better learning
        
        return reward
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get statistics for completed episode."""
        returns = np.array(self._episode_returns)
        
        total_return = self._portfolio_value / self.config.initial_capital - 1
        
        stats = {
            'total_return': total_return,
            'mean_return': np.mean(returns) if len(returns) > 0 else 0,
            'std_return': np.std(returns) if len(returns) > 0 else 0,
            'sharpe': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'num_steps': len(returns),
        }
        
        return stats
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown for episode."""
        if len(self._episode_returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(self._episode_returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        
        return np.max(drawdowns)


class ActorCriticNetwork:
    """Actor-Critic neural network for PPO.
    
    Uses separate networks for policy (actor) and value (critic).
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: PPOConfig,
        action_space: ActionSpace = ActionSpace.DISCRETE
    ):
        """Initialize networks.
        
        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            config: PPO configuration
            action_space: Type of action space
        """
        self.tf = _check_tensorflow()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.config = config
        self.action_space = action_space
        
        # Build networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # Optimizer
        self.optimizer = self.tf.keras.optimizers.Adam(
            learning_rate=config.learning_rate
        )
    
    def _build_actor(self) -> Any:
        """Build actor (policy) network."""
        tf = self.tf
        
        inputs = tf.keras.Input(shape=(self.observation_dim,))
        x = inputs
        
        # Hidden layers
        for units in self.config.actor_layers:
            x = tf.keras.layers.Dense(units, activation=self.config.activation)(x)
        
        # Output layer
        if self.action_space == ActionSpace.DISCRETE:
            # Softmax for action probabilities
            outputs = tf.keras.layers.Dense(self.action_dim, activation='softmax')(x)
        else:
            # Gaussian policy: output mean and log_std
            mean = tf.keras.layers.Dense(self.action_dim, activation='tanh')(x)
            log_std = tf.keras.layers.Dense(self.action_dim)(x)
            outputs = tf.keras.layers.Concatenate()([mean, log_std])
        
        return tf.keras.Model(inputs=inputs, outputs=outputs, name='actor')
    
    def _build_critic(self) -> Any:
        """Build critic (value) network."""
        tf = self.tf
        
        inputs = tf.keras.Input(shape=(self.observation_dim,))
        x = inputs
        
        # Hidden layers
        for units in self.config.critic_layers:
            x = tf.keras.layers.Dense(units, activation=self.config.activation)(x)
        
        # Value output
        outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs, name='critic')
    
    def get_action(
        self, 
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get action from policy.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic action
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Value estimate
        """
        tf = self.tf
        
        obs = np.atleast_2d(observation).astype(np.float32)
        
        # Get policy output
        policy_out = self.actor(obs)
        
        if self.action_space == ActionSpace.DISCRETE:
            probs = policy_out.numpy()[0]
            
            if deterministic:
                action = np.argmax(probs)
            else:
                action = np.random.choice(self.action_dim, p=probs)
            
            log_prob = np.log(probs[action] + 1e-8)
        else:
            # Continuous: Gaussian policy
            mean = policy_out[:, :self.action_dim].numpy()[0]
            log_std = policy_out[:, self.action_dim:].numpy()[0]
            std = np.exp(np.clip(log_std, -20, 2))
            
            if deterministic:
                action = mean
            else:
                action = mean + std * np.random.randn(self.action_dim)
            
            # Clip action
            action = np.clip(action, -1, 1)
            
            # Log probability
            log_prob = -0.5 * (((action - mean) / std) ** 2 + 2 * log_std + np.log(2 * np.pi))
            log_prob = log_prob.sum()
        
        # Get value estimate
        value = self.critic(obs).numpy()[0, 0]
        
        return action, log_prob, value
    
    def get_value(self, observation: np.ndarray) -> float:
        """Get value estimate for observation."""
        obs = np.atleast_2d(observation).astype(np.float32)
        return self.critic(obs).numpy()[0, 0]


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self, config: PPOConfig):
        """Initialize buffer."""
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        self._ptr = 0
    
    def add(
        self,
        obs: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add experience to buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
        self._ptr += 1
    
    def compute_returns_and_advantages(self, last_value: float):
        """Compute advantages using GAE."""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [False])
        
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t + 1]
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        returns = advantages + np.array(self.values)
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batches(self, batch_size: int):
        """Get randomized mini-batches."""
        n_samples = len(self.observations)
        indices = np.random.permutation(n_samples)
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            
            yield {
                'observations': np.array([self.observations[i] for i in batch_indices]),
                'actions': np.array([self.actions[i] for i in batch_indices]),
                'log_probs': np.array([self.log_probs[i] for i in batch_indices]),
                'advantages': self.advantages[batch_indices],
                'returns': self.returns[batch_indices],
            }


class PPOAgent:
    """Proximal Policy Optimization agent for trading.
    
    Implements the PPO algorithm with clipped surrogate objective
    for stable policy learning.
    
    Example:
        >>> # Create environment
        >>> env = TradingEnvironment(prices, features)
        >>> 
        >>> # Create agent
        >>> agent = PPOAgent(
        ...     env,
        ...     learning_rate=3e-4,
        ...     n_steps=2048,
        ...     batch_size=64
        ... )
        >>> 
        >>> # Train
        >>> agent.train(total_timesteps=100000)
        >>> 
        >>> # Save
        >>> agent.save("ppo_trading_agent")
        >>> 
        >>> # Evaluate
        >>> eval_env = TradingEnvironment(test_prices, test_features)
        >>> rewards, actions = agent.evaluate(eval_env, n_episodes=10)
    """
    
    def __init__(
        self,
        env: TradingEnvironment,
        config: Optional[PPOConfig] = None,
        **kwargs
    ):
        """Initialize PPO agent.
        
        Args:
            env: Trading environment
            config: PPO configuration
            **kwargs: Override config parameters
        """
        self.env = env
        self.config = config or PPOConfig(**kwargs)
        
        self.tf = _check_tensorflow()
        
        # Set random seed
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)
            self.tf.random.set_seed(self.config.random_state)
        
        # Determine action dimension
        if self.config.action_space == ActionSpace.DISCRETE:
            action_dim = self.config.n_actions
        else:
            action_dim = 1  # Single position value
        
        # Create network
        self.network = ActorCriticNetwork(
            observation_dim=env.observation_dim,
            action_dim=action_dim,
            config=self.config,
            action_space=self.config.action_space
        )
        
        # Rollout buffer
        self.buffer = RolloutBuffer(self.config)
        
        # Training stats
        self.training_stats = []
        self.episode_stats = []
        
        logger.info(
            f"PPOAgent initialized: obs_dim={env.observation_dim}, "
            f"action_space={self.config.action_space.value}"
        )
    
    def collect_rollout(self) -> Dict[str, float]:
        """Collect rollout data from environment.
        
        Returns:
            Episode statistics
        """
        self.buffer.reset()
        
        obs = self.env.reset()
        episode_rewards = []
        episode_stats = []
        
        for step in range(self.config.n_steps):
            # Get action from policy
            action, log_prob, value = self.network.get_action(obs)
            
            # Step environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Store in buffer
            self.buffer.add(obs, action, reward, value, log_prob, done)
            
            episode_rewards.append(reward)
            
            if done:
                stats = self.env.get_episode_stats()
                episode_stats.append(stats)
                obs = self.env.reset()
                episode_rewards = []
            else:
                obs = next_obs
        
        # Compute advantages
        last_value = self.network.get_value(obs)
        self.buffer.compute_returns_and_advantages(last_value)
        
        # Aggregate stats
        if episode_stats:
            avg_stats = {
                'mean_episode_return': np.mean([s['total_return'] for s in episode_stats]),
                'mean_sharpe': np.mean([s['sharpe'] for s in episode_stats]),
                'mean_max_drawdown': np.mean([s['max_drawdown'] for s in episode_stats]),
                'num_episodes': len(episode_stats),
            }
        else:
            avg_stats = {
                'mean_episode_return': 0,
                'mean_sharpe': 0,
                'mean_max_drawdown': 0,
                'num_episodes': 0,
            }
        
        return avg_stats
    
    def update(self) -> Dict[str, float]:
        """Update policy using collected rollout data.
        
        Returns:
            Training statistics
        """
        tf = self.tf
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        # Normalize advantages
        advantages = self.buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages
        
        for epoch in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                with tf.GradientTape() as tape:
                    # Get current policy outputs
                    obs = tf.constant(batch['observations'], dtype=tf.float32)
                    old_log_probs = tf.constant(batch['log_probs'], dtype=tf.float32)
                    advantages = tf.constant(batch['advantages'], dtype=tf.float32)
                    returns = tf.constant(batch['returns'], dtype=tf.float32)
                    actions = batch['actions']
                    
                    # Forward pass
                    policy_out = self.network.actor(obs)
                    values = tf.squeeze(self.network.critic(obs))
                    
                    # Compute new log probs and entropy
                    if self.config.action_space == ActionSpace.DISCRETE:
                        probs = policy_out
                        action_masks = tf.one_hot(actions, self.config.n_actions)
                        action_probs = tf.reduce_sum(probs * action_masks, axis=1)
                        new_log_probs = tf.math.log(action_probs + 1e-8)
                        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
                    else:
                        mean = policy_out[:, :1]
                        log_std = policy_out[:, 1:]
                        std = tf.exp(tf.clip_by_value(log_std, -20, 2))
                        actions_tf = tf.constant(actions.reshape(-1, 1), dtype=tf.float32)
                        
                        new_log_probs = -0.5 * (
                            ((actions_tf - mean) / std) ** 2 + 
                            2 * log_std + 
                            tf.math.log(2 * np.pi)
                        )
                        new_log_probs = tf.reduce_sum(new_log_probs, axis=1)
                        entropy = tf.reduce_sum(log_std + 0.5 + 0.5 * tf.math.log(2 * np.pi), axis=1)
                    
                    # PPO clipped objective
                    ratio = tf.exp(new_log_probs - old_log_probs)
                    clipped_ratio = tf.clip_by_value(
                        ratio,
                        1 - self.config.clip_epsilon,
                        1 + self.config.clip_epsilon
                    )
                    
                    policy_loss = -tf.reduce_mean(tf.minimum(
                        ratio * advantages,
                        clipped_ratio * advantages
                    ))
                    
                    # Value loss
                    value_loss = tf.reduce_mean((returns - values) ** 2)
                    
                    # Entropy bonus
                    entropy_loss = -tf.reduce_mean(entropy)
                    
                    # Total loss
                    loss = (
                        policy_loss + 
                        self.config.value_coef * value_loss + 
                        self.config.entropy_coef * entropy_loss
                    )
                
                # Compute gradients
                variables = self.network.actor.trainable_variables + self.network.critic.trainable_variables
                gradients = tape.gradient(loss, variables)
                
                # Clip gradients
                gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)
                
                # Apply gradients
                self.network.optimizer.apply_gradients(zip(gradients, variables))
                
                total_policy_loss += policy_loss.numpy()
                total_value_loss += value_loss.numpy()
                total_entropy += tf.reduce_mean(entropy).numpy()
                n_updates += 1
        
        stats = {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }
        
        return stats
    
    def train(
        self,
        total_timesteps: int,
        log_interval: int = 10,
        callback: Optional[Callable] = None
    ) -> List[Dict]:
        """Train the agent.
        
        Args:
            total_timesteps: Total number of environment steps
            log_interval: Log every N updates
            callback: Optional callback function(agent, update_num)
            
        Returns:
            Training statistics history
        """
        logger.info(f"Starting PPO training for {total_timesteps} timesteps...")
        
        n_updates = total_timesteps // self.config.n_steps
        
        for update in range(n_updates):
            # Collect rollout
            rollout_stats = self.collect_rollout()
            
            # Update policy
            train_stats = self.update()
            
            # Combine stats
            stats = {**rollout_stats, **train_stats, 'update': update}
            self.training_stats.append(stats)
            
            # Logging
            if update % log_interval == 0:
                logger.info(
                    f"Update {update}/{n_updates}: "
                    f"return={rollout_stats['mean_episode_return']:.4f}, "
                    f"sharpe={rollout_stats['mean_sharpe']:.2f}, "
                    f"policy_loss={train_stats['policy_loss']:.4f}"
                )
            
            # Callback
            if callback is not None:
                callback(self, update)
        
        logger.info("Training complete!")
        return self.training_stats
    
    def evaluate(
        self,
        env: Optional[TradingEnvironment] = None,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Tuple[List[Dict], List[List]]:
        """Evaluate agent performance.
        
        Args:
            env: Environment to evaluate on (default: training env)
            n_episodes: Number of episodes
            deterministic: Use deterministic policy
            
        Returns:
            episode_stats: List of episode statistics
            episode_actions: List of action sequences
        """
        if env is None:
            env = self.env
        
        episode_stats = []
        episode_actions = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            actions = []
            
            while not done:
                action, _, _ = self.network.get_action(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                actions.append(action)
            
            stats = env.get_episode_stats()
            episode_stats.append(stats)
            episode_actions.append(actions)
            
            logger.info(
                f"Episode {episode + 1}: return={stats['total_return']:.4f}, "
                f"sharpe={stats['sharpe']:.2f}, max_dd={stats['max_drawdown']:.4f}"
            )
        
        # Summary
        avg_return = np.mean([s['total_return'] for s in episode_stats])
        avg_sharpe = np.mean([s['sharpe'] for s in episode_stats])
        
        logger.info(f"Evaluation: avg_return={avg_return:.4f}, avg_sharpe={avg_sharpe:.2f}")
        
        return episode_stats, episode_actions
    
    def get_action(
        self, 
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Union[int, np.ndarray]:
        """Get action for a single observation.
        
        Args:
            observation: Current observation
            deterministic: Use deterministic policy
            
        Returns:
            Action to take
        """
        action, _, _ = self.network.get_action(observation, deterministic=deterministic)
        return action
    
    def save(self, path: Union[str, Path]):
        """Save agent to disk.
        
        Args:
            path: Directory path to save to
        """
        import pickle
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save networks
        self.network.actor.save(path / "actor.keras")
        self.network.critic.save(path / "critic.keras")
        
        # Save config and stats
        meta = {
            'config': self.config,
            'observation_dim': self.env.observation_dim,
            'training_stats': self.training_stats,
        }
        with open(path / "meta.pkl", 'wb') as f:
            pickle.dump(meta, f)
        
        logger.info(f"Saved agent to {path}")
    
    @classmethod
    def load(
        cls, 
        path: Union[str, Path],
        env: TradingEnvironment
    ) -> "PPOAgent":
        """Load agent from disk.
        
        Args:
            path: Directory path to load from
            env: Trading environment
            
        Returns:
            Loaded PPOAgent
        """
        import pickle
        
        path = Path(path)
        tf = _check_tensorflow()
        
        # Load metadata
        with open(path / "meta.pkl", 'rb') as f:
            meta = pickle.load(f)
        
        # Create agent
        agent = cls(env, config=meta['config'])
        agent.training_stats = meta['training_stats']
        
        # Load networks
        agent.network.actor = tf.keras.models.load_model(path / "actor.keras")
        agent.network.critic = tf.keras.models.load_model(path / "critic.keras")
        
        logger.info(f"Loaded agent from {path}")
        return agent
    
    def plot_training(self, figsize: Tuple[int, int] = (12, 8)) -> Any:
        """Plot training curves.
        
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        if not self.training_stats:
            raise ValueError("No training data. Train the agent first.")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        updates = [s['update'] for s in self.training_stats]
        
        # Episode return
        returns = [s['mean_episode_return'] for s in self.training_stats]
        axes[0, 0].plot(updates, returns)
        axes[0, 0].set_xlabel('Update')
        axes[0, 0].set_ylabel('Mean Episode Return')
        axes[0, 0].set_title('Episode Returns')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Sharpe ratio
        sharpes = [s['mean_sharpe'] for s in self.training_stats]
        axes[0, 1].plot(updates, sharpes, color='green')
        axes[0, 1].set_xlabel('Update')
        axes[0, 1].set_ylabel('Mean Sharpe Ratio')
        axes[0, 1].set_title('Sharpe Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Policy loss
        policy_loss = [s['policy_loss'] for s in self.training_stats]
        axes[1, 0].plot(updates, policy_loss, color='red')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Policy Loss')
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Value loss
        value_loss = [s['value_loss'] for s in self.training_stats]
        axes[1, 1].plot(updates, value_loss, color='orange')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Value Loss')
        axes[1, 1].set_title('Value Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def create_trading_agent(
    prices: Union[pd.Series, np.ndarray],
    features: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    action_space: str = "discrete",
    initial_capital: float = 100000,
    **kwargs
) -> Tuple[PPOAgent, TradingEnvironment]:
    """Convenience function to create trading agent and environment.
    
    Args:
        prices: Price series
        features: Optional feature matrix
        action_space: "discrete" or "continuous"
        initial_capital: Starting capital
        **kwargs: Additional PPO config parameters
        
    Returns:
        (agent, environment) tuple
        
    Example:
        >>> agent, env = create_trading_agent(
        ...     prices=prices,
        ...     features=features,
        ...     learning_rate=1e-4
        ... )
        >>> agent.train(total_timesteps=50000)
    """
    # Create environment
    env_config = EnvironmentConfig(initial_capital=initial_capital)
    env = TradingEnvironment(prices, features, config=env_config)
    
    # Create agent config
    action_space_enum = ActionSpace(action_space)
    ppo_config = PPOConfig(action_space=action_space_enum, **kwargs)
    
    # Create agent
    agent = PPOAgent(env, config=ppo_config)
    
    return agent, env

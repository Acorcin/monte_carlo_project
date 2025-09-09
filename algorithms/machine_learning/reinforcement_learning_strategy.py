"""
Reinforcement Learning Trading Strategy

An advanced trading strategy using reinforcement learning techniques
for optimal trading decision making in financial markets.

Features:
- Q-Learning algorithm for action selection
- State representation with technical indicators
- Reward function based on returns and risk
- Experience replay for improved learning
- Epsilon-greedy exploration strategy

Author: Advanced ML Trading Suite
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import sys
import os
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from collections import defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

# Add the algorithms directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_algorithm import TradingAlgorithm

class ReinforcementLearningStrategy(TradingAlgorithm):
    """
    Advanced reinforcement learning trading strategy.

    Uses Q-Learning algorithm to learn optimal trading decisions
    based on market states and reward signals.
    """

    def __init__(self,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 num_episodes: int = 1000,
                 state_bins: int = 10,
                 reward_function: str = 'returns_risk',
                 experience_replay_size: int = 10000,
                 batch_size: int = 32):
        """
        Initialize Reinforcement Learning Strategy.

        Args:
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration rate decay factor
            num_episodes: Number of training episodes
            state_bins: Number of bins for state discretization
            reward_function: Type of reward function ('returns', 'returns_risk', 'sharpe')
            experience_replay_size: Size of experience replay buffer
            batch_size: Batch size for experience replay
        """
        parameters = {
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay,
            'num_episodes': num_episodes,
            'state_bins': state_bins,
            'reward_function': reward_function,
            'experience_replay_size': experience_replay_size,
            'batch_size': batch_size
        }

        super().__init__(
            name="Reinforcement Learning Strategy",
            description="Q-Learning based strategy for optimal trading decisions",
            parameters=parameters
        )

        # Initialize Q-learning components
        self.q_table = defaultdict(lambda: np.zeros(3))  # 3 actions: sell, hold, buy
        self.epsilon = self.parameters['epsilon_start']
        self.episode_count = 0

        # Experience replay buffer
        self.experience_buffer = []
        self.buffer_size = self.parameters['experience_replay_size']

        # State discretization
        self.state_discretizer = None
        self.scaler = StandardScaler()

        # Performance tracking
        self.training_rewards = []
        self.epsilon_history = []
        self.q_table_size_history = []

        self.is_trained = False

    def _create_state_features(self, data: pd.DataFrame, idx: int) -> np.ndarray:
        """Create state representation from market data."""
        # Get recent data window
        window_size = 20
        start_idx = max(0, idx - window_size + 1)
        window_data = data.iloc[start_idx:idx+1]

        if len(window_data) < 5:
            return np.zeros(10)  # Return zero state if insufficient data

        # Price-based features
        close_prices = window_data['Close'].values
        returns = np.diff(close_prices) / close_prices[:-1]

        # Technical indicators
        sma_short = close_prices[-10:].mean() if len(close_prices) >= 10 else close_prices.mean()
        sma_long = close_prices[-20:].mean() if len(close_prices) >= 20 else close_prices.mean()
        sma_ratio = sma_short / sma_long if sma_long != 0 else 1.0

        # Volatility
        volatility = np.std(returns) if len(returns) > 0 else 0.0

        # RSI approximation
        gains = returns[returns > 0].sum() if len(returns[returns > 0]) > 0 else 0
        losses = abs(returns[returns < 0]).sum() if len(returns[returns < 0]) > 0 else 0
        rsi = 100 - (100 / (1 + (gains / losses) if losses != 0 else 100)) if (gains + losses) > 0 else 50

        # Volume trend
        volume = window_data['Volume'].values
        volume_trend = np.mean(volume[-5:]) / np.mean(volume[:5]) if len(volume) >= 10 and np.mean(volume[:5]) != 0 else 1.0

        # Price momentum
        momentum = (close_prices[-1] - close_prices[0]) / close_prices[0] if len(close_prices) > 1 else 0.0

        # Bollinger Band position approximation
        bb_middle = np.mean(close_prices)
        bb_std = np.std(close_prices)
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        bb_position = (close_prices[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0.5

        # Create state vector
        state = np.array([
            sma_ratio,           # Trend indicator
            volatility,          # Volatility measure
            rsi / 100.0,        # Normalized RSI
            volume_trend,        # Volume trend
            momentum,            # Price momentum
            bb_position,         # Bollinger Band position
            returns[-1] if len(returns) > 0 else 0,  # Last return
            np.mean(returns) if len(returns) > 0 else 0,  # Average return
            np.std(returns) if len(returns) > 1 else 0,   # Return volatility
            len(returns) / window_size  # Data completeness
        ])

        return state

    def _discretize_state(self, state: np.ndarray) -> str:
        """Discretize continuous state into discrete bins for Q-table."""
        if self.state_discretizer is None:
            # Initialize discretizer with training data
            self.state_discretizer = KBinsDiscretizer(
                n_bins=self.parameters['state_bins'],
                encode='ordinal',
                strategy='uniform'
            )
            # Fit on the current state (will be updated during training)
            self.state_discretizer.fit(state.reshape(1, -1))

        try:
            discretized = self.state_discretizer.transform(state.reshape(1, -1))[0]
            state_key = tuple(discretized.astype(int))
            return str(state_key)
        except:
            # Fallback if discretization fails
            return str(tuple(np.round(state, 2)))

    def _calculate_reward(self, action: int, returns: float, risk: float) -> float:
        """Calculate reward based on action and market outcome."""
        if self.parameters['reward_function'] == 'returns':
            # Simple returns-based reward
            reward = returns * 100  # Scale up for better learning

        elif self.parameters['reward_function'] == 'returns_risk':
            # Returns adjusted for risk
            if risk > 0:
                risk_adjusted_return = returns / risk
                reward = risk_adjusted_return * 100
            else:
                reward = returns * 100

        elif self.parameters['reward_function'] == 'sharpe':
            # Sharpe ratio approximation
            if risk > 0:
                sharpe = returns / risk
                reward = sharpe * 10  # Scale for learning
            else:
                reward = returns * 10

        else:
            reward = returns * 100

        # Action-based adjustments
        if action == 0:  # Hold
            reward *= 0.9  # Slight penalty for inaction
        elif action == 1 and returns > 0:  # Buy and market goes up
            reward *= 1.2  # Bonus for correct bullish call
        elif action == 2 and returns < 0:  # Sell and market goes down
            reward *= 1.2  # Bonus for correct bearish call

        return reward

    def _choose_action(self, state_key: str) -> int:
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, 2)
        else:
            # Exploitation: best action from Q-table
            q_values = self.q_table[state_key]
            return np.argmax(q_values)

    def _update_q_value(self, state_key: str, action: int, reward: float, next_state_key: str):
        """Update Q-value using Q-learning update rule."""
        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])

        # Q-learning update
        new_q = current_q + self.parameters['learning_rate'] * (
            reward + self.parameters['discount_factor'] * next_max_q - current_q
        )

        self.q_table[state_key][action] = new_q

    def _add_experience(self, experience: Tuple[str, int, float, str]):
        """Add experience to replay buffer."""
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)

    def _sample_experience(self) -> List[Tuple[str, int, float, str]]:
        """Sample random batch from experience replay buffer."""
        batch_size = min(self.parameters['batch_size'], len(self.experience_buffer))
        return random.sample(self.experience_buffer, batch_size)

    def train_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the reinforcement learning agent."""
        try:
            print("🧠 Training Reinforcement Learning agent...")

            if len(data) < 50:
                return {'status': 'error', 'message': 'Insufficient data for training'}

            # Initialize state discretizer with full dataset
            sample_states = []
            for idx in range(20, len(data)):
                state = self._create_state_features(data, idx)
                sample_states.append(state)

            if sample_states:
                sample_states = np.array(sample_states)
                self.state_discretizer = KBinsDiscretizer(
                    n_bins=self.parameters['state_bins'],
                    encode='ordinal',
                    strategy='uniform'
                )
                self.state_discretizer.fit(sample_states)

            # Training loop
            self.epsilon = self.parameters['epsilon_start']

            for episode in range(self.parameters['num_episodes']):
                total_reward = 0
                position = 0  # 0: no position, 1: long, -1: short

                # Start from a random point in the data
                start_idx = random.randint(20, len(data) - 20)
                current_idx = start_idx

                # Episode loop
                while current_idx < len(data) - 1:
                    # Get current state
                    current_state = self._create_state_features(data, current_idx)
                    state_key = self._discretize_state(current_state)

                    # Choose action
                    action = self._choose_action(state_key)

                    # Execute action (update position)
                    old_position = position
                    if action == 0:  # Hold
                        position = position
                    elif action == 1:  # Buy
                        position = 1
                    elif action == 2:  # Sell
                        position = -1

                    # Move to next state
                    current_idx += 1
                    next_state = self._create_state_features(data, current_idx)
                    next_state_key = self._discretize_state(next_state)

                    # Calculate reward
                    current_return = data['Close'].pct_change().iloc[current_idx]
                    risk = data['Close'].pct_change().rolling(20).std().iloc[current_idx]

                    # Position-based reward
                    if old_position == 1:  # Was long
                        reward = self._calculate_reward(1, current_return, risk)
                    elif old_position == -1:  # Was short
                        reward = self._calculate_reward(2, -current_return, risk)  # Inverse for shorts
                    else:  # Was flat
                        reward = self._calculate_reward(0, 0, risk) * 0.1  # Small reward for holding cash

                    total_reward += reward

                    # Update Q-value
                    self._update_q_value(state_key, action, reward, next_state_key)

                    # Add to experience replay
                    experience = (state_key, action, reward, next_state_key)
                    self._add_experience(experience)

                    # Experience replay learning
                    if len(self.experience_buffer) >= self.parameters['batch_size']:
                        replay_batch = self._sample_experience()
                        for replay_state, replay_action, replay_reward, replay_next_state in replay_batch:
                            self._update_q_value(replay_state, replay_action, replay_reward, replay_next_state)

                # Decay epsilon
                self.epsilon = max(
                    self.parameters['epsilon_end'],
                    self.epsilon * self.parameters['epsilon_decay']
                )

                # Track progress
                self.training_rewards.append(total_reward)
                self.epsilon_history.append(self.epsilon)
                self.q_table_size_history.append(len(self.q_table))

                # Progress reporting
                if (episode + 1) % 100 == 0:
                    avg_reward = np.mean(self.training_rewards[-10:])
                    print(f"Episode {episode + 1}/{self.parameters['num_episodes']}: "
                          f"Avg Reward = {avg_reward:.2f}, "
                          f"Epsilon = {self.epsilon:.3f}, "
                          f"Q-table size = {len(self.q_table)}")

            self.is_trained = True

            # Final training metrics
            final_avg_reward = np.mean(self.training_rewards[-50:]) if len(self.training_rewards) >= 50 else np.mean(self.training_rewards)
            q_table_size = len(self.q_table)

            print("✅ Reinforcement Learning agent trained successfully")
            print(".2f")
            print(f"   Q-table size: {q_table_size}")

            return {
                'status': 'success',
                'final_avg_reward': final_avg_reward,
                'q_table_size': q_table_size,
                'total_episodes': self.parameters['num_episodes'],
                'final_epsilon': self.epsilon
            }

        except Exception as e:
            print(f"❌ RL training failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using the trained RL agent.

        Args:
            data: OHLCV data with columns ['Open', 'High', 'Low', 'Close', 'Volume']

        Returns:
            pd.Series: Trading signals (1=buy, -1=sell, 0=hold)
        """
        try:
            if not self.is_trained:
                training_result = self.train_model(data)
                if training_result['status'] != 'success':
                    return self._generate_fallback_signals(data)

            signals = pd.Series(0, index=data.index)

            for idx in range(20, len(data)):
                # Get current state
                current_state = self._create_state_features(data, idx)
                state_key = self._discretize_state(current_state)

                # Get action from Q-table (pure exploitation, no exploration)
                q_values = self.q_table[state_key]
                action = np.argmax(q_values)

                # Convert action to signal
                if action == 1:  # Buy
                    signals.iloc[idx] = 1
                elif action == 2:  # Sell
                    signals.iloc[idx] = -1
                # Action 0 (Hold) remains as 0

            return signals

        except Exception as e:
            print(f"❌ Signal generation failed: {e}")
            return self._generate_fallback_signals(data)

    def _generate_fallback_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate simple trend-following signals when RL is not available."""
        signals = pd.Series(0, index=data.index)

        # Simple trend-following strategy as fallback
        sma_short = data['Close'].rolling(10).mean()
        sma_long = data['Close'].rolling(30).mean()

        # Generate signals based on moving average crossover
        signals[(sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))] = 1   # Golden cross
        signals[(sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))] = -1  # Death cross

        return signals

    def get_algorithm_type(self) -> str:
        """Return algorithm category."""
        return "machine_learning"

    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Define parameter configuration for optimization."""
        return {
            'learning_rate': {
                'type': 'float',
                'default': 0.1,
                'min': 0.01,
                'max': 0.5,
                'description': 'Learning rate for Q-value updates'
            },
            'discount_factor': {
                'type': 'float',
                'default': 0.95,
                'min': 0.8,
                'max': 0.99,
                'description': 'Discount factor for future rewards'
            },
            'epsilon_start': {
                'type': 'float',
                'default': 1.0,
                'min': 0.5,
                'max': 1.0,
                'description': 'Initial exploration rate'
            },
            'epsilon_end': {
                'type': 'float',
                'default': 0.01,
                'min': 0.001,
                'max': 0.1,
                'description': 'Final exploration rate'
            },
            'prediction_horizon': {
                'type': 'int',
                'default': 5,
                'min': 1,
                'max': 20,
                'description': 'Days ahead to predict price movements'
            },
            'confidence_threshold': {
                'type': 'float',
                'default': 0.6,
                'min': 0.4,
                'max': 0.8,
                'description': 'Minimum confidence required for signals'
            },
            'num_episodes': {
                'type': 'int',
                'default': 1000,
                'min': 100,
                'max': 5000,
                'description': 'Number of training episodes'
            }
        }

    def get_strategy_description(self) -> str:
        """Get detailed strategy description."""
        return """
        🚀 REINFORCEMENT LEARNING TRADING STRATEGY 🚀

        This cutting-edge strategy uses reinforcement learning techniques to learn
        optimal trading decisions through interaction with the market environment.

        🔬 ALGORITHM ARCHITECTURE:
        • Q-Learning algorithm for action-value estimation
        • Epsilon-greedy exploration strategy
        • Experience replay for improved sample efficiency
        • State discretization for continuous market data
        • Multiple reward function options

        📊 STATE REPRESENTATION:
        • Technical indicators (SMA, RSI, Bollinger Bands)
        • Price momentum and volatility measures
        • Volume trends and patterns
        • Market microstructure features
        • Risk-adjusted state variables

        🎯 ACTIONS & REWARDS:
        • Actions: Buy, Sell, Hold
        • Reward functions: Returns, Risk-adjusted returns, Sharpe ratio
        • Position-based reward scaling
        • Risk penalty for excessive volatility

        💡 ADVANTAGES:
        ✅ Learns optimal strategies through market interaction
        ✅ Adapts to changing market conditions
        ✅ Balances exploration vs exploitation
        ✅ Considers risk in decision making
        ✅ No assumptions about market distribution

        ⚠️ REQUIREMENTS:
        • Significant computational resources for training
        • Large historical dataset for learning
        • Careful hyperparameter tuning
        • Regular model retraining recommended

        🔧 CONFIGURATION TIPS:
        • Higher learning_rate for faster adaptation
        • Lower discount_factor for short-term focus
        • More episodes improve learning but increase training time
        • Adjust epsilon parameters for exploration balance
        • Choose reward_function based on risk preferences
        """

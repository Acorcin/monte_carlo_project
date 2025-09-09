"""
Transformer Trading Strategy

An advanced trading strategy using Transformer architecture for financial time series
prediction and trading signal generation.

Features:
- Multi-head self-attention mechanism
- Positional encoding for temporal patterns
- Technical indicator integration
- Multi-scale feature extraction
- Risk-adjusted signal generation

Author: Advanced ML Trading Suite
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import sys
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports with fallbacks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Dense, Dropout, LayerNormalization,
        MultiHeadAttention, GlobalAveragePooling1D,
        Add, Concatenate, Conv1D, MaxPooling1D
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - Transformer strategy will use simplified version")
    # Define dummy classes
    Model = None
    Input = None

# Add the algorithms directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_algorithm import TradingAlgorithm

class TransformerTradingStrategy(TradingAlgorithm):
    """
    Advanced Transformer-based trading strategy.

    Uses self-attention mechanism to identify complex patterns in financial time series
    and generate high-confidence trading signals.
    """

    def __init__(self,
                 sequence_length: int = 60,
                 num_heads: int = 8,
                 key_dim: int = 64,
                 num_transformer_blocks: int = 4,
                 ff_dim: int = 128,
                 dropout_rate: float = 0.1,
                 learning_rate: float = 0.001,
                 prediction_horizon: int = 3,
                 confidence_threshold: float = 0.65,
                 use_conv_stem: bool = True):
        """
        Initialize Transformer Trading Strategy.

        Args:
            sequence_length: Number of time steps to look back
            num_heads: Number of attention heads
            key_dim: Dimension of attention keys/queries
            num_transformer_blocks: Number of transformer encoder blocks
            ff_dim: Feed-forward network dimension
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimization
            prediction_horizon: Days ahead to predict
            confidence_threshold: Minimum confidence for signals
            use_conv_stem: Whether to use convolutional stem for feature extraction
        """
        parameters = {
            'sequence_length': sequence_length,
            'num_heads': num_heads,
            'key_dim': key_dim,
            'num_transformer_blocks': num_transformer_blocks,
            'ff_dim': ff_dim,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'prediction_horizon': prediction_horizon,
            'confidence_threshold': confidence_threshold,
            'use_conv_stem': use_conv_stem
        }

        super().__init__(
            name="Transformer Trading Strategy",
            description="Attention-based deep learning strategy using Transformer architecture",
            parameters=parameters
        )

        # Initialize components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.is_trained = False

        # Performance tracking
        self.training_history = []
        self.attention_weights = None

        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow not available - using simplified Transformer implementation")

    def _create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical features for the Transformer model."""
        df = data.copy()

        # Basic price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Multi-timeframe moving averages
        for period in [5, 10, 15, 20, 30, 50, 100]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()

        # Volatility measures
        for period in [10, 20, 30]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'high_low_range_{period}'] = (df['High'] - df['Low']).rolling(period).mean()

        # Momentum indicators
        for period in [5, 10, 14, 20]:
            df[f'ROC_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)
            df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1

        # RSI with multiple periods
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        for period in [7, 14, 21]:
            df[f'RSI_{period}'] = calculate_rsi(df['Close'], period)

        # MACD variants
        for fast, slow, signal in [(8, 21, 5), (12, 26, 9), (5, 35, 5)]:
            exp1 = df['Close'].ewm(span=fast).mean()
            exp2 = df['Close'].ewm(span=slow).mean()
            macd = exp1 - exp2
            df[f'MACD_{fast}_{slow}'] = macd
            df[f'MACD_signal_{fast}_{slow}_{signal}'] = macd.ewm(span=signal).mean()
            df[f'MACD_hist_{fast}_{slow}_{signal}'] = macd - df[f'MACD_signal_{fast}_{slow}_{signal}']

        # Bollinger Bands with different periods
        for period in [10, 20, 30]:
            middle = df['Close'].rolling(period).mean()
            std = df['Close'].rolling(period).std()
            df[f'BB_middle_{period}'] = middle
            df[f'BB_upper_{period}'] = middle + 2 * std
            df[f'BB_lower_{period}'] = middle - 2 * std
            df[f'BB_position_{period}'] = (df['Close'] - df[f'BB_lower_{period}']) / (df[f'BB_upper_{period}'] - df[f'BB_lower_{period}'])
            df[f'BB_width_{period}'] = (df[f'BB_upper_{period}'] - df[f'BB_lower_{period}']) / middle

        # Volume indicators
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        df['volume_trend'] = df['Volume'].pct_change(5)

        # Price patterns
        df['price_to_sma_20'] = df['Close'] / df['SMA_20']
        df['price_to_sma_50'] = df['Close'] / df['SMA_50']

        # Statistical measures
        for period in [10, 20]:
            df[f'skewness_{period}'] = df['returns'].rolling(period).skew()
            df[f'kurtosis_{period}'] = df['returns'].rolling(period).kurt()

        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

        return df

    def _positional_encoding(self, position: int, d_model: int) -> np.ndarray:
        """Generate positional encoding for temporal patterns."""
        angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
        angles = position * angle_rates

        # Apply sin to even indices, cos to odd indices
        pos_encoding = np.zeros(d_model)
        pos_encoding[0::2] = np.sin(angles[0::2])
        pos_encoding[1::2] = np.cos(angles[1::2])

        return pos_encoding

    def _create_positional_embeddings(self, seq_length: int, d_model: int) -> np.ndarray:
        """Create positional embeddings for the sequence."""
        positional_encodings = np.zeros((seq_length, d_model))

        for pos in range(seq_length):
            positional_encodings[pos] = self._positional_encoding(pos, d_model)

        return positional_encodings

    def _transformer_encoder(self, inputs, num_heads: int, key_dim: int, ff_dim: int, dropout_rate: float):
        """Create a transformer encoder block."""
        # Multi-head attention
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
        attention_output = Dropout(dropout_rate)(attention_output)
        attention_output = Add()([inputs, attention_output])  # Residual connection
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

        # Feed-forward network
        ffn_output = Dense(ff_dim, activation='relu')(attention_output)
        ffn_output = Dense(key_dim)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        ffn_output = Add()([attention_output, ffn_output])  # Residual connection
        ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output)

        return ffn_output

    def _build_transformer_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build the Transformer model architecture."""
        if not TENSORFLOW_AVAILABLE:
            return None

        inputs = Input(shape=input_shape)

        # Convolutional stem for feature extraction (optional)
        if self.parameters['use_conv_stem']:
            x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
            x = MaxPooling1D(pool_size=2)(x)
            x = Conv1D(128, kernel_size=3, activation='relu')(x)
            x = MaxPooling1D(pool_size=2)(x)
        else:
            x = inputs

        # Add positional encoding
        positional_embeddings = self._create_positional_embeddings(
            x.shape[1], x.shape[2]
        )
        positional_embeddings = tf.constant(positional_embeddings, dtype=tf.float32)
        positional_embeddings = tf.expand_dims(positional_embeddings, axis=0)
        x = Add()([x, positional_embeddings])

        # Transformer encoder blocks
        for _ in range(self.parameters['num_transformer_blocks']):
            x = self._transformer_encoder(
                x,
                self.parameters['num_heads'],
                self.parameters['key_dim'],
                self.parameters['ff_dim'],
                self.parameters['dropout_rate']
            )

        # Global pooling and dense layers
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(self.parameters['dropout_rate'])(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(self.parameters['dropout_rate'])(x)

        # Multi-output: direction and confidence
        direction_output = Dense(1, activation='sigmoid', name='direction')(x)
        confidence_output = Dense(1, activation='sigmoid', name='confidence')(x)

        model = Model(inputs=inputs, outputs=[direction_output, confidence_output])

        optimizer = Adam(learning_rate=self.parameters['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss={'direction': 'binary_crossentropy', 'confidence': 'mse'},
            metrics={'direction': 'accuracy', 'confidence': 'mae'}
        )

        return model

    def _prepare_sequences(self, data: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for Transformer training."""
        sequences = []
        targets = []

        feature_cols = [col for col in data.columns if col not in ['target', 'future_price']]
        scaled_features = self.feature_scaler.fit_transform(data[feature_cols])

        for i in range(len(data) - self.parameters['sequence_length']):
            sequences.append(scaled_features[i:i + self.parameters['sequence_length']])
            targets.append(target.iloc[i + self.parameters['sequence_length']])

        return np.array(sequences), np.array(targets)

    def _create_target(self, prices: pd.Series, horizon: int = 3) -> pd.Series:
        """Create prediction target with confidence weighting."""
        future_prices = prices.shift(-horizon)
        returns = (future_prices - prices) / prices

        # Create binary target with confidence based on return magnitude
        direction = (returns > 0).astype(int)
        confidence = np.abs(returns).clip(0, 0.1) / 0.1  # Scale to 0-1

        # Combine direction and confidence
        target = direction * confidence + (1 - direction) * (1 - confidence)

        return target

    def train_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the Transformer model on historical data."""
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow not available - skipping model training")
            return {'status': 'skipped', 'message': 'TensorFlow not available'}

        try:
            print("🧠 Training Transformer model...")

            # Create comprehensive features
            feature_data = self._create_advanced_features(data)

            # Create target with confidence weighting
            target = self._create_target(data['Close'], self.parameters['prediction_horizon'])

            # Remove NaN values
            valid_idx = ~(feature_data.isna().any(axis=1) | target.isna())
            feature_data = feature_data[valid_idx]
            target = target[valid_idx]

            if len(feature_data) < self.parameters['sequence_length'] + 10:
                return {'status': 'error', 'message': 'Insufficient data for training'}

            # Prepare sequences
            X, y = self._prepare_sequences(feature_data, target)

            # Build model
            self.model = self._build_transformer_model((X.shape[1], X.shape[2]))

            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

            # Split for validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Train model
            history = self.model.fit(
                X_train, [y_train, y_train],  # Same target for both outputs
                validation_data=(X_val, [y_val, y_val]),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping, lr_reducer],
                verbose=0
            )

            self.is_trained = True
            self.training_history = history.history

            print("✅ Transformer model trained successfully")
            print(".3f")
            print(".3f")

            return {
                'status': 'success',
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'epochs_trained': len(history.history['loss'])
            }

        except Exception as e:
            print(f"❌ Transformer training failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using the trained Transformer model.

        Args:
            data: OHLCV data with columns ['Open', 'High', 'Low', 'Close', 'Volume']

        Returns:
            pd.Series: Trading signals (1=buy, -1=sell, 0=hold)
        """
        try:
            if not self.is_trained or self.model is None:
                if TENSORFLOW_AVAILABLE:
                    # Train model if not trained
                    training_result = self.train_model(data)
                    if training_result['status'] != 'success':
                        return self._generate_fallback_signals(data)
                else:
                    return self._generate_fallback_signals(data)

            # Create features
            feature_data = self._create_advanced_features(data)

            # Prepare prediction sequences
            feature_cols = [col for col in feature_data.columns if col not in ['target', 'future_price']]
            scaled_features = self.feature_scaler.transform(feature_data[feature_cols])

            signals = pd.Series(0, index=data.index)

            # Generate predictions for each time step
            for i in range(self.parameters['sequence_length'], len(feature_data)):
                sequence = scaled_features[i - self.parameters['sequence_length']:i]
                sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])

                # Get predictions
                direction_pred, confidence_pred = self.model.predict(sequence, verbose=0)

                direction = direction_pred[0][0]  # Price direction (0-1)
                confidence = confidence_pred[0][0]  # Prediction confidence (0-1)

                # Generate signal based on prediction and confidence
                if confidence > self.parameters['confidence_threshold']:
                    if direction > 0.6:  # Strong bullish prediction
                        signals.iloc[i] = 1
                    elif direction < 0.4:  # Strong bearish prediction
                        signals.iloc[i] = -1

            return signals

        except Exception as e:
            print(f"❌ Signal generation failed: {e}")
            return self._generate_fallback_signals(data)

    def _generate_fallback_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trend-following signals when Transformer is not available."""
        signals = pd.Series(0, index=data.index)

        # Simple trend-following strategy as fallback
        sma_short = data['Close'].rolling(20).mean()
        sma_long = data['Close'].rolling(50).mean()

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
            'sequence_length': {
                'type': 'int',
                'default': 60,
                'min': 30,
                'max': 120,
                'description': 'Number of time steps for Transformer input'
            },
            'num_heads': {
                'type': 'int',
                'default': 8,
                'min': 4,
                'max': 16,
                'description': 'Number of attention heads'
            },
            'key_dim': {
                'type': 'int',
                'default': 64,
                'min': 32,
                'max': 128,
                'description': 'Dimension of attention keys/queries'
            },
            'num_transformer_blocks': {
                'type': 'int',
                'default': 4,
                'min': 2,
                'max': 8,
                'description': 'Number of transformer encoder blocks'
            },
            'prediction_horizon': {
                'type': 'int',
                'default': 3,
                'min': 1,
                'max': 10,
                'description': 'Days ahead to predict price movements'
            },
            'confidence_threshold': {
                'type': 'float',
                'default': 0.65,
                'min': 0.4,
                'max': 0.8,
                'description': 'Minimum confidence required for signals'
            }
        }

    def get_strategy_description(self) -> str:
        """Get detailed strategy description."""
        return """
        🚀 TRANSFORMER TRADING STRATEGY 🚀

        This cutting-edge strategy uses Transformer architecture with self-attention
        mechanism to identify complex patterns in financial time series.

        🔬 MODEL ARCHITECTURE:
        • Multi-head self-attention mechanism
        • Positional encoding for temporal patterns
        • Convolutional stem for feature extraction
        • Multiple transformer encoder blocks
        • Multi-output prediction (direction + confidence)

        📊 ADVANCED FEATURES:
        • Multi-timeframe technical indicators
        • Statistical measures (skewness, kurtosis)
        • Volume analysis and trends
        • Bollinger Bands positioning
        • RSI with multiple periods
        • MACD variants

        🎯 SIGNAL GENERATION:
        • Attention-based pattern recognition
        • Confidence-weighted predictions
        • Risk-adjusted signal filtering
        • Multi-scale feature integration

        💡 ADVANTAGES:
        ✅ Captures long-range dependencies in time series
        ✅ Self-attention identifies important patterns
        ✅ Handles variable-length sequences effectively
        ✅ Parallelizable training and inference
        ✅ State-of-the-art performance on sequential data

        ⚠️ REQUIREMENTS:
        • TensorFlow/Keras for full functionality
        • Significant computational resources
        • Large historical dataset for training
        • Regular model updates recommended

        🔧 CONFIGURATION TIPS:
        • Increase num_transformer_blocks for complex patterns
        • Adjust num_heads based on feature complexity
        • Use longer sequence_length for longer-term patterns
        • Monitor attention weights for pattern insights
        """

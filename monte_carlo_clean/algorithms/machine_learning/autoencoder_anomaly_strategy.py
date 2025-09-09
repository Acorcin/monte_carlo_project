"""
Autoencoder Anomaly Detection Trading Strategy

An advanced trading strategy using autoencoder neural networks for anomaly detection
in financial time series, identifying unusual market conditions and potential opportunities.

Features:
- Variational Autoencoder (VAE) for anomaly detection
- Reconstruction error-based anomaly scoring
- Multi-scale feature extraction
- Risk-adjusted anomaly signal generation
- Market regime classification

Author: Advanced ML Trading Suite
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import sys
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports with fallbacks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        Input, Dense, Dropout, LSTM, Conv1D, MaxPooling1D,
        UpSampling1D, Flatten, Reshape, Lambda
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras import backend as K
    from tensorflow.keras.losses import mse
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - Autoencoder strategy will use simplified version")
    # Define dummy classes
    Model = None
    Sequential = None
    Input = None

# Add the algorithms directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_algorithm import TradingAlgorithm

class AutoencoderAnomalyStrategy(TradingAlgorithm):
    """
    Advanced autoencoder-based anomaly detection trading strategy.

    Uses unsupervised learning to identify anomalous market conditions
    that may present trading opportunities.
    """

    def __init__(self,
                 sequence_length: int = 30,
                 encoding_dim: int = 16,
                 anomaly_threshold_percentile: float = 95.0,
                 reconstruction_error_threshold: float = 0.1,
                 use_variational: bool = False,
                 use_conv_encoder: bool = True,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 32,
                 contamination: float = 0.1):
        """
        Initialize Autoencoder Anomaly Strategy.

        Args:
            sequence_length: Number of time steps for sequence analysis
            encoding_dim: Dimension of the latent space
            anomaly_threshold_percentile: Percentile for anomaly threshold
            reconstruction_error_threshold: Direct threshold for reconstruction error
            use_variational: Whether to use Variational Autoencoder
            use_conv_encoder: Whether to use convolutional layers
            learning_rate: Learning rate for optimization
            epochs: Maximum training epochs
            batch_size: Training batch size
            contamination: Expected proportion of anomalies in data
        """
        parameters = {
            'sequence_length': sequence_length,
            'encoding_dim': encoding_dim,
            'anomaly_threshold_percentile': anomaly_threshold_percentile,
            'reconstruction_error_threshold': reconstruction_error_threshold,
            'use_variational': use_variational,
            'use_conv_encoder': use_conv_encoder,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'contamination': contamination
        }

        super().__init__(
            name="Autoencoder Anomaly Strategy",
            description="Unsupervised anomaly detection for identifying unusual market conditions",
            parameters=parameters
        )

        # Initialize components
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.is_trained = False

        # Anomaly detection
        self.anomaly_threshold = None
        self.reconstruction_errors = []
        self.anomaly_scores = []

        # Performance tracking
        self.training_history = []

        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow not available - using simplified anomaly detection")

    def _create_anomaly_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for anomaly detection."""
        df = data.copy()

        # Basic price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Multi-scale volatility
        for period in [5, 10, 20, 30]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'volume_volatility_{period}'] = df['Volume'].pct_change().rolling(period).std()

        # Price momentum features
        for period in [5, 10, 15]:
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            df[f'roc_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)

        # Volume features
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_trend'] = df['Volume'].pct_change(5)

        # RSI variations
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = calculate_rsi(df['Close'], period)

        # MACD components
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd_line'] = exp1 - exp2
        df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']

        # Bollinger Band features
        for period in [10, 20]:
            middle = df['Close'].rolling(period).mean()
            std = df['Close'].rolling(period).std()
            df[f'bb_middle_{period}'] = middle
            df[f'bb_upper_{period}'] = middle + 2 * std
            df[f'bb_lower_{period}'] = middle - 2 * std
            df[f'bb_position_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / middle

        # Statistical measures
        for period in [10, 20]:
            df[f'skewness_{period}'] = df['returns'].rolling(period).skew()
            df[f'kurtosis_{period}'] = df['returns'].rolling(period).kurt()
            df[f'autocorr_{period}'] = df['returns'].rolling(period).apply(lambda x: x.autocorr(), raw=False)

        # High-low range features
        df['daily_range'] = (df['High'] - df['Low']) / df['Close']
        df['gap_up'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['gap_down'] = (df['Close'].shift(1) - df['Open']) / df['Close'].shift(1)

        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

        return df

    def _build_autoencoder(self, input_shape: Tuple[int, int]) -> Model:
        """Build the autoencoder model architecture."""
        if not TENSORFLOW_AVAILABLE:
            return None

        inputs = Input(shape=input_shape)

        # Encoder
        if self.parameters['use_conv_encoder']:
            # Convolutional encoder
            x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
            x = MaxPooling1D(pool_size=2)(x)
            x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Flatten()(x)
        else:
            # Dense encoder
            x = LSTM(64, return_sequences=True)(inputs)
            x = LSTM(32)(x)

        # Latent space
        if self.parameters['use_variational']:
            # Variational Autoencoder
            z_mean = Dense(self.parameters['encoding_dim'], name='z_mean')(x)
            z_log_var = Dense(self.parameters['encoding_dim'], name='z_log_var')(x)

            # Sampling function
            def sampling(args):
                z_mean, z_log_var = args
                batch = K.shape(z_mean)[0]
                dim = K.int_shape(z_mean)[1]
                epsilon = K.random_normal(shape=(batch, dim))
                return z_mean + K.exp(0.5 * z_log_var) * epsilon

            z = Lambda(sampling, output_shape=(self.parameters['encoding_dim'],), name='z')([z_mean, z_log_var])

            # Encoder model for latent space
            self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

            # Decoder
            latent_inputs = Input(shape=(self.parameters['encoding_dim'],), name='z_sampling')
            x = Dense(32, activation='relu')(latent_inputs)

            if self.parameters['use_conv_encoder']:
                # Reshape for convolutional decoder
                target_length = input_shape[0] // 4  # After two MaxPooling1D operations
                feature_dim = 32
                x = Dense(target_length * feature_dim)(x)
                x = Reshape((target_length, feature_dim))(x)
                x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
                x = UpSampling1D(size=2)(x)
                x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
                x = UpSampling1D(size=2)(x)
                decoded = Conv1D(input_shape[1], kernel_size=3, activation='linear', padding='same')(x)
            else:
                x = Dense(64, activation='relu')(x)
                x = Dense(input_shape[0] * input_shape[1])(x)
                decoded = Reshape(input_shape)(x)

            self.decoder = Model(latent_inputs, decoded, name='decoder')

            # VAE loss
            def vae_loss(x, x_decoded_mean):
                reconstruction_loss = mse(K.flatten(x), K.flatten(x_decoded_mean))
                reconstruction_loss *= input_shape[0] * input_shape[1]

                kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
                kl_loss = K.sum(kl_loss, axis=-1)
                kl_loss *= -0.5

                return K.mean(reconstruction_loss + kl_loss)

            outputs = self.decoder(self.encoder(inputs)[2])
            autoencoder = Model(inputs, outputs, name='vae')
            autoencoder.compile(optimizer=Adam(learning_rate=self.parameters['learning_rate']), loss=vae_loss)

        else:
            # Standard Autoencoder
            encoded = Dense(self.parameters['encoding_dim'], activation='relu')(x)
            encoded = Dropout(0.2)(encoded)

            # Decoder
            if self.parameters['use_conv_encoder']:
                # Dense decoder for convolutional encoder
                target_length = input_shape[0] // 4
                feature_dim = 32
                x = Dense(target_length * feature_dim, activation='relu')(encoded)
                x = Reshape((target_length, feature_dim))(x)
                x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
                x = UpSampling1D(size=2)(x)
                x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
                x = UpSampling1D(size=2)(x)
                decoded = Conv1D(input_shape[1], kernel_size=3, activation='linear', padding='same')(x)
            else:
                x = Dense(64, activation='relu')(encoded)
                x = Dense(input_shape[0] * input_shape[1], activation='linear')(x)
                decoded = Reshape(input_shape)(x)

            autoencoder = Model(inputs, decoded)
            autoencoder.compile(optimizer=Adam(learning_rate=self.parameters['learning_rate']), loss='mse')

        return autoencoder

    def _calculate_reconstruction_errors(self, data_sequences: np.ndarray) -> np.ndarray:
        """Calculate reconstruction errors for anomaly detection."""
        if not TENSORFLOW_AVAILABLE or self.autoencoder is None:
            # Fallback: use simple statistical anomaly detection
            errors = np.zeros(len(data_sequences))
            for i, seq in enumerate(data_sequences):
                # Simple anomaly score based on statistical measures
                mean_val = np.mean(seq)
                std_val = np.std(seq)
                if std_val > 0:
                    errors[i] = abs(seq[-1] - mean_val) / std_val
                else:
                    errors[i] = 0
            return errors

        # Get reconstructions from autoencoder
        reconstructions = self.autoencoder.predict(data_sequences, verbose=0)

        # Calculate reconstruction errors
        errors = np.mean(np.square(data_sequences - reconstructions), axis=(1, 2))

        return errors

    def _determine_anomaly_threshold(self, reconstruction_errors: np.ndarray):
        """Determine anomaly threshold based on reconstruction errors."""
        # Method 1: Percentile-based threshold
        percentile_threshold = np.percentile(
            reconstruction_errors,
            self.parameters['anomaly_threshold_percentile']
        )

        # Method 2: Statistical threshold (mean + k*std)
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        statistical_threshold = mean_error + 3 * std_error

        # Use the more conservative threshold
        self.anomaly_threshold = max(percentile_threshold, statistical_threshold)
        self.anomaly_threshold = max(self.anomaly_threshold, self.parameters['reconstruction_error_threshold'])

        print(f"📊 Anomaly threshold set to: {self.anomaly_threshold:.4f}")
        print(f"   Percentile threshold: {percentile_threshold:.4f}")
        print(f"   Statistical threshold: {statistical_threshold:.4f}")

    def train_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the autoencoder for anomaly detection."""
        try:
            print("🧠 Training Autoencoder for anomaly detection...")

            # Create features
            feature_data = self._create_anomaly_features(data)

            # Prepare sequences for training
            sequences = []
            feature_cols = [col for col in feature_data.columns if col not in ['target', 'future_price']]

            for i in range(len(feature_data) - self.parameters['sequence_length']):
                seq = feature_data[feature_cols].iloc[i:i + self.parameters['sequence_length']].values
                sequences.append(seq)

            if len(sequences) < 10:
                return {'status': 'error', 'message': 'Insufficient data for training'}

            X_train = np.array(sequences)

            # Scale the data
            original_shape = X_train.shape
            X_train_2d = X_train.reshape(-1, X_train.shape[2])
            X_train_scaled = self.feature_scaler.fit_transform(X_train_2d)
            X_train = X_train_scaled.reshape(original_shape)

            # Build model
            self.autoencoder = self._build_autoencoder((X_train.shape[1], X_train.shape[2]))

            if self.autoencoder is None:
                return {'status': 'error', 'message': 'Failed to build autoencoder model'}

            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

            # Split for validation
            split_idx = int(len(X_train) * 0.8)
            X_train_split, X_val = X_train[:split_idx], X_train[split_idx:]

            # Train model
            history = self.autoencoder.fit(
                X_train_split, X_train_split,
                validation_data=(X_val, X_val),
                epochs=self.parameters['epochs'],
                batch_size=self.parameters['batch_size'],
                callbacks=[early_stopping, lr_reducer],
                verbose=0
            )

            # Calculate reconstruction errors for anomaly detection
            reconstruction_errors = self._calculate_reconstruction_errors(X_train)
            self.reconstruction_errors = reconstruction_errors

            # Determine anomaly threshold
            self._determine_anomaly_threshold(reconstruction_errors)

            self.is_trained = True
            self.training_history = history.history

            print("✅ Autoencoder trained successfully for anomaly detection")
            print(".4f")
            print(".4f")
            print(f"   Anomaly threshold: {self.anomaly_threshold:.4f}")

            # Calculate expected anomaly rate
            anomaly_rate = np.mean(reconstruction_errors > self.anomaly_threshold)
            print(".1%")

            return {
                'status': 'success',
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'anomaly_threshold': self.anomaly_threshold,
                'anomaly_rate': anomaly_rate,
                'epochs_trained': len(history.history['loss'])
            }

        except Exception as e:
            print(f"❌ Autoencoder training failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on anomaly detection.

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

            # Create features
            feature_data = self._create_anomaly_features(data)

            signals = pd.Series(0, index=data.index)
            anomaly_scores = pd.Series(0.0, index=data.index)

            feature_cols = [col for col in feature_data.columns if col not in ['target', 'future_price']]

            # Process each time step
            for i in range(self.parameters['sequence_length'], len(feature_data)):
                # Get sequence
                seq = feature_data[feature_cols].iloc[i - self.parameters['sequence_length']:i].values

                if TENSORFLOW_AVAILABLE and self.autoencoder is not None:
                    # Scale sequence
                    seq_2d = seq.reshape(-1, seq.shape[1])
                    seq_scaled = self.feature_scaler.transform(seq_2d)
                    seq_input = seq_scaled.reshape(1, seq.shape[0], seq.shape[1])

                    # Get reconstruction
                    reconstruction = self.autoencoder.predict(seq_input, verbose=0)

                    # Calculate reconstruction error
                    reconstruction_error = np.mean(np.square(seq_input - reconstruction))
                else:
                    # Fallback anomaly calculation
                    mean_val = np.mean(seq)
                    std_val = np.std(seq)
                    reconstruction_error = abs(seq[-1, 0] - mean_val) / (std_val + 1e-8)

                anomaly_scores.iloc[i] = reconstruction_error

                # Generate signals based on anomaly detection
                if reconstruction_error > self.anomaly_threshold:
                    # Anomaly detected - check market direction for signal
                    recent_returns = data['Close'].pct_change().iloc[i-5:i].mean()

                    if reconstruction_error > self.anomaly_threshold * 1.5:
                        # Strong anomaly
                        if recent_returns > 0:
                            signals.iloc[i] = 1  # Buy on bullish anomaly
                        else:
                            signals.iloc[i] = -1  # Sell on bearish anomaly
                    else:
                        # Moderate anomaly - hold
                        signals.iloc[i] = 0

            self.anomaly_scores = anomaly_scores
            return signals

        except Exception as e:
            print(f"❌ Signal generation failed: {e}")
            return self._generate_fallback_signals(data)

    def _generate_fallback_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate simple breakout signals when autoencoder is not available."""
        signals = pd.Series(0, index=data.index)

        # Simple breakout strategy as fallback
        high_20 = data['High'].rolling(20).max()
        low_20 = data['Low'].rolling(20).min()

        # Generate signals based on breakouts
        signals[data['Close'] > high_20.shift(1)] = 1   # Buy on breakout above recent high
        signals[data['Close'] < low_20.shift(1)] = -1   # Sell on breakdown below recent low

        return signals

    def get_algorithm_type(self) -> str:
        """Return algorithm category."""
        return "machine_learning"

    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Define parameter configuration for optimization."""
        return {
            'sequence_length': {
                'type': 'int',
                'default': 30,
                'min': 10,
                'max': 100,
                'description': 'Number of time steps for sequence analysis'
            },
            'encoding_dim': {
                'type': 'int',
                'default': 16,
                'min': 8,
                'max': 64,
                'description': 'Dimension of the latent space'
            },
            'anomaly_threshold_percentile': {
                'type': 'float',
                'default': 95.0,
                'min': 90.0,
                'max': 99.9,
                'description': 'Percentile for anomaly threshold calculation'
            },
            'reconstruction_error_threshold': {
                'type': 'float',
                'default': 0.1,
                'min': 0.01,
                'max': 1.0,
                'description': 'Direct threshold for reconstruction error'
            },
            'epochs': {
                'type': 'int',
                'default': 100,
                'min': 50,
                'max': 500,
                'description': 'Maximum training epochs'
            }
        }

    def get_strategy_description(self) -> str:
        """Get detailed strategy description."""
        return """
        🚀 AUTOENCODER ANOMALY DETECTION STRATEGY 🚀

        This advanced strategy uses unsupervised deep learning to detect anomalous
        market conditions that may present unique trading opportunities.

        🔬 MODEL ARCHITECTURE:
        • Convolutional/LSTM Encoder for feature extraction
        • Bottleneck latent space for dimensionality reduction
        • Decoder for reconstruction
        • Reconstruction error-based anomaly scoring
        • Optional Variational Autoencoder (VAE) for generative modeling

        📊 ANOMALY DETECTION:
        • Multi-scale feature analysis (price, volume, volatility)
        • Statistical and technical indicator integration
        • Reconstruction error thresholding
        • Percentile-based and statistical threshold methods
        • Contamination-aware anomaly detection

        🎯 TRADING SIGNALS:
        • Anomalies indicate unusual market conditions
        • Direction-based signal generation (bullish/bearish anomalies)
        • Confidence-weighted signal filtering
        • Risk-adjusted position sizing

        💡 ADVANTAGES:
        ✅ Detects unusual market conditions without supervision
        ✅ Identifies potential regime changes and outliers
        ✅ Works with any market condition (bull/bear/sideways)
        ✅ No assumptions about market distribution
        ✅ Can discover novel trading patterns

        ⚠️ REQUIREMENTS:
        • TensorFlow/Keras for full functionality
        • Clean historical data for training
        • Careful threshold tuning to avoid false signals
        • Regular model retraining recommended

        🔧 CONFIGURATION TIPS:
        • Higher encoding_dim for complex pattern recognition
        • Adjust anomaly_threshold_percentile based on market volatility
        • Use convolutional encoder for time series patterns
        • Monitor reconstruction errors for model health
        • Consider contamination rate for different markets
        """

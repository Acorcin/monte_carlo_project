"""
Ensemble Stacking Trading Strategy

An advanced ensemble trading strategy that combines multiple machine learning
models using stacking technique for improved prediction accuracy.

Features:
- Multiple base models (Random Forest, XGBoost, SVM, Neural Network)
- Meta-learner for final predictions
- Feature engineering and selection
- Confidence-based signal generation
- Model diversity and correlation analysis

Author: Advanced ML Trading Suite
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import sys
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Optional advanced libraries
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available - using RandomForest as alternative")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available - using ExtraTrees as alternative")

# Add the algorithms directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_algorithm import TradingAlgorithm

class EnsembleStackingStrategy(TradingAlgorithm):
    """
    Advanced ensemble stacking trading strategy.

    Combines multiple machine learning models using stacking technique
    to improve prediction accuracy and robustness.
    """

    def __init__(self,
                 prediction_horizon: int = 5,
                 confidence_threshold: float = 0.6,
                 use_feature_selection: bool = True,
                 num_base_models: int = 5,
                 cv_folds: int = 5,
                 meta_learner: str = 'logistic',
                 feature_lookback: int = 20):
        """
        Initialize Ensemble Stacking Strategy.

        Args:
            prediction_horizon: Days ahead to predict price movements
            confidence_threshold: Minimum confidence for signals
            use_feature_selection: Whether to perform feature selection
            num_base_models: Number of base models to use
            cv_folds: Number of cross-validation folds
            meta_learner: Type of meta-learner ('logistic', 'rf', 'mlp')
            feature_lookback: Lookback period for feature engineering
        """
        parameters = {
            'prediction_horizon': prediction_horizon,
            'confidence_threshold': confidence_threshold,
            'use_feature_selection': use_feature_selection,
            'num_base_models': num_base_models,
            'cv_folds': cv_folds,
            'meta_learner': meta_learner,
            'feature_lookback': feature_lookback
        }

        super().__init__(
            name="Ensemble Stacking Strategy",
            description="Advanced ensemble strategy combining multiple ML models with stacking",
            parameters=parameters
        )

        # Initialize components
        self.base_models = []
        self.meta_learner = None
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.is_trained = False

        # Performance tracking
        self.model_weights = {}
        self.feature_importance = {}
        self.training_metrics = {}

        self._initialize_base_models()

    def _initialize_base_models(self):
        """Initialize the base models for the ensemble."""
        self.base_models = []

        # Random Forest
        rf_model = {
            'name': 'RandomForest',
            'model': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'weight': 1.0
        }
        self.base_models.append(rf_model)

        # Extra Trees
        et_model = {
            'name': 'ExtraTrees',
            'model': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'weight': 1.0
        }
        self.base_models.append(et_model)

        # SVM
        svm_model = {
            'name': 'SVM',
            'model': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'weight': 1.0
        }
        self.base_models.append(svm_model)

        # Neural Network
        mlp_model = {
            'name': 'MLP',
            'model': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42
            ),
            'weight': 1.0
        }
        self.base_models.append(mlp_model)

        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            xgb_model = {
                'name': 'XGBoost',
                'model': XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                ),
                'weight': 1.2  # Higher weight for XGBoost
            }
            self.base_models.append(xgb_model)

        # LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            lgb_model = {
                'name': 'LightGBM',
                'model': LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                ),
                'weight': 1.2  # Higher weight for LightGBM
            }
            self.base_models.append(lgb_model)

        # Limit to specified number of models
        self.base_models = self.base_models[:self.parameters['num_base_models']]

    def _create_ensemble_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for the ensemble models."""
        df = data.copy()

        # Basic price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()

        # Volatility measures
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_50'] = df['returns'].rolling(50).std()

        # Momentum indicators
        for period in [10, 20]:
            df[f'ROC_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)

        # RSI
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        df['RSI_14'] = calculate_rsi(df['Close'])

        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(20).mean()
        df['BB_std'] = df['Close'].rolling(20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # Volume features
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']

        # Price patterns
        df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['open_close_ratio'] = (df['Close'] - df['Open']) / df['Open']

        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume_ratio'].shift(lag)

        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

        return df

    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Perform feature selection to identify most important features."""
        if not self.parameters['use_feature_selection']:
            return list(X.columns)

        try:
            # Use Random Forest for feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)

            # Get feature importance
            feature_importance = dict(zip(X.columns, rf.feature_importances_))
            self.feature_importance = feature_importance

            # Select top features (top 70%)
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            num_features = max(10, int(len(sorted_features) * 0.7))
            selected_features = [f[0] for f in sorted_features[:num_features]]

            print(f"📊 Selected {len(selected_features)} out of {len(X.columns)} features")
            return selected_features

        except Exception as e:
            print(f"⚠️ Feature selection failed: {e}")
            return list(X.columns)

    def _initialize_meta_learner(self):
        """Initialize the meta-learner based on configuration."""
        if self.parameters['meta_learner'] == 'logistic':
            self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        elif self.parameters['meta_learner'] == 'rf':
            self.meta_learner = RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
            )
        elif self.parameters['meta_learner'] == 'mlp':
            self.meta_learner = MLPClassifier(
                hidden_layer_sizes=(50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42
            )
        else:
            self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)

    def _calculate_model_weights(self, base_predictions: np.ndarray, y_true: np.ndarray):
        """Calculate optimal weights for base models based on their performance."""
        weights = {}

        for i, model_info in enumerate(self.base_models):
            model_name = model_info['name']
            predictions = base_predictions[:, i]

            # Calculate accuracy
            accuracy = accuracy_score(y_true, (predictions > 0.5).astype(int))

            # Calculate precision and recall for positive class
            precision = precision_score(y_true, (predictions > 0.5).astype(int), zero_division=0)
            recall = recall_score(y_true, (predictions > 0.5).astype(int), zero_division=0)

            # F1 score as primary metric
            f1 = f1_score(y_true, (predictions > 0.5).astype(int), zero_division=0)

            # Weight based on F1 score with base model weight
            weights[model_name] = f1 * model_info['weight']

            print(f"📊 {model_name}: Accuracy={accuracy:.3f}, F1={f1:.3f}, Weight={weights[model_name]:.3f}")

        self.model_weights = weights
        return weights

    def train_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the ensemble stacking model."""
        try:
            print("🧠 Training Ensemble Stacking model...")

            # Create features
            feature_data = self._create_ensemble_features(data)

            # Create target
            future_prices = data['Close'].shift(-self.parameters['prediction_horizon'])
            returns = (future_prices - data['Close']) / data['Close']
            target = (returns > 0).astype(int)

            # Remove NaN values
            valid_idx = ~(feature_data.isna().any(axis=1) | target.isna())
            feature_data = feature_data[valid_idx]
            target = target[valid_idx]

            if len(feature_data) < 100:
                return {'status': 'error', 'message': 'Insufficient data for training'}

            # Feature selection
            feature_cols = [col for col in feature_data.columns if col not in ['target', 'future_price']]
            selected_features = self._select_features(feature_data[feature_cols], target)

            X = feature_data[selected_features]
            y = target

            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)

            # Train base models and generate predictions
            print("🎯 Training base models...")
            base_predictions = np.zeros((len(X), len(self.base_models)))

            tscv = TimeSeriesSplit(n_splits=self.parameters['cv_folds'])

            for i, model_info in enumerate(self.base_models):
                model_name = model_info['name']
                model = model_info['model']

                print(f"   Training {model_name}...")

                # Generate cross-validated predictions
                try:
                    cv_predictions = cross_val_predict(
                        model, X_scaled, y, cv=tscv, method='predict_proba'
                    )[:, 1]  # Get probability of positive class

                    base_predictions[:, i] = cv_predictions

                    # Train final model on full dataset
                    model.fit(X_scaled, y)

                except Exception as e:
                    print(f"   ⚠️ {model_name} training failed: {e}")
                    base_predictions[:, i] = np.mean(y)  # Fallback to mean prediction

            # Calculate model weights
            self._calculate_model_weights(base_predictions, y)

            # Initialize and train meta-learner
            print("🎯 Training meta-learner...")
            self._initialize_meta_learner()
            self.meta_learner.fit(base_predictions, y)

            self.is_trained = True

            # Calculate training metrics
            meta_predictions = self.meta_learner.predict(base_predictions)
            accuracy = accuracy_score(y, meta_predictions)
            f1 = f1_score(y, meta_predictions)

            print("✅ Ensemble model trained successfully")
            print(".3f")
            print(".3f")

            return {
                'status': 'success',
                'accuracy': accuracy,
                'f1_score': f1,
                'num_features': len(selected_features),
                'num_models': len(self.base_models)
            }

        except Exception as e:
            print(f"❌ Ensemble training failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using the trained ensemble model.

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
            feature_data = self._create_ensemble_features(data)

            # Select features (use same selection as training)
            feature_cols = [col for col in feature_data.columns if col not in ['target', 'future_price']]
            if hasattr(self, 'feature_importance') and self.feature_importance:
                # Use previously selected features if available
                selected_features = list(self.feature_importance.keys())[:len(feature_cols)//2]
            else:
                selected_features = feature_cols

            X = feature_data[selected_features]

            # Scale features
            X_scaled = self.feature_scaler.transform(X)

            signals = pd.Series(0, index=data.index)

            # Generate predictions for each time step
            for i in range(len(X)):
                if i >= len(X_scaled):
                    continue

                sample = X_scaled[i:i+1]

                # Get base model predictions
                base_predictions = np.zeros(len(self.base_models))

                for j, model_info in enumerate(self.base_models):
                    try:
                        model = model_info['model']
                        pred_proba = model.predict_proba(sample)[0][1]  # Probability of positive class
                        base_predictions[j] = pred_proba
                    except Exception as e:
                        print(f"⚠️ Base model {model_info['name']} prediction failed: {e}")
                        base_predictions[j] = 0.5  # Neutral prediction

                # Get meta-learner prediction
                try:
                    ensemble_prediction = self.meta_learner.predict_proba(base_predictions.reshape(1, -1))[0][1]
                    confidence = max(base_predictions)  # Use highest base model confidence
                except Exception as e:
                    print(f"⚠️ Meta-learner prediction failed: {e}")
                    ensemble_prediction = np.mean(base_predictions)
                    confidence = np.std(base_predictions)  # Use diversity as confidence measure

                # Generate signal based on ensemble prediction and confidence
                if confidence > self.parameters['confidence_threshold']:
                    if ensemble_prediction > 0.55:  # Bullish prediction
                        signals.iloc[i] = 1
                    elif ensemble_prediction < 0.45:  # Bearish prediction
                        signals.iloc[i] = -1

            return signals

        except Exception as e:
            print(f"❌ Signal generation failed: {e}")
            return self._generate_fallback_signals(data)

    def _generate_fallback_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate simple mean-reversion signals when ensemble is not available."""
        signals = pd.Series(0, index=data.index)

        # Simple mean-reversion strategy as fallback
        sma = data['Close'].rolling(20).mean()
        std = data['Close'].rolling(20).std()

        # Generate signals based on Bollinger Band position
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std

        signals[data['Close'] < lower_band] = 1   # Buy when price below lower band
        signals[data['Close'] > upper_band] = -1  # Sell when price above upper band

        return signals

    def get_algorithm_type(self) -> str:
        """Return algorithm category."""
        return "machine_learning"

    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Define parameter configuration for optimization."""
        return {
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
            'num_base_models': {
                'type': 'int',
                'default': 5,
                'min': 3,
                'max': 8,
                'description': 'Number of base models in ensemble'
            },
            'cv_folds': {
                'type': 'int',
                'default': 5,
                'min': 3,
                'max': 10,
                'description': 'Number of cross-validation folds'
            }
        }

    def get_strategy_description(self) -> str:
        """Get detailed strategy description."""
        return """
        🚀 ENSEMBLE STACKING TRADING STRATEGY 🚀

        This advanced strategy combines multiple machine learning models using
        stacking technique to improve prediction accuracy and robustness.

        🔬 MODEL ARCHITECTURE:
        • Multiple base models (Random Forest, XGBoost, SVM, Neural Network)
        • Meta-learner for final predictions
        • Feature selection and engineering
        • Model weighting based on performance
        • Cross-validation for robust training

        📊 BASE MODELS INCLUDE:
        • Random Forest - Ensemble of decision trees
        • Extra Trees - Extremely randomized trees
        • SVM - Support Vector Machine with RBF kernel
        • MLP - Multi-layer Perceptron neural network
        • XGBoost - Gradient boosting (if available)
        • LightGBM - Light gradient boosting (if available)

        🎯 SIGNAL GENERATION:
        • Base model predictions with confidence scores
        • Meta-learner combines predictions optimally
        • Confidence-based signal filtering
        • Model diversity improves robustness

        💡 ADVANTAGES:
        ✅ Combines strengths of different ML algorithms
        ✅ Reduces overfitting through ensemble approach
        ✅ Handles different market regimes effectively
        ✅ Provides confidence scores for risk management
        ✅ Feature selection improves model efficiency

        ⚠️ REQUIREMENTS:
        • Multiple ML libraries (scikit-learn, optional: XGBoost, LightGBM)
        • Sufficient historical data for training
        • Computational resources for ensemble training
        • Regular model updates recommended

        🔧 CONFIGURATION TIPS:
        • Increase num_base_models for more diversity
        • Adjust confidence_threshold based on risk tolerance
        • Use feature_selection=True for better performance
        • Monitor individual model weights for insights
        """

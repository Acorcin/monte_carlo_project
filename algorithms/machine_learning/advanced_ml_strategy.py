"""
Advanced ML Trading Algorithm

A sophisticated machine learning trading strategy that includes:
- Rich feature engineering (price, volume, volatility, technical indicators)
- Market regime detection using Hidden Markov Models or Gaussian Mixture Models
- Walk-forward ML pipeline with XGBoost/GradientBoosting
- Advanced position sizing using Kelly criterion and volatility targeting
- Probability calibration for better predictions

This algorithm is adapted for the Monte Carlo simulation framework.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import sys
import os

# Add the algorithms directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_algorithm import TradingAlgorithm

# Core ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.mixture import GaussianMixture

# Optional advanced libraries (graceful fallbacks if not available)
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    import ta  # technical analysis library
except ImportError:
    ta = None


class AdvancedMLStrategy(TradingAlgorithm):
    """
    Advanced Machine Learning Trading Strategy
    
    Features:
    - Comprehensive feature engineering
    - Market regime detection
    - Walk-forward ML training with hyperparameter optimization
    - Kelly criterion position sizing with volatility targeting
    - Transaction cost modeling
    """
    
    def __init__(self):
        super().__init__("Advanced ML Strategy")
        self.description = "ML-based strategy with regime detection and advanced position sizing"
        
        # Algorithm parameters
        self.params = {
            'n_splits': 3,  # Reduced for faster backtesting
            'n_iter': 15,   # Reduced for faster backtesting
            'target_vol': 0.15,
            'max_leverage': 2.0,
            'cost_bps': 2.0,
            'slippage_bps': 1.0,
            'n_regimes': 3,
            'min_train_samples': 50,  # Reduced for backtesting compatibility
            'prediction_threshold': 0.55  # Probability threshold for trades
        }
        
        # State variables
        self.features_df = None
        self.model = None
        self.scaler = None
        self.regime_model = None
        self.asset_vol = None
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading signals using ML predictions.
        
        Args:
            data: OHLCV data
            
        Returns:
            DataFrame with signals
        """
        try:
            print(f"   🧠 Advanced ML Strategy: Engineering features...")
            
            # Step 1: Feature Engineering
            features = self._engineer_features(data.copy())
            
            if len(features) < self.params['min_train_samples']:
                print(f"   ⚠️  Insufficient data: {len(features)} < {self.params['min_train_samples']}")
                return pd.DataFrame(index=data.index, columns=['signal', 'position_size', 'probability'])
            
            # Step 2: Regime Detection
            print(f"   🔍 Detecting market regimes...")
            regimes = self._detect_regimes(features)
            features['regime'] = regimes
            
            # Step 3: Prepare ML dataset
            X, y = self._prepare_ml_dataset(features)
            
            if len(X) < self.params['min_train_samples']:
                print(f"   ⚠️  Insufficient ML data: {len(X)} < {self.params['min_train_samples']}")
                return pd.DataFrame(index=data.index, columns=['signal', 'position_size', 'probability'])
            
            # Step 4: Walk-forward ML predictions
            print(f"   🚀 Running walk-forward ML predictions...")
            probabilities = self._walk_forward_predict(X, y)
            
            # Step 5: Convert probabilities to positions
            print(f"   💰 Converting probabilities to position sizes...")
            positions = self._probabilities_to_positions(probabilities, features)
            
            # Step 6: Generate signals
            signals_df = pd.DataFrame(index=data.index)
            signals_df['signal'] = 0
            signals_df['position_size'] = 0.0
            signals_df['probability'] = 0.5
            
            # Align positions with original data index
            for idx in positions.index:
                if idx in signals_df.index:
                    prob = probabilities.get(idx, 0.5)
                    pos_size = positions.get(idx, 0.0)
                    
                    # Generate signal based on probability threshold
                    if prob > self.params['prediction_threshold']:
                        signals_df.loc[idx, 'signal'] = 1  # Buy
                    elif prob < (1 - self.params['prediction_threshold']):
                        signals_df.loc[idx, 'signal'] = -1  # Sell
                    
                    signals_df.loc[idx, 'position_size'] = pos_size
                    signals_df.loc[idx, 'probability'] = prob
            
            print(f"   ✅ Generated {len(signals_df[signals_df['signal'] != 0])} trading signals")
            return signals_df
            
        except Exception as e:
            print(f"   ❌ ML Strategy error: {e}")
            # Return neutral signals on error
            return pd.DataFrame(index=data.index, columns=['signal', 'position_size', 'probability'])
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive feature set."""
        out = df.copy()
        close = out['Close']
        high = out['High']
        low = out['Low']
        open_ = out['Open']
        volume = out['Volume']
        
        # Basic returns & volatility
        out['ret_1'] = close.pct_change()
        out['ret_5'] = close.pct_change(5)
        out['ret_10'] = close.pct_change(10)
        out['logret_1'] = np.log(close / close.shift(1))
        out['rvol_10'] = out['logret_1'].rolling(10).std() * np.sqrt(252)
        out['rvol_20'] = out['logret_1'].rolling(20).std() * np.sqrt(252)
        
        # Moving averages and momentum
        for w in [5, 10, 20, 50]:
            out[f'sma_{w}'] = close.rolling(w).mean()
            out[f'ema_{w}'] = close.ewm(span=w, adjust=False).mean()
            out[f'roc_{w}'] = close.pct_change(w)
            
            # Bollinger Bands
            bb_std = close.rolling(w).std()
            out[f'bb_upper_{w}'] = out[f'sma_{w}'] + 2 * bb_std
            out[f'bb_lower_{w}'] = out[f'sma_{w}'] - 2 * bb_std
            out[f'bb_position_{w}'] = (close - out[f'bb_lower_{w}']) / (out[f'bb_upper_{w}'] - out[f'bb_lower_{w}'])
        
        # Price range features
        out['true_range'] = np.maximum(high - low, 
                           np.maximum(abs(high - close.shift(1)), 
                                    abs(low - close.shift(1))))
        out['hl_ratio'] = (high - low) / close
        out['oc_ratio'] = (close - open_) / open_
        out['gap'] = (open_ - close.shift(1)) / close.shift(1)
        
        # Volume features
        out['volume_sma_5'] = volume.rolling(5).mean()
        out['volume_sma_20'] = volume.rolling(20).mean()
        out['volume_ratio'] = out['volume_sma_5'] / (out['volume_sma_20'] + 1e-9)
        
        # Technical indicators
        if ta is not None:
            # Use TA library if available
            out['rsi_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()
            macd = ta.trend.MACD(close)
            out['macd'] = macd.macd()
            out['macd_signal'] = macd.macd_signal()
            out['macd_histogram'] = macd.macd_diff()
        else:
            # Simple RSI implementation
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            out['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Simple MACD
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            out['macd'] = ema_12 - ema_26
            out['macd_signal'] = out['macd'].ewm(span=9).mean()
            out['macd_histogram'] = out['macd'] - out['macd_signal']
        
        # Statistical features
        out['returns_skew_20'] = out['logret_1'].rolling(20).skew()
        out['returns_kurt_20'] = out['logret_1'].rolling(20).kurt()
        
        # Price momentum features
        out['momentum_5'] = (close / close.shift(5) - 1)
        out['momentum_20'] = (close / close.shift(20) - 1)
        
        # Clean up
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna()
        
        return out
    
    def _detect_regimes(self, features: pd.DataFrame) -> pd.Series:
        """Detect market regimes using HMM or GMM."""
        try:
            # Use log returns and volatility for regime detection
            X = np.column_stack([
                features['logret_1'].values,
                features['rvol_20'].values
            ])
            
            if GaussianHMM is not None:
                # Prefer HMM if available
                model = GaussianHMM(
                    n_components=self.params['n_regimes'], 
                    covariance_type='full', 
                    n_iter=100,
                    random_state=42
                )
                model.fit(X)
                regimes = model.predict(X)
            else:
                # Fallback to GMM
                model = GaussianMixture(
                    n_components=self.params['n_regimes'], 
                    covariance_type='full',
                    random_state=42
                )
                regimes = model.fit_predict(X)
            
            # Sort regimes by volatility (0=low vol, 1=med, 2=high)
            vol_by_regime = pd.Series(features['rvol_20'].values).groupby(regimes).mean()
            regime_order = vol_by_regime.sort_values().index
            regime_mapping = {old: new for new, old in enumerate(regime_order)}
            sorted_regimes = [regime_mapping[r] for r in regimes]
            
            return pd.Series(sorted_regimes, index=features.index, name='regime')
            
        except Exception as e:
            print(f"   ⚠️  Regime detection failed: {e}, using single regime")
            return pd.Series(0, index=features.index, name='regime')
    
    def _prepare_ml_dataset(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels for ML."""
        # Create binary labels: 1 if next period return > 0, else 0
        forward_return = features['Close'].pct_change().shift(-1)
        y = (forward_return > 0).astype(int)
        
        # Select features (exclude price levels and future-looking variables)
        feature_cols = [col for col in features.columns if col not in [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
            'ret_1', 'logret_1'  # Exclude current period returns to avoid lookahead
        ]]
        
        X = features[feature_cols].copy()
        
        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Remove any remaining NaN values
        valid_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        return X, y
    
    def _build_model(self) -> Tuple[Pipeline, dict]:
        """Build ML model pipeline and parameter distributions."""
        if XGBClassifier is not None:
            # Use XGBoost if available
            clf = XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                n_estimators=200,  # Reduced for speed
                random_state=42,
                n_jobs=1  # Single thread for stability
            )
            param_dist = {
                'clf__max_depth': [3, 4, 5],
                'clf__learning_rate': [0.01, 0.05, 0.1, 0.15],
                'clf__subsample': [0.8, 0.9, 1.0],
                'clf__colsample_bytree': [0.8, 0.9, 1.0]
            }
        else:
            # Fallback to GradientBoosting
            clf = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
            param_dist = {
                'clf__learning_rate': [0.05, 0.1, 0.15],
                'clf__max_depth': [3, 4, 5],
                'clf__subsample': [0.8, 0.9, 1.0]
            }
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', clf)
        ])
        
        return pipeline, param_dist
    
    def _walk_forward_predict(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Walk-forward training and prediction."""
        if len(X) < self.params['min_train_samples']:
            return pd.Series(0.5, index=X.index)
        
        tscv = TimeSeriesSplit(n_splits=self.params['n_splits'])
        predictions = pd.Series(index=X.index, dtype=float)
        
        pipeline, param_dist = self._build_model()
        
        fold_count = 0
        for train_idx, test_idx in tscv.split(X):
            fold_count += 1
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Skip if insufficient training data
            if len(X_train) < 20:
                continue
            
            try:
                # Hyperparameter search
                search = RandomizedSearchCV(
                    pipeline, 
                    param_distributions=param_dist,
                    n_iter=self.params['n_iter'],
                    cv=3,
                    random_state=42,
                    scoring='roc_auc',
                    n_jobs=1,
                    verbose=0
                )
                
                search.fit(X_train, y_train)
                
                # Calibrate probabilities
                best_model = search.best_estimator_
                calibrated = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
                calibrated.fit(X_train, y_train)
                
                # Predict probabilities
                test_proba = calibrated.predict_proba(X_test)[:, 1]
                test_pred = pd.Series(test_proba, index=X.index[test_idx])
                
                predictions.loc[test_pred.index] = test_pred
                
                # Calculate metrics
                auc = roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) > 1 else 0.5
                print(f"   Fold {fold_count}: AUC={auc:.3f}")
                
            except Exception as e:
                print(f"   ⚠️  Fold {fold_count} failed: {e}")
                # Fill with neutral predictions
                predictions.iloc[test_idx] = 0.5
        
        # Fill any remaining NaN with neutral prediction
        predictions = predictions.fillna(0.5)
        return predictions
    
    def _estimate_payoff_ratio(self, returns: pd.Series) -> float:
        """Estimate up/down payoff ratio for Kelly sizing."""
        if len(returns) < 10:
            return 1.0
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return 1.0
        
        avg_up = positive_returns.mean()
        avg_down = abs(negative_returns.mean())
        
        if avg_down == 0:
            return 1.0
        
        ratio = avg_up / avg_down
        return max(0.5, min(2.0, ratio))  # Clamp to reasonable range
    
    def _probabilities_to_positions(self, probabilities: pd.Series, features: pd.DataFrame) -> pd.Series:
        """Convert prediction probabilities to position sizes using Kelly criterion."""
        if len(probabilities) == 0:
            return pd.Series(dtype=float)
        
        # Estimate asset volatility
        returns = features['ret_1'].dropna()
        if len(returns) > 20:
            asset_vol = returns.std() * np.sqrt(252)
        else:
            asset_vol = 0.2  # Default assumption
        
        # Estimate payoff ratio
        payoff_ratio = self._estimate_payoff_ratio(returns)
        
        # Convert probabilities to positions
        p = probabilities.clip(0.01, 0.99)  # Avoid extreme values
        
        # Kelly fraction: f* = (p * (b + 1) - 1) / b
        kelly_fractions = (p * (payoff_ratio + 1) - 1) / payoff_ratio
        
        # Edge (conviction): how far from 50/50
        edge = np.abs(2 * p - 1)
        
        # Raw position size
        raw_positions = np.sign(2 * p - 1) * edge * np.abs(kelly_fractions)
        
        # Volatility targeting
        vol_scale = self.params['target_vol'] / max(asset_vol, 0.01)
        scaled_positions = raw_positions * vol_scale
        
        # Apply leverage limit
        final_positions = scaled_positions.clip(
            -self.params['max_leverage'], 
            self.params['max_leverage']
        )
        
        return final_positions
    
    def get_parameters(self) -> Dict:
        """Return algorithm parameters."""
        return self.params.copy()
    
    def set_parameters(self, params: Dict):
        """Set algorithm parameters."""
        for key, value in params.items():
            if key in self.params:
                self.params[key] = value
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals (required by base class).
        
        Args:
            data: OHLCV data
            
        Returns:
            pd.Series: Signal values aligned with data index (1=buy, -1=sell, 0=hold)
        """
        signals_df = self.calculate_signals(data)
        
        # Create a Series aligned with the original data index
        signal_series = pd.Series(0, index=data.index, name='signal')
        
        # Map the calculated signals to the data index
        for date in signals_df.index:
            if date in data.index:
                signal_series.loc[date] = signals_df.loc[date, 'signal']
        
        return signal_series
    
    def get_algorithm_type(self) -> str:
        """Return the algorithm type/category."""
        return "Machine Learning"

"""
Demo: Advanced ML Algorithms Showcase

This script demonstrates all the new advanced machine learning algorithms
added to the Monte Carlo trading system.

Features:
- LSTM Trading Strategy (Deep Learning for Time Series)
- Transformer Trading Strategy (Attention-based Forecasting)
- Ensemble Stacking Strategy (Multiple ML Model Combination)
- Reinforcement Learning Strategy (Q-Learning for Trading)
- Autoencoder Anomaly Strategy (Unsupervised Anomaly Detection)

Author: Advanced ML Trading Suite Demo
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_sample_data(symbol: str = "AAPL", days: int = 500) -> pd.DataFrame:
    """Create sample financial data for demonstration."""
    print(f"📊 Creating sample {days}-day dataset for {symbol}...")

    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start_date, end_date, freq='D')

    # Generate realistic price series with trends and volatility
    np.random.seed(42)

    # Base price
    base_price = 150.0

    # Generate returns with some autocorrelation
    returns = np.random.normal(0.0005, 0.025, len(dates))

    # Add some trending periods
    for i in range(0, len(returns), 100):
        if i + 50 < len(returns):
            trend = np.linspace(0, 0.02, 50) if np.random.random() > 0.5 else np.linspace(0, -0.02, 50)
            returns[i:i+50] += trend

    # Calculate prices
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLCV data
    high_mult = 1 + np.abs(np.random.normal(0, 0.015, len(dates)))
    low_mult = 1 - np.abs(np.random.normal(0, 0.015, len(dates)))
    volume = np.random.randint(1000000, 10000000, len(dates))

    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'High': prices * high_mult,
        'Low': prices * low_mult,
        'Close': prices,
        'Volume': volume
    }, index=dates)

    print(f"✅ Generated {len(data)} records from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(".2f")
    print(".0f")

    return data

def test_algorithm_loading():
    """Test that all advanced ML algorithms load correctly."""
    print("\n🔍 Testing Algorithm Loading...")
    print("=" * 40)

    try:
        from algorithms.algorithm_manager import AlgorithmManager

        manager = AlgorithmManager()

        print(f"✅ Algorithm manager initialized")
        print(f"📊 Total algorithms loaded: {len(manager.algorithms)}")

        # Get categories
        categories = manager.get_algorithm_categories()

        # Focus on machine learning algorithms
        if 'machine_learning' in categories:
            ml_algorithms = categories['machine_learning']
            print(f"\n🤖 Machine Learning Algorithms ({len(ml_algorithms)}):")

            advanced_ml_found = []
            for algo_name in ml_algorithms:
                info = manager.get_algorithm_info(algo_name)
                if info:
                    print(f"  ✅ {info['name']}")
                    print(f"     {info['description'][:60]}...")

                    # Check if it's one of our advanced algorithms
                    if any(keyword in algo_name.lower() for keyword in
                          ['lstm', 'transformer', 'ensemble', 'reinforcement', 'autoencoder']):
                        advanced_ml_found.append(algo_name)

            if advanced_ml_found:
                print(f"\n🎯 Advanced ML Algorithms Detected: {len(advanced_ml_found)}")
                for algo in advanced_ml_found:
                    print(f"   • {algo}")
            else:
                print("\n⚠️  No advanced ML algorithms detected")

        return manager

    except Exception as e:
        print(f"❌ Algorithm loading test failed: {e}")
        return None

def demonstrate_lstm_strategy(manager, data):
    """Demonstrate LSTM Trading Strategy."""
    print("\n🧠 Testing LSTM Trading Strategy")
    print("-" * 35)

    try:
        # Create LSTM strategy instance
        lstm_config = {
            'sequence_length': 30,  # Shorter for demo
            'lstm_units': 64,       # Smaller network
            'epochs': 10,           # Fewer epochs for demo
            'learning_rate': 0.001
        }

        lstm_algo = manager.create_algorithm('LSTMTradingStrategy', lstm_config)

        if lstm_algo:
            print("✅ LSTM strategy created successfully")

            # Try to generate signals (will train if needed)
            print("🚀 Generating signals...")
            signals = lstm_algo.generate_signals(data)

            signal_counts = signals.value_counts()
            print(f"📊 Signals generated: {len(signals)} total")
            print(f"   Buy signals: {signal_counts.get(1, 0)}")
            print(f"   Sell signals: {signal_counts.get(-1, 0)}")
            print(f"   Hold signals: {signal_counts.get(0, 0)}")

            return True
        else:
            print("❌ Failed to create LSTM strategy")
            return False

    except Exception as e:
        print(f"❌ LSTM demonstration failed: {e}")
        return False

def demonstrate_transformer_strategy(manager, data):
    """Demonstrate Transformer Trading Strategy."""
    print("\n🔄 Testing Transformer Trading Strategy")
    print("-" * 40)

    try:
        # Create Transformer strategy instance
        transformer_config = {
            'sequence_length': 20,   # Shorter for demo
            'num_heads': 4,          # Fewer heads
            'key_dim': 32,           # Smaller dimension
            'epochs': 5,             # Fewer epochs
            'num_transformer_blocks': 2
        }

        transformer_algo = manager.create_algorithm('TransformerTradingStrategy', transformer_config)

        if transformer_algo:
            print("✅ Transformer strategy created successfully")

            # Try to generate signals
            print("🚀 Generating signals...")
            signals = transformer_algo.generate_signals(data)

            signal_counts = signals.value_counts()
            print(f"📊 Signals generated: {len(signals)} total")
            print(f"   Buy signals: {signal_counts.get(1, 0)}")
            print(f"   Sell signals: {signal_counts.get(-1, 0)}")
            print(f"   Hold signals: {signal_counts.get(0, 0)}")

            return True
        else:
            print("❌ Failed to create Transformer strategy")
            return False

    except Exception as e:
        print(f"❌ Transformer demonstration failed: {e}")
        return False

def demonstrate_ensemble_strategy(manager, data):
    """Demonstrate Ensemble Stacking Strategy."""
    print("\n🎯 Testing Ensemble Stacking Strategy")
    print("-" * 38)

    try:
        # Create Ensemble strategy instance
        ensemble_config = {
            'num_base_models': 3,    # Fewer models for demo
            'cv_folds': 3,           # Fewer folds
            'prediction_horizon': 3
        }

        ensemble_algo = manager.create_algorithm('EnsembleStackingStrategy', ensemble_config)

        if ensemble_algo:
            print("✅ Ensemble strategy created successfully")

            # Try to generate signals
            print("🚀 Generating signals...")
            signals = ensemble_algo.generate_signals(data)

            signal_counts = signals.value_counts()
            print(f"📊 Signals generated: {len(signals)} total")
            print(f"   Buy signals: {signal_counts.get(1, 0)}")
            print(f"   Sell signals: {signal_counts.get(-1, 0)}")
            print(f"   Hold signals: {signal_counts.get(0, 0)}")

            return True
        else:
            print("❌ Failed to create Ensemble strategy")
            return False

    except Exception as e:
        print(f"❌ Ensemble demonstration failed: {e}")
        return False

def demonstrate_rl_strategy(manager, data):
    """Demonstrate Reinforcement Learning Strategy."""
    print("\n🎮 Testing Reinforcement Learning Strategy")
    print("-" * 42)

    try:
        # Create RL strategy instance
        rl_config = {
            'num_episodes': 50,      # Fewer episodes for demo
            'epsilon_start': 0.8,    # Start with less exploration
            'learning_rate': 0.2,
            'state_bins': 5          # Fewer bins for faster computation
        }

        rl_algo = manager.create_algorithm('ReinforcementLearningStrategy', rl_config)

        if rl_algo:
            print("✅ RL strategy created successfully")

            # Try to generate signals
            print("🚀 Generating signals...")
            signals = rl_algo.generate_signals(data)

            signal_counts = signals.value_counts()
            print(f"📊 Signals generated: {len(signals)} total")
            print(f"   Buy signals: {signal_counts.get(1, 0)}")
            print(f"   Sell signals: {signal_counts.get(-1, 0)}")
            print(f"   Hold signals: {signal_counts.get(0, 0)}")

            return True
        else:
            print("❌ Failed to create RL strategy")
            return False

    except Exception as e:
        print(f"❌ RL demonstration failed: {e}")
        return False

def demonstrate_autoencoder_strategy(manager, data):
    """Demonstrate Autoencoder Anomaly Strategy."""
    print("\n🔍 Testing Autoencoder Anomaly Strategy")
    print("-" * 40)

    try:
        # Create Autoencoder strategy instance
        ae_config = {
            'sequence_length': 15,   # Shorter for demo
            'encoding_dim': 8,       # Smaller latent space
            'epochs': 5,             # Fewer epochs
            'anomaly_threshold_percentile': 90.0  # More sensitive
        }

        ae_algo = manager.create_algorithm('AutoencoderAnomalyStrategy', ae_config)

        if ae_algo:
            print("✅ Autoencoder strategy created successfully")

            # Try to generate signals
            print("🚀 Generating signals...")
            signals = ae_algo.generate_signals(data)

            signal_counts = signals.value_counts()
            print(f"📊 Signals generated: {len(signals)} total")
            print(f"   Buy signals: {signal_counts.get(1, 0)}")
            print(f"   Sell signals: {signal_counts.get(-1, 0)}")
            print(f"   Hold signals: {signal_counts.get(0, 0)}")

            return True
        else:
            print("❌ Failed to create Autoencoder strategy")
            return False

    except Exception as e:
        print(f"❌ Autoencoder demonstration failed: {e}")
        return False

def main():
    """Main demonstration function."""
    print("🚀 Advanced ML Algorithms Demonstration")
    print("=" * 50)
    print("This demo showcases the new cutting-edge ML algorithms")
    print("added to your Monte Carlo trading system.\n")

    # Test algorithm loading
    manager = test_algorithm_loading()
    if not manager:
        print("❌ Cannot continue without algorithm manager")
        return

    # Create sample data
    data = create_sample_data()

    # Test each algorithm
    results = {}

    print("\n🔬 Testing Advanced ML Algorithms")
    print("=" * 40)

    # Test LSTM
    results['LSTM'] = demonstrate_lstm_strategy(manager, data)

    # Test Transformer
    results['Transformer'] = demonstrate_transformer_strategy(manager, data)

    # Test Ensemble
    results['Ensemble'] = demonstrate_ensemble_strategy(manager, data)

    # Test Reinforcement Learning
    results['RL'] = demonstrate_rl_strategy(manager, data)

    # Test Autoencoder
    results['Autoencoder'] = demonstrate_autoencoder_strategy(manager, data)

    # Summary
    print("\n📊 Demonstration Summary")
    print("=" * 30)

    successful = sum(results.values())
    total = len(results)

    print(f"✅ Algorithms tested: {total}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {total - successful}")

    if successful > 0:
        print("
🎯 Successfully demonstrated advanced ML algorithms:"        for algo, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {algo}")

        print("
🚀 Your trading system now includes:"        print("   • Deep Learning (LSTM, Transformer)"        print("   • Ensemble Methods (Stacking)"        print("   • Reinforcement Learning (Q-Learning)"        print("   • Unsupervised Learning (Autoencoder)"        print("   • Traditional ML (Random Forest, SVM, etc.)"
    else:
        print("\n⚠️  Some algorithms failed to run")
        print("   This may be due to missing dependencies:")
        print("   - pip install tensorflow")
        print("   - pip install xgboost lightgbm")
        print("   - Or run with fallback implementations")

    print("
🎉 Advanced ML integration complete!"    print("   Use these algorithms in your Strategy tab!")

if __name__ == "__main__":
    main()

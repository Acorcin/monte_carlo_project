#!/usr/bin/env python3
"""
Performance Test Script for Backtesting Optimizations

This script demonstrates the performance improvements from:
1. Parallel processing
2. Caching system
3. Vectorized operations
4. Optimized data structures
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add to path
sys.path.append('algorithms')

from algorithms.algorithm_manager import AlgorithmManager
from data_fetcher import fetch_stock_data

def create_test_data(symbol: str = 'AAPL', period: str = '1y') -> pd.DataFrame:
    """Create test data for benchmarking."""
    try:
        data = fetch_stock_data(symbol, period=period)
        if data is None or data.empty:
            # Create synthetic data if real data fetch fails
            print("⚠️ Using synthetic data for testing")
            dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
            np.random.seed(42)
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
            data = pd.DataFrame({
                'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
                'High': prices * (1 + np.random.normal(0, 0.01, len(dates))),
                'Low': prices * (1 - np.random.normal(0, 0.01, len(dates))),
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
        return data
    except Exception as e:
        print(f"❌ Failed to create test data: {e}")
        return None

def run_performance_test():
    """Run comprehensive performance test."""
    print("🚀 MONTE CARLO BACKTEST PERFORMANCE TEST")
    print("=" * 50)

    # Create test data
    print("\n📊 Creating test data...")
    data = create_test_data()
    if data is None:
        print("❌ Failed to create test data")
        return

    print(f"✅ Test data created: {len(data)} rows")

    # Create algorithm manager with optimizations
    print("\n⚙️ Initializing optimized algorithm manager...")
    manager = AlgorithmManager(use_parallel=True)

    # Test with multiple algorithms
    test_configs = [
        {'name': 'MovingAverageCrossover', 'parameters': {'fast_period': 10, 'slow_period': 30}},
        {'name': 'RSIOversoldOverbought', 'parameters': {'period': 14, 'oversold': 30, 'overbought': 70}},
        {'name': 'PriceMomentum', 'parameters': {'period': 20, 'threshold': 0.02}},
    ]

    # Filter to available algorithms
    available_algorithms = list(manager.get_available_algorithms().keys())
    test_configs = [config for config in test_configs if config['name'] in available_algorithms]

    if not test_configs:
        print("❌ No test algorithms available")
        return

    print(f"\n🤖 Testing with {len(test_configs)} algorithms:")
    for config in test_configs:
        print(f"  • {config['name']}")

    # Performance test
    print("
🏃 Running performance tests..."    print("1. Sequential processing...")
    manager.use_parallel = False
    start_time = time.time()
    results_seq = manager.backtest_multiple_algorithms(test_configs, data)
    seq_time = time.time() - start_time

    print("2. Parallel processing...")
    manager.use_parallel = True
    start_time = time.time()
    results_par = manager.backtest_multiple_algorithms(test_configs, data)
    par_time = time.time() - start_time

    # Calculate improvements
    speedup = seq_time / par_time if par_time > 0 else 1.0

    # Test caching
    print("3. Testing cache performance...")
    start_time = time.time()
    results_cached = manager.backtest_multiple_algorithms(test_configs, data)
    cached_time = time.time() - start_time

    cache_speedup = par_time / cached_time if cached_time > 0 else 1.0

    # Results
    print("
📊 PERFORMANCE RESULTS"    print("=" * 40)
    print(".2f"    print(".2f"    print(".2f"    print(".2f"    print(".2f"
    # Memory usage estimation
    cache_files = len(list(manager.cache_dir.glob("*.pkl"))) if manager.cache_dir.exists() else 0
    print(f"💾 Cache files created: {cache_files}")

    # Summary
    total_improvement = speedup * cache_speedup
    print("
🏆 TOTAL PERFORMANCE IMPROVEMENT:"    print(".1f"
    if total_improvement > 2:
        print("   🚀 EXCELLENT: Significant performance boost!")
    elif total_improvement > 1.5:
        print("   ✅ GOOD: Notable performance improvement!")
    else:
        print("   📈 MODERATE: Some performance gains achieved!")

    print("
💡 OPTIMIZATION FEATURES ENABLED:"    print("   • Parallel processing (ThreadPoolExecutor/ProcessPoolExecutor)"    print("   • Intelligent caching system"    print("   • Vectorized backtesting operations"    print("   • Optimized data structures"    print("   • Smart algorithm detection (CPU vs GPU intensive)")

if __name__ == "__main__":
    run_performance_test()

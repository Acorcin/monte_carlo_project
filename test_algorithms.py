"""
Test Script for Trading Algorithms

Simple test to verify the algorithm backtesting system works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))

from algorithms.algorithm_manager import algorithm_manager
from data_fetcher import fetch_stock_data

def test_algorithm_system():
    """Test the complete algorithm system."""
    print("🧪 TESTING ALGORITHM BACKTESTING SYSTEM")
    print("=" * 50)
    
    # Test 1: Algorithm Discovery
    print("\n🔍 Test 1: Algorithm Discovery")
    print("-" * 30)
    
    available_algos = algorithm_manager.get_available_algorithms()
    print(f"✅ Found {len(available_algos)} algorithms:")
    for name, algo_type in available_algos.items():
        print(f"   • {name} ({algo_type})")
    
    # Test 2: Fetch Test Data
    print("\n📊 Test 2: Data Fetching")
    print("-" * 25)
    
    try:
        data = fetch_stock_data("SPY", period="1mo", interval="1d")
        print(f"✅ Fetched {len(data)} data points from SPY")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    except Exception as e:
        print(f"❌ Data fetch failed: {e}")
        return
    
    # Test 3: Individual Algorithm Backtest
    print("\n🤖 Test 3: Individual Algorithm Backtest")
    print("-" * 40)
    
    test_algorithms = [
        {'name': 'MovingAverageCrossover', 'parameters': {'fast_period': 5, 'slow_period': 20}},
        {'name': 'RSIOversoldOverbought', 'parameters': {'rsi_period': 14}},
        {'name': 'PriceMomentum', 'parameters': {'momentum_period': 10}}
    ]
    
    results = {}
    for config in test_algorithms:
        algo_name = config['name']
        parameters = config['parameters']
        
        print(f"\n   Testing {algo_name}...")
        
        result = algorithm_manager.backtest_algorithm(
            algorithm_name=algo_name,
            data=data,
            parameters=parameters,
            initial_capital=10000
        )
        
        if result:
            results[algo_name] = result
            metrics = result['metrics']
            print(f"   ✅ {algo_name}:")
            print(f"      Total Return: {result['total_return']:.2f}%")
            print(f"      Total Trades: {metrics['total_trades']}")
            print(f"      Win Rate: {metrics['win_rate']:.1f}%")
            print(f"      Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        else:
            print(f"   ❌ {algo_name}: Failed")
    
    # Test 4: Results Summary
    print("\n📊 Test 4: Results Summary")
    print("-" * 30)
    
    if results:
        print(f"\n{'Algorithm':<25} {'Return':<10} {'Trades':<8} {'Win Rate':<10}")
        print("-" * 55)
        
        for algo_name, result in results.items():
            metrics = result['metrics']
            print(f"{algo_name:<25} {result['total_return']:>7.2f}% {metrics['total_trades']:>6} {metrics['win_rate']:>8.1f}%")
        
        # Find best performer
        best_algo = max(results.items(), key=lambda x: x[1]['total_return'])
        print(f"\n🏆 Best Performer: {best_algo[0]} ({best_algo[1]['total_return']:.2f}% return)")
    else:
        print("❌ No successful backtests")
    
    # Test 5: Algorithm Parameter Info
    print("\n⚙️  Test 5: Algorithm Configuration")
    print("-" * 35)
    
    for algo_name in available_algos.keys():
        info = algorithm_manager.get_algorithm_info(algo_name)
        if info and info['parameters']:
            print(f"\n   {algo_name} parameters:")
            for param_name, param_info in info['parameters'].items():
                print(f"      {param_name}: {param_info['description']}")
                print(f"         Default: {param_info['default']}, Type: {param_info['type']}")
    
    print(f"\n🎯 TESTING COMPLETE!")
    print(f"   Algorithm system is {'✅ WORKING' if results else '❌ FAILED'}")
    
    return results

if __name__ == "__main__":
    test_results = test_algorithm_system()
    
    if test_results:
        print(f"\n💡 QUICK START GUIDE:")
        print(f"   1. Run: python backtest_algorithms.py")
        print(f"   2. Select SPY data source")
        print(f"   3. Choose algorithms to test")
        print(f"   4. View results and analysis")
        print(f"   5. Optional: Generate plots and Monte Carlo analysis")

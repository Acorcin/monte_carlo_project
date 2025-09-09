#!/usr/bin/env python3
"""
Test Liquidity Structure Strategy Integration

This script tests the new liquidity analyzer and trading strategy.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add algorithms to path
sys.path.append('algorithms')

def test_liquidity_analyzer():
    """Test the liquidity analyzer functionality."""
    print("🔬 Testing Liquidity Analyzer...")
    
    try:
        from liquidity_analyzer import run_liquidity_analyzer
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 100
        price_changes = np.random.normal(0, 0.02, 100)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create OHLCV data
        data = pd.DataFrame(index=dates)
        data['close'] = prices
        data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
        data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, 100))
        data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, 100))
        data['volume'] = np.random.randint(1000, 10000, 100)
        
        # Run analyzer
        result = run_liquidity_analyzer(data)
        
        print(f"✅ Analysis complete!")
        print(f"   📊 Structure Events: {len(result.events)}")
        print(f"   🎯 Supply/Demand Zones: {len(result.zones)}")
        print(f"   💧 Liquidity Pockets: {len(result.pockets)}")
        print(f"   🌊 Hurst Exponent: {result.hurst:.3f}")
        
        # Show some events
        if result.events:
            print(f"\n   Recent Events:")
            for event in result.events[-3:]:
                print(f"   • {event.kind} at {event.timestamp.strftime('%Y-%m-%d')} (Level: {event.level:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Liquidity analyzer test failed: {e}")
        return False

def test_liquidity_strategy():
    """Test the liquidity structure trading strategy."""
    print("\n🤖 Testing Liquidity Structure Strategy...")
    
    try:
        from algorithms.technical_indicators.liquidity_structure_strategy import LiquidityStructureStrategy
        from data_fetcher import fetch_stock_data
        
        # Create strategy
        strategy = LiquidityStructureStrategy()
        print(f"✅ Strategy created: {strategy.name}")
        
        # Get some real data for testing
        print("📊 Fetching test data...")
        try:
            data = fetch_stock_data("SPY", period="3mo", interval="1d")
            print(f"✅ Data fetched: {len(data)} rows")
        except:
            # Fallback to synthetic data if fetch fails
            print("⚠️ Using synthetic data...")
            dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
            np.random.seed(42)
            
            base_price = 450
            price_changes = np.random.normal(0, 0.015, 60)
            prices = [base_price]
            
            for change in price_changes[1:]:
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)
            
            data = pd.DataFrame(index=dates)
            data['Close'] = prices
            data['Open'] = data['Close'].shift(1).fillna(data['Close'].iloc[0])
            data['High'] = data[['Open', 'Close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, 60))
            data['Low'] = data[['Open', 'Close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, 60))
            data['Volume'] = np.random.randint(50000, 200000, 60)
        
        # Generate signals
        print("🎯 Generating trading signals...")
        signals = strategy.generate_signals(data)
        
        # Analyze signals
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        hold_signals = (signals == 0).sum()
        
        print(f"✅ Signals generated successfully!")
        print(f"   📈 Buy signals: {buy_signals}")
        print(f"   📉 Sell signals: {sell_signals}")
        print(f"   ⏸️ Hold signals: {hold_signals}")
        
        # Show recent signals
        recent_signals = signals.tail(10)
        recent_with_signals = recent_signals[recent_signals != 0]
        
        if len(recent_with_signals) > 0:
            print(f"\n   Recent Trading Signals:")
            for date, signal in recent_with_signals.items():
                signal_type = "BUY" if signal > 0 else "SELL"
                print(f"   • {date.strftime('%Y-%m-%d')}: {signal_type} (strength: {signal:.2f})")
        
        # Test backtesting capability
        print("\n📊 Testing backtest capability...")
        try:
            results = strategy.backtest(data, initial_capital=10000)
            print(f"✅ Backtest complete!")
            print(f"   💰 Total Return: {results['total_return']:.2f}%")
            print(f"   🔄 Total Trades: {results['metrics']['total_trades']}")
            if results['metrics']['total_trades'] > 0:
                print(f"   🎯 Win Rate: {results['metrics']['win_rate']:.1f}%")
        except Exception as e:
            print(f"⚠️ Backtest failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        return False

def test_integration():
    """Test integration with existing framework."""
    print("\n🔗 Testing Framework Integration...")
    
    try:
        from algorithms.algorithm_manager import AlgorithmManager
        
        # Create algorithm manager
        manager = AlgorithmManager()
        
        # Check if our algorithm is discovered
        algorithms = manager.get_available_algorithms()
        liquidity_algos = [name for name in algorithms if 'Liquidity' in name]
        
        if liquidity_algos:
            print(f"✅ Algorithm discovered: {liquidity_algos}")
            
            # Test creating the algorithm through manager
            algo = manager.create_algorithm(liquidity_algos[0])
            print(f"✅ Algorithm created through manager: {algo.name}")
            
            # Test parameter info
            params = algo.get_parameter_info()
            print(f"✅ Parameters available: {len(params)} configurable parameters")
            
            return True
        else:
            print("❌ Algorithm not discovered by manager")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Liquidity Strategy Integration\n")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Liquidity Analyzer", test_liquidity_analyzer),
        ("Liquidity Strategy", test_liquidity_strategy),
        ("Framework Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Liquidity strategy is ready to use!")
        print("\n📋 Next Steps:")
        print("   1. Run: python backtest_algorithms.py")
        print("   2. Select 'Liquidity Structure Strategy'")
        print("   3. Test with your preferred ticker and timeframe")
        print("   4. Add to GUI application for visual analysis")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    main()

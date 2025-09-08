#!/usr/bin/env python3
"""
Test script to verify the enhanced drawdown calculation functionality.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add algorithms to path
sys.path.append('algorithms')

# Import our modules
from data_fetcher import fetch_stock_data
from algorithms.algorithm_manager import algorithm_manager
from backtest_algorithms import display_results, display_drawdown_analysis

def test_drawdown_calculation():
    """Test the enhanced drawdown calculation functionality."""
    print("🧪 TESTING ENHANCED DRAWDOWN CALCULATION")
    print("=" * 60)
    print("🎯 Formula: [(Highest Peak - Lowest Trough) / Highest Peak] × 100")
    print("=" * 60)
    
    try:
        # Get some sample data
        print("📊 Loading sample data...")
        data = fetch_stock_data("SPY", period="3mo", interval="1d")
        print(f"✅ Loaded {len(data)} data points")
        
        # Run a backtest with one algorithm
        print("\n🤖 Running backtest with RSIOversoldOverbought...")
        
        # Get the algorithm
        available_algos = algorithm_manager.get_available_algorithms()
        if 'RSIOversoldOverbought' in available_algos:
            algo_config = {'name': 'RSIOversoldOverbought', 'parameters': {}}
            
            results = algorithm_manager.backtest_multiple_algorithms(
                algorithm_configs=[algo_config],
                data=data,
                initial_capital=10000
            )
            
            if results and 'RSIOversoldOverbought' in results:
                print("✅ Backtest completed successfully")
                
                # Test the drawdown functionality
                result = results['RSIOversoldOverbought']
                metrics = result.get('metrics', {})
                
                print(f"\n📊 Algorithm generated {len(result.get('trades', []))} trades")
                print(f"📈 Total return: {result.get('total_return', 0):.2f}%")
                
                # Display enhanced drawdown metrics
                print("\n🎯 Testing Enhanced Drawdown Calculation:")
                print("=" * 60)
                
                if metrics:
                    print(f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
                    print(f"Peak Value: ${metrics.get('drawdown_peak_value', 0):,.2f}")
                    print(f"Trough Value: ${metrics.get('drawdown_trough_value', 0):,.2f}")
                    print(f"Dollar Loss: ${metrics.get('drawdown_peak_value', 0) - metrics.get('drawdown_trough_value', 0):,.2f}")
                    print(f"Duration: {metrics.get('drawdown_duration_days', 0)} periods")
                    print(f"Time Underwater: {metrics.get('time_underwater_pct', 0):.1f}%")
                    
                    # Verify the formula
                    peak = metrics.get('drawdown_peak_value', 0)
                    trough = metrics.get('drawdown_trough_value', 0)
                    if peak > 0:
                        calculated_dd = ((peak - trough) / peak) * 100
                        reported_dd = metrics.get('max_drawdown', 0)
                        print(f"\n✅ FORMULA VERIFICATION:")
                        print(f"   Calculated: {calculated_dd:.2f}%")
                        print(f"   Reported: {reported_dd:.2f}%")
                        print(f"   Match: {'✅ YES' if abs(calculated_dd - reported_dd) < 0.01 else '❌ NO'}")
                
                # Test the new display function
                print("\n🎯 Testing Enhanced Display Function:")
                display_drawdown_analysis(metrics, 'RSIOversoldOverbought')
                
            else:
                print("❌ Backtest failed or no results")
        else:
            print("❌ RSIOversoldOverbought not found")
            print(f"Available algorithms: {list(available_algos.keys())}")
    
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

def test_manual_drawdown():
    """Test drawdown calculation with manual data."""
    print("\n🧪 MANUAL DRAWDOWN TEST")
    print("=" * 40)
    
    # Create sample portfolio values with known drawdown
    portfolio_values = np.array([10000, 11000, 12000, 9000, 8000, 10000, 11500])
    initial_capital = 10000
    
    print(f"Sample portfolio values: {portfolio_values}")
    print(f"Expected max drawdown from 12000 to 8000: {((12000-8000)/12000)*100:.2f}%")
    
    # Calculate returns from portfolio values
    returns = (portfolio_values[1:] / portfolio_values[:-1]) - 1
    
    # Import base algorithm to test the method directly
    from algorithms.base_algorithm import TradingAlgorithm
    
    # Create a dummy algorithm to test the method
    class TestAlgorithm(TradingAlgorithm):
        def generate_signals(self, data): return pd.Series([0])
        def get_algorithm_type(self): return 'test'
    
    test_algo = TestAlgorithm("Test")
    drawdown_metrics = test_algo._calculate_enhanced_drawdown(returns, initial_capital)
    
    print(f"\n📊 CALCULATED METRICS:")
    print(f"Max Drawdown: {drawdown_metrics['max_drawdown']:.2f}%")
    print(f"Peak Value: ${drawdown_metrics['peak_value']:,.2f}")
    print(f"Trough Value: ${drawdown_metrics['trough_value']:,.2f}")
    
    print(f"\n✅ Formula verification: {((drawdown_metrics['peak_value'] - drawdown_metrics['trough_value']) / drawdown_metrics['peak_value']) * 100:.2f}%")

if __name__ == "__main__":
    test_drawdown_calculation()
    test_manual_drawdown()



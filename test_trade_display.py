#!/usr/bin/env python3
"""
Test script to verify that the trade display functionality works correctly.
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
from backtest_algorithms import display_results, display_individual_trades

def test_trade_display():
    """Test the trade display functionality."""
    print("🧪 TESTING TRADE DISPLAY FUNCTIONALITY")
    print("=" * 50)
    
    try:
        # Get some sample data
        print("📊 Loading sample data...")
        data = fetch_stock_data("SPY", period="1mo", interval="1d")
        print(f"✅ Loaded {len(data)} data points")
        
        # Run a simple backtest with one algorithm
        print("\n🤖 Running backtest with AdvancedMLStrategy...")
        
        # Get the algorithm
        available_algos = algorithm_manager.get_available_algorithms()
        if 'AdvancedMLStrategy' in available_algos:
            algo_config = {'name': 'AdvancedMLStrategy', 'parameters': {}}
            
            results = algorithm_manager.backtest_multiple_algorithms(
                algorithm_configs=[algo_config],
                data=data,
                initial_capital=10000
            )
            
            if results and 'AdvancedMLStrategy' in results:
                print("✅ Backtest completed successfully")
                
                # Test the display functionality
                print("\n📋 Testing individual trade display...")
                result = results['AdvancedMLStrategy']
                trades = result.get('trades', [])
                
                print(f"Algorithm generated {len(trades)} trades")
                
                if trades:
                    print("\n📊 Sample trade structure:")
                    for key, value in trades[0].items():
                        print(f"  {key}: {value} ({type(value)})")
                
                # Test the new display function
                print("\n🎯 Testing new display_individual_trades function:")
                display_individual_trades(trades, 'AdvancedMLStrategy')
                
                # Test the updated display_results function
                print("\n🎯 Testing updated display_results function:")
                display_results(results)
                
            else:
                print("❌ Backtest failed or no results")
        else:
            print("❌ AdvancedMLStrategy not found")
            print(f"Available algorithms: {list(available_algos.keys())}")
    
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_trade_display()



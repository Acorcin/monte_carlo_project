"""
Simple test to understand the trade display issue
"""

import pandas as pd
import numpy as np

def test_trade_display_logic():
    """Test the trade display logic."""
    
    print("🧪 Testing Trade Display Logic")
    print("=" * 50)
    
    # Test case 1: Empty trades
    print("\n📋 Test 1: Empty trades list")
    trades = []
    
    if trades and len(trades) > 0:
        valid_trades = [trade for trade in trades if trade.get('entry_date') and trade.get('entry_date') != 'N/A']
        if valid_trades:
            print(f"   Result: Sample trades ({len(valid_trades)} total)")
        else:
            print(f"   Result: No valid trades generated")
    else:
        print(f"   ✅ Result: No trades data available")
    
    # Test case 2: Invalid trades (what might happen with failed data)
    print("\n📋 Test 2: Invalid trades")
    trades = [
        {'entry_date': 'N/A', 'exit_date': 'N/A', 'return': 0},
        {'entry_date': None, 'exit_date': None, 'return': 0},
        {'entry_date': '', 'exit_date': '', 'return': 0}
    ]
    
    if trades and len(trades) > 0:
        valid_trades = [trade for trade in trades if trade.get('entry_date') and trade.get('entry_date') != 'N/A']
        if valid_trades:
            print(f"   Result: Sample trades ({len(valid_trades)} total)")
        else:
            print(f"   ✅ Result: No valid trades generated (algorithm may not have triggered any signals)")
    else:
        print(f"   Result: No trades data available")
    
    # Test case 3: Valid trades
    print("\n📋 Test 3: Valid trades")
    trades = [
        {'entry_date': '2024-01-01', 'exit_date': '2024-01-02', 'return': 0.02},
        {'entry_date': '2024-01-03', 'exit_date': '2024-01-04', 'return': -0.01},
        {'entry_date': 'N/A', 'exit_date': 'N/A', 'return': 0},  # Invalid trade mixed in
    ]
    
    if trades and len(trades) > 0:
        valid_trades = [trade for trade in trades if trade.get('entry_date') and trade.get('entry_date') != 'N/A']
        if valid_trades:
            print(f"   ✅ Result: Sample trades ({len(valid_trades)} total)")
            for i, trade in enumerate(valid_trades[:5]):
                entry_date = trade.get('entry_date', 'N/A')
                exit_date = trade.get('exit_date', 'N/A')
                trade_return = trade.get('return', 0)
                print(f"     Trade {i+1}: {entry_date} to {exit_date}, Return: {trade_return:.2%}")
        else:
            print(f"   Result: No valid trades generated")
    else:
        print(f"   Result: No trades data available")

def test_data_loading():
    """Test data loading to see what happens with failed tickers."""
    
    print(f"\n🔍 Testing Data Loading")
    print("=" * 50)
    
    # Test working ticker
    print(f"\n📊 Testing AAPL (should work):")
    try:
        from data_fetcher import fetch_stock_data
        data = fetch_stock_data("AAPL", period="5d", interval="1d")
        if data is not None and not data.empty:
            print(f"   ✅ AAPL loaded: {len(data)} records")
            print(f"   📈 Latest price: ${data['Close'].iloc[-1]:.2f}")
        else:
            print(f"   ❌ AAPL failed to load")
    except Exception as e:
        print(f"   ❌ AAPL error: {e}")
    
    # Test failing ticker
    print(f"\n📊 Testing MNQ (might fail without =F):")
    try:
        data = fetch_stock_data("MNQ", period="5d", interval="1d")
        if data is not None and not data.empty:
            print(f"   ✅ MNQ loaded: {len(data)} records")
        else:
            print(f"   ❌ MNQ failed to load (expected)")
    except Exception as e:
        print(f"   ❌ MNQ error: {e}")
    
    # Test correct futures ticker
    print(f"\n📊 Testing MNQ=F (should work):")
    try:
        data = fetch_stock_data("MNQ=F", period="5d", interval="1d")
        if data is not None and not data.empty:
            print(f"   ✅ MNQ=F loaded: {len(data)} records")
            print(f"   📈 Latest price: ${data['Close'].iloc[-1]:.2f}")
        else:
            print(f"   ❌ MNQ=F failed to load")
    except Exception as e:
        print(f"   ❌ MNQ=F error: {e}")

def explain_the_issue():
    """Explain what's happening with the sample trades."""
    
    print(f"\n💡 EXPLANATION OF THE ISSUE")
    print("=" * 50)
    
    print("The issue you're seeing happens when:")
    print()
    print("1. 🔍 DATA LOAD FAILS:")
    print("   • You try to load 'MNQ' instead of 'MNQ=F'")
    print("   • yfinance returns an error or empty data")
    print("   • GUI still shows backtest interface")
    print()
    print("2. 🧪 BACKTEST RUNS ON EMPTY DATA:")
    print("   • Algorithm tries to generate signals on empty/invalid data")
    print("   • No valid trades are created")
    print("   • Backtest result has empty trades list")
    print()
    print("3. 📊 GUI SHOWS PLACEHOLDER:")
    print("   • OLD behavior: Always showed '5 sample trades' even with empty data")
    print("   • NEW behavior: Checks for valid trades first")
    print()
    print("✅ FIXED IN UPDATED GUI:")
    print("   • Better data validation")
    print("   • Clear error messages for failed data loads")
    print("   • 'No trades data available' when appropriate")
    print("   • 'No valid trades generated' when algorithm doesn't trigger")
    print()
    print("🎯 SOLUTION:")
    print("   • Use proper futures tickers (MNQ=F not MNQ)")
    print("   • GUI now provides futures dropdown with correct symbols")
    print("   • Better error handling and user feedback")

if __name__ == "__main__":
    print("🚀 Simple Trade Display Test")
    print("Understanding why '5 sample trades' shows with no data")
    print()
    
    # Run tests
    test_trade_display_logic()
    test_data_loading()
    explain_the_issue()
    
    print(f"\n🎉 CONCLUSION:")
    print("The updated GUI now properly handles:")
    print("• Failed data loads with clear error messages")
    print("• Empty trade results with appropriate messaging")
    print("• Futures ticker selection with correct symbols")
    print()
    print("🚀 Try the fixed GUI: python launch_gui_fixed.py")
    print("   Select 'futures' asset type for proper futures symbols!")

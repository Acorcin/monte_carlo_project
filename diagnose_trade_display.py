"""
Diagnostic tool to understand why sample trades show with no data
"""

import pandas as pd
import numpy as np
from data_fetcher import fetch_stock_data
from algorithms.algorithm_manager import AlgorithmManager

def test_backtest_process():
    """Test the complete backtest process to see where trades come from."""
    
    print("🔍 Diagnosing Trade Display Issues")
    print("=" * 60)
    
    # Test with successful data first
    print("\n📊 Testing with AAPL (should work):")
    try:
        # Load data
        data = fetch_stock_data("AAPL", period="3mo", interval="1d")
        print(f"   ✅ Data loaded: {len(data)} records")
        
        # Test with algorithm manager
        algorithm_manager = AlgorithmManager()
        
        # Test a simple algorithm
        algorithm_name = "MovingAverageCrossover"
        if algorithm_name in algorithm_manager.algorithms:
            result = algorithm_manager.backtest_algorithm(algorithm_name, data, initial_capital=10000)
            
            if result:
                print(f"   ✅ Backtest completed")
                print(f"   📊 Result keys: {list(result.keys())}")
                print(f"   💰 Final capital: ${result.get('final_capital', 0):,.2f}")
                print(f"   📈 Total return: {result.get('total_return', 0):.2%}")
                
                # Check trades
                trades = result.get('trades', [])
                print(f"   🔄 Trades: {len(trades)} total")
                
                if trades:
                    print(f"   📋 First trade structure:")
                    for key, value in trades[0].items():
                        print(f"     {key}: {value}")
                    
                    # Check for valid trades
                    valid_trades = [trade for trade in trades if trade.get('entry_date') and trade.get('entry_date') != 'N/A']
                    print(f"   ✅ Valid trades: {len(valid_trades)}")
                    
                    if valid_trades:
                        print(f"   📋 Valid trade example:")
                        trade = valid_trades[0]
                        print(f"     Entry: {trade.get('entry_date')}")
                        print(f"     Exit: {trade.get('exit_date')}")
                        print(f"     Return: {trade.get('return', 0):.2%}")
                else:
                    print(f"   ❌ No trades generated")
                
                # Check returns
                returns = result.get('returns', [])
                print(f"   📊 Returns: {len(returns)} values")
                if returns:
                    print(f"     Sample: {returns[:3]}")
                
            else:
                print(f"   ❌ Backtest failed")
        else:
            print(f"   ❌ Algorithm not found: {algorithm_name}")
            print(f"   Available: {list(algorithm_manager.algorithms.keys())}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test with failed data
    print(f"\n📊 Testing with MNQ (might fail):")
    try:
        # Try to load MNQ data
        data = fetch_stock_data("MNQ", period="3mo", interval="1d")
        if data is not None and not data.empty:
            print(f"   ✅ MNQ data loaded: {len(data)} records")
        else:
            print(f"   ❌ MNQ data failed to load")
            
            # Test what happens with empty backtest
            print(f"   🔍 Testing empty backtest behavior...")
            
            # Simulate what might happen with no data
            empty_result = {
                'algorithm_name': 'TestAlgorithm',
                'initial_capital': 10000,
                'final_capital': 10000,
                'total_return': 0.0,
                'trades': [],
                'returns': [],
                'metrics': {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'avg_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'profit_factor': 0.0
                }
            }
            
            print(f"   📊 Empty result structure created")
            print(f"   🔄 Trades: {len(empty_result['trades'])}")
            print(f"   📊 Returns: {len(empty_result['returns'])}")
            
            # Test trade display logic
            trades = empty_result['trades']
            if trades and len(trades) > 0:
                valid_trades = [trade for trade in trades if trade.get('entry_date') and trade.get('entry_date') != 'N/A']
                print(f"   ✅ Would show: {len(valid_trades)} valid trades")
            else:
                print(f"   ✅ Would show: 'No trades data available'")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test specific algorithm behavior
    print(f"\n🧪 Testing Algorithm Signal Generation:")
    
    try:
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        sample_data = pd.DataFrame({
            'Open': 100 + np.cumsum(np.random.randn(50) * 0.5),
            'High': 100 + np.cumsum(np.random.randn(50) * 0.5) + 1,
            'Low': 100 + np.cumsum(np.random.randn(50) * 0.5) - 1,
            'Close': 100 + np.cumsum(np.random.randn(50) * 0.5),
            'Volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
        
        # Ensure High >= Close >= Low
        sample_data['High'] = sample_data[['Open', 'Close', 'High']].max(axis=1)
        sample_data['Low'] = sample_data[['Open', 'Close', 'Low']].min(axis=1)
        
        print(f"   📊 Sample data created: {len(sample_data)} records")
        
        # Test algorithm
        algorithm_manager = AlgorithmManager()
        for algo_name in ['MovingAverageCrossover', 'RSIOversoldOverbought']:
            if algo_name in algorithm_manager.algorithms:
                print(f"   🧪 Testing {algo_name}:")
                
                try:
                    algorithm = algorithm_manager.create_algorithm(algo_name)
                    signals = algorithm.generate_signals(sample_data)
                    
                    buy_signals = (signals == 1).sum()
                    sell_signals = (signals == -1).sum()
                    hold_signals = (signals == 0).sum()
                    
                    print(f"     📈 Buy signals: {buy_signals}")
                    print(f"     📉 Sell signals: {sell_signals}")
                    print(f"     🔄 Hold signals: {hold_signals}")
                    
                    if buy_signals > 0 or sell_signals > 0:
                        print(f"     ✅ Algorithm generates signals")
                    else:
                        print(f"     ❌ No signals generated")
                        
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_gui_display_logic():
    """Test the GUI display logic specifically."""
    
    print(f"\n🎨 Testing GUI Display Logic:")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            'name': 'Empty trades list',
            'trades': [],
            'expected': 'No trades data available'
        },
        {
            'name': 'Trades with no valid data',
            'trades': [
                {'entry_date': 'N/A', 'exit_date': 'N/A', 'return': 0},
                {'entry_date': None, 'exit_date': None, 'return': 0}
            ],
            'expected': 'No valid trades generated'
        },
        {
            'name': 'Valid trades',
            'trades': [
                {'entry_date': '2024-01-01', 'exit_date': '2024-01-02', 'return': 0.02},
                {'entry_date': '2024-01-03', 'exit_date': '2024-01-04', 'return': -0.01}
            ],
            'expected': 'Sample trades'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        trades = test_case['trades']
        
        # Simulate GUI logic
        if trades and len(trades) > 0:
            valid_trades = [trade for trade in trades if trade.get('entry_date') and trade.get('entry_date') != 'N/A']
            
            if valid_trades:
                result = f"Sample trades ({len(valid_trades)} total)"
                print(f"   ✅ Result: '{result}'")
                for i, trade in enumerate(valid_trades[:5]):
                    entry_date = trade.get('entry_date', 'N/A')
                    exit_date = trade.get('exit_date', 'N/A')
                    trade_return = trade.get('return', 0)
                    print(f"     Trade {i+1}: {entry_date} to {exit_date}, Return: {trade_return:.2%}")
            else:
                result = "No valid trades generated (algorithm may not have triggered any signals)"
                print(f"   ✅ Result: '{result}'")
        else:
            result = "No trades data available"
            print(f"   ✅ Result: '{result}'")
        
        expected = test_case['expected']
        if expected in result:
            print(f"   ✅ PASS - Contains expected text")
        else:
            print(f"   ❌ FAIL - Expected '{expected}', got '{result}'")

if __name__ == "__main__":
    print("🧪 Trade Display Diagnostic Tool")
    print("This will help understand why sample trades show with no data")
    print()
    
    # Run tests
    test_backtest_process()
    test_gui_display_logic()
    
    print(f"\n📋 SUMMARY:")
    print("The issue likely occurs when:")
    print("1. Data fails to load (like MNQ without =F)")
    print("2. Algorithm doesn't generate any signals")
    print("3. Backtest returns empty/invalid trade data")
    print()
    print("💡 SOLUTION:")
    print("The updated GUI now:")
    print("• Checks for valid trade data before displaying")
    print("• Shows helpful messages when no trades available")
    print("• Provides better error handling for failed data loads")
    print("• Filters out invalid/empty trades")
    
    print(f"\n🚀 Test the fixed GUI: python launch_gui_fixed.py")

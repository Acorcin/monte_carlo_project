"""
Test the Advanced ML Trading Algorithm

This script tests the new ML algorithm and demonstrates its capabilities.
"""

import pandas as pd
import numpy as np
from algorithms.machine_learning.advanced_ml_strategy import AdvancedMLStrategy
from data_fetcher import fetch_stock_data

def test_ml_algorithm():
    """Test the Advanced ML Trading Algorithm."""
    print("🧠 TESTING ADVANCED ML TRADING ALGORITHM")
    print("="*60)
    
    # Fetch test data
    print("📊 Fetching test data (SPY, 6 months)...")
    try:
        data = fetch_stock_data('SPY', period='6mo', interval='1d')
        print(f"   ✅ Loaded {len(data)} data points")
        print(f"   📅 Date range: {data.index[0]} to {data.index[-1]}")
    except Exception as e:
        print(f"   ❌ Data fetch failed: {e}")
        return
    
    # Initialize algorithm
    print("\n🤖 Initializing Advanced ML Strategy...")
    algorithm = AdvancedMLStrategy()
    
    # Display algorithm info
    print(f"   Algorithm: {algorithm.name}")
    print(f"   Description: {algorithm.description}")
    
    # Show parameters
    print(f"\n⚙️  Algorithm Parameters:")
    for key, value in algorithm.get_parameters().items():
        print(f"   {key}: {value}")
    
    # Test signal generation
    print(f"\n🔄 Generating trading signals...")
    try:
        signals = algorithm.calculate_signals(data)
        
        print(f"   ✅ Generated signals for {len(signals)} periods")
        
        # Analyze signals
        total_signals = len(signals[signals['signal'] != 0])
        buy_signals = len(signals[signals['signal'] == 1])
        sell_signals = len(signals[signals['signal'] == -1])
        
        print(f"\n📈 Signal Analysis:")
        print(f"   Total signals: {total_signals}")
        print(f"   Buy signals: {buy_signals}")
        print(f"   Sell signals: {sell_signals}")
        print(f"   Signal rate: {total_signals/len(signals):.1%}")
        
        if total_signals > 0:
            avg_probability = signals[signals['signal'] != 0]['probability'].mean()
            avg_position_size = signals[signals['signal'] != 0]['position_size'].abs().mean()
            print(f"   Average probability: {avg_probability:.3f}")
            print(f"   Average position size: {avg_position_size:.3f}")
        
        # Show sample signals
        print(f"\n📋 Sample Signals (last 10 non-zero):")
        sample_signals = signals[signals['signal'] != 0].tail(10)
        if len(sample_signals) > 0:
            for idx, row in sample_signals.iterrows():
                signal_type = "BUY" if row['signal'] == 1 else "SELL"
                print(f"   {idx.date()}: {signal_type} | "
                      f"Prob: {row['probability']:.3f} | "
                      f"Size: {row['position_size']:.3f}")
        else:
            print("   No trading signals generated")
        
        return signals
        
    except Exception as e:
        print(f"   ❌ Signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_simple_backtest(data, signals):
    """Run a simple backtest of the ML algorithm."""
    if signals is None or len(signals) == 0:
        return
    
    print(f"\n💰 RUNNING SIMPLE BACKTEST")
    print("="*40)
    
    # Simple backtest logic
    portfolio_value = 10000  # Starting capital
    position = 0  # Current position
    trades = []
    
    for i, (date, row) in enumerate(signals.iterrows()):
        if i == 0:
            continue  # Skip first row
        
        current_price = data.loc[date, 'Close']
        
        # Execute trades based on signals
        if row['signal'] == 1 and position <= 0:  # Buy signal
            shares = portfolio_value / current_price
            position = shares
            trades.append({
                'date': date,
                'action': 'BUY',
                'price': current_price,
                'shares': shares,
                'portfolio_value': portfolio_value
            })
            
        elif row['signal'] == -1 and position >= 0:  # Sell signal
            if position > 0:
                portfolio_value = position * current_price
                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'portfolio_value': portfolio_value
                })
                position = 0
    
    # Calculate final portfolio value
    if position > 0:
        final_price = data.iloc[-1]['Close']
        final_value = position * final_price
    else:
        final_value = portfolio_value
    
    total_return = (final_value - 10000) / 10000
    
    print(f"📊 Backtest Results:")
    print(f"   Initial Capital: $10,000")
    print(f"   Final Value: ${final_value:,.2f}")
    print(f"   Total Return: {total_return:.2%}")
    print(f"   Number of Trades: {len(trades)}")
    
    if len(trades) > 0:
        print(f"\n📋 Trade History:")
        for trade in trades[-5:]:  # Show last 5 trades
            print(f"   {trade['date'].date()}: {trade['action']} at ${trade['price']:.2f}")

if __name__ == "__main__":
    # Test the algorithm
    signals = test_ml_algorithm()
    
    if signals is not None:
        # Get the data again for backtesting
        try:
            data = fetch_stock_data('SPY', period='6mo', interval='1d')
            run_simple_backtest(data, signals)
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
    
    print(f"\n✅ ML ALGORITHM TEST COMPLETE!")
    print("The Advanced ML Strategy is now available in the algorithm selection menu.")

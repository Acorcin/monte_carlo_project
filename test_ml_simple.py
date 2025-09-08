import sys
sys.path.append('algorithms')

from machine_learning.advanced_ml_strategy import AdvancedMLStrategy
from data_fetcher import fetch_stock_data

print('🧠 TESTING ADVANCED ML ALGORITHM')
print('='*50)

# Test with sample data
print('📊 Fetching test data...')
data = fetch_stock_data('SPY', period='6mo', interval='1d')
print(f'   ✅ Loaded {len(data)} data points')

# Initialize algorithm
algo = AdvancedMLStrategy()
print(f'\n🤖 Algorithm: {algo.name}')
print(f'   Type: {algo.get_algorithm_type()}')

# Show some parameters
params = algo.get_parameters()
print(f'\n⚙️  Key Parameters:')
print(f'   Target Vol: {params["target_vol"]}')
print(f'   Max Leverage: {params["max_leverage"]}')
print(f'   ML Splits: {params["n_splits"]}')

print('\n🔄 Running ML signal generation (this may take a moment)...')
try:
    signals = algo.calculate_signals(data)
    total_signals = len(signals[signals.signal != 0])
    
    print(f'   ✅ Generated {total_signals} trading signals from {len(signals)} periods')
    
    if total_signals > 0:
        buy_signals = len(signals[signals.signal == 1])
        sell_signals = len(signals[signals.signal == -1])
        avg_prob = signals[signals.signal != 0]['probability'].mean()
        avg_pos_size = signals[signals.signal != 0]['position_size'].abs().mean()
        
        print(f'   📈 Buy signals: {buy_signals}')
        print(f'   📉 Sell signals: {sell_signals}')
        print(f'   🎯 Average probability: {avg_prob:.3f}')
        print(f'   📊 Average position size: {avg_pos_size:.3f}')
    
    print('\n✅ Advanced ML Algorithm test successful!')
    print('🚀 Algorithm is ready for backtesting and Monte Carlo simulation!')
    
except Exception as e:
    print(f'   ❌ Error: {e}')
    import traceback
    traceback.print_exc()

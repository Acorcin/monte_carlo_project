import sys
sys.path.append('algorithms')
from market_scenario_simulation import generate_market_scenarios, test_strategy_across_scenarios
from machine_learning.advanced_ml_strategy import AdvancedMLStrategy
from data_fetcher import fetch_stock_data

print('🔧 TESTING WITH LONGER SCENARIOS')
print('='*40)

# Get more data
data = fetch_stock_data('SPY', period='1y', interval='1d')
print(f'Loaded {len(data)} data points')

# Generate longer scenarios that have enough data for ML
scenarios, _ = generate_market_scenarios(data, num_scenarios=4, scenario_length=150)  # Longer scenarios

# Test ML strategy
ml_algo = AdvancedMLStrategy()

def ml_strategy_wrapper(price_data):
    try:
        trades = ml_algo.generate_signals(price_data)
        return trades
    except Exception as e:
        print(f'   ❌ ML Error: {str(e)[:50]}...')
        return []

print(f'\n🚀 Testing with longer scenarios (150 periods)...')
results = test_strategy_across_scenarios(ml_strategy_wrapper, scenarios, initial_capital=10000)

if results:
    print(f'\n✅ SUCCESS! Results:')
    for regime, stats in results.items():
        print(f'   {regime}: {stats["num_scenarios"]} scenarios')
        print(f'     Avg return: {stats["mean_return"]:.2%}')
        print(f'     Win rate: {stats["win_rate"]:.1%}')
else:
    print('❌ Still no results - ML algorithm needs even more data')

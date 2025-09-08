import sys
sys.path.append('algorithms')
from market_scenario_simulation import generate_market_scenarios, test_strategy_across_scenarios
from machine_learning.advanced_ml_strategy import AdvancedMLStrategy
from data_fetcher import fetch_stock_data

print('🔧 TESTING FIXED MARKET SCENARIO INTEGRATION')
print('='*50)

# Get data
data = fetch_stock_data('SPY', period='3mo', interval='1d')
print(f'Loaded {len(data)} data points')

# Generate small test scenarios
scenarios, _ = generate_market_scenarios(data, num_scenarios=4, scenario_length=30)
print(f'Generated scenarios: {[f"{k}: {len(v)}" for k, v in scenarios.items()]}')

# Test ML strategy wrapper
ml_algo = AdvancedMLStrategy()

def ml_strategy_wrapper(price_data):
    try:
        return ml_algo.generate_signals(price_data)
    except Exception as e:
        print(f'   ❌ ML Error: {e}')
        return []

# Test one scenario manually
if 'bull' in scenarios and len(scenarios['bull']) > 0:
    print('\n🧪 Testing single bull scenario...')
    bull_scenario = scenarios['bull'][0]
    
    # Convert to prices and create OHLCV
    prices = [100]
    for ret in bull_scenario:
        prices.append(prices[-1] * (1 + ret))
    
    import pandas as pd
    from market_scenario_simulation import create_synthetic_ohlcv_from_prices
    price_series = pd.Series(prices[1:], index=pd.date_range('2024-01-01', periods=len(prices)-1))
    ohlcv_data = create_synthetic_ohlcv_from_prices(price_series)
    
    print(f'   📊 Created OHLCV data: {len(ohlcv_data)} periods')
    print(f'   📋 Columns: {list(ohlcv_data.columns)}')
    
    # Test ML algorithm
    trades = ml_strategy_wrapper(ohlcv_data)
    print(f'   ✅ ML generated {len(trades)} trades')

print('\n🚀 Testing full scenario integration...')
results = test_strategy_across_scenarios(ml_strategy_wrapper, scenarios, initial_capital=10000)

if results:
    print(f'✅ SUCCESS! Scenario testing completed for {len(results)} market types')
    for regime, stats in results.items():
        print(f'   {regime}: {stats["num_scenarios"]} scenarios, avg return {stats["mean_return"]:.2%}')
else:
    print('❌ No results generated')

import sys
sys.path.append('algorithms')
from market_scenario_simulation import generate_market_scenarios
from data_fetcher import fetch_stock_data
import pandas as pd

# Check what the market scenario data looks like
print('🔍 INVESTIGATING THE FAILURE...')
print('='*50)

# Get original data format
original_data = fetch_stock_data('SPY', period='3mo', interval='1d')
print('📊 Original SPY data columns:')
print(f'   {list(original_data.columns)}')
print(f'   Sample row:')
for col in original_data.columns:
    print(f'     {col}: {original_data.iloc[0][col]:.2f}')

# Generate market scenario
scenarios, _ = generate_market_scenarios(original_data, num_scenarios=1, scenario_length=10)

# Check what scenario data looks like
print('\n🎲 Market scenario data structure:')
bull_scenario = scenarios['bull'][0]  # First bull scenario
print(f'   Type: {type(bull_scenario)}')
print(f'   Shape: {bull_scenario.shape}')
print(f'   Sample values: {bull_scenario[:3]}')

# Show how scenario testing creates price data
print('\n🏗️  How scenario testing works:')
scenario_returns = bull_scenario
prices = [100]
for ret in scenario_returns:
    prices.append(prices[-1] * (1 + ret))

price_data = pd.DataFrame({
    'Close': prices[1:],  # Remove initial price
    'Date': pd.date_range('2024-01-01', periods=len(prices)-1)
})

print('   Created price_data columns:', list(price_data.columns))
print('   Missing columns needed by ML:', ['Open', 'High', 'Low', 'Volume'])

print('\n❌ PROBLEM IDENTIFIED:')
print('   • Original data: Full OHLCV DataFrame')
print('   • Scenario data: Only return arrays → Only Close prices')
print('   • ML Algorithm needs: OHLCV columns for feature engineering')
print('\n✅ SOLUTION NEEDED:')
print('   • Create synthetic OHLCV data from scenario prices')
print('   • Or modify ML algorithm to work with price-only data')

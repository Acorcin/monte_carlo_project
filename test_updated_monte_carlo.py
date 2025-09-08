import sys
sys.path.append('algorithms')
from algorithms.algorithm_manager import AlgorithmManager
from data_fetcher import fetch_stock_data
from backtest_algorithms import monte_carlo_integration

print('🧪 TESTING UPDATED MONTE CARLO INTEGRATION')
print('='*50)

# Get data and run backtest
data = fetch_stock_data('SPY', period='1y', interval='1d')
manager = AlgorithmManager()

# Run quick backtest
results = {'AdvancedMLStrategy': manager.backtest_algorithm('AdvancedMLStrategy', data, initial_capital=10000)}

if results['AdvancedMLStrategy'] and results['AdvancedMLStrategy']['returns']:
    print(f'\n📊 Backtest completed:')
    print(f'   Trades: {len(results["AdvancedMLStrategy"]["returns"])}')
    
    print(f'\n🎲 Running Monte Carlo with fixed display...')
    monte_carlo_integration(results, num_simulations=500, simulation_method='synthetic_returns')
else:
    print('❌ No trades generated for testing')

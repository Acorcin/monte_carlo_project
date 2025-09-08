import sys
sys.path.append('algorithms')
from algorithms.algorithm_manager import AlgorithmManager
from data_fetcher import fetch_stock_data

print('🔧 TESTING FIXED ADVANCED ML STRATEGY')
print('='*50)

# Initialize manager
manager = AlgorithmManager()

# Get test data with more data points
data = fetch_stock_data('SPY', period='1y', interval='1d')
print(f'Data loaded: {len(data)} points')

# Test Advanced ML Strategy
print('\n🧠 Testing Fixed Advanced ML Strategy...')
try:
    results = manager.backtest_algorithm('AdvancedMLStrategy', data, initial_capital=10000)
    
    print(f'✅ Backtest completed successfully!')
    print(f'   Algorithm: {results["algorithm_name"]}')
    print(f'   Total trades: {results["metrics"]["total_trades"]}')
    print(f'   Final capital: ${results["final_capital"]:,.2f}')
    print(f'   Total return: {results["total_return"]:.2%}')
    
    if results['metrics']['total_trades'] > 0:
        print(f'   Win rate: {results["metrics"]["win_rate"]:.1%}')
        print(f'   Sharpe ratio: {results["metrics"]["sharpe_ratio"]:.3f}')
        print(f'   Max drawdown: {results["metrics"]["max_drawdown"]:.2%}')
        print(f'\n📈 Sample trades:')
        for i, trade in enumerate(results['trades'][:3]):  # Show first 3 trades
            entry_date = trade.get('entry_date', 'N/A')
            exit_date = trade.get('exit_date', 'N/A')
            trade_return = trade.get('return', 0)
            print(f'   Trade {i+1}: {entry_date} to {exit_date}, Return: {trade_return:.2%}')
    
    print('\n🎯 Advanced ML Strategy is now working in the backtesting system!')
    
except Exception as e:
    print(f'❌ Backtest still failed: {e}')
    import traceback
    traceback.print_exc()

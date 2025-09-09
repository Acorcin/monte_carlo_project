"""
Integration tests for full workflow scenarios.

Tests complete end-to-end workflows from data loading to Monte Carlo analysis.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

# Import modules for integration testing
import data_fetcher
import monte_carlo_trade_simulation
from algorithms.algorithm_manager import AlgorithmManager


class TestDataToBacktestWorkflow:
    """Test integration from data fetching to backtesting."""

    @pytest.mark.integration
    def test_complete_data_to_backtest_workflow(self, mock_yfinance_data):
        """Test complete workflow from data fetching to backtesting."""
        with patch('yfinance.download', side_effect=mock_yfinance_data):
            # Step 1: Fetch data
            data = data_fetcher.fetch_stock_data('AAPL', period='1y', interval='1d')
            
            assert data is not None
            assert not data.empty
            
            # Step 2: Initialize algorithm manager
            manager = AlgorithmManager()
            assert len(manager.algorithms) > 0
            
            # Step 3: Create algorithm
            algo_name = list(manager.algorithms.keys())[0]
            algorithm = manager.create_algorithm(algo_name)
            assert algorithm is not None
            
            # Step 4: Generate signals
            signals = algorithm.generate_signals(data)
            assert isinstance(signals, pd.Series)
            assert len(signals) == len(data)
            
            # Step 5: Basic backtest calculation
            returns = data['Close'].pct_change().fillna(0)
            strategy_returns = signals.shift(1) * returns
            cumulative_returns = (1 + strategy_returns).cumprod()
            total_return = cumulative_returns.iloc[-1] - 1
            
            assert isinstance(total_return, (int, float))

    @pytest.mark.integration
    def test_multi_algorithm_workflow(self, mock_yfinance_data):
        """Test workflow with multiple algorithms."""
        with patch('yfinance.download', side_effect=mock_yfinance_data):
            data = data_fetcher.fetch_stock_data('AAPL', period='6mo', interval='1d')
            
            manager = AlgorithmManager()
            
            # Test first 3 algorithms
            algorithm_names = list(manager.algorithms.keys())[:3]
            results = {}
            
            for algo_name in algorithm_names:
                algorithm = manager.create_algorithm(algo_name)
                signals = algorithm.generate_signals(data)
                
                # Calculate simple performance metrics
                returns = data['Close'].pct_change().fillna(0)
                strategy_returns = signals.shift(1) * returns
                total_return = (1 + strategy_returns).cumprod().iloc[-1] - 1
                
                results[algo_name] = {
                    'total_return': total_return,
                    'num_signals': (signals != 0).sum(),
                    'signals': signals
                }
            
            assert len(results) == len(algorithm_names)
            for algo_name, result in results.items():
                assert 'total_return' in result
                assert 'num_signals' in result
                assert isinstance(result['total_return'], (int, float))


class TestBacktestToMonteCarloWorkflow:
    """Test integration from backtesting to Monte Carlo simulation."""

    @pytest.mark.integration
    def test_backtest_to_monte_carlo_workflow(self, sample_stock_data, mock_backtest_results):
        """Test workflow from backtest results to Monte Carlo simulation."""
        # Step 1: Use mock backtest results
        backtest_results = mock_backtest_results
        
        # Step 2: Extract returns for Monte Carlo
        returns_data = backtest_results['returns']
        assert len(returns_data) > 0
        
        # Step 3: Run Monte Carlo simulation
        mc_results = monte_carlo_trade_simulation.random_trade_order_simulation(
            returns=returns_data,
            num_simulations=100,
            initial_capital=backtest_results['initial_capital']
        )
        
        assert isinstance(mc_results, pd.DataFrame)
        assert mc_results.shape[1] == 100  # Number of simulations
        assert (mc_results.iloc[0] == backtest_results['initial_capital']).all()
        
        # Step 4: Calculate risk metrics
        final_values = mc_results.iloc[-1]
        var_95 = np.percentile(final_values, 5)
        cvar_95 = final_values[final_values <= var_95].mean()
        
        assert var_95 <= backtest_results['initial_capital']
        assert cvar_95 <= var_95

    @pytest.mark.integration
    def test_consensus_strategy_workflow(self, sample_stock_data):
        """Test consensus strategy workflow."""
        manager = AlgorithmManager()
        
        if len(manager.algorithms) >= 2:
            # Step 1: Select multiple algorithms
            algo_names = list(manager.algorithms.keys())[:2]
            algorithms = [manager.create_algorithm(name) for name in algo_names]
            
            # Step 2: Generate signals for each algorithm
            all_signals = {}
            for i, algorithm in enumerate(algorithms):
                signals = algorithm.generate_signals(sample_stock_data)
                all_signals[algo_names[i]] = signals
            
            # Step 3: Generate consensus signals (all must agree)
            consensus_signals = pd.Series(0, index=sample_stock_data.index)
            
            # Find positions where all algorithms agree
            for idx in consensus_signals.index:
                signals_at_time = [signals.loc[idx] for signals in all_signals.values()]
                
                # All must agree and not be zero
                if len(set(signals_at_time)) == 1 and signals_at_time[0] != 0:
                    consensus_signals.loc[idx] = signals_at_time[0]
            
            # Step 4: Calculate consensus performance
            returns = sample_stock_data['Close'].pct_change().fillna(0)
            consensus_strategy_returns = consensus_signals.shift(1) * returns
            consensus_total_return = (1 + consensus_strategy_returns).cumprod().iloc[-1] - 1
            
            assert isinstance(consensus_total_return, (int, float))
            # Consensus should typically have fewer signals
            assert (consensus_signals != 0).sum() <= min(
                (signals != 0).sum() for signals in all_signals.values()
            )


class TestDataValidationWorkflow:
    """Test data validation throughout the workflow."""

    @pytest.mark.integration
    def test_data_quality_validation_workflow(self, sample_stock_data):
        """Test data quality validation at each step."""
        # Step 1: Validate input data
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        assert all(col in sample_stock_data.columns for col in required_columns)
        assert not sample_stock_data.isnull().any().any()
        
        # Step 2: Validate price relationships
        assert (sample_stock_data['High'] >= sample_stock_data['Low']).all()
        assert (sample_stock_data['High'] >= sample_stock_data['Open']).all()
        assert (sample_stock_data['High'] >= sample_stock_data['Close']).all()
        assert (sample_stock_data['Low'] <= sample_stock_data['Open']).all()
        assert (sample_stock_data['Low'] <= sample_stock_data['Close']).all()
        
        # Step 3: Validate through algorithm processing
        manager = AlgorithmManager()
        if len(manager.algorithms) > 0:
            algo_name = list(manager.algorithms.keys())[0]
            algorithm = manager.create_algorithm(algo_name)
            signals = algorithm.generate_signals(sample_stock_data)
            
            # Validate signals
            assert len(signals) == len(sample_stock_data)
            assert all(signal in [-1, 0, 1] for signal in signals)
        
        # Step 4: Validate returns calculation
        returns = sample_stock_data['Close'].pct_change()
        assert not returns.iloc[1:].isnull().any()  # First return is NaN, rest should be valid
        assert (returns.iloc[1:].abs() < 0.5).all()  # No extreme single-day moves

    @pytest.mark.integration
    def test_error_handling_workflow(self):
        """Test error handling throughout the workflow."""
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'Close': [100, np.nan, 102, np.inf, 104],  # Contains NaN and inf
            'Volume': [1000000, 1100000, -1000, 1300000, 1400000]  # Contains negative
        })
        
        manager = AlgorithmManager()
        
        if len(manager.algorithms) > 0:
            algo_name = list(manager.algorithms.keys())[0]
            algorithm = manager.create_algorithm(algo_name)
            
            # Algorithm should handle invalid data gracefully
            try:
                signals = algorithm.generate_signals(invalid_data)
                # If it succeeds, signals should be valid
                assert len(signals) == len(invalid_data)
                assert all(signal in [-1, 0, 1] for signal in signals if not pd.isna(signal))
            except (ValueError, IndexError):
                # Or it should raise a reasonable error
                pass


class TestPerformanceWorkflow:
    """Test performance aspects of the complete workflow."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_dataset_workflow(self, mock_yfinance_data):
        """Test workflow with large datasets."""
        # Create large mock data
        def large_mock_data(*args, **kwargs):
            dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
            np.random.seed(42)
            
            initial_price = 100.0
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = [initial_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            return pd.DataFrame({
                'Open': prices,
                'High': [p * 1.01 for p in prices],
                'Low': [p * 0.99 for p in prices],
                'Close': prices,
                'Volume': np.random.randint(1000000, 5000000, len(dates))
            }, index=dates)
        
        with patch('yfinance.download', side_effect=large_mock_data):
            import time
            start_time = time.time()
            
            # Full workflow with large dataset
            data = data_fetcher.fetch_stock_data('AAPL', period='5y', interval='1d')
            
            manager = AlgorithmManager()
            algo_name = list(manager.algorithms.keys())[0]
            algorithm = manager.create_algorithm(algo_name)
            
            signals = algorithm.generate_signals(data)
            
            # Calculate returns
            returns = data['Close'].pct_change().fillna(0)
            strategy_returns = signals.shift(1) * returns
            trade_returns = strategy_returns[strategy_returns != 0]
            
            if len(trade_returns) > 0:
                # Run Monte Carlo with subset of returns
                mc_results = monte_carlo_trade_simulation.random_trade_order_simulation(
                    returns=trade_returns.tolist()[:100],  # Limit to first 100 trades
                    num_simulations=100,
                    initial_capital=10000
                )
                
                assert isinstance(mc_results, pd.DataFrame)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should complete within reasonable time even with large data
            assert execution_time < 60  # 1 minute max

    @pytest.mark.integration
    def test_memory_efficiency_workflow(self, sample_stock_data):
        """Test memory efficiency of the complete workflow."""
        import gc
        
        # Run workflow multiple times to test for memory leaks
        for i in range(5):
            manager = AlgorithmManager()
            
            if len(manager.algorithms) > 0:
                algo_name = list(manager.algorithms.keys())[0]
                algorithm = manager.create_algorithm(algo_name)
                signals = algorithm.generate_signals(sample_stock_data)
                
                returns = sample_stock_data['Close'].pct_change().fillna(0)
                strategy_returns = signals.shift(1) * returns
                
                # Force garbage collection
                del signals, strategy_returns
                gc.collect()
        
        # Memory test would require more sophisticated monitoring
        # This documents the expected behavior


class TestRobustnessWorkflow:
    """Test robustness of the workflow under various conditions."""

    @pytest.mark.integration
    def test_workflow_with_missing_data(self):
        """Test workflow with missing data points."""
        # Create data with gaps
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
        data = pd.DataFrame({
            'Open': range(100, 100 + len(dates)),
            'High': range(102, 102 + len(dates)),
            'Low': range(98, 98 + len(dates)),
            'Close': range(101, 101 + len(dates)),
            'Volume': [1000000] * len(dates)
        }, index=dates)
        
        # Remove some days to create gaps
        data_with_gaps = data.drop(data.index[10:15])  # Remove 5 days
        
        manager = AlgorithmManager()
        
        if len(manager.algorithms) > 0:
            algo_name = list(manager.algorithms.keys())[0]
            algorithm = manager.create_algorithm(algo_name)
            
            # Should handle data gaps gracefully
            signals = algorithm.generate_signals(data_with_gaps)
            
            assert len(signals) == len(data_with_gaps)
            assert all(signal in [-1, 0, 1] for signal in signals)

    @pytest.mark.integration
    def test_workflow_with_extreme_market_conditions(self):
        """Test workflow with extreme market conditions."""
        # Create data with extreme movements
        extreme_data = pd.DataFrame({
            'Open': [100, 150, 75, 200, 50],  # Extreme price swings
            'High': [105, 160, 80, 210, 55],
            'Low': [95, 140, 70, 190, 45],
            'Close': [150, 75, 200, 50, 100],
            'Volume': [10000000, 20000000, 15000000, 25000000, 5000000]
        }, index=pd.date_range('2024-01-01', periods=5))
        
        manager = AlgorithmManager()
        
        if len(manager.algorithms) > 0:
            algo_name = list(manager.algorithms.keys())[0]
            algorithm = manager.create_algorithm(algo_name)
            
            # Should handle extreme conditions without crashing
            signals = algorithm.generate_signals(extreme_data)
            
            assert len(signals) == len(extreme_data)
            assert all(signal in [-1, 0, 1] for signal in signals)


if __name__ == '__main__':
    pytest.main([__file__])

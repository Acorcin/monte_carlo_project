"""
Unit tests for trading algorithms.

Tests individual algorithms, algorithm manager, and algorithm validation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import importlib
import sys
import os

# Import modules to test
from algorithms.algorithm_manager import AlgorithmManager
from algorithms.base_algorithm import TradingAlgorithm


class TestAlgorithmManager:
    """Test class for AlgorithmManager functionality."""

    @pytest.mark.unit
    def test_algorithm_manager_initialization(self):
        """Test AlgorithmManager initialization."""
        manager = AlgorithmManager()
        
        assert hasattr(manager, 'algorithms')
        assert isinstance(manager.algorithms, dict)
        assert len(manager.algorithms) > 0  # Should discover some algorithms

    @pytest.mark.unit
    def test_algorithm_discovery(self):
        """Test automatic algorithm discovery."""
        manager = AlgorithmManager()
        
        # Should find basic algorithms
        expected_algorithms = [
            'MovingAverageCrossover',
            'RSIOversoldOverbought', 
            'PriceMomentum'
        ]
        
        for algo_name in expected_algorithms:
            assert algo_name in manager.algorithms, f"Algorithm {algo_name} not found"

    @pytest.mark.unit
    def test_create_algorithm(self):
        """Test algorithm creation."""
        manager = AlgorithmManager()
        
        # Test creating a known algorithm
        if 'MovingAverageCrossover' in manager.algorithms:
            algorithm = manager.create_algorithm('MovingAverageCrossover')
            
            assert algorithm is not None
            assert hasattr(algorithm, 'generate_signals')
            assert hasattr(algorithm, 'get_algorithm_type')
            assert callable(algorithm.generate_signals)

    @pytest.mark.unit
    def test_create_nonexistent_algorithm(self):
        """Test creating a non-existent algorithm."""
        manager = AlgorithmManager()
        
        with pytest.raises((KeyError, ValueError)):
            manager.create_algorithm('NonExistentAlgorithm')

    @pytest.mark.unit
    def test_get_algorithm_types(self):
        """Test getting algorithm types."""
        manager = AlgorithmManager()
        
        if len(manager.algorithms) > 0:
            for algo_name in list(manager.algorithms.keys())[:3]:  # Test first 3
                algorithm = manager.create_algorithm(algo_name)
                algo_type = algorithm.get_algorithm_type()
                
                assert isinstance(algo_type, str)
                assert len(algo_type) > 0
                assert algo_type in ['momentum', 'mean_reversion', 'trend_following', 
                                   'machine_learning', 'technical_indicators']

    @pytest.mark.unit
    def test_algorithm_validation(self):
        """Test algorithm validation."""
        manager = AlgorithmManager()
        
        for algo_name in list(manager.algorithms.keys())[:3]:  # Test first 3
            algorithm = manager.create_algorithm(algo_name)
            
            # Test that algorithm has required methods
            assert hasattr(algorithm, 'generate_signals')
            assert hasattr(algorithm, 'get_algorithm_type')
            assert hasattr(algorithm, 'name')


class TestBaseAlgorithm:
    """Test class for base TradingAlgorithm functionality."""

    @pytest.mark.unit
    def test_base_algorithm_initialization(self):
        """Test base algorithm initialization."""
        # Since TradingAlgorithm is abstract, we'll create a simple implementation
        class TestAlgorithm(TradingAlgorithm):
            def generate_signals(self, data):
                return pd.Series(0, index=data.index)
            
            def get_algorithm_type(self):
                return "test"
        
        algorithm = TestAlgorithm("Test Algorithm")
        
        assert algorithm.name == "Test Algorithm"
        assert hasattr(algorithm, 'generate_signals')
        assert hasattr(algorithm, 'get_algorithm_type')

    @pytest.mark.unit
    def test_algorithm_interface_compliance(self):
        """Test that algorithms comply with the expected interface."""
        manager = AlgorithmManager()
        
        if len(manager.algorithms) > 0:
            algo_name = list(manager.algorithms.keys())[0]
            algorithm = manager.create_algorithm(algo_name)
            
            # Test interface compliance
            assert callable(algorithm.generate_signals)
            assert callable(algorithm.get_algorithm_type)
            
            # Test with sample data
            sample_data = pd.DataFrame({
                'Open': [100, 101, 102, 103, 104],
                'High': [102, 103, 104, 105, 106],
                'Low': [99, 100, 101, 102, 103],
                'Close': [101, 102, 103, 104, 105],
                'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
            }, index=pd.date_range('2024-01-01', periods=5))
            
            signals = algorithm.generate_signals(sample_data)
            
            assert isinstance(signals, pd.Series)
            assert len(signals) == len(sample_data)
            assert all(signal in [-1, 0, 1] for signal in signals)


class TestSpecificAlgorithms:
    """Test class for specific algorithm implementations."""

    @pytest.mark.unit
    def test_moving_average_crossover(self, sample_stock_data):
        """Test Moving Average Crossover algorithm."""
        manager = AlgorithmManager()
        
        if 'MovingAverageCrossover' in manager.algorithms:
            algorithm = manager.create_algorithm('MovingAverageCrossover')
            signals = algorithm.generate_signals(sample_stock_data)
            
            assert isinstance(signals, pd.Series)
            assert len(signals) == len(sample_stock_data)
            assert all(signal in [-1, 0, 1] for signal in signals)
            
            # Test that signals make sense (some buy/sell signals generated)
            unique_signals = signals.unique()
            assert len(unique_signals) > 1  # Should have different signals

    @pytest.mark.unit
    def test_rsi_algorithm(self, sample_stock_data):
        """Test RSI Oversold/Overbought algorithm."""
        manager = AlgorithmManager()
        
        if 'RSIOversoldOverbought' in manager.algorithms:
            algorithm = manager.create_algorithm('RSIOversoldOverbought')
            signals = algorithm.generate_signals(sample_stock_data)
            
            assert isinstance(signals, pd.Series)
            assert len(signals) == len(sample_stock_data)
            assert all(signal in [-1, 0, 1] for signal in signals)

    @pytest.mark.unit
    def test_momentum_algorithm(self, sample_stock_data):
        """Test Price Momentum algorithm."""
        manager = AlgorithmManager()
        
        if 'PriceMomentum' in manager.algorithms:
            algorithm = manager.create_algorithm('PriceMomentum')
            signals = algorithm.generate_signals(sample_stock_data)
            
            assert isinstance(signals, pd.Series)
            assert len(signals) == len(sample_stock_data)
            assert all(signal in [-1, 0, 1] for signal in signals)

    @pytest.mark.unit
    def test_algorithm_with_insufficient_data(self):
        """Test algorithm behavior with insufficient data."""
        manager = AlgorithmManager()
        
        # Create minimal data (insufficient for most algorithms)
        minimal_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        if len(manager.algorithms) > 0:
            algo_name = list(manager.algorithms.keys())[0]
            algorithm = manager.create_algorithm(algo_name)
            
            # Should handle insufficient data gracefully
            signals = algorithm.generate_signals(minimal_data)
            
            assert isinstance(signals, pd.Series)
            assert len(signals) == len(minimal_data)

    @pytest.mark.unit
    def test_algorithm_with_flat_prices(self):
        """Test algorithm behavior with flat (unchanging) prices."""
        # Create data with no price movement
        flat_data = pd.DataFrame({
            'Open': [100] * 50,
            'High': [100] * 50,
            'Low': [100] * 50,
            'Close': [100] * 50,
            'Volume': [1000000] * 50
        }, index=pd.date_range('2024-01-01', periods=50))
        
        manager = AlgorithmManager()
        
        if len(manager.algorithms) > 0:
            algo_name = list(manager.algorithms.keys())[0]
            algorithm = manager.create_algorithm(algo_name)
            
            signals = algorithm.generate_signals(flat_data)
            
            assert isinstance(signals, pd.Series)
            assert len(signals) == len(flat_data)
            # With flat prices, should generate mostly hold signals
            assert (signals == 0).sum() > len(signals) * 0.8


class TestAdvancedAlgorithms:
    """Test class for advanced ML algorithms (if available)."""

    @pytest.mark.unit
    def test_ml_algorithms_availability(self):
        """Test availability of ML algorithms."""
        manager = AlgorithmManager()
        
        ml_algorithms = [
            'AdvancedMLStrategy',
            'LSTMTradingStrategy', 
            'TransformerTradingStrategy',
            'ReinforcementLearningStrategy'
        ]
        
        available_ml = []
        for algo in ml_algorithms:
            if algo in manager.algorithms:
                available_ml.append(algo)
        
        print(f"Available ML algorithms: {available_ml}")
        # Don't require ML algorithms to be available (they might need TensorFlow)

    @pytest.mark.unit
    def test_ml_algorithm_fallback(self, sample_stock_data):
        """Test ML algorithm fallback behavior when libraries unavailable."""
        manager = AlgorithmManager()
        
        # Test algorithms that might have fallback implementations
        potential_ml_algorithms = ['AdvancedMLStrategy', 'LSTMTradingStrategy']
        
        for algo_name in potential_ml_algorithms:
            if algo_name in manager.algorithms:
                algorithm = manager.create_algorithm(algo_name)
                
                # Should work even without full ML libraries
                signals = algorithm.generate_signals(sample_stock_data)
                
                assert isinstance(signals, pd.Series)
                assert len(signals) == len(sample_stock_data)
                assert all(signal in [-1, 0, 1] for signal in signals)


class TestAlgorithmBacktesting:
    """Test class for algorithm backtesting functionality."""

    @pytest.mark.unit
    def test_backtest_algorithm_basic(self, sample_stock_data):
        """Test basic algorithm backtesting."""
        manager = AlgorithmManager()
        
        if len(manager.algorithms) > 0:
            algo_name = list(manager.algorithms.keys())[0]
            
            # Test if backtest_algorithm method exists
            if hasattr(manager, 'backtest_algorithm'):
                results = manager.backtest_algorithm(
                    algo_name, 
                    sample_stock_data, 
                    initial_capital=10000
                )
                
                assert isinstance(results, dict)
                assert 'total_return' in results
                assert 'metrics' in results
                assert isinstance(results['total_return'], (int, float))

    @pytest.mark.unit
    def test_backtest_invalid_algorithm(self, sample_stock_data):
        """Test backtesting with invalid algorithm."""
        manager = AlgorithmManager()
        
        if hasattr(manager, 'backtest_algorithm'):
            with pytest.raises((KeyError, ValueError)):
                manager.backtest_algorithm(
                    'NonExistentAlgorithm', 
                    sample_stock_data, 
                    initial_capital=10000
                )

    @pytest.mark.unit
    def test_backtest_with_no_signals(self):
        """Test backtesting when algorithm generates no trading signals."""
        # Create a mock algorithm that generates no signals
        class NoSignalAlgorithm(TradingAlgorithm):
            def generate_signals(self, data):
                return pd.Series(0, index=data.index)  # All hold signals
            
            def get_algorithm_type(self):
                return "test"
        
        # Test would depend on AlgorithmManager implementation
        # This documents expected behavior for edge cases


class TestAlgorithmPerformance:
    """Test class for algorithm performance characteristics."""

    @pytest.mark.unit
    @pytest.mark.slow
    def test_algorithm_execution_time(self, sample_stock_data):
        """Test algorithm execution time with large datasets."""
        manager = AlgorithmManager()
        
        # Create larger dataset
        large_data = pd.concat([sample_stock_data] * 10)  # 10x larger
        
        if len(manager.algorithms) > 0:
            algo_name = list(manager.algorithms.keys())[0]
            algorithm = manager.create_algorithm(algo_name)
            
            import time
            start_time = time.time()
            
            signals = algorithm.generate_signals(large_data)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should complete within reasonable time
            assert execution_time < 10  # 10 seconds max for reasonable algorithm
            assert len(signals) == len(large_data)

    @pytest.mark.unit
    def test_algorithm_memory_usage(self, sample_stock_data):
        """Test algorithm memory usage."""
        manager = AlgorithmManager()
        
        if len(manager.algorithms) > 0:
            algo_name = list(manager.algorithms.keys())[0]
            algorithm = manager.create_algorithm(algo_name)
            
            # Generate signals multiple times to test for memory leaks
            for _ in range(10):
                signals = algorithm.generate_signals(sample_stock_data)
                assert len(signals) == len(sample_stock_data)
            
            # Memory test would require more sophisticated monitoring
            # This documents the expected behavior


if __name__ == '__main__':
    pytest.main([__file__])

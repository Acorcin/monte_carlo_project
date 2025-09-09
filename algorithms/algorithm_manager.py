"""
Algorithm Manager

Manages all trading algorithms, handles discovery, loading, and backtesting.
"""

import os
import sys
import importlib
import inspect
from typing import Dict, List, Type, Any, Optional, Tuple
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import hashlib
import pickle
import time

from base_algorithm import TradingAlgorithm

class AlgorithmManager:
    """
    Manages trading algorithms and provides backtesting functionality.
    """

    def __init__(self, cache_dir: str = ".backtest_cache", use_parallel: bool = True):
        """Initialize the algorithm manager."""
        self.algorithms: Dict[str, Type[TradingAlgorithm]] = {}
        self.algorithm_instances: Dict[str, TradingAlgorithm] = {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.use_parallel = use_parallel
        self.max_workers = min(mp.cpu_count(), 8)  # Limit to 8 workers max

        # Create lock for thread-safe operations
        self._lock = threading.Lock()

        self._discover_algorithms()
    
    def _discover_algorithms(self):
        """Automatically discover all algorithms in the algorithms folder."""
        algorithms_dir = Path(__file__).parent
        
        # Search through all subdirectories
        for category_dir in algorithms_dir.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith('_'):
                self._load_algorithms_from_directory(category_dir)
    
    def _load_algorithms_from_directory(self, directory: Path):
        """Load algorithms from a specific directory."""
        for file_path in directory.glob("*.py"):
            if file_path.name.startswith('_'):
                continue
                
            try:
                # Import the module
                module_name = file_path.stem
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                sys.path.insert(0, str(file_path.parent))
                spec.loader.exec_module(module)
                sys.path.pop(0)
                
                # Find TradingAlgorithm classes in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, TradingAlgorithm) and 
                        obj != TradingAlgorithm and 
                        not inspect.isabstract(obj)):
                        
                        self.algorithms[name] = obj
                        print(f"✅ Loaded algorithm: {name}")
                        
            except Exception as e:
                print(f"❌ Failed to load algorithm from {file_path}: {e}")

    def _get_cache_key(self, algorithm_name: str, data_hash: str, params_hash: str) -> str:
        """Generate a unique cache key for backtest results."""
        combined = f"{algorithm_name}_{data_hash}_{params_hash}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Generate a hash of the dataframe for cache validation."""
        # Use shape, columns, and a sample of data for hash
        key_data = f"{df.shape}_{list(df.columns)}_{df.head(10).to_string()}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _hash_parameters(self, params: Dict[str, Any]) -> str:
        """Generate a hash of parameters for cache validation."""
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load backtest result from cache if available."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                # If cache file is corrupted, remove it
                cache_file.unlink()
        return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Save backtest result to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"⚠️ Failed to cache result: {e}")

    def _backtest_single_algorithm_parallel(self, args) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Helper function for parallel backtesting of a single algorithm."""
        algorithm_name, data, parameters, initial_capital = args

        try:
            # Check cache first
            data_hash = self._hash_dataframe(data)
            params_hash = self._hash_parameters(parameters or {})
            cache_key = self._get_cache_key(algorithm_name, data_hash, params_hash)

            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                print(f"✅ {algorithm_name}: Loaded from cache")
                return algorithm_name, cached_result

            # Run backtest
            result = self.backtest_algorithm(
                algorithm_name=algorithm_name,
                data=data,
                parameters=parameters,
                initial_capital=initial_capital
            )

            # Cache the result
            if result:
                self._save_to_cache(cache_key, result)

            return algorithm_name, result

        except Exception as e:
            print(f"❌ {algorithm_name}: Backtest failed - {e}")
            return algorithm_name, None

    def get_available_algorithms(self) -> Dict[str, str]:
        """
        Get list of available algorithms with their types.
        
        Returns:
            dict: Algorithm name -> algorithm type mapping
        """
        result = {}
        for name, algo_class in self.algorithms.items():
            try:
                # Create temporary instance to get type
                temp_instance = algo_class()
                result[name] = temp_instance.get_algorithm_type()
            except Exception:
                result[name] = "unknown"
        return result
    
    def get_algorithm_info(self, algorithm_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an algorithm.
        
        Args:
            algorithm_name (str): Name of the algorithm
            
        Returns:
            dict: Algorithm information including parameters
        """
        if algorithm_name not in self.algorithms:
            return None
        
        try:
            algo_class = self.algorithms[algorithm_name]
            temp_instance = algo_class()
            
            return {
                'name': temp_instance.name,
                'type': temp_instance.get_algorithm_type(),
                'description': temp_instance.description,
                'parameters': temp_instance.get_parameter_info(),
                'default_parameters': temp_instance.parameters
            }
        except Exception as e:
            print(f"❌ Error getting info for {algorithm_name}: {e}")
            return None
    
    def create_algorithm(self, algorithm_name: str, parameters: Dict[str, Any] = None) -> Optional[TradingAlgorithm]:
        """
        Create an instance of the specified algorithm.
        
        Args:
            algorithm_name (str): Name of the algorithm
            parameters (dict): Algorithm parameters
            
        Returns:
            TradingAlgorithm: Algorithm instance
        """
        if algorithm_name not in self.algorithms:
            print(f"❌ Algorithm '{algorithm_name}' not found")
            return None
        
        try:
            algo_class = self.algorithms[algorithm_name]
            
            # Create instance with parameters
            if parameters:
                # Try to pass parameters to constructor
                try:
                    instance = algo_class(**parameters)
                except TypeError:
                    # If constructor doesn't accept parameters, create default and set them
                    instance = algo_class()
                    instance.set_parameters(parameters)
            else:
                instance = algo_class()
            
            return instance
            
        except Exception as e:
            print(f"❌ Error creating algorithm {algorithm_name}: {e}")
            return None
    
    def backtest_algorithm(self, algorithm_name: str, data: pd.DataFrame,
                          parameters: Dict[str, Any] = None,
                          initial_capital: float = 10000,
                          commission: float = 0.001,
                          slippage: float = 0.0001) -> Optional[Dict[str, Any]]:
        """
        Backtest a specific algorithm on the provided data.

        Args:
            algorithm_name (str): Name of the algorithm
            data (pd.DataFrame): OHLCV data
            parameters (dict): Algorithm parameters
            initial_capital (float): Starting capital
            commission (float): Commission rate
            slippage (float): Slippage rate

        Returns:
            dict: Backtest results
        """
        # Check cache first
        data_hash = self._hash_dataframe(data)
        params_hash = self._hash_parameters(parameters or {})
        cache_key = self._get_cache_key(algorithm_name, data_hash, params_hash)

        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            print(f"✅ {algorithm_name}: Loaded from cache")
            return cached_result

        # Create algorithm instance
        algorithm = self.create_algorithm(algorithm_name, parameters)
        if algorithm is None:
            return None

        try:
            # Run backtest
            results = algorithm.backtest(
                data=data,
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage
            )

            # Cache the result
            if results:
                self._save_to_cache(cache_key, results)

            return results

        except Exception as e:
            print(f"❌ Backtest failed for {algorithm_name}: {e}")
            return None
    
    def backtest_multiple_algorithms(self, algorithm_configs: List[Dict[str, Any]],
                                   data: pd.DataFrame,
                                   initial_capital: float = 10000) -> Dict[str, Dict[str, Any]]:
        """
        Backtest multiple algorithms and compare results.

        Args:
            algorithm_configs (list): List of algorithm configurations
                Each config should have: {'name': str, 'parameters': dict}
            data (pd.DataFrame): OHLCV data
            initial_capital (float): Starting capital

        Returns:
            dict: Results for each algorithm
        """
        results = {}

        if not self.use_parallel or len(algorithm_configs) <= 2:
            # Use sequential processing for small number of algorithms
            return self._backtest_sequential(algorithm_configs, data, initial_capital)

        # Use parallel processing for larger sets
        print(f"🚀 Starting parallel backtesting with {self.max_workers} workers...")

        # Prepare arguments for parallel processing
        args_list = []
        for config in algorithm_configs:
            algorithm_name = config['name']
            parameters = config.get('parameters', {})
            args_list.append((algorithm_name, data, parameters, initial_capital))

        # Use ThreadPoolExecutor for I/O bound operations (most algorithms)
        # Use ProcessPoolExecutor for CPU intensive ML algorithms
        has_ml_algorithms = any(config['name'] in [
            'LSTMTradingStrategy', 'TransformerTradingStrategy',
            'AdvancedMLStrategy', 'EnsembleStackingStrategy'
        ] for config in algorithm_configs)

        if has_ml_algorithms:
            # Use process pool for ML algorithms
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_config = {
                    executor.submit(self._backtest_single_algorithm_parallel, args): args[0]
                    for args in args_list
                }

                for future in as_completed(future_to_config):
                    algorithm_name = future_to_config[future]
                    try:
                        result_name, result = future.result()
                        if result:
                            results[result_name] = result
                    except Exception as e:
                        print(f"❌ Parallel backtest failed for {algorithm_name}: {e}")
        else:
            # Use thread pool for technical indicators
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_config = {
                    executor.submit(self._backtest_single_algorithm_parallel, args): args[0]
                    for args in args_list
                }

                for future in as_completed(future_to_config):
                    algorithm_name = future_to_config[future]
                    try:
                        result_name, result = future.result()
                        if result:
                            results[result_name] = result
                    except Exception as e:
                        print(f"❌ Parallel backtest failed for {algorithm_name}: {e}")

        return results

    def _backtest_sequential(self, algorithm_configs: List[Dict[str, Any]],
                           data: pd.DataFrame,
                           initial_capital: float = 10000) -> Dict[str, Dict[str, Any]]:
        """Sequential backtesting for small numbers of algorithms."""
        results = {}

        for config in algorithm_configs:
            algorithm_name = config['name']
            parameters = config.get('parameters', {})

            print(f"🔄 Backtesting {algorithm_name}...")

            # Check cache first
            data_hash = self._hash_dataframe(data)
            params_hash = self._hash_parameters(parameters)
            cache_key = self._get_cache_key(algorithm_name, data_hash, params_hash)

            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                print(f"✅ {algorithm_name}: Loaded from cache")
                results[algorithm_name] = cached_result
                continue

            result = self.backtest_algorithm(
                algorithm_name=algorithm_name,
                data=data,
                parameters=parameters,
                initial_capital=initial_capital
            )

            if result:
                results[algorithm_name] = result
                # Cache the result
                self._save_to_cache(cache_key, result)
                print(f"✅ {algorithm_name}: {result['total_return']:.2f}% return")
            else:
                print(f"❌ {algorithm_name}: Backtest failed")

        return results
    
    def get_algorithm_categories(self) -> Dict[str, List[str]]:
        """
        Group algorithms by their categories.
        
        Returns:
            dict: Category -> list of algorithm names
        """
        categories = {}
        
        for name, algo_class in self.algorithms.items():
            try:
                temp_instance = algo_class()
                category = temp_instance.get_algorithm_type()
                
                if category not in categories:
                    categories[category] = []
                categories[category].append(name)
                
            except Exception:
                if 'unknown' not in categories:
                    categories['unknown'] = []
                categories['unknown'].append(name)
        
        return categories
    
    def print_available_algorithms(self):
        """Print all available algorithms organized by category."""
        print("📊 AVAILABLE TRADING ALGORITHMS")
        print("=" * 50)
        
        categories = self.get_algorithm_categories()
        
        for category, algorithms in categories.items():
            print(f"\n📈 {category.upper().replace('_', ' ')}")
            print("-" * 30)
            
            for algo_name in algorithms:
                info = self.get_algorithm_info(algo_name)
                if info:
                    print(f"  • {info['name']}")
                    print(f"    {info['description']}")
                else:
                    print(f"  • {algo_name}")
        
        print(f"\n💡 Total algorithms available: {len(self.algorithms)}")

# Global instance for easy access
algorithm_manager = AlgorithmManager(use_parallel=True)

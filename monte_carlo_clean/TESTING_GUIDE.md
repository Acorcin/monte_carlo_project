# Testing Guide for Monte Carlo Trading Application

## Overview 🧪

This guide covers the comprehensive testing framework for the Monte Carlo Trading Application, including unit tests, integration tests, and GUI tests.

## Test Structure 📁

```
tests/
├── __init__.py                    # Test package initialization
├── unit/                          # Unit tests for individual components
│   ├── __init__.py
│   ├── test_data_fetcher.py       # Data fetching tests
│   ├── test_monte_carlo_simulation.py  # Monte Carlo simulation tests
│   ├── test_algorithms.py         # Algorithm tests
│   └── test_risk_management.py    # Risk management tests
├── integration/                   # Integration tests for workflows
│   ├── __init__.py
│   ├── test_full_workflow.py      # End-to-end workflow tests
│   └── test_gui_integration.py    # GUI integration tests
└── fixtures/                      # Test data and utilities
    ├── sample_data.py             # Sample market data
    └── mock_responses.py          # Mock API responses
```

## Running Tests 🚀

### Quick Start

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-xdist

# Run all tests
python run_tests.py

# Quick test suite (unit tests only)
python run_tests.py --quick

# Full test suite
python run_tests.py --full

# Tests with coverage
python run_tests.py --coverage
```

### Specific Test Categories

```bash
# Unit tests only
python run_tests.py --unit

# Integration tests only
python run_tests.py --integration

# GUI tests only (requires display)
python run_tests.py --gui

# Include slow tests
python run_tests.py --full --slow
```

### Specific Test Files or Functions

```bash
# Run specific test file
python run_tests.py --file unit/test_data_fetcher.py

# Run specific test function
python run_tests.py --function test_fetch_stock_data

# Run with verbose output
python run_tests.py --unit --verbose
```

### Parallel Execution

```bash
# Run tests in parallel (faster)
python run_tests.py --parallel

# Parallel with coverage
python run_tests.py --coverage --parallel
```

## Test Categories 🏷️

### Unit Tests (`@pytest.mark.unit`)

Test individual components in isolation:

- **Data Fetcher Tests**: Validate data fetching, caching, and error handling
- **Algorithm Tests**: Test trading algorithms and signal generation
- **Monte Carlo Tests**: Validate simulation logic and statistical properties
- **Risk Management Tests**: Test risk calculations and validation

### Integration Tests (`@pytest.mark.integration`)

Test component interactions and workflows:

- **Full Workflow Tests**: End-to-end data → algorithm → Monte Carlo
- **GUI Integration Tests**: Test GUI component interactions
- **Multi-Algorithm Tests**: Test consensus and multi-strategy workflows

### GUI Tests (`@pytest.mark.gui`)

Test GUI functionality (may require display):

- **Component Creation**: Test GUI initialization and component setup
- **Data Flow**: Test data passing between GUI components
- **User Interactions**: Test button clicks, form submissions, etc.
- **Error Handling**: Test GUI error display and validation

### Slow Tests (`@pytest.mark.slow`)

Tests that take longer to run:

- **Large Dataset Tests**: Test with multi-year data
- **High-Simulation Monte Carlo**: Test with 10k+ simulations
- **Performance Tests**: Test execution time and memory usage

## Test Fixtures 🔧

### Available Fixtures

```python
@pytest.fixture
def sample_stock_data():
    """Generate sample OHLCV data for testing."""
    
@pytest.fixture  
def mock_backtest_results():
    """Mock backtest results with metrics and trades."""
    
@pytest.fixture
def mock_algorithm():
    """Mock trading algorithm for testing."""
    
@pytest.fixture
def mock_yfinance_data():
    """Mock yfinance data for testing data fetching."""
```

### Using Fixtures

```python
def test_algorithm_with_data(sample_stock_data, mock_algorithm):
    """Test algorithm with sample data."""
    signals = mock_algorithm.generate_signals(sample_stock_data)
    assert len(signals) == len(sample_stock_data)
```

## Writing Tests ✍️

### Unit Test Example

```python
import pytest
import pandas as pd
from algorithms.algorithm_manager import AlgorithmManager

class TestAlgorithmManager:
    @pytest.mark.unit
    def test_algorithm_discovery(self):
        """Test that algorithms are discovered correctly."""
        manager = AlgorithmManager()
        
        assert len(manager.algorithms) > 0
        assert 'MovingAverageCrossover' in manager.algorithms
    
    @pytest.mark.unit
    def test_create_algorithm(self):
        """Test algorithm creation."""
        manager = AlgorithmManager()
        algorithm = manager.create_algorithm('MovingAverageCrossover')
        
        assert hasattr(algorithm, 'generate_signals')
        assert callable(algorithm.generate_signals)
```

### Integration Test Example

```python
@pytest.mark.integration
def test_data_to_monte_carlo_workflow(mock_yfinance_data):
    """Test complete workflow from data to Monte Carlo."""
    with patch('yfinance.download', side_effect=mock_yfinance_data):
        # Step 1: Fetch data
        data = data_fetcher.fetch_stock_data('AAPL', period='1y')
        
        # Step 2: Run algorithm
        manager = AlgorithmManager()
        algorithm = manager.create_algorithm('MovingAverageCrossover')
        signals = algorithm.generate_signals(data)
        
        # Step 3: Calculate returns
        returns = data['Close'].pct_change().fillna(0)
        strategy_returns = signals.shift(1) * returns
        
        # Step 4: Monte Carlo simulation
        results = monte_carlo_simulation.random_trade_order_simulation(
            returns=strategy_returns[strategy_returns != 0].tolist(),
            num_simulations=100,
            initial_capital=10000
        )
        
        assert isinstance(results, pd.DataFrame)
        assert results.shape[1] == 100
```

### GUI Test Example

```python
@pytest.mark.gui
def test_gui_data_loading(mock_gui_root, mock_yfinance_data):
    """Test GUI data loading workflow."""
    with patch('yfinance.download', side_effect=mock_yfinance_data):
        gui = MonteCarloGUI(mock_gui_root)
        
        # Mock GUI inputs
        gui.ticker_var.get.return_value = 'AAPL'
        gui.period_var.get.return_value = '1y'
        
        # Test data loading
        gui.load_data()
        
        assert gui.current_data is not None
        assert not gui.current_data.empty
```

## Test Configuration ⚙️

### pytest.ini

```ini
[tool:pytest]
testpaths = tests
markers =
    unit: Unit tests for individual components
    integration: Integration tests for workflows
    gui: GUI tests (may require display)
    slow: Tests that take longer to run

addopts = -v --tb=short --strict-markers
```

### conftest.py

Contains shared fixtures and configuration:
- Sample data generators
- Mock objects and responses
- Test environment setup
- Custom markers and hooks

## Coverage Reports 📊

### Generate Coverage

```bash
# HTML coverage report
python run_tests.py --coverage

# View coverage report
# Open htmlcov/index.html in browser
```

### Coverage Targets

- **Overall Coverage**: Target 80%+
- **Core Modules**: Target 90%+
  - `data_fetcher.py`
  - `monte_carlo_trade_simulation.py`
  - `algorithms/algorithm_manager.py`
- **GUI Modules**: Target 70%+ (harder to test)

### Coverage Exclusions

```python
# Exclude GUI main loops
if __name__ == '__main__':  # pragma: no cover
    main()

# Exclude abstract methods
def abstract_method(self):  # pragma: no cover
    raise NotImplementedError
```

## Continuous Integration 🔄

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-xdist
    
    - name: Run unit tests
      run: python run_tests.py --unit --coverage
    
    - name: Run integration tests
      run: python run_tests.py --integration
```

## Test Data Management 💾

### Mock Data Strategy

- **Use fixtures for predictable data**
- **Mock external APIs (yfinance)**
- **Generate realistic but controlled data**
- **Version control test data files**

### Test Data Isolation

```python
# Use temporary directories for file operations
@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

# Reset global state between tests
@pytest.fixture(autouse=True)
def reset_state():
    yield
    # Cleanup code here
```

## Performance Testing 🚀

### Performance Test Example

```python
@pytest.mark.slow
def test_large_simulation_performance():
    """Test Monte Carlo performance with large simulations."""
    import time
    
    start_time = time.time()
    results = monte_carlo_simulation.random_trade_order_simulation(
        returns=returns_data,
        num_simulations=10000,
        initial_capital=10000
    )
    end_time = time.time()
    
    # Should complete within 30 seconds
    assert end_time - start_time < 30
    assert results.shape[1] == 10000
```

### Memory Testing

```python
def test_memory_usage():
    """Test memory usage during operations."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Perform memory-intensive operation
    large_simulation()
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Should not increase memory by more than 500MB
    assert memory_increase < 500 * 1024 * 1024
```

## Debugging Tests 🐛

### Running Tests in Debug Mode

```bash
# Run with pdb on failure
python -m pytest tests/ --pdb

# Run single test with verbose output
python -m pytest tests/unit/test_data_fetcher.py::test_fetch_stock_data -v -s

# Print output during tests
python -m pytest tests/ -s
```

### Test Debugging Tips

1. **Use `pytest.set_trace()`** for breakpoints
2. **Use `-s` flag** to see print statements
3. **Use `--tb=long`** for detailed tracebacks
4. **Use `--lf`** to run only last failed tests
5. **Use `--sw`** to stop on first failure

## Best Practices 📋

### Test Organization

- **One test class per component**
- **Descriptive test names**
- **Arrange-Act-Assert pattern**
- **Independent tests (no dependencies)**

### Test Quality

- **Test both happy path and edge cases**
- **Use meaningful assertions**
- **Test error conditions**
- **Keep tests simple and focused**

### Mocking Guidelines

- **Mock external dependencies**
- **Don't mock the system under test**
- **Use fixtures for complex mocks**
- **Verify mock interactions when relevant**

### Performance Considerations

- **Mark slow tests with `@pytest.mark.slow`**
- **Use smaller datasets for unit tests**
- **Run performance tests separately**
- **Monitor test execution time**

## Troubleshooting 🔧

### Common Issues

**Tests fail with import errors:**
```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**GUI tests fail in headless environment:**
```bash
# Skip GUI tests in CI
python run_tests.py --unit --integration
```

**Coverage report not generated:**
```bash
# Install pytest-cov
pip install pytest-cov
```

**Tests timeout:**
```bash
# Install pytest-timeout
pip install pytest-timeout
```

### Getting Help

- **Check test output for detailed error messages**
- **Use `--tb=long` for full tracebacks**
- **Check fixture setup in `conftest.py`**
- **Verify all dependencies are installed**

## Advanced Testing 🎓

### Property-Based Testing

```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=1))
def test_monte_carlo_with_random_returns(returns):
    """Test Monte Carlo with property-based random returns."""
    results = monte_carlo_simulation.random_trade_order_simulation(
        returns=returns,
        num_simulations=10,
        initial_capital=10000
    )
    
    assert isinstance(results, pd.DataFrame)
    assert results.shape[1] == 10
```

### Mutation Testing

```bash
# Install mutmut
pip install mutmut

# Run mutation testing
mutmut run --paths-to-mutate=algorithms/

# View results
mutmut results
```

### Test Metrics

- **Test Coverage**: % of code covered by tests
- **Test Speed**: Average test execution time
- **Test Reliability**: % of tests that pass consistently
- **Test Maintenance**: Time to update tests for code changes

---

## Summary 📝

This comprehensive testing framework ensures:

✅ **Reliability**: Catch bugs before they reach production  
✅ **Confidence**: Safe refactoring and feature additions  
✅ **Documentation**: Tests serve as living documentation  
✅ **Quality**: Maintain high code quality standards  
✅ **Performance**: Monitor and optimize application performance

Happy testing! 🧪✨

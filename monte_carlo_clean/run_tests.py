"""
Test runner for Monte Carlo Trading Application.

Provides convenient commands for running different types of tests.
"""

import pytest
import sys
import os
import argparse
from pathlib import Path


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run tests for Monte Carlo Trading Application')
    
    parser.add_argument('--unit', action='store_true', 
                       help='Run only unit tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('--gui', action='store_true',
                       help='Run only GUI tests')
    parser.add_argument('--slow', action='store_true',
                       help='Include slow tests')
    parser.add_argument('--coverage', action='store_true',
                       help='Run with coverage report')
    parser.add_argument('--parallel', action='store_true',
                       help='Run tests in parallel')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test suite (unit tests only, no slow tests)')
    parser.add_argument('--full', action='store_true',
                       help='Run full test suite including slow tests')
    parser.add_argument('--file', type=str,
                       help='Run specific test file')
    parser.add_argument('--function', type=str,
                       help='Run specific test function')
    
    args = parser.parse_args()
    
    # Build pytest arguments
    pytest_args = []
    
    # Test selection
    if args.unit:
        pytest_args.extend(['-m', 'unit'])
    elif args.integration:
        pytest_args.extend(['-m', 'integration'])
    elif args.gui:
        pytest_args.extend(['-m', 'gui'])
    elif args.quick:
        pytest_args.extend(['-m', 'unit and not slow'])
    elif args.full:
        pytest_args.extend(['-m', 'unit or integration'])
        if not args.slow:
            pytest_args.extend(['and not slow'])
    
    # Include slow tests if specified
    if args.slow and not args.quick:
        if '-m' in pytest_args:
            idx = pytest_args.index('-m')
            pytest_args[idx + 1] += ' or slow'
        else:
            pytest_args.extend(['-m', 'slow'])
    elif not args.slow and '-m' not in pytest_args:
        pytest_args.extend(['-m', 'not slow'])
    
    # Specific file or function
    if args.file:
        pytest_args.append(f"tests/{args.file}")
    if args.function:
        pytest_args.extend(['-k', args.function])
    
    # Coverage
    if args.coverage:
        pytest_args.extend(['--cov=.', '--cov-report=html', '--cov-report=term-missing'])
    
    # Parallel execution
    if args.parallel:
        pytest_args.extend(['-n', 'auto'])
    
    # Verbose output
    if args.verbose:
        pytest_args.append('-v')
    
    # Default test directory if no specific file
    if not args.file:
        pytest_args.append('tests/')
    
    print("🧪 Monte Carlo Trading Application Test Runner")
    print("=" * 50)
    print(f"Running tests with arguments: {' '.join(pytest_args)}")
    print()
    
    # Run pytest
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    return exit_code


def run_quick_tests():
    """Run quick test suite for development."""
    print("🚀 Running Quick Test Suite (Unit Tests Only)")
    print("=" * 45)
    
    exit_code = pytest.main([
        'tests/',
        '-m', 'unit and not slow',
        '-v',
        '--tb=short'
    ])
    
    return exit_code


def run_full_tests():
    """Run full test suite."""
    print("🔬 Running Full Test Suite")
    print("=" * 30)
    
    exit_code = pytest.main([
        'tests/',
        '-v',
        '--tb=short'
    ])
    
    return exit_code


def run_coverage_tests():
    """Run tests with coverage report."""
    print("📊 Running Tests with Coverage")
    print("=" * 35)
    
    # Check if pytest-cov is available
    try:
        import pytest_cov
        coverage_available = True
    except ImportError:
        coverage_available = False
        print("⚠️  pytest-cov not installed. Installing...")
        os.system("pip install pytest-cov")
        coverage_available = True
    
    if coverage_available:
        exit_code = pytest.main([
            'tests/',
            '--cov=.',
            '--cov-report=html',
            '--cov-report=term-missing',
            '--cov-report=xml',
            '-v'
        ])
        
        print("\n📈 Coverage report generated:")
        print("   📄 HTML: htmlcov/index.html")
        print("   📄 XML: coverage.xml")
    else:
        print("❌ Coverage testing not available")
        exit_code = 1
    
    return exit_code


def check_test_environment():
    """Check if test environment is properly set up."""
    print("🔍 Checking Test Environment")
    print("=" * 30)
    
    # Check if pytest is available
    try:
        import pytest
        print(f"✅ pytest version: {pytest.__version__}")
    except ImportError:
        print("❌ pytest not installed")
        return False
    
    # Check if test directories exist
    test_dir = Path("tests")
    if test_dir.exists():
        print(f"✅ Test directory found: {test_dir}")
        
        # Count test files
        unit_tests = list(test_dir.glob("unit/test_*.py"))
        integration_tests = list(test_dir.glob("integration/test_*.py"))
        
        print(f"   📁 Unit tests: {len(unit_tests)} files")
        print(f"   📁 Integration tests: {len(integration_tests)} files")
    else:
        print("❌ Test directory not found")
        return False
    
    # Check if conftest.py exists
    conftest = Path("conftest.py")
    if conftest.exists():
        print("✅ pytest configuration found")
    else:
        print("⚠️  conftest.py not found")
    
    # Check optional dependencies
    optional_deps = {
        'pytest-cov': 'Coverage testing',
        'pytest-xdist': 'Parallel test execution',
        'pytest-timeout': 'Test timeout handling',
        'pytest-mock': 'Enhanced mocking'
    }
    
    print("\n🔧 Optional Dependencies:")
    for dep, description in optional_deps.items():
        try:
            __import__(dep.replace('-', '_'))
            print(f"   ✅ {dep}: {description}")
        except ImportError:
            print(f"   ⚠️  {dep}: {description} (not installed)")
    
    return True


if __name__ == '__main__':
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        print("🧪 Monte Carlo Trading Application Test Runner")
        print("=" * 50)
        print("\nQuick Commands:")
        print("  python run_tests.py --quick      # Quick unit tests")
        print("  python run_tests.py --full       # Full test suite")
        print("  python run_tests.py --coverage   # Tests with coverage")
        print("  python run_tests.py --unit       # Unit tests only")
        print("  python run_tests.py --integration # Integration tests only")
        print("  python run_tests.py --gui        # GUI tests only")
        print("\nSpecial Commands:")
        print("  python run_tests.py --file unit/test_data_fetcher.py")
        print("  python run_tests.py --function test_fetch_stock_data")
        print("\nFor full options: python run_tests.py --help")
        print("\n🔍 Checking test environment...")
        check_test_environment()
        sys.exit(0)
    
    # Handle special commands
    if '--check' in sys.argv:
        check_test_environment()
        sys.exit(0)
    
    if '--quick-run' in sys.argv:
        sys.exit(run_quick_tests())
    
    if '--full-run' in sys.argv:
        sys.exit(run_full_tests())
    
    if '--coverage-run' in sys.argv:
        sys.exit(run_coverage_tests())
    
    # Run main parser
    sys.exit(main())

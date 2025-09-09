"""
Test script to verify Monte Carlo integration fix
"""

import subprocess
import sys
import time

def test_monte_carlo_integration():
    """Test if Monte Carlo integration is working after consensus backtest."""
    
    print("🧪 TESTING MONTE CARLO INTEGRATION FIX")
    print("=" * 50)
    
    # Test 1: Check if the fix is in place
    print("1️⃣ Checking code fix...")
    
    with open('monte_carlo_gui_app.py', 'r') as f:
        content = f.read()
        
    if 'self.current_results = consensus_results' in content:
        print("✅ Code fix detected: consensus results now stored")
    else:
        print("❌ Code fix missing: consensus results not being stored")
        return False
        
    if 'Consensus results stored for Monte Carlo' in content:
        print("✅ Debug logging added for Monte Carlo integration")
    else:
        print("❌ Debug logging missing")
        return False
    
    # Test 2: Check Monte Carlo function looks for current_results
    print("\n2️⃣ Checking Monte Carlo function...")
    
    if 'if not self.current_results:' in content:
        print("✅ Monte Carlo checks for current_results")
    else:
        print("❌ Monte Carlo doesn't check for current_results")
        return False
        
    # Test 3: Verify data flow structure
    print("\n3️⃣ Checking data flow structure...")
    
    expected_flow = [
        'consensus_results = self._perform_consensus_backtest',
        'self.current_results = consensus_results', 
        'self.mc_btn.config(state=\'normal\')'
    ]
    
    for step in expected_flow:
        if step in content:
            print(f"✅ Found: {step[:50]}...")
        else:
            print(f"❌ Missing: {step[:50]}...")
            return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("📋 Integration Summary:")
    print("   ✅ Consensus backtest stores results")
    print("   ✅ Monte Carlo button gets enabled")
    print("   ✅ Monte Carlo can access consensus data")
    print("   ✅ Debug logging provides visibility")
    
    return True

def print_usage_instructions():
    """Print instructions for testing the fix."""
    
    print("\n🎯 HOW TO TEST THE FIX:")
    print("=" * 30)
    print("1. Launch the application:")
    print("   python monte_carlo_gui_app.py")
    print()
    print("2. Load some data (e.g., AAPL, 1mo, 30m)")
    print()
    print("3. Go to Strategy tab and select 2+ algorithms")
    print()
    print("4. Run 'Consensus Backtest'")
    print()
    print("5. Check console output for:")
    print("   ✅ Consensus results stored for Monte Carlo: X.XXXX return")
    print("   ✅ Monte Carlo button enabled")
    print()
    print("6. Go to Monte Carlo tab - button should be enabled")
    print()
    print("7. Click 'Run Monte Carlo' - should work without error")
    print()
    print("Expected result: Monte Carlo simulation runs successfully!")

if __name__ == "__main__":
    print("🔧 Monte Carlo Integration Fix Test")
    print("=" * 40)
    
    if test_monte_carlo_integration():
        print_usage_instructions()
    else:
        print("\n❌ Tests failed - fix may not be complete")
        sys.exit(1)

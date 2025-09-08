"""
Interactive Monte Carlo Trade Simulation

This version allows you to customize simulation parameters including
the number of simulations to run.
"""

import numpy as np
from monte_carlo_trade_simulation import random_trade_order_simulation, plot_trade_order_simulations
from data_fetcher import fetch_stock_data, calculate_returns_from_ohlcv

def get_user_input():
    """Get simulation parameters from user input."""
    print("🎛️  MONTE CARLO SIMULATION - INTERACTIVE MODE")
    print("=" * 55)
    
    # Get number of simulations
    while True:
        try:
            num_sims = input("\n📊 Number of simulations (default: 1000): ").strip()
            if not num_sims:
                num_sims = 1000
            else:
                num_sims = int(num_sims)
            
            if num_sims <= 0:
                print("❌ Please enter a positive number")
                continue
            if num_sims > 10000:
                confirm = input(f"⚠️  {num_sims} simulations might be slow. Continue? (y/n): ")
                if confirm.lower() != 'y':
                    continue
            break
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Get initial capital
    while True:
        try:
            capital = input(f"\n💰 Initial capital (default: $10,000): ").strip()
            if not capital:
                capital = 10000
            else:
                capital = float(capital.replace('$', '').replace(',', ''))
            
            if capital <= 0:
                print("❌ Please enter a positive amount")
                continue
            break
        except ValueError:
            print("❌ Please enter a valid amount")
    
    # Choose data source
    print(f"\n📈 Data Source Options:")
    print(f"  1. Use example trade returns")
    print(f"  2. Fetch real market data (SPY ETF)")
    print(f"  3. Enter custom trade returns")
    
    while True:
        choice = input(f"\nSelect option (1, 2, or 3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("❌ Please enter 1, 2, or 3")
    
    return num_sims, capital, choice

def get_custom_returns():
    """Get custom trade returns from user."""
    print(f"\n✏️  Enter your trade returns (in decimal format)")
    print(f"   Examples: 0.05 = 5%, -0.02 = -2%, 0.1 = 10%")
    print(f"   Enter one return per line, press Enter twice when done:")
    
    returns = []
    while True:
        try:
            line = input(f"Return #{len(returns)+1} (or press Enter to finish): ").strip()
            if not line:
                if len(returns) == 0:
                    print("❌ Please enter at least one return")
                    continue
                break
            
            ret = float(line)
            if abs(ret) > 1:
                confirm = input(f"⚠️  {ret:.2%} is a large return. Did you mean {ret/100:.2%}? (y/n): ")
                if confirm.lower() == 'y':
                    ret = ret / 100
            
            returns.append(ret)
            print(f"   Added: {ret:.2%}")
            
        except ValueError:
            print("❌ Please enter a valid decimal number")
    
    return np.array(returns)

def run_interactive_simulation():
    """Run the interactive simulation with user-specified parameters."""
    
    # Get user preferences
    num_sims, capital, data_choice = get_user_input()
    
    # Get trade returns based on choice
    if data_choice == '1':
        # Example returns
        trade_returns = np.array([0.03, -0.02, 0.05, -0.01, 0.04, -0.03, 0.02, 0.01, -0.015, 0.025])
        data_source = "Example trade returns"
        print(f"\n📊 Using example returns: {[f'{r:.1%}' for r in trade_returns]}")
        
    elif data_choice == '2':
        # Real market data
        try:
            print(f"\n📡 Fetching SPY ETF data...")
            period = input("Period (1mo, 3mo, 6mo, 1y, default: 3mo): ").strip() or "3mo"
            interval = input("Interval (1d, 1h, default: 1d): ").strip() or "1d"
            
            spy_data = fetch_stock_data("SPY", period=period, interval=interval)
            trade_returns = calculate_returns_from_ohlcv(spy_data, remove_outliers=True)
            data_source = f"SPY ETF ({period}, {interval})"
            print(f"✅ Fetched {len(trade_returns)} returns from SPY")
            
        except Exception as e:
            print(f"❌ Failed to fetch data: {e}")
            print("📊 Falling back to example data...")
            trade_returns = np.array([0.03, -0.02, 0.05, -0.01, 0.04, -0.03, 0.02])
            data_source = "Example data (fallback)"
            
    else:
        # Custom returns
        trade_returns = get_custom_returns()
        data_source = "Custom user returns"
        print(f"✅ Using {len(trade_returns)} custom returns")
    
    # Show simulation setup
    print(f"\n🎯 SIMULATION SETUP")
    print(f"   {'=' * 30}")
    print(f"   Simulations:     {num_sims:,}")
    print(f"   Initial Capital: ${capital:,.2f}")
    print(f"   Data Source:     {data_source}")
    print(f"   Number of Trades: {len(trade_returns)}")
    if len(trade_returns) <= 10:
        print(f"   Returns:         {[f'{r:.1%}' for r in trade_returns]}")
    
    # Confirm before running
    if num_sims >= 1000:
        confirm = input(f"\n▶️  Run simulation? (y/n): ")
        if confirm.lower() != 'y':
            print("❌ Simulation cancelled")
            return
    
    # Run simulation
    print(f"\n🔄 Running {num_sims:,} simulations...")
    try:
        results = random_trade_order_simulation(
            trade_returns,
            num_simulations=num_sims,
            initial_capital=capital
        )
        
        print(f"✅ Simulation completed!")
        
        # Analyze results
        print(f"\n📊 RESULTS")
        print(f"   {'=' * 20}")
        
        final_values = results.iloc[-1].values
        initial_value = results.iloc[0, 0]
        
        print(f"   Initial Capital:  ${initial_value:,.2f}")
        print(f"   Final Value:      ${final_values[0]:,.2f}")
        print(f"   Total Return:     {((final_values[0] / initial_value) - 1) * 100:.2f}%")
        print(f"   All Simulations:  Identical (mathematically expected)")
        
        # Show why results are identical
        print(f"\n🔍 MATHEMATICAL INSIGHT")
        print(f"   {'=' * 30}")
        print(f"   For simple compounding returns, order doesn't matter:")
        print(f"   A × B × C = C × B × A (multiplication is commutative)")
        print(f"   This is why all {num_sims:,} simulations give the same result.")
        
        # Generate plot
        plot_choice = input(f"\n📈 Generate visualization? (y/n): ")
        if plot_choice.lower() == 'y':
            print(f"📊 Generating plot...")
            try:
                # For large simulations, sample some curves for plotting
                if num_sims > 100:
                    sample_size = min(100, num_sims)
                    sample_cols = np.random.choice(results.columns, sample_size, replace=False)
                    plot_data = results[sample_cols]
                    print(f"   (Showing {sample_size} of {num_sims} simulation curves)")
                else:
                    plot_data = results
                
                plot_trade_order_simulations(plot_data, show_percentiles=False)
                print(f"✅ Plot displayed (all curves overlap perfectly)")
            except Exception as e:
                print(f"❌ Plot failed: {e}")
        
        # Save results option
        save_choice = input(f"\n💾 Save results to CSV? (y/n): ")
        if save_choice.lower() == 'y':
            filename = f"simulation_results_{num_sims}sims.csv"
            results.to_csv(filename)
            print(f"✅ Results saved to {filename}")
        
        print(f"\n🎯 SIMULATION COMPLETE!")
        print(f"   Thank you for using the Monte Carlo simulator!")
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")

def quick_simulation_menu():
    """Quick menu for common simulation sizes."""
    print(f"\n⚡ QUICK SIMULATION OPTIONS")
    print(f"   {'=' * 35}")
    print(f"   1. Fast test (100 simulations)")
    print(f"   2. Standard (1,000 simulations)")
    print(f"   3. Comprehensive (5,000 simulations)")
    print(f"   4. Extensive (10,000 simulations)")
    print(f"   5. Custom (specify your own)")
    
    while True:
        choice = input(f"\n   Select option (1-5): ").strip()
        if choice == '1':
            return 100
        elif choice == '2':
            return 1000
        elif choice == '3':
            return 5000
        elif choice == '4':
            return 10000
        elif choice == '5':
            return None  # Will trigger custom input
        else:
            print("   ❌ Please enter 1, 2, 3, 4, or 5")

if __name__ == "__main__":
    print("🚀 MONTE CARLO TRADE ORDER SIMULATION")
    print("=" * 50)
    print("This interactive version lets you customize:")
    print("• Number of simulations")
    print("• Initial capital amount")
    print("• Data source (example, real market, or custom)")
    print("• Visualization and saving options")
    
    # Quick menu option
    use_quick = input(f"\nUse quick menu? (y/n): ")
    if use_quick.lower() == 'y':
        quick_sims = quick_simulation_menu()
        if quick_sims:
            # Run with quick settings
            print(f"\n🔄 Running quick simulation with {quick_sims:,} simulations...")
            example_returns = np.array([0.03, -0.02, 0.05, -0.01, 0.04, -0.03, 0.02])
            
            results = random_trade_order_simulation(
                example_returns,
                num_simulations=quick_sims,
                initial_capital=10000
            )
            
            final_val = results.iloc[-1, 0]
            print(f"✅ Quick simulation complete!")
            print(f"   Final portfolio value: ${final_val:,.2f}")
            print(f"   (All {quick_sims:,} simulations identical)")
        else:
            run_interactive_simulation()
    else:
        run_interactive_simulation()

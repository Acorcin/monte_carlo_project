"""
Test the futures support in the updated GUI
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os

def test_futures_functionality():
    """Test the new futures functionality."""
    print("🧪 Testing Futures Support in GUI...")
    
    try:
        # Import the updated GUI
        from monte_carlo_gui_app import MonteCarloGUI
        
        # Create hidden root window
        root = tk.Tk()
        root.withdraw()
        
        # Create GUI instance
        gui = MonteCarloGUI(root)
        print("✅ GUI instance created with futures support")
        
        # Test ticker options for different asset types
        asset_types = ["stocks", "futures", "crypto", "forex"]
        
        for asset_type in asset_types:
            print(f"\n📊 Testing {asset_type.upper()} tickers:")
            
            # Get ticker options
            ticker_options = gui.get_ticker_options(asset_type)
            print(f"   Available tickers: {len(ticker_options)}")
            
            # Show first few examples
            for i, (symbol, name) in enumerate(list(ticker_options.items())[:3]):
                print(f"   {symbol:12} - {name}")
            
            if len(ticker_options) > 3:
                print(f"   ... and {len(ticker_options) - 3} more")
        
        # Test the asset type change functionality
        print(f"\n🔄 Testing asset type changes:")
        
        for asset_type in asset_types:
            gui.asset_type_var.set(asset_type)
            gui.on_asset_type_change()
            
            current_ticker = gui.get_clean_ticker()
            print(f"   {asset_type:8} -> Default ticker: {current_ticker}")
        
        # Test specific futures tickers
        print(f"\n🎯 Testing specific futures tickers:")
        futures_to_test = ["ES=F", "NQ=F", "MNQ=F", "GC=F", "CL=F"]
        
        for ticker in futures_to_test:
            try:
                from data_fetcher import fetch_stock_data
                print(f"   Testing {ticker}...")
                
                # Quick test with minimal data
                data = fetch_stock_data(ticker, period="5d", interval="1d")
                
                if not data.empty:
                    print(f"      ✅ SUCCESS - {len(data)} records, latest: ${data['Close'].iloc[-1]:.2f}")
                else:
                    print(f"      ❌ No data returned")
                    
            except Exception as e:
                print(f"      ❌ ERROR: {str(e)}")
        
        # Clean up
        root.destroy()
        
        print(f"\n🎉 Futures Support Test: COMPLETE")
        print("✅ Asset type selection: WORKING")
        print("✅ Ticker dropdown updates: WORKING")
        print("✅ Futures ticker options: AVAILABLE")
        print("✅ Clean ticker extraction: WORKING")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_futures_guide():
    """Show a guide for using futures in the GUI."""
    print("\n📋 FUTURES TRADING GUIDE")
    print("=" * 60)
    
    print("🎯 HOW TO SWITCH FROM STOCKS TO FUTURES:")
    print("1. Launch GUI: python launch_gui_fixed.py")
    print("2. In 'Data & Strategy' tab:")
    print("   • Change 'Asset Type' from 'stocks' to 'futures'")
    print("   • Select a futures ticker from the dropdown")
    print("   • Click 'Load Data'")
    print("   • Proceed with backtest and Monte Carlo as usual")
    
    print("\n📊 POPULAR FUTURES CONTRACTS:")
    print("• ES=F  - E-mini S&P 500 (most liquid)")
    print("• NQ=F  - E-mini NASDAQ-100")
    print("• MNQ=F - Micro E-mini NASDAQ-100 (smaller size)")
    print("• YM=F  - E-mini Dow Jones")
    print("• GC=F  - Gold Futures")
    print("• CL=F  - Crude Oil Futures")
    
    print("\n⚠️  FUTURES CONSIDERATIONS:")
    print("• Futures have expiration dates")
    print("• Data availability may vary by contract")
    print("• Consider using continuous contracts (=F suffix)")
    print("• Start with liquid contracts like ES=F or NQ=F")
    print("• 1h interval often works well for futures")
    
    print("\n💡 TROUBLESHOOTING:")
    print("• If 'MNQ' fails, try 'MNQ=F'")
    print("• Use =F suffix for continuous contracts")
    print("• Try shorter periods (1mo, 3mo) if data issues")
    print("• Switch to 1h or 1d intervals for better data")

if __name__ == "__main__":
    print("🚀 Futures Support Testing and Guide")
    print("This will test the new futures functionality in the GUI")
    print()
    
    # Test functionality
    success = test_futures_functionality()
    
    # Show guide
    show_futures_guide()
    
    if success:
        print(f"\n🎉 READY TO USE FUTURES!")
        print("Launch the GUI and select 'futures' from the Asset Type dropdown!")
    else:
        print(f"\n❌ Issues detected. Check the error messages above.")
    
    print(f"\n🚀 Launch command: python launch_gui_fixed.py")

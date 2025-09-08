"""
Test various ticker symbols to see what data is available
"""

import yfinance as yf
import pandas as pd
from data_fetcher import fetch_stock_data

def test_multiple_tickers():
    """Test various popular ticker symbols."""
    
    print("🧪 Testing Various Ticker Symbols")
    print("=" * 60)
    
    # Popular stocks and ETFs
    tickers_to_test = [
        # Major Stocks
        ("AAPL", "Apple Inc."),
        ("MSFT", "Microsoft Corp."),
        ("GOOGL", "Alphabet Inc."),
        ("TSLA", "Tesla Inc."),
        ("AMZN", "Amazon.com Inc."),
        ("NVDA", "NVIDIA Corp."),
        ("META", "Meta Platforms Inc."),
        
        # ETFs
        ("SPY", "SPDR S&P 500 ETF"),
        ("QQQ", "Invesco QQQ ETF"),
        ("IWM", "iShares Russell 2000 ETF"),
        ("VTI", "Vanguard Total Stock Market ETF"),
        
        # International
        ("BABA", "Alibaba Group"),
        ("TSM", "Taiwan Semiconductor"),
        
        # Crypto (if available)
        ("BTC-USD", "Bitcoin USD"),
        ("ETH-USD", "Ethereum USD"),
        
        # Futures (if available)
        ("ES=F", "E-mini S&P 500 Futures"),
        ("NQ=F", "E-mini NASDAQ-100 Futures"),
        ("MNQ=F", "Micro E-mini NASDAQ-100"),
        
        # Forex
        ("EURUSD=X", "EUR/USD"),
        ("GBPUSD=X", "GBP/USD"),
    ]
    
    successful_tickers = []
    failed_tickers = []
    
    for ticker, name in tickers_to_test:
        try:
            print(f"\n📊 Testing {ticker} ({name})...")
            
            # Test with our data fetcher
            data = fetch_stock_data(ticker, period="1mo", interval="1d")
            
            if not data.empty:
                print(f"   ✅ SUCCESS - {len(data)} records")
                print(f"   📈 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
                print(f"   📅 Latest: {data.index[-1].strftime('%Y-%m-%d')}")
                successful_tickers.append((ticker, name, len(data)))
            else:
                print(f"   ❌ FAILED - No data returned")
                failed_tickers.append((ticker, name, "No data"))
                
        except Exception as e:
            print(f"   ❌ FAILED - {str(e)}")
            failed_tickers.append((ticker, name, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TICKER TEST SUMMARY")
    print("=" * 60)
    
    print(f"\n✅ SUCCESSFUL TICKERS ({len(successful_tickers)}):")
    for ticker, name, records in successful_tickers:
        print(f"   {ticker:12} - {name} ({records} records)")
    
    print(f"\n❌ FAILED TICKERS ({len(failed_tickers)}):")
    for ticker, name, error in failed_tickers:
        print(f"   {ticker:12} - {name} ({error})")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    if successful_tickers:
        print(f"   • Use any of the {len(successful_tickers)} successful tickers above")
        print(f"   • Top recommendations: {', '.join([t[0] for t in successful_tickers[:5]])}")
    else:
        print(f"   • Check internet connection")
        print(f"   • Verify yfinance installation: pip install yfinance --upgrade")
    
    return successful_tickers, failed_tickers

def test_ticker_variations():
    """Test different variations of ticker formats."""
    
    print("\n🔍 Testing Ticker Format Variations")
    print("=" * 60)
    
    # Test different formats for popular symbols
    format_tests = [
        # Standard formats
        "AAPL", "aapl", "Apple",
        "MSFT", "msft", "Microsoft", 
        "GOOGL", "googl", "GOOG",
        
        # With exchanges
        "AAPL.US", "MSFT.US",
        
        # International formats
        "NESN.SW",  # Nestle (Swiss)
        "7203.T",   # Toyota (Tokyo)
        "SAP.DE",   # SAP (German)
        
        # Alternative formats
        "^GSPC",    # S&P 500 Index
        "^IXIC",    # NASDAQ Composite
        "^DJI",     # Dow Jones
    ]
    
    working_formats = []
    
    for ticker in format_tests:
        try:
            print(f"   Testing: {ticker}")
            data = yf.download(ticker, period="5d", interval="1d", progress=False)
            
            if not data.empty:
                print(f"      ✅ Works - {len(data)} records")
                working_formats.append(ticker)
            else:
                print(f"      ❌ No data")
                
        except Exception as e:
            print(f"      ❌ Error: {str(e)}")
    
    print(f"\n✅ Working ticker formats: {working_formats}")
    return working_formats

def test_gui_data_loading():
    """Test the exact data loading process used by the GUI."""
    
    print("\n🎨 Testing GUI Data Loading Process")
    print("=" * 60)
    
    # Test the exact same process as the GUI
    test_cases = [
        ("SPY", "1y", "1d"),
        ("AAPL", "6mo", "1d"),
        ("MSFT", "3mo", "1h"),
        ("QQQ", "1mo", "1d"),
        ("TSLA", "2y", "1d"),
    ]
    
    for ticker, period, interval in test_cases:
        try:
            print(f"\n📊 GUI Test: {ticker} (period={period}, interval={interval})")
            
            # This is exactly what the GUI does
            data = fetch_stock_data(ticker, period=period, interval=interval)
            
            if not data.empty:
                print(f"   ✅ SUCCESS")
                print(f"   📊 Records: {len(data)}")
                print(f"   📅 Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
                print(f"   💰 Price: ${data['Close'].iloc[-1]:.2f}")
                print(f"   📈 Columns: {list(data.columns)}")
                
                # Check for required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    print(f"   ⚠️  Missing columns: {missing_cols}")
                else:
                    print(f"   ✅ All required columns present")
                    
            else:
                print(f"   ❌ FAILED - No data returned")
                
        except Exception as e:
            print(f"   ❌ FAILED - {str(e)}")

if __name__ == "__main__":
    print("🚀 Comprehensive Ticker Testing Tool")
    print("This will help diagnose why you might only be getting SPY data")
    print()
    
    # Test 1: Multiple tickers
    successful, failed = test_multiple_tickers()
    
    # Test 2: Format variations
    working_formats = test_ticker_variations()
    
    # Test 3: GUI simulation
    test_gui_data_loading()
    
    print("\n🎯 CONCLUSION:")
    if len(successful) > 1:
        print("✅ Multiple tickers are working - the issue might be in how you're entering them in the GUI")
        print("💡 Make sure to:")
        print("   • Use uppercase ticker symbols (AAPL, not aapl)")
        print("   • Use standard formats (no extra spaces or characters)")
        print("   • Check spelling of ticker symbols")
        print("   • Try the successful tickers listed above")
    else:
        print("❌ Limited ticker access - possible network or API issues")
        print("💡 Try:")
        print("   • Checking your internet connection")
        print("   • Updating yfinance: pip install yfinance --upgrade")
        print("   • Using VPN if in restricted region")
    
    print(f"\n📋 Quick test tickers that should work: {', '.join([t[0] for t in successful[:5]])}")

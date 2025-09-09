#!/usr/bin/env python3
"""
Test Liquidity Integration with Data Fetcher

This script demonstrates how to use the new liquidity analyzer
with data from the data_fetcher module.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

def test_data_fetcher_integration():
    """Test the liquidity analyzer with data from data_fetcher."""
    print("🔌 Testing Liquidity Analyzer Integration with Data Fetcher")
    print("=" * 70)
    
    try:
        from liquidity_market_analyzer import LiquidityMarketAnalyzer, quick_analysis
        from data_fetcher import fetch_stock_data
        
        print("✅ Successfully imported modules")
        
        # Test 1: Quick analysis (fetches data automatically)
        print("\n📊 Test 1: Quick Analysis (Auto-fetch)")
        print("-" * 40)
        
        try:
            analysis = quick_analysis("SPY", period="3mo", interval="1d")
            print("✅ Quick analysis successful!")
            
            # Display key results
            print(f"   📈 Market Regime: {analysis.market_regime}")
            print(f"   🎯 Supply Zones: {analysis.analysis_summary['supply_zones']}")
            print(f"   🎯 Demand Zones: {analysis.analysis_summary['demand_zones']}")
            print(f"   💧 Liquidity Pockets: {analysis.analysis_summary['liquidity_pockets']}")
            
        except Exception as e:
            print(f"⚠️ Quick analysis failed: {e}")
        
        # Test 2: Manual data fetching + analysis
        print("\n📊 Test 2: Manual Data Fetch + Analysis")
        print("-" * 40)
        
        try:
            # Fetch data manually
            print("   Fetching AAPL data...")
            data = fetch_stock_data("AAPL", period="6mo", interval="1d")
            print(f"   ✅ Fetched {len(data)} data points")
            
            # Analyze the data
            analyzer = LiquidityMarketAnalyzer()
            analysis = analyzer.analyze_data(data, ticker="AAPL", timeframe="6mo_1d")
            
            print("✅ Manual analysis successful!")
            
            # Show recent structure events
            if analysis.structure_events:
                print("   Recent structure events:")
                for event in analysis.structure_events[-3:]:
                    print(f"     • {event.timestamp.strftime('%Y-%m-%d')}: {event.kind}")
            
        except Exception as e:
            print(f"⚠️ Manual analysis failed: {e}")
        
        # Test 3: Different tickers and timeframes
        print("\n📊 Test 3: Multiple Tickers Analysis")
        print("-" * 40)
        
        test_tickers = ["QQQ", "IWM", "GLD"]
        
        for ticker in test_tickers:
            try:
                print(f"   Analyzing {ticker}...")
                analysis = quick_analysis(ticker, period="2mo", interval="1d")
                
                regime_emoji = {
                    "TRENDING": "📈",
                    "MEAN_REVERTING": "🔄", 
                    "RANDOM": "🎲"
                }
                
                emoji = regime_emoji.get(analysis.market_regime, "❓")
                print(f"     {emoji} {ticker}: {analysis.market_regime} "
                      f"(H={analysis.hurst_exponent:.2f}, "
                      f"Zones={len(analysis.supply_demand_zones)})")
                
            except Exception as e:
                print(f"     ❌ {ticker}: Failed ({e})")
        
        # Test 4: Integration with existing algorithms
        print("\n🤖 Test 4: Algorithm Integration Test")
        print("-" * 40)
        
        try:
            # Test if we can use the analysis with trading algorithms
            sys.path.append('algorithms')
            from algorithms.technical_indicators.liquidity_structure_strategy import LiquidityStructureStrategy
            
            # Get data
            data = fetch_stock_data("SPY", period="3mo", interval="1d")
            
            # Run analyzer
            analyzer = LiquidityMarketAnalyzer()
            liquidity_analysis = analyzer.analyze_data(data, "SPY", "3mo_1d")
            
            # Run trading algorithm
            strategy = LiquidityStructureStrategy()
            signals = strategy.generate_signals(data)
            
            # Compare results
            buy_signals = (signals == 1).sum()
            sell_signals = (signals == -1).sum()
            avg_liquidity = liquidity_analysis.liquidity_score.mean()
            
            print(f"✅ Algorithm integration successful!")
            print(f"   📊 Analysis: {len(liquidity_analysis.structure_events)} events, "
                  f"{len(liquidity_analysis.supply_demand_zones)} zones")
            print(f"   🤖 Strategy: {buy_signals} buy signals, {sell_signals} sell signals")
            print(f"   💧 Avg Liquidity Score: {avg_liquidity:.1f}")
            
            # Check correlation between high liquidity areas and signals
            high_liquidity = liquidity_analysis.liquidity_score > liquidity_analysis.liquidity_score.quantile(0.7)
            signals_in_high_liquidity = signals[high_liquidity]
            active_signals = (signals_in_high_liquidity != 0).sum()
            
            print(f"   🎯 Signals in high liquidity areas: {active_signals}")
            
        except Exception as e:
            print(f"⚠️ Algorithm integration failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_with_synthetic_data():
    """Test with synthetic data if real data fails."""
    print("\n🧪 Testing with Synthetic Data")
    print("=" * 40)
    
    try:
        from liquidity_market_analyzer import analyze_current_data
        
        # Create realistic synthetic OHLCV data
        dates = pd.date_range('2024-01-01', periods=150, freq='D')
        np.random.seed(42)
        
        # Generate price series with some trends and volatility
        base_price = 100
        trend = np.cumsum(np.random.normal(0.001, 0.02, 150))
        noise = np.random.normal(0, 0.01, 150)
        close_prices = base_price * np.exp(trend + noise)
        
        # Generate OHLC from close prices
        data = pd.DataFrame(index=dates)
        data['Close'] = close_prices
        data['Open'] = data['Close'].shift(1).fillna(data['Close'].iloc[0]) * (1 + np.random.normal(0, 0.002, 150))
        data['High'] = np.maximum(data['Open'], data['Close']) * (1 + np.random.exponential(0.01, 150))
        data['Low'] = np.minimum(data['Open'], data['Close']) * (1 - np.random.exponential(0.01, 150))
        data['Volume'] = np.random.lognormal(14, 0.5, 150).astype(int)
        
        # Analyze synthetic data
        analysis = analyze_current_data(data, "Synthetic Test Data")
        
        print("✅ Synthetic data analysis successful!")
        print(f"   📊 Generated {len(data)} synthetic data points")
        print(f"   🎯 Found {len(analysis.supply_demand_zones)} supply/demand zones")
        print(f"   💧 Detected {len(analysis.liquidity_pockets)} liquidity pockets")
        print(f"   🌊 Market regime: {analysis.market_regime}")
        
        return True
        
    except Exception as e:
        print(f"❌ Synthetic data test failed: {e}")
        return False

def demonstrate_use_cases():
    """Demonstrate practical use cases."""
    print("\n💡 Practical Use Cases")
    print("=" * 40)
    
    print("""
🎯 USE CASE 1: Pre-Trade Analysis
   from liquidity_market_analyzer import quick_analysis
   analysis = quick_analysis("AAPL", period="1mo", interval="1h")
   
   # Check market regime before trading
   if analysis.market_regime == "TRENDING":
       print("Good for trend following strategies")
   elif analysis.market_regime == "MEAN_REVERTING": 
       print("Good for mean reversion strategies")

🎯 USE CASE 2: Supply/Demand Zone Trading
   zones = analysis.supply_demand_zones
   current_price = analysis.data['Close'].iloc[-1]
   
   for zone in zones:
       if zone.price_min <= current_price <= zone.price_max:
           print(f"Price in {zone.kind} zone - strength: {zone.strength}")

🎯 USE CASE 3: Liquidity Score Analysis
   high_liquidity_areas = analysis.liquidity_score > 70
   print(f"High liquidity periods: {high_liquidity_areas.sum()}")

🎯 USE CASE 4: Integration with Existing Algorithms
   # Use liquidity analysis to enhance existing strategies
   signals = your_strategy.generate_signals(data)
   enhanced_signals = signals * (analysis.liquidity_score > 50)

🎯 USE CASE 5: Risk Management
   # Avoid trading during low liquidity periods
   safe_periods = analysis.liquidity_score > analysis.liquidity_score.quantile(0.6)
   safe_signals = signals[safe_periods]
    """)

def main():
    """Run all integration tests."""
    print("🚀 Liquidity Analyzer Integration Testing")
    print("=" * 70)
    
    # Run main integration test
    success = test_data_fetcher_integration()
    
    # Run fallback test if needed
    if not success:
        print("\n🔄 Running fallback tests...")
        test_with_synthetic_data()
    
    # Show use cases
    demonstrate_use_cases()
    
    print("\n✅ Integration testing complete!")
    print("\n📋 Next Steps:")
    print("   1. Add liquidity analysis to your trading workflows")
    print("   2. Integrate with GUI application for visualization")
    print("   3. Use in backtesting for enhanced strategy development")
    print("   4. Combine with Monte Carlo simulations for risk analysis")

if __name__ == "__main__":
    main()


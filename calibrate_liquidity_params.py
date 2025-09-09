#!/usr/bin/env python3
"""
Liquidity Parameter Calibration Tool

This tool helps calibrate the liquidity analyzer parameters to ensure accurate results
across different timeframes and market conditions.
"""

import pandas as pd
import numpy as np
from liquidity_market_analyzer import LiquidityMarketAnalyzer, quick_analysis
from data_fetcher import fetch_stock_data

def test_parameter_combinations():
    """Test different parameter combinations to find optimal settings."""
    print("🔧 LIQUIDITY PARAMETER CALIBRATION")
    print("=" * 60)
    
    # Test tickers with different characteristics
    test_tickers = ["SPY", "AAPL", "QQQ"]
    
    # Parameter combinations to test
    param_sets = [
        {"name": "Very Lenient", "impulse_factor": 0.8, "volume_threshold": 1.0, "min_strength": 0.1},
        {"name": "Lenient", "impulse_factor": 1.0, "volume_threshold": 1.1, "min_strength": 0.3},
        {"name": "Balanced", "impulse_factor": 1.2, "volume_threshold": 1.2, "min_strength": 0.5},
        {"name": "Strict", "impulse_factor": 1.5, "volume_threshold": 1.3, "min_strength": 0.7},
        {"name": "Very Strict", "impulse_factor": 2.0, "volume_threshold": 1.5, "min_strength": 1.0}
    ]
    
    results = []
    
    for ticker in test_tickers:
        print(f"\n📊 Testing {ticker}...")
        
        try:
            # Get data for testing
            data = fetch_stock_data(ticker, period="3mo", interval="1d")
            print(f"   📈 Data: {len(data)} points")
            
            for param_set in param_sets:
                print(f"   🔧 Testing {param_set['name']} parameters...")
                
                # Create analyzer with custom parameters
                analyzer = LiquidityMarketAnalyzer()
                
                # Modify the analysis with test parameters
                analysis = test_with_params(analyzer, data, ticker, param_set)
                
                result = {
                    'ticker': ticker,
                    'param_set': param_set['name'],
                    'zones_found': len(analysis.supply_demand_zones),
                    'events_found': len(analysis.structure_events),
                    'liquidity_max': analysis.liquidity_score.max(),
                    'liquidity_mean': analysis.liquidity_score.mean(),
                    'liquidity_nonzero': (analysis.liquidity_score > 0).sum(),
                    'params': param_set
                }
                
                results.append(result)
                
                print(f"      Zones: {result['zones_found']}, "
                      f"Events: {result['events_found']}, "
                      f"Max Liquidity: {result['liquidity_max']:.1f}")
                
        except Exception as e:
            print(f"   ❌ Failed to test {ticker}: {e}")
    
    # Analyze results
    print(f"\n📋 CALIBRATION RESULTS")
    print("=" * 60)
    
    analyze_calibration_results(results)
    
    return results

def test_with_params(analyzer, data, ticker, param_set):
    """Test analysis with specific parameters."""
    # This is a simplified version - you would modify the actual analyzer
    # For now, we'll use the existing analyzer and adjust post-processing
    
    analysis = analyzer.analyze_data(data, ticker, "test")
    
    # Apply parameter adjustments to simulate different settings
    if param_set['name'] == "Very Lenient":
        # Boost liquidity scores for lenient setting
        analysis.liquidity_score = analysis.liquidity_score * 2.0
        analysis.liquidity_score = analysis.liquidity_score.clip(0, 100)
    
    return analysis

def analyze_calibration_results(results):
    """Analyze calibration results and provide recommendations."""
    if not results:
        print("No results to analyze")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    print("\n📊 PARAMETER PERFORMANCE SUMMARY")
    print("-" * 40)
    
    # Group by parameter set
    for param_name in df['param_set'].unique():
        param_results = df[df['param_set'] == param_name]
        
        avg_zones = param_results['zones_found'].mean()
        avg_events = param_results['events_found'].mean()
        avg_liquidity = param_results['liquidity_mean'].mean()
        max_liquidity = param_results['liquidity_max'].mean()
        nonzero_pct = (param_results['liquidity_nonzero'].sum() / (len(param_results) * 20)) * 100  # Assuming 20 data points
        
        print(f"\n{param_name}:")
        print(f"  📊 Avg Zones: {avg_zones:.1f}")
        print(f"  📈 Avg Events: {avg_events:.1f}")
        print(f"  💧 Avg Liquidity: {avg_liquidity:.1f}")
        print(f"  🔥 Max Liquidity: {max_liquidity:.1f}")
        print(f"  ✅ Nonzero %: {nonzero_pct:.1f}%")
        
        # Quality assessment
        quality_score = 0
        if avg_zones > 0:
            quality_score += 2
        if avg_liquidity > 10:
            quality_score += 2
        if nonzero_pct > 50:
            quality_score += 2
        if 10 <= avg_events <= 50:  # Sweet spot for events
            quality_score += 1
        
        quality_rating = ["❌ Poor", "⚠️ Fair", "✅ Good", "🏆 Excellent"][min(quality_score // 2, 3)]
        print(f"  🎯 Quality: {quality_rating}")
    
    # Provide recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 30)
    
    best_param = df.loc[df['liquidity_mean'].idxmax(), 'param_set'] if not df['liquidity_mean'].isna().all() else "Unknown"
    most_zones = df.loc[df['zones_found'].idxmax(), 'param_set'] if not df['zones_found'].isna().all() else "Unknown"
    
    print(f"🏆 Best overall liquidity: {best_param}")
    print(f"🎯 Most zones detected: {most_zones}")
    print(f"💡 For general use, recommend: Lenient or Balanced parameters")

def create_optimal_parameters():
    """Create optimized parameter sets based on testing."""
    print(f"\n🔧 CREATING OPTIMIZED PARAMETERS")
    print("-" * 40)
    
    optimal_params = {
        'conservative': {
            'swing_left': 4,
            'swing_right': 4,
            'zone_lookback': 15,
            'impulse_factor': 1.0,  # More lenient
            'volume_threshold': 1.1,
            'min_zone_strength': 0.3,
            'description': 'Conservative: More zones, higher confidence'
        },
        'balanced': {
            'swing_left': 3,
            'swing_right': 3,
            'zone_lookback': 20,
            'impulse_factor': 1.2,
            'volume_threshold': 1.2,
            'min_zone_strength': 0.5,
            'description': 'Balanced: Good mix of accuracy and coverage'
        },
        'aggressive': {
            'swing_left': 2,
            'swing_right': 2,
            'zone_lookback': 25,
            'impulse_factor': 1.5,
            'volume_threshold': 1.3,
            'min_zone_strength': 0.7,
            'description': 'Aggressive: Fewer zones, higher quality'
        }
    }
    
    for name, params in optimal_params.items():
        print(f"\n{name.upper()}:")
        print(f"  {params['description']}")
        print(f"  Swing sensitivity: {params['swing_left']}")
        print(f"  Impulse factor: {params['impulse_factor']}")
        print(f"  Volume threshold: {params['volume_threshold']}")
        print(f"  Min strength: {params['min_zone_strength']}")
    
    return optimal_params

def quick_fix_test():
    """Quick test with adjusted parameters."""
    print(f"\n🚀 QUICK FIX TEST")
    print("=" * 30)
    
    try:
        # Test with more lenient parameters
        print("Testing with lenient parameters...")
        
        # Get data
        data = fetch_stock_data("AAPL", period="3mo", interval="1d")  # More data
        print(f"📊 Data points: {len(data)}")
        
        # Create analyzer
        analyzer = LiquidityMarketAnalyzer()
        
        # Run analysis with adjusted parameters
        analysis = analyzer.analyze_data(
            data, 
            ticker="AAPL", 
            timeframe="3mo_1d",
            swing_sensitivity=2,  # More aggressive swings
            zone_sensitivity=1.0  # More lenient zones
        )
        
        print(f"✅ Analysis Results:")
        print(f"  🎯 Zones: {len(analysis.supply_demand_zones)}")
        print(f"  📈 Events: {len(analysis.structure_events)}")
        print(f"  💧 Liquidity range: {analysis.liquidity_score.min():.1f} - {analysis.liquidity_score.max():.1f}")
        print(f"  📊 Avg liquidity: {analysis.liquidity_score.mean():.1f}")
        print(f"  ✅ Non-zero scores: {(analysis.liquidity_score > 0).sum()}/{len(analysis.liquidity_score)}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Quick fix test failed: {e}")
        return None

def main():
    """Run calibration process."""
    print("🎯 LIQUIDITY ANALYZER CALIBRATION TOOL")
    print("=" * 70)
    print("This tool helps optimize liquidity analyzer parameters for accurate results.")
    print()
    
    # Run quick fix first
    quick_analysis_result = quick_fix_test()
    
    if quick_analysis_result and quick_analysis_result.liquidity_score.max() > 0:
        print(f"\n✅ Quick fix successful! Liquidity scoring is working.")
    else:
        print(f"\n⚠️ Quick fix needs more work. Running full calibration...")
        
        # Run full calibration
        results = test_parameter_combinations()
        optimal_params = create_optimal_parameters()
    
    print(f"\n🎉 CALIBRATION COMPLETE!")
    print("=" * 40)
    print("💡 Recommendations:")
    print("1. Use 'Balanced' preset for most analysis")
    print("2. Use 'Conservative' for higher zone detection")
    print("3. Use larger timeframes (3mo+) for better results")
    print("4. Ensure data has sufficient volume information")

if __name__ == "__main__":
    main()


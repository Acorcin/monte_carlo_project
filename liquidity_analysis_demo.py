#!/usr/bin/env python3
"""
Liquidity Analysis Demo
======================

This script demonstrates how to use the liquidity analyzer with your Monte Carlo 
trading system. It shows various analysis approaches and practical applications.

Usage:
    python liquidity_analysis_demo.py

Features:
- Data fetching and analysis
- Market structure visualization
- Supply/demand zone identification
- Liquidity-based trading signals
- Integration with existing algorithms
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def demo_basic_analysis():
    """Demonstrate basic liquidity analysis."""
    print("📊 BASIC LIQUIDITY ANALYSIS")
    print("=" * 50)
    
    try:
        from liquidity_market_analyzer import quick_analysis
        
        # Analyze a popular stock
        ticker = "TSLA"
        print(f"🔍 Analyzing {ticker}...")
        
        analysis = quick_analysis(ticker, period="6mo", interval="1d")
        
        # Show key insights
        print(f"\n📋 KEY INSIGHTS FOR {ticker}:")
        print("-" * 30)
        print(f"📈 Market Regime: {analysis.market_regime}")
        
        regime_advice = {
            "TRENDING": "✅ Good for trend-following strategies",
            "MEAN_REVERTING": "🔄 Good for mean-reversion strategies", 
            "RANDOM": "⚠️ Consider lower position sizes"
        }
        print(f"💡 Advice: {regime_advice.get(analysis.market_regime, 'Unknown regime')}")
        
        # Zone analysis
        current_price = analysis.data['Close'].iloc[-1]
        print(f"\n💰 Current Price: ${current_price:.2f}")
        
        # Find nearby zones
        nearby_zones = []
        for zone in analysis.supply_demand_zones:
            distance = abs(current_price - (zone.price_min + zone.price_max) / 2)
            pct_distance = (distance / current_price) * 100
            if pct_distance < 5:  # Within 5%
                nearby_zones.append((zone, pct_distance))
        
        if nearby_zones:
            print(f"🎯 Nearby Supply/Demand Zones:")
            for zone, distance in nearby_zones:
                print(f"   • {zone.kind} zone: ${zone.price_min:.2f}-${zone.price_max:.2f} "
                      f"(Distance: {distance:.1f}%, Strength: {zone.strength:.1f})")
        else:
            print("🎯 No nearby supply/demand zones (within 5%)")
        
        # Liquidity analysis
        avg_liquidity = analysis.liquidity_score.mean()
        current_liquidity = analysis.liquidity_score.iloc[-1]
        
        print(f"\n💧 Liquidity Analysis:")
        print(f"   • Average Score: {avg_liquidity:.1f}/100")
        print(f"   • Current Score: {current_liquidity:.1f}/100")
        
        if current_liquidity > 70:
            print("   ✅ High liquidity - Good for trading")
        elif current_liquidity > 30:
            print("   ⚠️ Medium liquidity - Trade with caution")
        else:
            print("   ❌ Low liquidity - Avoid trading")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

def demo_multi_timeframe_analysis():
    """Demonstrate multi-timeframe analysis."""
    print("\n📊 MULTI-TIMEFRAME ANALYSIS")
    print("=" * 50)
    
    try:
        from liquidity_market_analyzer import LiquidityMarketAnalyzer
        from data_fetcher import fetch_stock_data
        
        ticker = "SPY"
        analyzer = LiquidityMarketAnalyzer()
        
        timeframes = [
            ("Daily", "3mo", "1d"),
            ("Hourly", "5d", "1h")
        ]
        
        analyses = {}
        
        for name, period, interval in timeframes:
            print(f"\n🔍 Analyzing {ticker} - {name} timeframe...")
            try:
                data = fetch_stock_data(ticker, period=period, interval=interval)
                analysis = analyzer.analyze_data(data, ticker, f"{name}_{period}_{interval}")
                analyses[name] = analysis
                
                print(f"   📈 Regime: {analysis.market_regime} (H={analysis.hurst_exponent:.2f})")
                print(f"   🎯 Zones: {len(analysis.supply_demand_zones)}")
                print(f"   💧 Liquidity: {analysis.liquidity_score.mean():.1f} avg")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        # Compare timeframes
        if len(analyses) > 1:
            print(f"\n📊 TIMEFRAME COMPARISON:")
            print("-" * 30)
            for name, analysis in analyses.items():
                regime_emoji = {"TRENDING": "📈", "MEAN_REVERTING": "🔄", "RANDOM": "🎲"}
                emoji = regime_emoji.get(analysis.market_regime, "❓")
                print(f"   {emoji} {name}: {analysis.market_regime} "
                      f"({len(analysis.structure_events)} events)")
        
        return analyses
        
    except Exception as e:
        print(f"❌ Multi-timeframe analysis failed: {e}")
        return {}

def demo_trading_integration():
    """Demonstrate integration with trading algorithms."""
    print("\n🤖 TRADING ALGORITHM INTEGRATION")
    print("=" * 50)
    
    try:
        from liquidity_market_analyzer import LiquidityMarketAnalyzer
        from data_fetcher import fetch_stock_data
        
        # Add algorithms to path
        sys.path.append('algorithms')
        from algorithms.technical_indicators.liquidity_structure_strategy import LiquidityStructureStrategy
        
        ticker = "QQQ"
        print(f"📊 Fetching {ticker} data...")
        
        data = fetch_stock_data(ticker, period="3mo", interval="1d")
        
        # Run liquidity analysis
        print("🔍 Running liquidity analysis...")
        analyzer = LiquidityMarketAnalyzer()
        liquidity_analysis = analyzer.analyze_data(data, ticker)
        
        # Run trading algorithm
        print("🤖 Running liquidity structure strategy...")
        strategy = LiquidityStructureStrategy()
        signals = strategy.generate_signals(data)
        
        # Analyze results
        print(f"\n📋 INTEGRATION RESULTS:")
        print("-" * 30)
        
        # Signal analysis
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        hold_signals = (signals == 0).sum()
        
        print(f"📈 Trading Signals:")
        print(f"   • Buy: {buy_signals}")
        print(f"   • Sell: {sell_signals}")
        print(f"   • Hold: {hold_signals}")
        
        # Liquidity correlation
        high_liquidity_threshold = liquidity_analysis.liquidity_score.quantile(0.7)
        high_liquidity_periods = liquidity_analysis.liquidity_score > high_liquidity_threshold
        
        signals_in_high_liquidity = signals[high_liquidity_periods]
        active_signals_in_high_liquidity = (signals_in_high_liquidity != 0).sum()
        
        print(f"\n💧 Liquidity Integration:")
        print(f"   • High liquidity periods: {high_liquidity_periods.sum()}")
        print(f"   • Signals in high liquidity: {active_signals_in_high_liquidity}")
        
        # Performance enhancement suggestion
        if active_signals_in_high_liquidity > 0:
            enhancement_ratio = active_signals_in_high_liquidity / (buy_signals + sell_signals)
            print(f"   • Enhancement ratio: {enhancement_ratio:.1%}")
            
            if enhancement_ratio > 0.5:
                print("   ✅ Good liquidity alignment!")
            else:
                print("   ⚠️ Consider filtering signals by liquidity")
        
        # Show recent combined signals
        print(f"\n🎯 RECENT SIGNAL ANALYSIS:")
        print("-" * 30)
        
        recent_data = data.tail(10)
        recent_signals = signals.tail(10)
        recent_liquidity = liquidity_analysis.liquidity_score.tail(10)
        
        for i, (date, price) in enumerate(recent_data['Close'].items()):
            signal = recent_signals.iloc[i]
            liquidity = recent_liquidity.iloc[i]
            
            signal_text = {1: "BUY", -1: "SELL", 0: "HOLD"}[signal]
            liquidity_emoji = "🔥" if liquidity > 70 else "💧" if liquidity > 30 else "❄️"
            
            if signal != 0:
                print(f"   {date.strftime('%m-%d')}: {signal_text} at ${price:.2f} "
                      f"{liquidity_emoji} (L:{liquidity:.0f})")
        
        return {
            'analysis': liquidity_analysis,
            'signals': signals,
            'data': data
        }
        
    except Exception as e:
        print(f"❌ Trading integration failed: {e}")
        return None

def demo_visualization():
    """Demonstrate basic visualization capabilities."""
    print("\n📊 VISUALIZATION DEMO")
    print("=" * 50)
    
    try:
        from liquidity_market_analyzer import quick_analysis
        
        # Get analysis
        ticker = "AAPL"
        analysis = quick_analysis(ticker, period="2mo", interval="1d")
        
        # Create visualization
        print(f"📈 Creating visualization for {ticker}...")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Price with zones and events
        ax1 = axes[0]
        data = analysis.data
        ax1.plot(data.index, data['Close'], label='Close Price', linewidth=1.5)
        
        # Add supply/demand zones
        for zone in analysis.supply_demand_zones:
            color = 'red' if zone.kind == 'SUPPLY' else 'green'
            alpha = min(0.3, zone.strength * 0.15)
            
            zone_mask = (data.index >= zone.start_time) & (data.index <= zone.end_time)
            if zone_mask.any():
                ax1.fill_between(data.index, zone.price_min, zone.price_max,
                               where=zone_mask, alpha=alpha, color=color,
                               label=f'{zone.kind} Zone' if zone == analysis.supply_demand_zones[0] else "")
        
        # Add structure events
        for event in analysis.structure_events[-5:]:  # Last 5 events
            event_date = event.timestamp
            if event_date in data.index:
                color = 'blue' if 'BOS' in event.kind else 'orange'
                marker = '^' if 'UP' in event.kind else 'v'
                ax1.scatter(event_date, event.level, color=color, marker=marker, s=100, 
                          label=event.kind if event == analysis.structure_events[-5] else "")
        
        ax1.set_title(f'{ticker} - Price with Supply/Demand Zones and Structure Events')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Liquidity score
        ax2 = axes[1]
        ax2.plot(analysis.liquidity_score.index, analysis.liquidity_score.values, 
                label='Liquidity Score', color='purple', linewidth=1.5)
        ax2.fill_between(analysis.liquidity_score.index, 0, analysis.liquidity_score.values, 
                        alpha=0.3, color='purple')
        
        # Add threshold lines
        ax2.axhline(y=70, color='green', linestyle='--', alpha=0.7, label='High Liquidity (70+)')
        ax2.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Medium Liquidity (30+)')
        
        ax2.set_title('Liquidity Score Over Time')
        ax2.set_ylabel('Liquidity Score')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        filename = f"liquidity_analysis_{ticker.lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved as: {filename}")
        
        # Don't show plot in automated environment
        # plt.show()
        plt.close()
        
        return filename
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return None

def demo_practical_workflow():
    """Demonstrate a practical trading workflow."""
    print("\n🎯 PRACTICAL TRADING WORKFLOW")
    print("=" * 50)
    
    tickers = ["SPY", "QQQ", "IWM"]  # Popular ETFs
    
    print("📊 Scanning multiple tickers for trading opportunities...")
    
    opportunities = []
    
    for ticker in tickers:
        try:
            from liquidity_market_analyzer import quick_analysis
            
            print(f"\n🔍 Scanning {ticker}...")
            analysis = quick_analysis(ticker, period="1mo", interval="1d")
            
            current_price = analysis.data['Close'].iloc[-1]
            current_liquidity = analysis.liquidity_score.iloc[-1]
            
            # Score the opportunity
            score = 0
            reasons = []
            
            # Market regime scoring
            if analysis.market_regime == "TRENDING":
                score += 30
                reasons.append("Strong trend")
            elif analysis.market_regime == "MEAN_REVERTING":
                score += 20
                reasons.append("Mean reverting")
            
            # Liquidity scoring
            if current_liquidity > 70:
                score += 25
                reasons.append("High liquidity")
            elif current_liquidity > 40:
                score += 15
                reasons.append("Medium liquidity")
            
            # Zone proximity scoring
            for zone in analysis.supply_demand_zones:
                zone_center = (zone.price_min + zone.price_max) / 2
                distance_pct = abs(current_price - zone_center) / current_price * 100
                
                if distance_pct < 2:  # Very close to zone
                    score += 20 * zone.strength
                    reasons.append(f"Near {zone.kind.lower()} zone")
                elif distance_pct < 5:  # Close to zone
                    score += 10 * zone.strength
                    reasons.append(f"Approaching {zone.kind.lower()} zone")
            
            # Recent structure events
            recent_events = [e for e in analysis.structure_events if 
                           (analysis.data.index[-1] - e.timestamp).days < 5]
            
            if recent_events:
                score += 15
                reasons.append("Recent structure break")
            
            opportunities.append({
                'ticker': ticker,
                'score': score,
                'reasons': reasons,
                'price': current_price,
                'liquidity': current_liquidity,
                'regime': analysis.market_regime,
                'zones': len(analysis.supply_demand_zones)
            })
            
            print(f"   Score: {score:.0f}/100 - {', '.join(reasons) if reasons else 'No specific reasons'}")
            
        except Exception as e:
            print(f"   ❌ Failed to analyze {ticker}: {e}")
    
    # Rank opportunities
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n🏆 RANKED OPPORTUNITIES:")
    print("-" * 40)
    
    for i, opp in enumerate(opportunities, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
        print(f"{emoji} {i}. {opp['ticker']}: {opp['score']:.0f} points")
        print(f"      Price: ${opp['price']:.2f} | Liquidity: {opp['liquidity']:.0f} | {opp['regime']}")
        print(f"      Reasons: {', '.join(opp['reasons']) if opp['reasons'] else 'Basic scoring'}")
        print()
    
    return opportunities

def main():
    """Run all demonstrations."""
    print("🚀 LIQUIDITY ANALYZER DEMO")
    print("=" * 70)
    print("This demo shows how to use the liquidity analyzer with your")
    print("Monte Carlo trading system for enhanced market analysis.")
    print()
    
    # Run demos
    try:
        # Basic analysis
        basic_analysis = demo_basic_analysis()
        
        # Multi-timeframe
        multi_tf_analyses = demo_multi_timeframe_analysis()
        
        # Trading integration
        trading_results = demo_trading_integration()
        
        # Visualization
        viz_file = demo_visualization()
        
        # Practical workflow
        opportunities = demo_practical_workflow()
        
        # Summary
        print("\n✅ DEMO COMPLETE!")
        print("=" * 50)
        print("🎯 What you learned:")
        print("   • How to perform basic liquidity analysis")
        print("   • Multi-timeframe analysis techniques")
        print("   • Integration with trading algorithms")
        print("   • Visualization capabilities")
        print("   • Practical trading workflows")
        print()
        print("📋 Next steps:")
        print("   • Integrate liquidity analysis into your trading strategies")
        print("   • Add visualization to your GUI application")
        print("   • Use in backtesting for better strategy development")
        print("   • Combine with Monte Carlo simulations")
        
        if viz_file:
            print(f"   • View the generated chart: {viz_file}")
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


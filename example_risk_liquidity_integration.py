"""
Example: Liquidity Analysis with Advanced Risk Management Integration

This example demonstrates how to use the enhanced liquidity analyzer
that integrates with the advanced risk management system.

Features Demonstrated:
- Risk-adjusted zone strength scoring
- Position sizing based on Kelly criterion
- Stop-loss and take-profit calculations
- Comprehensive risk reporting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Import our enhanced modules
from liquidity_analyzer import detect_zones, Zone, RISK_MANAGEMENT_AVAILABLE
from risk_management import RiskManager, RiskLimits, RiskMetrics

def create_sample_data(symbol: str = "AAPL", period: str = "6mo") -> pd.DataFrame:
    """Create sample market data for demonstration."""
    try:
        # Try to fetch real data
        data = yf.download(symbol, period=period, interval="1d")
        if not data.empty:
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            return data
    except:
        pass

    # Fallback: generate synthetic data
    print("⚠️  Using synthetic data for demonstration")
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    np.random.seed(42)

    # Generate realistic price series
    base_price = 150.0
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    # Add OHLC and volume
    high_mult = 1 + np.abs(np.random.normal(0, 0.01, len(dates)))
    low_mult = 1 - np.abs(np.random.normal(0, 0.01, len(dates)))
    volume = np.random.randint(1000000, 10000000, len(dates))

    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'high': prices * high_mult,
        'low': prices * low_mult,
        'close': prices,
        'volume': volume
    }, index=dates)

    return df

def demonstrate_risk_liquidity_integration():
    """Demonstrate the integration of liquidity analysis with risk management."""

    print("🚀 Liquidity Analysis with Risk Management Integration")
    print("=" * 60)

    # 1. Create sample market data
    print("\n📊 Step 1: Loading Market Data...")
    data = create_sample_data()
    print(f"✅ Loaded {len(data)} days of data")
    print(".2f")

    # 2. Initialize Risk Manager
    print("\n🛡️  Step 2: Initializing Risk Management...")
    risk_limits = RiskLimits(
        max_portfolio_volatility=0.25,  # 25% max volatility
        max_single_position=0.15,       # 15% max single position
        max_drawdown_limit=0.20,        # 20% max drawdown
        var_limit_95=0.12,             # 12% VaR limit at 95%
    )

    risk_manager = RiskManager(risk_limits)
    print("✅ Risk manager initialized with conservative limits")

    # 3. Detect zones with risk management enhancement
    print("\n🎯 Step 3: Detecting Supply/Demand Zones with Risk Analysis...")

    if RISK_MANAGEMENT_AVAILABLE:
        zones = detect_zones(
            data,
            lookback=20,
            impulse_factor=1.5,
            risk_manager=risk_manager
        )
        print(f"✅ Detected {len(zones)} zones with risk management enhancement")
    else:
        zones = detect_zones(data, lookback=20, impulse_factor=1.5)
        print(f"✅ Detected {len(zones)} basic zones (risk management not available)")

    # 4. Analyze and display results
    print("\n📈 Step 4: Risk-Enhanced Zone Analysis")
    print("-" * 50)

    if zones:
        # Separate demand and supply zones
        demand_zones = [z for z in zones if z.kind == 'DEMAND']
        supply_zones = [z for z in zones if z.kind == 'SUPPLY']

        print(f"📈 Demand Zones: {len(demand_zones)}")
        print(f"📉 Supply Zones: {len(supply_zones)}")

        # Display top zones by risk-adjusted strength
        if RISK_MANAGEMENT_AVAILABLE:
            print("
🎯 Top Zones by Risk-Adjusted Strength:"            print("-" * 80)

            # Sort zones by risk-adjusted strength
            sorted_zones = sorted(zones, key=lambda z: z.risk_adjusted_strength, reverse=True)

            for i, zone in enumerate(sorted_zones[:5], 1):  # Show top 5
                print(f"{i}. {zone.kind} Zone")
                print(".3f")
                print(".2%")
                print(".2f")
                if zone.stop_loss_price:
                    print(".2f")
                    print(".2f")
                    print(".1f")
                print()

        # Create sample portfolio positions based on zones
        if RISK_MANAGEMENT_AVAILABLE:
            print("\n💼 Step 5: Portfolio Risk Analysis")
            print("-" * 40)

            # Create sample positions from the strongest zones
            sample_positions = []
            for zone in sorted_zones[:3]:  # Use top 3 zones
                current_price = data['close'].iloc[-1]
                position_value = 10000 * zone.position_size_recommendation  # $10K base

                sample_positions.append({
                    'zone': zone,
                    'symbol': f"{zone.kind}_{zone.origin_idx}",
                    'position_value': position_value,
                    'current_price': current_price
                })

            # Create portfolio returns (simplified)
            portfolio_returns = data['close'].pct_change().fillna(0)
            risk_manager.update_portfolio_history(portfolio_returns)

            # Calculate comprehensive risk metrics
            metrics = risk_manager.calculate_portfolio_risk_metrics()

            print("📊 Portfolio Risk Metrics:")
            print(".2%")
            print(".2%")
            print(".2%")
            print(".2%")
            print(".2%")
            print(".2f")

            # Generate risk report
            print("
📋 Detailed Risk Report:"            print("-" * 30)
            risk_report = risk_manager.generate_risk_report(metrics, {}, [])
            print(risk_report)

    print("\n🎉 Analysis Complete!")
    print("\n💡 Key Benefits of Risk-Enhanced Liquidity Analysis:")
    print("   • Risk-adjusted zone strength scoring")
    print("   • Kelly criterion position sizing")
    print("   • Automated stop-loss and take-profit levels")
    print("   • Risk-reward ratio optimization")
    print("   • Portfolio-level risk management")

    return zones, data

def create_risk_comparison_demo():
    """Demonstrate the difference between basic and risk-enhanced analysis."""

    print("\n🔄 Risk Enhancement Comparison Demo")
    print("=" * 50)

    data = create_sample_data()

    # Basic analysis (without risk management)
    basic_zones = detect_zones(data, lookback=20, impulse_factor=1.5)

    # Risk-enhanced analysis
    risk_limits = RiskLimits()
    risk_manager = RiskManager(risk_limits)

    if RISK_MANAGEMENT_AVAILABLE:
        enhanced_zones = detect_zones(
            data,
            lookback=20,
            impulse_factor=1.5,
            risk_manager=risk_manager
        )

        print("📊 Analysis Comparison:")
        print(f"Basic Zones: {len(basic_zones)}")
        print(f"Risk-Enhanced Zones: {len(enhanced_zones)}")

        print("
✨ Risk Enhancement Features:"        print("   • Volatility-adjusted strength scoring")
        print("   • Kelly criterion position sizing")
        print("   • Stop-loss optimization")
        print("   • Risk-reward analysis")
        print("   • Expected return calculations")

        # Show sample enhanced zone
        if enhanced_zones:
            top_zone = max(enhanced_zones, key=lambda z: z.risk_adjusted_strength)
            print("
🎯 Sample Enhanced Zone:"            print(f"   Type: {top_zone.kind}")
            print(".3f")
            print(".2%")
            print(".2f")
            if top_zone.stop_loss_price:
                print(".2f")
                print(".1f")
    else:
        print("⚠️  Risk management module not available")
        print("   Run: pip install -r requirements.txt")

if __name__ == "__main__":
    # Run the main demonstration
    zones, data = demonstrate_risk_liquidity_integration()

    # Run comparison demo
    create_risk_comparison_demo()

    print("\n🔗 Integration Complete!")
    print("Your liquidity analyzer now includes advanced risk management features!")

"""
Liquidity Analyzer with Risk Management Integration
---------------------------------------------------
A Python module for supply/demand zone detection, market structure (BOS/CHOCH),
latent liquidity estimation, and fractality (Hurst exponent), with multi-timeframe support.

Enhanced with Advanced Risk Management:
- Portfolio-level risk metrics (VaR, CVaR, Sharpe, Sortino)
- Position sizing based on liquidity analysis
- Risk-adjusted zone strength scoring
- Stress testing for liquidity events

Dependencies: numpy, pandas, matplotlib, scipy, scikit-learn
Optional: ta-lib, seaborn, plotly

Author: Angel
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import math
import warnings
warnings.filterwarnings('ignore')

# Import risk management components
try:
    from risk_management import RiskManager, RiskLimits, RiskMetrics, PositionRisk
    RISK_MANAGEMENT_AVAILABLE = True
except ImportError:
    RISK_MANAGEMENT_AVAILABLE = False
    print("⚠️  Risk management module not available - using basic functionality")

# -------------------------
# Utility & Validation
# -------------------------

def _require_cols(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLCV DataFrame to a higher timeframe.
    df must have columns: ['open','high','low','close','volume'] and a DateTimeIndex.
    """
    _require_cols(df, ['open','high','low','close','volume'])
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DateTimeIndex for resampling.")
    o = df['open'].resample(rule).first()
    h = df['high'].resample(rule).max()
    l = df['low'].resample(rule).min()
    c = df['close'].resample(rule).last()
    v = df['volume'].resample(rule).sum()
    out = pd.concat([o,h,l,c,v], axis=1)
    out.columns = ['open','high','low','close','volume']
    out.dropna(inplace=True)
    return out


# -------------------------
# Swing / Fractal Detection
# -------------------------

def find_swings(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    """
    Detect swing highs/lows using simple fractal logic:
    - Swing high: high greater than previous 'left' highs and following 'right' highs
    - Swing low: low lower than previous 'left' lows and following 'right' lows

    Returns df with boolean columns 'swing_high' and 'swing_low'.
    """
    _require_cols(df, ['high', 'low'])
    highs = df['high']
    lows = df['low']

    swing_high = pd.Series(False, index=df.index)
    swing_low  = pd.Series(False, index=df.index)

    for i in range(left, len(df) - right):
        if highs.iloc[i] == highs.iloc[i-left:i+right+1].max() and \
           highs.iloc[i] > highs.iloc[i-left:i].max() and \
           highs.iloc[i] > highs.iloc[i+1:i+right+1].max():
            swing_high.iloc[i] = True
        if lows.iloc[i] == lows.iloc[i-left:i+right+1].min() and \
           lows.iloc[i] < lows.iloc[i-left:i].min() and \
           lows.iloc[i] < lows.iloc[i+1:i+right+1].min():
            swing_low.iloc[i] = True

    out = df.copy()
    out['swing_high'] = swing_high
    out['swing_low'] = swing_low
    return out


# -------------------------
# Market Structure & BOS/CHOCH
# -------------------------

@dataclass
class StructureEvent:
    timestamp: pd.Timestamp
    kind: str          # 'BOS_UP', 'BOS_DOWN', 'CHOCH_UP', 'CHOCH_DOWN'
    level: float
    ref_idx: int       # index position of the event
    comment: str = ""


def label_structure(df: pd.DataFrame,
                    left: int = 2,
                    right: int = 2,
                    close_col: str = 'close') -> Tuple[pd.DataFrame, List[StructureEvent]]:
    """
    Label market structure based on swing points and detect BOS/CHOCH.
    Logic:
      - Track last confirmed swing high and swing low.
      - BOS_UP when closing price breaks above last swing high.
      - BOS_DOWN when closing price breaks below last swing low.
      - CHOCH when the BOS direction flips relative to previous trend.
    """
    _require_cols(df, ['high','low', close_col])
    work = find_swings(df, left, right)
    swings = []
    for i, row in enumerate(work.itertuples(index=True)):
        if getattr(row, 'swing_high'):
            swings.append(('H', i, row.Index, row.high))
        if getattr(row, 'swing_low'):
            swings.append(('L', i, row.Index, row.low))

    last_high = None
    last_low = None
    events: List[StructureEvent] = []
    trend = None  # 'UP' or 'DOWN'

    for i in range(len(work)):
        price = work.iloc[i][close_col]

        # Update last swings
        if work.iloc[i]['swing_high']:
            last_high = (i, work.index[i], work.iloc[i]['high'])
        if work.iloc[i]['swing_low']:
            last_low = (i, work.index[i], work.iloc[i]['low'])

        # Detect breaks
        if last_high and price > last_high[2]:
            kind = 'BOS_UP' if trend in (None, 'UP') else 'CHOCH_UP'
            events.append(StructureEvent(work.index[i], kind, last_high[2], i))
            trend = 'UP'
            last_low = last_low  # unchanged
        elif last_low and price < last_low[2]:
            kind = 'BOS_DOWN' if trend in (None, 'DOWN') else 'CHOCH_DOWN'
            events.append(StructureEvent(work.index[i], kind, last_low[2], i))
            trend = 'DOWN'
            last_high = last_high  # unchanged

    work['trend'] = pd.Series(index=work.index, dtype='object')
    for e in events:
        work.loc[e.timestamp:, 'trend'] = 'UP' if 'UP' in e.kind else 'DOWN'
    work['trend'].ffill(inplace=True)
    return work, events


# -------------------------
# Supply & Demand Zones
# -------------------------

@dataclass
class Zone:
    kind: str            # 'SUPPLY' or 'DEMAND'
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    price_min: float
    price_max: float
    strength: float      # heuristic score
    origin_idx: int      # index where impulse began
    # Risk management enhancements
    risk_adjusted_strength: float = 0.0  # Risk-adjusted score
    position_size_recommendation: float = 0.0  # Recommended position size
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    risk_reward_ratio: float = 0.0
    expected_return: float = 0.0
    max_drawdown_risk: float = 0.0


def detect_zones(df: pd.DataFrame,
                 lookback: int = 20,
                 impulse_factor: float = 1.5,
                 risk_manager: Optional['RiskManager'] = None) -> List[Zone]:
    """
    Detect supply/demand zones using impulse logic with risk management integration.

    Enhanced Features:
    - Risk-adjusted zone strength scoring
    - Position sizing recommendations based on Kelly criterion
    - Stop-loss and take-profit calculations
    - Risk-reward ratio analysis
    - Volatility-adjusted scoring

    Parameters:
      lookback: rolling window for average true range proxy.
      impulse_factor: multiplier above avg range to qualify as impulse.
      risk_manager: Optional RiskManager instance for enhanced analysis.
    """
    _require_cols(df, ['open','high','low','close'])
    high = df['high']
    low = df['low']
    close = df['close']
    rng = (high - low).rolling(lookback).mean()

    # Calculate volatility for risk adjustments
    returns = close.pct_change().dropna()
    volatility = returns.rolling(lookback).std() * np.sqrt(252)  # Annualized

    zones: List[Zone] = []

    for i in range(lookback, len(df)-1):
        current_volatility = volatility.iloc[i] if i < len(volatility) else volatility.iloc[-1]

        # Impulse up (Demand zone)
        if (close.iloc[i] - close.iloc[i-1]) > impulse_factor * (rng.iloc[i] or 1e-9):
            base_idx = max(0, i-3)
            base_high = df['high'].iloc[base_idx:i].max()
            base_low = df['low'].iloc[base_idx:i].min()

            # Create basic zone
            zone = Zone('DEMAND', df.index[base_idx], df.index[i], base_low, base_high,
                       strength=1.0, origin_idx=i)

            # Apply risk management enhancements if available
            if RISK_MANAGEMENT_AVAILABLE and risk_manager:
                zone = _enhance_zone_with_risk_management(zone, df, i, current_volatility, risk_manager)

            zones.append(zone)

        # Impulse down (Supply zone)
        if (close.iloc[i-1] - close.iloc[i]) > impulse_factor * (rng.iloc[i] or 1e-9):
            base_idx = max(0, i-3)
            base_high = df['high'].iloc[base_idx:i].max()
            base_low = df['low'].iloc[base_idx:i].min()

            # Create basic zone
            zone = Zone('SUPPLY', df.index[base_idx], df.index[i], base_low, base_high,
                       strength=1.0, origin_idx=i)

            # Apply risk management enhancements if available
            if RISK_MANAGEMENT_AVAILABLE and risk_manager:
                zone = _enhance_zone_with_risk_management(zone, df, i, current_volatility, risk_manager)

            zones.append(zone)

    # Deduplicate overlapping zones by kind and proximity
    def _merge(zs: List[Zone]) -> List[Zone]:
        zs = sorted(zs, key=lambda z: (z.kind, z.start_time))
        merged: List[Zone] = []
        for z in zs:
            if not merged:
                merged.append(z)
                continue
            last = merged[-1]
            if z.kind == last.kind and not (z.price_min > last.price_max or z.price_max < last.price_min):
                # overlap → merge bounding box and extend time
                new = Zone(
                    kind=z.kind,
                    start_time=min(last.start_time, z.start_time),
                    end_time=max(last.end_time, z.end_time),
                    price_min=min(last.price_min, z.price_min),
                    price_max=max(last.price_max, z.price_max),
                    strength=max(last.strength, z.strength) + 0.2,
                    origin_idx=min(last.origin_idx, z.origin_idx),
                )
                merged[-1] = new
            else:
                merged.append(z)
        return merged

    return _merge_zones_risk_aware(zones)


def _enhance_zone_with_risk_management(zone: Zone, df: pd.DataFrame, idx: int,
                                     volatility: float, risk_manager: 'RiskManager') -> Zone:
    """
    Enhance zone with comprehensive risk management calculations.

    Features Added:
    - Risk-adjusted strength scoring based on volatility
    - Kelly criterion position sizing
    - Stop-loss and take-profit levels
    - Risk-reward ratio analysis
    - Expected return calculations
    """
    try:
        current_price = df['close'].iloc[idx]
        zone_center = (zone.price_min + zone.price_max) / 2
        zone_width = zone.price_max - zone.price_min

        # Calculate risk-adjusted strength based on multiple factors
        base_strength = zone.strength

        # 1. Volatility adjustment (lower volatility = higher strength)
        vol_adjustment = max(0.1, 1.0 - volatility / 0.5)  # Normalize volatility impact

        # 2. Zone width adjustment (narrower zones = higher strength)
        avg_range = (df['high'] - df['low']).rolling(20).mean().iloc[idx]
        width_ratio = zone_width / avg_range if avg_range > 0 else 1.0
        width_adjustment = max(0.1, 1.0 - width_ratio)

        # 3. Recency adjustment (more recent zones = higher strength)
        time_decay = max(0.1, 1.0 - (len(df) - idx) / len(df))

        # 4. Volume confirmation (if available)
        volume_strength = 1.0
        if 'volume' in df.columns:
            recent_volume = df['volume'].iloc[idx-5:idx].mean()
            avg_volume = df['volume'].rolling(20).mean().iloc[idx]
            if avg_volume > 0:
                volume_strength = min(2.0, recent_volume / avg_volume)

        # Calculate final risk-adjusted strength
        zone.risk_adjusted_strength = (base_strength * vol_adjustment *
                                     width_adjustment * time_decay * volume_strength)

        # Position sizing using Kelly criterion
        if hasattr(risk_manager, 'risk_limits'):
            max_risk_per_trade = risk_manager.risk_limits.max_single_position

            # Use risk-adjusted strength as win probability proxy
            win_probability = min(0.95, max(0.1, zone.risk_adjusted_strength))
            risk_reward_ratio = 2.0  # Conservative 1:2 risk-reward assumption

            # Kelly formula: (win_prob * (R+1) - 1) / R
            kelly_fraction = (win_probability * (risk_reward_ratio + 1) - 1) / risk_reward_ratio
            kelly_fraction = max(0, min(kelly_fraction, max_risk_per_trade))

            zone.position_size_recommendation = kelly_fraction

        # Stop-loss and take-profit calculations
        risk_buffer = zone_width * 0.1  # 10% buffer

        if zone.kind == 'DEMAND':
            # Long position from demand zone
            zone.stop_loss_price = zone.price_min - risk_buffer
            zone.take_profit_price = zone.price_max + zone_width * 2.0  # 2:1 reward target
        else:
            # Short position from supply zone
            zone.stop_loss_price = zone.price_max + risk_buffer
            zone.take_profit_price = zone.price_min - zone_width * 2.0  # 2:1 reward target

        # Risk-reward ratio calculation
        if zone.stop_loss_price and zone.take_profit_price:
            risk = abs(current_price - zone.stop_loss_price)
            reward = abs(zone.take_profit_price - current_price)
            zone.risk_reward_ratio = reward / risk if risk > 0 else 0

            # Expected return calculation
            win_probability = zone.risk_adjusted_strength
            zone.expected_return = win_probability * reward - (1 - win_probability) * risk

            # Maximum drawdown risk estimation
            zone.max_drawdown_risk = risk * zone.position_size_recommendation

        return zone

    except Exception as e:
        print(f"⚠️  Risk management enhancement error: {e}")
        return zone


def _merge_zones_risk_aware(zones: List[Zone]) -> List[Zone]:
    """
    Merge overlapping zones with risk management consideration.

    Prioritizes zones with higher risk-adjusted strength and combines
    risk metrics appropriately.
    """
    if not zones:
        return zones

    zones = sorted(zones, key=lambda z: (z.kind, z.start_time))
    merged: List[Zone] = []

    for zone in zones:
        if not merged:
            merged.append(zone)
            continue

        last = merged[-1]

        # Check for significant overlap
        if (zone.kind == last.kind and
            zone.start_time <= last.end_time and
            abs((zone.price_min + zone.price_max)/2 -
                (last.price_min + last.price_max)/2) /
            ((zone.price_max - zone.price_min) or 1e-9) < 0.2):

            # Merge zones - keep the stronger one based on risk-adjusted strength
            if zone.risk_adjusted_strength > last.risk_adjusted_strength:
                # Replace with stronger zone
                merged[-1] = zone
            else:
                # Update existing zone with combined risk metrics
                last.position_size_recommendation = max(
                    last.position_size_recommendation,
                    zone.position_size_recommendation
                )
                last.expected_return = max(last.expected_return, zone.expected_return)
                last.risk_adjusted_strength = max(
                    last.risk_adjusted_strength,
                    zone.risk_adjusted_strength
                )
        else:
            merged.append(zone)

    return merged


# -------------------------
# Latent Liquidity (Heuristics without L2)
# -------------------------

@dataclass
class LiquidityPocket:
    side: str            # 'ABOVE' (stops above highs) or 'BELOW' (stops below lows)
    level: float
    width: float
    score: float
    timestamp: pd.Timestamp


def infer_latent_liquidity(df: pd.DataFrame,
                           swing_left: int = 2,
                           swing_right: int = 2,
                           buffer_frac: float = 0.0005,
                           lookback: int = 200) -> List[LiquidityPocket]:
    """
    Approximate latent liquidity pockets where stops cluster:
      - Just above recent swing highs and just below swing lows.
      - Optionally add width proportional to recent volatility.
    """
    work = find_swings(df, left=swing_left, right=swing_right)
    pockets: List[LiquidityPocket] = []

    atr_proxy = (df['high'] - df['low']).rolling(14).mean().fillna(method='bfill')
    buf = (atr_proxy * 0.5).fillna(atr_proxy.median())
    recent = work.iloc[-lookback:] if len(work) > lookback else work

    for i, row in enumerate(recent.itertuples(index=True)):
        ts = row.Index
        if getattr(row, 'swing_high'):
            level = row.high * (1 + buffer_frac)
            pockets.append(LiquidityPocket('ABOVE', float(level), float(buf.loc[ts]), 1.0, ts))
        if getattr(row, 'swing_low'):
            level = row.low * (1 - buffer_frac)
            pockets.append(LiquidityPocket('BELOW', float(level), float(buf.loc[ts]), 1.0, ts))

    return pockets


# -------------------------
# Fractality (Hurst Exponent)
# -------------------------

def hurst_exponent(series: pd.Series, min_window: int = 8, max_window: int = 128) -> float:
    """
    Estimate Hurst exponent using rescaled range (R/S) over multiple windows.
    H in (0,1): H>0.5 trending/persistent, H<0.5 mean-reverting, ~0.5 random.
    """
    s = series.dropna().values.astype(float)
    if len(s) < max_window*2:
        max_window = max(8, len(s)//4)
    if max_window <= min_window:
        max_window = min_window + 2
    window_sizes = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), num=10).astype(int))
    rs = []
    for w in window_sizes:
        if w < 8:
            continue
        n = len(s) // w
        if n < 2:
            continue
        chunks = s[:n*w].reshape(n, w)
        rs_vals = []
        for c in chunks:
            mean = c.mean()
            dev = c - mean
            z = np.cumsum(dev)
            R = z.max() - z.min()
            S = c.std(ddof=1)
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            rs.append((w, np.mean(rs_vals)))
    if len(rs) < 2:
        return 0.5
    x = np.log([w for w, _ in rs])
    y = np.log([v for _, v in rs])
    # Linear regression
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


# -------------------------
# Scoring & Heatmap
# -------------------------

def score_liquidity(df: pd.DataFrame,
                    zones: List[Zone],
                    pockets: List[LiquidityPocket],
                    events: List[StructureEvent]) -> pd.DataFrame:
    """
    Create a per-bar liquidity score combining:
      - Proximity to nearest supply/demand zone
      - Presence of pockets near price
      - Recent BOS/CHOCH events
    Returns df with 'liquidity_score' column (0..100).
    """
    work = df.copy()
    work['liquidity_score'] = 0.0
    prices = work['close']

    # Zone proximity score
    for z in zones:
        center = (z.price_min + z.price_max)/2
        width = (z.price_max - z.price_min) or (prices.std() * 0.5)
        dist = (prices - center).abs() / width
        contrib = np.clip(1.0 - dist, 0, 1) * (12 if z.kind=='SUPPLY' else 10) * z.strength
        work['liquidity_score'] += contrib

    # Pockets proximity
    for p in pockets:
        width = max(p.width, prices.std() * 0.25)
        dist = (prices - p.level).abs() / width
        contrib = np.clip(1.0 - dist, 0, 1) * (8 if p.side=='ABOVE' else 8) * p.score
        work['liquidity_score'] += contrib

    # Recent structure events
    for e in events:
        window = 20
        idx = e.ref_idx
        weight = 6 if 'BOS' in e.kind else 9  # CHOCH is impactful
        if idx < len(work):
            start = max(0, idx - 2)
            end = min(len(work)-1, idx + window)
            work.iloc[start:end+1, work.columns.get_loc('liquidity_score')] += np.linspace(weight, 0, end-start+1)

    # Normalize to 0..100
    if (m := work['liquidity_score'].max()) > 0:
        work['liquidity_score'] = (work['liquidity_score'] / m) * 100.0
    return work


# -------------------------
# Multi-timeframe Confluence
# -------------------------

def project_zones_to_lower_tf(zones_htf: List[Zone], lower_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame mask for projected high timeframe zones on a lower timeframe index.
    Returns a DataFrame with columns for each zone id indicating zone presence (1/0),
    and 'zone_floor'/'zone_ceiling' columns for aggregated min/max across HTF zones.
    """
    mask = pd.DataFrame(index=lower_df.index)
    floors = []
    ceils = []
    for i, z in enumerate(zones_htf):
        col = f"zone_{z.kind}_{i}"
        present = (lower_df.index >= z.start_time) & (lower_df.index <= z.end_time)
        mask[col] = present.astype(int)
        floors.append(np.where(present, z.price_min, np.nan))
        ceils.append(np.where(present, z.price_max, np.nan))
    if floors:
        mask['zone_floor'] = np.nanmin(np.vstack(floors), axis=0)
        mask['zone_ceiling'] = np.nanmax(np.vstack(ceils), axis=0)
        mask[['zone_floor','zone_ceiling']] = mask[['zone_floor','zone_ceiling']].ffill()
    else:
        mask['zone_floor'] = np.nan
        mask['zone_ceiling'] = np.nan
    return mask


# -------------------------
# High-level Pipeline
# -------------------------

@dataclass
class AnalyzerOutput:
    work: pd.DataFrame
    events: List[StructureEvent]
    zones: List[Zone]
    pockets: List[LiquidityPocket]
    hurst: float


def run_liquidity_analyzer(df: pd.DataFrame,
                           swing_left: int = 2,
                           swing_right: int = 2,
                           zone_lookback: int = 20,
                           impulse_factor: float = 1.5) -> AnalyzerOutput:
    """
    Execute the full pipeline on a single timeframe OHLCV DataFrame.
    """
    _require_cols(df, ['open','high','low','close','volume'])
    # Structure
    struct_df, events = label_structure(df, left=swing_left, right=swing_right)

    # Zones
    zones = detect_zones(df, lookback=zone_lookback, impulse_factor=impulse_factor)

    # Liquidity pockets
    pockets = infer_latent_liquidity(df, swing_left, swing_right)

    # Fractality via Hurst
    hurst = hurst_exponent(df['close'].pct_change())

    # Score
    scored = score_liquidity(struct_df, zones, pockets, events)

    return AnalyzerOutput(scored, events, zones, pockets, hurst)


# -------------------------
# Minimal Matplotlib Visualization
# -------------------------

def plot_with_zones(df: pd.DataFrame, zones: List[Zone], title: str = "Liquidity Analyzer"):
    """
    Plot close price with supply/demand zones shaded. Uses matplotlib.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['close'], label='Close')
    for z in zones:
        ax.fill_between(df.index, z.price_min, z.price_max,
                        where=(df.index >= z.start_time) & (df.index <= z.end_time),
                        alpha=0.2, label=z.kind)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    return fig, ax


# -------------------------
# Alias for compatibility
# -------------------------

# Import the main analyzer class for easier access
try:
    from liquidity_market_analyzer import LiquidityMarketAnalyzer as _LiquidityMarketAnalyzer
    LiquidityAnalyzer = _LiquidityMarketAnalyzer  # Alias for backward compatibility
except ImportError:
    # Fallback if liquidity_market_analyzer is not available
    class LiquidityAnalyzer:
        """Fallback LiquidityAnalyzer class when liquidity_market_analyzer is not available."""
        def __init__(self):
            print("⚠️  LiquidityMarketAnalyzer not available, using basic functionality")

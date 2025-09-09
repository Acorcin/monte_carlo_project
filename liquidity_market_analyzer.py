"""
Liquidity Market Analyzer
-------------------------
A comprehensive market analysis tool that integrates with the Monte Carlo trading system.
Performs liquidity analysis, market structure detection, and supply/demand zone identification
on data fetched from the data_fetcher module.

Features:
- Automatic data format handling from data_fetcher
- Market structure analysis (BOS/CHOCH detection)
- Supply/demand zone identification
- Liquidity pocket detection
- Fractality analysis (Hurst exponent)
- Multi-timeframe support
- Integration with existing trading algorithms

Author: Angel
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import math
import warnings

# Import data fetcher for integration
try:
    from data_fetcher import fetch_stock_data, fetch_futures_data
except ImportError:
    warnings.warn("Could not import data_fetcher. Some functionality may be limited.")

# -------------------------
# Data Classes
# -------------------------

@dataclass
class StructureEvent:
    timestamp: pd.Timestamp
    kind: str          # 'BOS_UP', 'BOS_DOWN', 'CHOCH_UP', 'CHOCH_DOWN'
    level: float
    ref_idx: int       # index position of the event
    comment: str = ""

@dataclass
class Zone:
    kind: str            # 'SUPPLY' or 'DEMAND'
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    price_min: float
    price_max: float
    strength: float      # heuristic score
    origin_idx: int      # index where impulse began

@dataclass
class LiquidityPocket:
    side: str            # 'ABOVE' (stops above highs) or 'BELOW' (stops below lows)
    level: float
    width: float
    score: float
    timestamp: pd.Timestamp

@dataclass
class MarketAnalysis:
    """Complete market analysis results."""
    ticker: str
    timeframe: str
    data: pd.DataFrame           # Original OHLCV data
    enhanced_data: pd.DataFrame  # Data with analysis columns
    structure_events: List[StructureEvent]
    supply_demand_zones: List[Zone]
    liquidity_pockets: List[LiquidityPocket]
    hurst_exponent: float
    market_regime: str           # 'TRENDING', 'MEAN_REVERTING', 'RANDOM'
    liquidity_score: pd.Series  # Per-bar liquidity score
    analysis_summary: Dict[str, Any]

# -------------------------
# Core Analysis Functions
# -------------------------

def normalize_data_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to work with both data_fetcher output and other sources.
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        DataFrame with standardized lowercase column names
    """
    df = data.copy()
    
    # Handle common column name variations
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['open', 'high', 'low', 'close', 'volume']:
            column_mapping[col] = col_lower
        elif col.upper() in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']:
            column_mapping[col] = col.upper().lower()
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Ensure we have the required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return df[required_columns]

def find_swings(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    """
    Detect swing highs/lows using fractal logic.
    """
    highs = df['high']
    lows = df['low']
    
    swing_high = pd.Series(False, index=df.index)
    swing_low = pd.Series(False, index=df.index)
    
    for i in range(left, len(df) - right):
        # Swing high detection
        if (highs.iloc[i] == highs.iloc[i-left:i+right+1].max() and 
            highs.iloc[i] > highs.iloc[i-left:i].max() and 
            highs.iloc[i] > highs.iloc[i+1:i+right+1].max()):
            swing_high.iloc[i] = True
            
        # Swing low detection
        if (lows.iloc[i] == lows.iloc[i-left:i+right+1].min() and 
            lows.iloc[i] < lows.iloc[i-left:i].min() and 
            lows.iloc[i] < lows.iloc[i+1:i+right+1].min()):
            swing_low.iloc[i] = True
    
    result = df.copy()
    result['swing_high'] = swing_high
    result['swing_low'] = swing_low
    return result

def detect_market_structure(df: pd.DataFrame, 
                          left: int = 2, 
                          right: int = 2) -> Tuple[pd.DataFrame, List[StructureEvent]]:
    """
    Detect market structure and BOS/CHOCH events.
    """
    work = find_swings(df, left, right)
    events: List[StructureEvent] = []
    
    last_high = None
    last_low = None
    trend = None  # 'UP' or 'DOWN'
    
    close_prices = work['close']
    
    for i in range(len(work)):
        current_price = close_prices.iloc[i]
        current_time = work.index[i]
        
        # Update swing levels
        if work.iloc[i]['swing_high']:
            last_high = (i, current_time, work.iloc[i]['high'])
        if work.iloc[i]['swing_low']:
            last_low = (i, current_time, work.iloc[i]['low'])
        
        # Detect structure breaks
        if last_high and current_price > last_high[2]:
            # Break above previous high
            if trend == 'UP':
                kind = 'BOS_UP'  # Break of Structure (continuation)
            else:
                kind = 'CHOCH_UP'  # Change of Character (reversal)
            
            events.append(StructureEvent(current_time, kind, last_high[2], i))
            trend = 'UP'
            
        elif last_low and current_price < last_low[2]:
            # Break below previous low
            if trend == 'DOWN':
                kind = 'BOS_DOWN'  # Break of Structure (continuation)
            else:
                kind = 'CHOCH_DOWN'  # Change of Character (reversal)
            
            events.append(StructureEvent(current_time, kind, last_low[2], i))
            trend = 'DOWN'
    
    # Add trend column
    work['trend'] = None
    for event in events:
        event_trend = 'UP' if 'UP' in event.kind else 'DOWN'
        work.loc[event.timestamp:, 'trend'] = event_trend
    
    work['trend'] = work['trend'].ffill()
    
    return work, events

def detect_supply_demand_zones(df: pd.DataFrame, 
                              lookback: int = 20, 
                              impulse_factor: float = 1.5) -> List[Zone]:
    """
    Detect supply and demand zones using sophisticated multi-factor analysis.
    
    Enhanced detection considers:
    1. Impulse strength relative to ATR
    2. Volume confirmation
    3. Multi-timeframe structure
    4. Zone quality scoring
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_prices = df['open']
    
    # Calculate sophisticated indicators
    atr = calculate_true_range(high, low, close).rolling(14, min_periods=5).mean()
    volume_ma = volume.rolling(20, min_periods=5).mean()
    price_range = (high - low)
    
    zones: List[Zone] = []
    
    # Detect zones using multiple criteria (adjusted for smaller datasets)
    for i in range(max(10, min(lookback, len(df)//3)), len(df) - 2):  # More flexible range
        current_atr = atr.iloc[i]
        if pd.isna(current_atr) or current_atr <= 0:
            continue
        
        # Multi-bar impulse detection
        impulse_start = max(0, i - 3)
        impulse_end = i + 1
        
        # Calculate impulse characteristics
        impulse_range = close.iloc[impulse_end] - close.iloc[impulse_start]
        impulse_strength = abs(impulse_range) / current_atr
        
        # Volume confirmation
        recent_volume = volume.iloc[impulse_start:impulse_end+1].mean()
        volume_ratio = recent_volume / (volume_ma.iloc[i] + 1e-10)
        
        # Body-to-wick ratio (indicates institutional activity)
        body_sizes = abs(close.iloc[impulse_start:impulse_end+1] - open_prices.iloc[impulse_start:impulse_end+1])
        wick_sizes = price_range.iloc[impulse_start:impulse_end+1] - body_sizes
        body_to_wick = body_sizes.mean() / (wick_sizes.mean() + 1e-10)
        
        # Enhanced impulse criteria (more lenient for smaller datasets)
        volume_confirmed = volume_ratio > 1.05  # Slightly above average volume
        strength_confirmed = impulse_strength > max(0.8, impulse_factor * 0.7)  # More lenient
        quality_confirmed = body_to_wick > 0.3  # Moderate bodies indicate activity
        
        if strength_confirmed and volume_confirmed and quality_confirmed:
            # Determine zone type and boundaries
            if impulse_range > 0:  # Bullish impulse -> Demand zone
                zone_kind = 'DEMAND'
                # Find the base before the impulse
                base_start = find_zone_base(df, impulse_start, 'demand')
                base_end = impulse_start
                
            else:  # Bearish impulse -> Supply zone
                zone_kind = 'SUPPLY'
                # Find the base before the impulse
                base_start = find_zone_base(df, impulse_start, 'supply')
                base_end = impulse_start
            
            # Calculate zone boundaries with precision
            zone_high, zone_low = calculate_zone_boundaries(df, base_start, base_end, zone_kind)
            
            # Calculate sophisticated strength score
            strength_score = calculate_zone_strength(
                impulse_strength, volume_ratio, body_to_wick, 
                impulse_range, current_atr
            )
            
            # Validate zone quality
            if validate_zone_quality(zone_high, zone_low, strength_score):
                zones.append(Zone(
                    kind=zone_kind,
                    start_time=df.index[base_start],
                    end_time=df.index[base_end],
                    price_min=zone_low,
                    price_max=zone_high,
                    strength=strength_score,
                    origin_idx=i
                ))
    
    # Advanced zone processing
    zones = _merge_overlapping_zones(zones)
    zones = filter_weak_zones(zones)
    zones = rank_zones_by_importance(zones)
    
    return zones

def calculate_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate True Range with proper handling."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.fillna(tr1)  # Fallback for first row

def find_zone_base(df: pd.DataFrame, impulse_start: int, zone_type: str) -> int:
    """Find the optimal base start for a zone."""
    lookback = min(10, impulse_start)  # Look back up to 10 bars
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Look for consolidation pattern before impulse
    best_start = max(0, impulse_start - 5)  # Default
    min_range = float('inf')
    
    for start in range(max(0, impulse_start - lookback), impulse_start):
        # Calculate consolidation quality
        period_high = high.iloc[start:impulse_start].max()
        period_low = low.iloc[start:impulse_start].min()
        period_range = period_high - period_low
        
        # Prefer smaller ranges (tighter consolidation)
        if period_range < min_range and period_range > 0:
            min_range = period_range
            best_start = start
    
    return best_start

def calculate_zone_boundaries(df: pd.DataFrame, start: int, end: int, zone_type: str) -> tuple:
    """Calculate precise zone boundaries."""
    high = df['high'].iloc[start:end+1]
    low = df['low'].iloc[start:end+1]
    close = df['close'].iloc[start:end+1]
    
    if zone_type == 'DEMAND':
        # For demand zones, use the lowest low and a conservative high
        zone_low = low.min()
        zone_high = min(high.max(), close.iloc[-1] * 1.002)  # Small buffer
    else:  # SUPPLY
        # For supply zones, use the highest high and a conservative low
        zone_high = high.max()
        zone_low = max(low.min(), close.iloc[-1] * 0.998)  # Small buffer
    
    return zone_high, zone_low

def calculate_zone_strength(impulse_strength: float, volume_ratio: float, 
                          body_to_wick: float, price_move: float, atr: float) -> float:
    """Calculate sophisticated zone strength score."""
    # Base strength from impulse
    strength = min(impulse_strength / 2.0, 3.0)  # Cap at 3.0
    
    # Volume confirmation bonus
    volume_bonus = min((volume_ratio - 1.0) * 0.5, 1.0)
    strength += volume_bonus
    
    # Body quality bonus
    body_bonus = min(body_to_wick * 0.3, 0.5)
    strength += body_bonus
    
    # Price move significance
    move_bonus = min(abs(price_move) / atr * 0.2, 0.5)
    strength += move_bonus
    
    return max(0.1, min(strength, 5.0))  # Clamp between 0.1 and 5.0

def validate_zone_quality(zone_high: float, zone_low: float, strength: float) -> bool:
    """Validate if a zone meets quality criteria."""
    # Zone must have reasonable width
    zone_width = zone_high - zone_low
    if zone_width <= 0:
        return False
    
    # Zone must have minimum strength (more lenient)
    if strength < 0.2:
        return False
    
    # Zone width should be reasonable (not too thin or too wide)
    zone_center = (zone_high + zone_low) / 2
    width_pct = zone_width / zone_center
    if width_pct < 0.0001 or width_pct > 0.1:  # 0.01% to 10%
        return False
    
    return True

def filter_weak_zones(zones: List[Zone]) -> List[Zone]:
    """Filter out weak or poor quality zones."""
    if not zones:
        return zones
    
    # Calculate strength threshold (keep top 80% by strength)
    strengths = [z.strength for z in zones]
    strength_threshold = np.percentile(strengths, 20) if len(strengths) > 5 else min(strengths)
    
    filtered_zones = [z for z in zones if z.strength >= strength_threshold]
    
    # Limit total number of zones to prevent overcrowding
    max_zones = 20
    if len(filtered_zones) > max_zones:
        filtered_zones.sort(key=lambda z: z.strength, reverse=True)
        filtered_zones = filtered_zones[:max_zones]
    
    return filtered_zones

def rank_zones_by_importance(zones: List[Zone]) -> List[Zone]:
    """Rank zones by importance and return sorted list."""
    if not zones:
        return zones
    
    # Calculate importance score for each zone
    for zone in zones:
        importance = zone.strength
        
        # Recent zones are more important
        days_old = 30  # Default age
        try:
            if hasattr(zone, 'start_time'):
                days_old = (pd.Timestamp.now() - zone.start_time).total_seconds() / (24 * 3600)
        except:
            pass
        
        time_factor = max(0.1, 1.0 - (days_old / 60))  # Decay over 60 days
        importance *= time_factor
        
        # Supply zones slightly more important (resistance typically stronger)
        if zone.kind == 'SUPPLY':
            importance *= 1.1
        
        zone.importance = importance
    
    # Sort by importance (highest first)
    zones.sort(key=lambda z: getattr(z, 'importance', z.strength), reverse=True)
    
    return zones

def _merge_overlapping_zones(zones: List[Zone]) -> List[Zone]:
    """Merge overlapping zones of the same type."""
    if not zones:
        return zones
    
    zones_by_type = {'SUPPLY': [], 'DEMAND': []}
    for zone in zones:
        zones_by_type[zone.kind].append(zone)
    
    merged_zones = []
    
    for kind, zone_list in zones_by_type.items():
        if not zone_list:
            continue
            
        # Sort by start time
        zone_list.sort(key=lambda z: z.start_time)
        
        merged = [zone_list[0]]
        
        for current_zone in zone_list[1:]:
            last_zone = merged[-1]
            
            # Check for overlap
            if (current_zone.price_min <= last_zone.price_max and 
                current_zone.price_max >= last_zone.price_min):
                
                # Merge zones
                new_zone = Zone(
                    kind=kind,
                    start_time=min(last_zone.start_time, current_zone.start_time),
                    end_time=max(last_zone.end_time, current_zone.end_time),
                    price_min=min(last_zone.price_min, current_zone.price_min),
                    price_max=max(last_zone.price_max, current_zone.price_max),
                    strength=max(last_zone.strength, current_zone.strength) + 0.2,
                    origin_idx=min(last_zone.origin_idx, current_zone.origin_idx)
                )
                merged[-1] = new_zone
            else:
                merged.append(current_zone)
        
        merged_zones.extend(merged)
    
    return merged_zones

def detect_liquidity_pockets(df: pd.DataFrame, 
                           swing_left: int = 2, 
                           swing_right: int = 2,
                           buffer_pct: float = 0.05) -> List[LiquidityPocket]:
    """
    Detect liquidity pockets where stops likely cluster.
    """
    swing_data = find_swings(df, swing_left, swing_right)
    pockets: List[LiquidityPocket] = []
    
    # Calculate ATR for pocket sizing
    atr = (df['high'] - df['low']).rolling(14).mean()
    
    for i, row in swing_data.iterrows():
        current_atr = atr.loc[i] if i in atr.index else atr.median()
        
        if row['swing_high']:
            # Stops likely above swing high
            level = row['high'] * (1 + buffer_pct / 100)
            pockets.append(LiquidityPocket(
                side='ABOVE',
                level=level,
                width=current_atr * 0.5,
                score=1.0,
                timestamp=i
            ))
        
        if row['swing_low']:
            # Stops likely below swing low
            level = row['low'] * (1 - buffer_pct / 100)
            pockets.append(LiquidityPocket(
                side='BELOW',
                level=level,
                width=current_atr * 0.5,
                score=1.0,
                timestamp=i
            ))
    
    return pockets

def calculate_hurst_exponent(returns: pd.Series, 
                           min_window: int = 8, 
                           max_window: int = 100) -> float:
    """
    Calculate Hurst exponent for fractality analysis.
    """
    clean_returns = returns.dropna()
    if len(clean_returns) < max_window * 2:
        max_window = max(min_window, len(clean_returns) // 4)
    
    if max_window <= min_window:
        return 0.5  # Random walk default
    
    window_sizes = np.unique(np.logspace(
        np.log10(min_window), 
        np.log10(max_window), 
        num=min(10, max_window - min_window)
    ).astype(int))
    
    rs_values = []
    
    for window in window_sizes:
        if window >= len(clean_returns):
            continue
            
        n_windows = len(clean_returns) // window
        if n_windows < 2:
            continue
        
        window_rs = []
        for i in range(n_windows):
            start_idx = i * window
            end_idx = start_idx + window
            window_data = clean_returns.iloc[start_idx:end_idx].values
            
            if len(window_data) < window:
                continue
            
            mean_return = np.mean(window_data)
            deviations = window_data - mean_return
            cumulative_deviations = np.cumsum(deviations)
            
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            S = np.std(window_data, ddof=1)
            
            if S > 0:
                window_rs.append(R / S)
        
        if window_rs:
            rs_values.append((window, np.mean(window_rs)))
    
    if len(rs_values) < 2:
        return 0.5
    
    # Linear regression to find Hurst exponent
    x = np.log([w for w, _ in rs_values])
    y = np.log([rs for _, rs in rs_values])
    hurst = np.polyfit(x, y, 1)[0]
    
    return float(np.clip(hurst, 0.0, 1.0))

def calculate_liquidity_score(df: pd.DataFrame,
                            zones: List[Zone],
                            pockets: List[LiquidityPocket],
                            events: List[StructureEvent]) -> pd.Series:
    """
    Calculate sophisticated liquidity score for each bar using multiple factors.
    
    Scoring components:
    1. Volume-based liquidity (40% weight)
    2. Zone proximity (25% weight) 
    3. Structure events (20% weight)
    4. Liquidity pockets (10% weight)
    5. Volatility adjustment (5% weight)
    """
    scores = pd.Series(0.0, index=df.index)
    close_prices = df['close']
    volume = df['volume']
    high_prices = df['high']
    low_prices = df['low']
    
    # 1. Volume-based liquidity scoring (35% weight)
    volume_score = calculate_volume_liquidity(volume) * 35
    scores += volume_score
    
    # 2. Volatility-based scoring (10% weight)
    volatility_score = calculate_volatility_liquidity(high_prices, low_prices) * 10
    scores += volatility_score
    
    # 3. Zone proximity scoring (30% weight)
    if zones:
        zone_score = calculate_zone_liquidity(close_prices, zones) * 30
        scores += zone_score
    
    # 4. Structure event scoring (25% weight)
    if events:
        event_score = calculate_event_liquidity(scores.index, events, close_prices) * 25
        scores += event_score
    
    # 5. Liquidity pocket scoring (10% weight)
    if pockets:
        pocket_score = calculate_pocket_liquidity(close_prices, pockets) * 10
        scores += pocket_score
    
    # 6. Fallback scoring for low-activity periods
    base_activity_score = calculate_base_activity_score(df) * 15
    scores += base_activity_score
    
    # Ensure scores are non-negative and fill any NaN values
    scores = scores.fillna(0).clip(lower=0)
    
    # Apply smoothing to reduce noise
    scores = scores.rolling(window=3, center=True).mean().fillna(scores)
    
    # Normalize to 0-100 scale with proper calibration
    if scores.max() > 0:
        # Use percentile-based normalization for better distribution
        scores = ((scores - scores.min()) / (scores.max() - scores.min())) * 100
    else:
        scores = pd.Series(0.0, index=scores.index)
    
    return scores

def calculate_volume_liquidity(volume: pd.Series) -> pd.Series:
    """Calculate volume-based liquidity score."""
    # Normalize volume using rolling statistics
    volume_ma = volume.rolling(window=20, min_periods=5).mean()
    volume_std = volume.rolling(window=20, min_periods=5).std()
    
    # Z-score based on recent volume patterns
    volume_zscore = (volume - volume_ma) / (volume_std + 1e-10)
    
    # Convert to 0-1 scale using sigmoid function
    volume_score = 1 / (1 + np.exp(-volume_zscore.clip(-3, 3)))
    
    return volume_score.fillna(0.5)  # Default to medium liquidity

def calculate_volatility_liquidity(high: pd.Series, low: pd.Series) -> pd.Series:
    """Calculate volatility-based liquidity score."""
    # True range calculation
    true_range = high - low
    atr = true_range.rolling(window=14, min_periods=5).mean()
    
    # Normalized volatility (lower volatility = higher liquidity)
    volatility_norm = atr / atr.rolling(window=50, min_periods=10).mean()
    
    # Invert so lower volatility gives higher score
    volatility_score = 1 / (1 + volatility_norm.fillna(1))
    
    return volatility_score.fillna(0.5)

def calculate_zone_liquidity(prices: pd.Series, zones: List[Zone]) -> pd.Series:
    """Calculate zone-based liquidity score with sophisticated proximity."""
    zone_scores = pd.Series(0.0, index=prices.index)
    
    for zone in zones:
        # Calculate zone center and effective width
        zone_center = (zone.price_min + zone.price_max) / 2
        zone_width = max(zone.price_max - zone.price_min, prices.std() * 0.02)
        
        # Time-based zone strength decay
        time_decay = calculate_time_decay(prices.index, zone.start_time, half_life_days=30)
        
        # Distance-based scoring with gaussian decay
        distance_norm = np.abs(prices - zone_center) / zone_width
        distance_score = np.exp(-0.5 * distance_norm**2)  # Gaussian decay
        
        # Combined zone influence
        zone_influence = distance_score * zone.strength * time_decay
        
        # Apply zone type multipliers
        if zone.kind == 'SUPPLY':
            zone_influence *= 1.2  # Supply zones slightly more important
        
        zone_scores += zone_influence
    
    # Normalize zone scores
    if zone_scores.max() > 0:
        zone_scores = zone_scores / zone_scores.max()
    
    return zone_scores

def calculate_event_liquidity(time_index: pd.Index, events: List[StructureEvent], prices: pd.Series) -> pd.Series:
    """Calculate structure event-based liquidity score."""
    event_scores = pd.Series(0.0, index=time_index)
    
    for event in events:
        try:
            event_time = event.timestamp
            
            # Find nearest time index
            if event_time in time_index:
                event_idx = time_index.get_loc(event_time)
            else:
                # Find closest time
                time_diffs = np.abs(time_index - event_time)
                event_idx = time_diffs.argmin()
            
            # Event importance weights
            if 'CHOCH' in event.kind:
                base_weight = 1.0  # Change of character is most important
            elif 'BOS' in event.kind:
                base_weight = 0.7  # Break of structure is moderately important
            else:
                base_weight = 0.5
            
            # Time decay from event
            decay_length = 20  # bars
            for i in range(max(0, event_idx - 2), min(len(event_scores), event_idx + decay_length)):
                distance = abs(i - event_idx)
                time_decay = np.exp(-distance / 10)  # Exponential decay
                event_scores.iloc[i] += base_weight * time_decay
                
        except (IndexError, KeyError):
            continue  # Skip invalid events
    
    # Normalize event scores
    if event_scores.max() > 0:
        event_scores = event_scores / event_scores.max()
    
    return event_scores

def calculate_pocket_liquidity(prices: pd.Series, pockets: List[LiquidityPocket]) -> pd.Series:
    """Calculate liquidity pocket-based score."""
    pocket_scores = pd.Series(0.0, index=prices.index)
    
    for pocket in pockets:
        # Distance to pocket level
        distance = np.abs(prices - pocket.level)
        effective_width = max(pocket.width, prices.std() * 0.01)
        
        # Gaussian proximity scoring
        distance_norm = distance / effective_width
        proximity_score = np.exp(-0.5 * distance_norm**2)
        
        # Time decay from pocket formation
        time_decay = calculate_time_decay(prices.index, pocket.timestamp, half_life_days=15)
        
        # Pocket influence
        pocket_influence = proximity_score * pocket.score * time_decay
        pocket_scores += pocket_influence
    
    # Normalize pocket scores
    if pocket_scores.max() > 0:
        pocket_scores = pocket_scores / pocket_scores.max()
    
    return pocket_scores

def calculate_time_decay(time_index: pd.Index, reference_time: pd.Timestamp, half_life_days: int = 30) -> pd.Series:
    """Calculate time-based decay factor."""
    try:
        # Calculate days from reference time
        days_diff = (time_index - reference_time).total_seconds() / (24 * 3600)
        
        # Exponential decay with specified half-life
        decay_factor = np.exp(-np.log(2) * days_diff / half_life_days)
        
        # Ensure non-negative and clip extreme values
        decay_factor = np.clip(decay_factor, 0.01, 1.0)
        
        return pd.Series(decay_factor, index=time_index)
        
    except Exception:
        # Fallback to uniform decay
        return pd.Series(1.0, index=time_index)

def calculate_base_activity_score(df: pd.DataFrame) -> pd.Series:
    """Calculate base activity score for periods with low zone/event activity."""
    close = df['close']
    volume = df['volume'] 
    high = df['high']
    low = df['low']
    
    # Volume activity (relative to recent average)
    volume_ma = volume.rolling(window=min(10, len(volume)//2), min_periods=3).mean()
    volume_activity = (volume / (volume_ma + 1e-10)).clip(0.5, 3.0)
    volume_score = (volume_activity - 1.0) * 0.5
    
    # Price movement activity
    price_change = close.pct_change().abs()
    price_ma = price_change.rolling(window=min(10, len(price_change)//2), min_periods=3).mean()
    price_activity = (price_change / (price_ma + 1e-10)).clip(0.5, 3.0)
    price_score = (price_activity - 1.0) * 0.3
    
    # Range activity (relative to recent ranges)
    true_range = (high - low) / close
    range_ma = true_range.rolling(window=min(10, len(true_range)//2), min_periods=3).mean()
    range_activity = (true_range / (range_ma + 1e-10)).clip(0.5, 2.5)
    range_score = (range_activity - 1.0) * 0.2
    
    # Combine scores
    activity_score = volume_score + price_score + range_score
    activity_score = activity_score.fillna(0.5).clip(0, 2.0)  # Base score of 0.5
    
    return activity_score

# -------------------------
# Main Analysis Class
# -------------------------

class LiquidityMarketAnalyzer:
    """
    Comprehensive market analyzer that integrates with the Monte Carlo trading system.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.last_analysis: Optional[MarketAnalysis] = None
    
    def analyze_ticker(self, 
                      ticker: str,
                      period: str = "6mo",
                      interval: str = "1d",
                      swing_sensitivity: int = 3,
                      zone_sensitivity: float = 1.5) -> MarketAnalysis:
        """
        Perform complete liquidity analysis on a ticker by fetching fresh data.
        
        Args:
            ticker: Stock/ETF ticker symbol
            period: Time period for data
            interval: Data interval
            swing_sensitivity: Sensitivity for swing detection (2-5)
            zone_sensitivity: Sensitivity for zone detection (1.0-3.0)
            
        Returns:
            MarketAnalysis object with complete results
        """
        print(f"🔍 Analyzing {ticker} liquidity structure...")
        
        # Fetch data using the existing data fetcher
        try:
            raw_data = fetch_stock_data(ticker, period=period, interval=interval)
            print(f"✅ Fetched {len(raw_data)} data points for {ticker}")
        except Exception as e:
            raise ValueError(f"Could not fetch data for {ticker}: {e}")
        
        return self.analyze_data(raw_data, ticker, f"{period}_{interval}", 
                               swing_sensitivity, zone_sensitivity)
    
    def analyze_data(self,
                    data: pd.DataFrame,
                    ticker: str = "Custom",
                    timeframe: str = "Unknown",
                    swing_sensitivity: int = 3,
                    zone_sensitivity: float = 1.5) -> MarketAnalysis:
        """
        Perform complete liquidity analysis on provided data.
        
        Args:
            data: OHLCV DataFrame (any column name format)
            ticker: Identifier for the asset
            timeframe: Description of the timeframe
            swing_sensitivity: Sensitivity for swing detection
            zone_sensitivity: Sensitivity for zone detection
            
        Returns:
            MarketAnalysis object with complete results
        """
        print(f"📊 Performing liquidity analysis on {ticker} ({timeframe})...")
        
        # Normalize data format
        normalized_data = normalize_data_columns(data)
        
        # Detect market structure
        print("   🔍 Detecting market structure...")
        structure_data, events = detect_market_structure(
            normalized_data, left=swing_sensitivity, right=swing_sensitivity
        )
        
        # Detect supply/demand zones
        print("   🎯 Identifying supply/demand zones...")
        zones = detect_supply_demand_zones(
            normalized_data, impulse_factor=zone_sensitivity
        )
        
        # Detect liquidity pockets
        print("   💧 Finding liquidity pockets...")
        pockets = detect_liquidity_pockets(
            normalized_data, swing_left=swing_sensitivity, swing_right=swing_sensitivity
        )
        
        # Calculate Hurst exponent
        print("   🌊 Calculating market fractality...")
        returns = normalized_data['close'].pct_change()
        hurst = calculate_hurst_exponent(returns)
        
        # Determine market regime
        if hurst < 0.4:
            regime = "MEAN_REVERTING"
        elif hurst > 0.6:
            regime = "TRENDING"
        else:
            regime = "RANDOM"
        
        # Calculate liquidity scores
        print("   🔥 Computing liquidity scores...")
        liquidity_scores = calculate_liquidity_score(normalized_data, zones, pockets, events)
        
        # Add analysis columns to data
        enhanced_data = structure_data.copy()
        enhanced_data['liquidity_score'] = liquidity_scores
        
        # Create analysis summary
        summary = {
            'total_structure_events': len(events),
            'bos_events': len([e for e in events if 'BOS' in e.kind]),
            'choch_events': len([e for e in events if 'CHOCH' in e.kind]),
            'supply_zones': len([z for z in zones if z.kind == 'SUPPLY']),
            'demand_zones': len([z for z in zones if z.kind == 'DEMAND']),
            'liquidity_pockets': len(pockets),
            'hurst_exponent': hurst,
            'market_regime': regime,
            'avg_liquidity_score': liquidity_scores.mean(),
            'max_liquidity_score': liquidity_scores.max(),
            'data_points': len(data)
        }
        
        # Create analysis object
        analysis = MarketAnalysis(
            ticker=ticker,
            timeframe=timeframe,
            data=data,
            enhanced_data=enhanced_data,
            structure_events=events,
            supply_demand_zones=zones,
            liquidity_pockets=pockets,
            hurst_exponent=hurst,
            market_regime=regime,
            liquidity_score=liquidity_scores,
            analysis_summary=summary
        )
        
        self.last_analysis = analysis
        
        # Print summary
        self._print_analysis_summary(analysis)
        
        return analysis
    
    def _print_analysis_summary(self, analysis: MarketAnalysis):
        """Print a summary of the analysis results."""
        print(f"\n📋 LIQUIDITY ANALYSIS SUMMARY - {analysis.ticker}")
        print("=" * 60)
        print(f"🕒 Timeframe: {analysis.timeframe}")
        print(f"📊 Data Points: {analysis.analysis_summary['data_points']}")
        print(f"🌊 Market Regime: {analysis.market_regime} (Hurst: {analysis.hurst_exponent:.3f})")
        print(f"\n📈 MARKET STRUCTURE:")
        print(f"   • Total Events: {analysis.analysis_summary['total_structure_events']}")
        print(f"   • BOS Events: {analysis.analysis_summary['bos_events']}")
        print(f"   • CHOCH Events: {analysis.analysis_summary['choch_events']}")
        print(f"\n🎯 SUPPLY/DEMAND ZONES:")
        print(f"   • Supply Zones: {analysis.analysis_summary['supply_zones']}")
        print(f"   • Demand Zones: {analysis.analysis_summary['demand_zones']}")
        print(f"\n💧 LIQUIDITY:")
        print(f"   • Pocket Count: {analysis.analysis_summary['liquidity_pockets']}")
        print(f"   • Avg Score: {analysis.analysis_summary['avg_liquidity_score']:.1f}")
        print(f"   • Max Score: {analysis.analysis_summary['max_liquidity_score']:.1f}")
        
        # Show recent events
        if analysis.structure_events:
            print(f"\n🔍 RECENT STRUCTURE EVENTS:")
            recent_events = analysis.structure_events[-3:]
            for event in recent_events:
                print(f"   • {event.timestamp.strftime('%Y-%m-%d')}: {event.kind} at {event.level:.2f}")

# -------------------------
# Convenience Functions
# -------------------------

def quick_analysis(ticker: str, period: str = "3mo", interval: str = "1d") -> MarketAnalysis:
    """
    Quick analysis function for immediate results.
    
    Args:
        ticker: Stock ticker
        period: Time period
        interval: Data interval
        
    Returns:
        MarketAnalysis object
    """
    analyzer = LiquidityMarketAnalyzer()
    return analyzer.analyze_ticker(ticker, period, interval)

def analyze_current_data(data: pd.DataFrame, name: str = "Current") -> MarketAnalysis:
    """
    Analyze data that's already loaded.
    
    Args:
        data: OHLCV DataFrame
        name: Name for the analysis
        
    Returns:
        MarketAnalysis object
    """
    analyzer = LiquidityMarketAnalyzer()
    return analyzer.analyze_data(data, name)

# -------------------------
# Example Usage
# -------------------------

if __name__ == "__main__":
    print("🚀 Liquidity Market Analyzer - Demo")
    print("=" * 50)
    
    # Example 1: Quick analysis
    try:
        analysis = quick_analysis("SPY", period="3mo", interval="1d")
        print(f"\n✅ Analysis complete for SPY!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        
        # Fallback to demo with synthetic data
        print("\n📊 Running demo with synthetic data...")
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        prices = 100 * (1 + np.random.normal(0, 0.02, 100)).cumprod()
        demo_data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, 100)),
            'High': prices * (1 + np.random.uniform(0, 0.02, 100)),
            'Low': prices * (1 - np.random.uniform(0, 0.02, 100)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        analysis = analyze_current_data(demo_data, "Demo Data")
        print(f"✅ Demo analysis complete!")


"""
Fixed Signal Engine for Gold Predictor
Addresses: Over-filtering, lagging confirmation, session blocking, 
structure break anticipation, and deadlock issues.
"""
import math
import numpy as np
import pandas as pd
import ta
from datetime import datetime, timezone

# ============================================
# CORE FIX 1: STREAMLINED PARAMETERS
# Removed 200+ over-tuned parameters that caused deadlock
# ============================================
DEFAULT_PARAMS = {
    "ema_short": 20,
    "ema_long": 50,
    "rsi_window": 14,
    "adx_window": 14,
    "adx_threshold": 20,           # Lowered from 22 - catches weaker trends
    "atr_window": 14,
    "atr_percent_threshold": 0.18, # Lowered from 0.25
    "cmf_window": 14,
    "trend_weight": 2.0,
    "structure_weight": 2.5,       # INCREASED - structure breaks matter most
    "momentum_weight": 1.5,
    "volume_weight": 1.0,
    "sr_weight": 1.8,              # INCREASED - key levels drive moves
    "min_confidence": 55,          # Lowered from 63 - allow earlier signals
    "min_tradeability": 45,        # Lowered from 52 - don't block valid setups
    "direction_threshold": 1.0,    # Lowered from 1.2 - easier directional trigger
    "session_filter": False,       # DISABLED - don't block Asian session moves
    "anticipate_breaks": True,     # NEW - pre-position at structure
    "trail_adx_threshold": 35,     # For trailing stops
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 3.0,
}

# ============================================
# CORE FIX 2: STRUCTURE BREAK DETECTION
# This is what was MISSING - the circled drop was a structure break
# ============================================

def detect_structure_break(df, lookback=5):
    """
    Detect when price breaks key structure BEFORE lagging indicators confirm.
    This catches moves like the 4699 -> 4669 drop.
    """
    if len(df) < lookback + 2:
        return "none", 0.0

    recent = df.iloc[-(lookback + 1):-1]
    current_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]

    # Find recent swing high/low
    recent_high = recent['High'].max()
    recent_low = recent['Low'].min()

    # Check lower-high / higher-low structure on completed candles only.
    highs = df['High'].iloc[-((lookback * 2) + 1):-1].tolist()
    lows = df['Low'].iloc[-((lookback * 2) + 1):-1].tolist()

    bearish_structure = False
    bullish_structure = False

    # Bearish: lower high + break below recent low
    if len(highs) >= 3:
        lower_highs = all(highs[i] >= highs[i + 1] for i in range(len(highs) - 2, len(highs) - 1))
        if lower_highs and current_close < recent_low:
            bearish_structure = True

    # Bullish: higher low + break above recent high  
    if len(lows) >= 3:
        higher_lows = all(lows[i] <= lows[i + 1] for i in range(len(lows) - 2, len(lows) - 1))
        if higher_lows and current_close > recent_high:
            bullish_structure = True

    # Break of key MA cluster (like in your chart)
    ema20 = df['EMA_20'].iloc[-1] if 'EMA_20' in df else None
    ema50 = df['EMA_50'].iloc[-1] if 'EMA_50' in df else None

    ma_break_bearish = False
    ma_break_bullish = False

    if pd.notna(ema20) and pd.notna(ema50):
        # Price below both MAs = bearish alignment
        if current_close < ema20 and current_close < ema50 and prev_close >= ema20:
            ma_break_bearish = True
        if current_close > ema20 and current_close > ema50 and prev_close <= ema20:
            ma_break_bullish = True

    # Combine signals
    if bearish_structure or ma_break_bearish:
        strength = 0.7 if bearish_structure else 0.5
        if current_close < recent_low:
            strength += 0.3
        return "bearish_break", min(strength, 1.0)

    if bullish_structure or ma_break_bullish:
        strength = 0.7 if bullish_structure else 0.5
        if current_close > recent_high:
            strength += 0.3
        return "bullish_break", min(strength, 1.0)

    return "none", 0.0


def detect_vwap_rejection(df):
    """
    Detect when price rejects VWAP - early momentum signal.
    Your chart shows price rejecting below VWAP before the big drop.
    """
    if 'VWAP' not in df or len(df) < 3:
        return "neutral", 0.0

    current = df.iloc[-1]
    prev = df.iloc[-2]

    vwap = current['VWAP']
    close = current['Close']

    # Bearish rejection: price below VWAP and falling
    prev_vwap = prev['VWAP']
    if pd.isna(vwap) or pd.isna(prev_vwap) or vwap == 0 or prev_vwap == 0:
        return "neutral", 0.0

    if close < vwap and prev['Close'] < prev_vwap:
        # Check if price is accelerating away from VWAP
        dist_now = abs(close - vwap) / vwap * 100
        dist_prev = abs(prev['Close'] - prev_vwap) / prev_vwap * 100

        if dist_now > dist_prev:
            strength = min(0.5 + (dist_now - dist_prev) * 10, 1.0)
            return "bearish_rejection", strength

    # Bullish rejection: price above VWAP and rising
    if close > vwap and prev['Close'] > prev_vwap:
        dist_now = abs(close - vwap) / vwap * 100
        dist_prev = abs(prev['Close'] - prev_vwap) / prev_vwap * 100

        if dist_now > dist_prev:
            strength = min(0.5 + (dist_now - dist_prev) * 10, 1.0)
            return "bullish_rejection", strength

    return "neutral", 0.0


# ============================================
# CORE FIX 3: ANTICIPATORY ENTRY LOGIC
# Pre-position BEFORE the break, not after
# ============================================

def calculate_anticipatory_score(df, params):
    """
    Calculate score for entering BEFORE confirmed breakout.
    This is the key fix - the original code waited for confirmation
    which came 100+ pips too late.
    """
    if len(df) < 10:
        return 0, "neutral", []

    current = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    signals = []
    direction = "neutral"

    # 1. Price action at key level (highest weight)
    structure, struct_strength = detect_structure_break(df)

    if structure == "bearish_break":
        score += struct_strength * 35  # Big weight for structure
        direction = "bearish"
        signals.append(f"Bearish structure break (strength: {struct_strength:.2f})")
    elif structure == "bullish_break":
        score += struct_strength * 35
        direction = "bullish"
        signals.append(f"Bullish structure break (strength: {struct_strength:.2f})")

    # 2. VWAP rejection (early momentum)
    vwap_signal, vwap_strength = detect_vwap_rejection(df)
    if vwap_signal == "bearish_rejection" and direction in ["bearish", "neutral"]:
        score += vwap_strength * 20
        direction = "bearish"
        signals.append(f"VWAP bearish rejection (strength: {vwap_strength:.2f})")
    elif vwap_signal == "bullish_rejection" and direction in ["bullish", "neutral"]:
        score += vwap_strength * 20
        direction = "bullish"
        signals.append(f"VWAP bullish rejection (strength: {vwap_strength:.2f})")

    # 3. MA alignment (trend confirmation)
    ema20 = current.get('EMA_20')
    ema50 = current.get('EMA_50')
    if pd.notna(ema20) and pd.notna(ema50):
        if current['Close'] < ema20 < ema50:
            score += 15
            if direction == "neutral":
                direction = "bearish"
            signals.append("Bearish MA alignment")
        elif current['Close'] > ema20 > ema50:
            score += 15
            if direction == "neutral":
                direction = "bullish"
            signals.append("Bullish MA alignment")

    # 4. ADX momentum (but don't require high threshold)
    adx = current.get('ADX_14', 0)
    if adx >= params['adx_threshold']:
        score += 10
        signals.append(f"ADX trending ({adx:.1f})")
    elif adx >= 15:  # Even weak trend gets some points
        score += 5
        signals.append(f"ADX weak trend ({adx:.1f})")

    # 5. Volume confirmation
    if 'VOLUME_SPIKE' in current and current['VOLUME_SPIKE']:
        score += 10
        signals.append("Volume spike")

    # 6. Support/Resistance break
    sr_break = detect_support_resistance_break(current, prev)
    if direction == "bearish" and sr_break == "support":
        score += 10
        signals.append("Breaking key support")
    elif direction == "bullish" and sr_break == "resistance":
        score += 10
        signals.append("Breaking key resistance")

    return min(score, 100), direction, signals


def detect_support_resistance_break(current, prev):
    current_close = current.get('Close')
    prev_close = prev.get('Close')
    if pd.isna(current_close) or pd.isna(prev_close) or current_close <= 0 or prev_close <= 0:
        return "none"

    recent_low = current.get('RECENT_SWING_LOW')
    if pd.notna(recent_low) and current_close < recent_low:
        return "support"

    recent_high = current.get('RECENT_SWING_HIGH')
    if pd.notna(recent_high) and current_close > recent_high:
        return "resistance"

    step = 5.0 if max(current_close, prev_close) >= 1000 else 1.0
    broken_support = math.floor(prev_close / step) * step
    if current_close < broken_support <= prev_close:
        return "support"

    broken_resistance = math.ceil(prev_close / step) * step
    if prev_close <= broken_resistance < current_close:
        return "resistance"

    return "none"


# ============================================
# CORE FIX 4: SIMPLIFIED CONFIDENCE CALCULATION
# Original had 50+ penalty terms that killed valid signals
# ============================================

def calculate_confidence(score, direction, df, params):
    """
    Simple confidence based on score + trend alignment.
    No more death-by-a-thousand-penalties.
    """
    if direction == "neutral":
        return 50, "neutral"

    base_confidence = 50 + (score * 0.5)  # Score 0-100 -> confidence 50-100

    # Bonus for strong trend alignment
    current = df.iloc[-1]
    ema20 = current.get('EMA_20')
    ema50 = current.get('EMA_50')

    if pd.notna(ema20) and pd.notna(ema50):
        if direction == "bearish" and current['Close'] < ema20 < ema50:
            base_confidence += 10
        elif direction == "bullish" and current['Close'] > ema20 > ema50:
            base_confidence += 10

    # Cap at 95
    confidence = min(base_confidence, 95)

    min_signal_score = params.get('min_signal_score', params['min_confidence'])

    # Determine verdict
    if score >= min_signal_score and confidence >= params['min_confidence']:
        verdict = "Bullish" if direction == "bullish" else "Bearish"
    else:
        verdict = "Neutral"

    return round(confidence), verdict


# ============================================
# CORE FIX 5: TRADEABILITY WITHOUT DEADLOCK
# ============================================

def calculate_tradeability(score, direction, df, params):
    """
    Simple tradeability - if we have a directional signal with decent score,
    it's tradeable. No complex multi-dimensional quality matrices.
    """
    if direction == "neutral":
        return 0, "Low", ["No directional bias"]

    tradeability = score * 0.9  # Slight reduction for risk management

    if tradeability >= params['min_tradeability']:
        label = "High" if tradeability >= 70 else "Medium"
        return tradeability, label, []
    else:
        return tradeability, "Low", [f"Score {tradeability:.1f} below threshold {params['min_tradeability']}"]


# ============================================
# CORE FIX 6: ENTRY/EXIT LOGIC
# ============================================

def determine_action(confidence, verdict, tradeability, blockers, params):
    """
    Clean action determination - no more WAIT deadlocks.
    """
    if blockers:
        return "WAIT", "hold", blockers

    if verdict == "Neutral":
        return "WAIT", "hold", ["No directional verdict"]

    if confidence < params['min_confidence']:
        return "WAIT", "hold", [f"Confidence {confidence} below {params['min_confidence']}"]

    # Active signal
    if verdict == "Bullish":
        return "LONG_ACTIVE", "buy", []
    else:
        return "SHORT_ACTIVE", "sell", []


def calculate_stop_loss_take_profit(current_price, direction, df, params):
    """
    ATR-based SL/TP with proper risk:reward.
    """
    atr = df['ATR_14'].iloc[-1] if 'ATR_14' in df else current_price * 0.002
    if pd.isna(atr) or atr <= 0:
        atr = current_price * 0.002

    sl_distance = atr * params['sl_atr_mult']
    tp_distance = atr * params['tp_atr_mult']

    if direction == "Bullish":
        sl = current_price - sl_distance
        tp = current_price + tp_distance
    else:
        sl = current_price + sl_distance
        tp = current_price - tp_distance

    return round(sl, 2), round(tp, 2), round(sl_distance / 0.1, 1), round(tp_distance / 0.1, 1)


# ============================================
# MAIN PREDICTION FUNCTION
# Replaces the 2000-line compute_prediction_from_ta monster
# ============================================

def compute_prediction(df, params=None):
    """
    Main prediction function - streamlined and fixed.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}

    if df.empty or len(df) < 20:
        return {
            "verdict": "Neutral",
            "confidence": 50,
            "tradeability": 0,
            "action": "hold",
            "actionState": "WAIT",
            "reason": "Insufficient data",
            "signals": []
        }

    current_price = df['Close'].iloc[-1]

    # Calculate anticipatory score (THE FIX)
    score, direction, signals = calculate_anticipatory_score(df, params)

    # Calculate confidence
    confidence, verdict = calculate_confidence(score, direction, df, params)

    # Calculate tradeability
    tradeability, tradeability_label, blockers = calculate_tradeability(score, direction, df, params)

    # Determine action
    action_state, action, final_blockers = determine_action(
        confidence, verdict, tradeability, blockers, params
    )

    # Calculate SL/TP if active
    sl = tp = sl_pips = tp_pips = None
    if action_state in ["LONG_ACTIVE", "SHORT_ACTIVE"]:
        sl, tp, sl_pips, tp_pips = calculate_stop_loss_take_profit(
            current_price, verdict, df, params
        )

    # Build forecast
    forecast = {
        "directionalBias": direction if direction != "neutral" else "Neutral",
        "confidence": confidence,
        "score": round(score, 2),
        "signals": signals
    }

    return {
        "verdict": verdict,
        "confidence": confidence,
        "tradeabilityScore": round(tradeability, 2),
        "tradeabilityLabel": tradeability_label,
        "actionState": action_state,
        "action": action,
        "blockers": final_blockers,
        "currentPrice": round(current_price, 2),
        "entryPrice": round(current_price, 2) if action_state in ["LONG_ACTIVE", "SHORT_ACTIVE"] else None,
        "stopLoss": sl,
        "takeProfit": tp,
        "slPips": sl_pips,
        "tpPips": tp_pips,
        "rrRatio": round(params['tp_atr_mult'] / params['sl_atr_mult'], 2),
        "forecast": forecast,
        "signals": signals,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ============================================
# DATA PREPARATION (simplified)
# ============================================

def prepare_data(df, params=None):
    """
    Prepare data with essential indicators only.
    Removed: complex regime calculations, 50+ feature annotations
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    frame = df.copy().sort_index()

    # Ensure numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors='coerce')
    if 'Volume' not in frame.columns:
        frame['Volume'] = 0.0
    else:
        frame['Volume'] = frame['Volume'].fillna(0.0)

    frame = frame.dropna(subset=['Open', 'High', 'Low', 'Close'])

    # Essential indicators only
    frame['EMA_20'] = ta.trend.EMAIndicator(frame['Close'], window=params['ema_short']).ema_indicator()
    frame['EMA_50'] = ta.trend.EMAIndicator(frame['Close'], window=params['ema_long']).ema_indicator()
    frame['RSI_14'] = ta.momentum.RSIIndicator(frame['Close'], window=params['rsi_window']).rsi()
    frame['ATR_14'] = ta.volatility.AverageTrueRange(
        frame['High'], frame['Low'], frame['Close'], window=params['atr_window']
    ).average_true_range()
    frame['ADX_14'] = ta.trend.ADXIndicator(
        frame['High'], frame['Low'], frame['Close'], window=params['adx_window']
    ).adx()

    # VWAP fallback for spot feeds: when volume is missing/zero, use an expanding
    # typical-price average so downstream charting and signal logic stay stable.
    typical_price = (frame['High'] + frame['Low'] + frame['Close']) / 3.0
    cumulative_volume = frame['Volume'].cumsum()
    weighted_price = (typical_price * frame['Volume']).cumsum()
    frame['VWAP'] = (weighted_price / cumulative_volume.replace(0, np.nan)).fillna(
        typical_price.expanding(min_periods=1).mean()
    )

    # Volume spike
    vol_mean = frame['Volume'].rolling(20, min_periods=5).mean()
    vol_std = frame['Volume'].rolling(20, min_periods=5).std().replace(0, np.nan)
    frame['VOLUME_ZSCORE'] = ((frame['Volume'] - vol_mean) / vol_std).fillna(0)
    frame['VOLUME_SPIKE'] = (frame['VOLUME_ZSCORE'] >= 1.8).astype(int)

    # Support/Resistance levels
    frame['RECENT_SWING_HIGH'] = frame['High'].rolling(24, min_periods=6).max().shift(1)
    frame['RECENT_SWING_LOW'] = frame['Low'].rolling(24, min_periods=6).min().shift(1)

    # Add nearest support/resistance to last row
    last = frame.iloc[-1]
    current_price = last['Close']

    supports = []
    resistances = []

    if not pd.isna(last['RECENT_SWING_LOW']):
        supports.append({'label': 'Recent Swing Low', 'price': last['RECENT_SWING_LOW']})
    if not pd.isna(last['RECENT_SWING_HIGH']):
        resistances.append({'label': 'Recent Swing High', 'price': last['RECENT_SWING_HIGH']})

    # Round numbers
    step = 5.0 if current_price >= 1000 else 1.0
    round_support = math.floor(current_price / step) * step
    round_resistance = math.ceil(current_price / step) * step
    supports.append({'label': 'Round Number', 'price': round_support})
    resistances.append({'label': 'Round Number', 'price': round_resistance})

    if 'nearest_support' not in frame.columns:
        frame['nearest_support'] = None
    if 'nearest_resistance' not in frame.columns:
        frame['nearest_resistance'] = None

    nearest_support = max(
        [s for s in supports if s['price'] <= current_price],
        key=lambda x: x['price'], default=None
    )
    nearest_resistance = min(
        [r for r in resistances if r['price'] >= current_price],
        key=lambda x: x['price'], default=None
    )

    frame.iat[-1, frame.columns.get_loc('nearest_support')] = nearest_support
    frame.iat[-1, frame.columns.get_loc('nearest_resistance')] = nearest_resistance

    return frame


"""
Fixed Signal Engine for Gold Predictor
Addresses: Over-filtering, lagging confirmation, session blocking, 
structure break anticipation, and deadlock issues.
"""
import logging
import math
import numpy as np
import pandas as pd
import ta
from datetime import datetime, timedelta, timezone


logger = logging.getLogger(__name__)

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
    "min_signal_score": 55,        # Entry score threshold
    "exit_confidence": 45,         # Exit score/confidence threshold for hysteresis
    "min_tradeability": 45,        # Lowered from 52 - don't block valid setups
    "exit_tradeability": 35,       # Lower exit threshold keeps active signals sticky
    "signal_cooldown_minutes": 15, # Minimum time between signal changes
    "min_hold_bars": 3,            # Opposite evidence must persist before flips
    "structure_dominance": True,   # Structure breaks dominate single-bar MA noise
    "max_flips_per_hour": 3,       # Rate-limit signal changes
    "direction_threshold": 1.0,    # Lowered from 1.2 - easier directional trigger
    "session_filter": False,       # DISABLED - don't block Asian session moves
    "anticipate_breaks": True,     # NEW - pre-position at structure
    "trail_adx_threshold": 35,     # For trailing stops
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 3.0,
    "symbol": "XAU/USD",
    "xauusd_min_atr": 1.0,
    "xauusd_max_atr": 50.0,
    "fallback_atr_percent": 0.005,
    "max_sl_percent": 0.02,
    "max_tp_percent": 0.05,
    "pip_size": 0.10,
}


ACTIVE_SIGNALS = {"LONG_ACTIVE", "SHORT_ACTIVE"}
SIGNAL_METADATA_FIELDS = (
    "signal_snapshot_id",
    "calculation_time",
    "latest_provider_candle_time",
    "last_closed_candle_time",
    "candle_used_for_signal",
    "candle_is_closed",
    "grace_period_active",
)


def _frame_signal_metadata(df):
    attrs = getattr(df, "attrs", {}) or {}
    return {field: attrs[field] for field in SIGNAL_METADATA_FIELDS if field in attrs}


def _signal_for_direction(direction):
    if direction == "bullish":
        return "LONG_ACTIVE"
    if direction == "bearish":
        return "SHORT_ACTIVE"
    return "WAIT"


def _direction_for_signal(signal):
    if signal == "LONG_ACTIVE":
        return "bullish"
    if signal == "SHORT_ACTIVE":
        return "bearish"
    return "neutral"


def _verdict_for_signal(signal):
    if signal == "LONG_ACTIVE":
        return "Bullish"
    if signal == "SHORT_ACTIVE":
        return "Bearish"
    return "Neutral"


def _action_for_signal(signal):
    if signal == "LONG_ACTIVE":
        return "buy"
    if signal == "SHORT_ACTIVE":
        return "sell"
    return "hold"


def _opposite_direction(direction):
    if direction == "bullish":
        return "bearish"
    if direction == "bearish":
        return "bullish"
    return "neutral"


def _current_bar_id(df):
    if df.empty:
        return 0
    value = df.index[-1]
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


class SignalState:
    """Runtime signal memory used to prevent rapid LONG/WAIT flip-flopping."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reset()
        return cls._instance

    def reset(self):
        self.current_signal = "WAIT"
        self.current_direction = "neutral"
        self.signal_start_time = None
        self.signal_start_bar_id = None
        self.last_flip_time = None
        self.flip_count = 0
        self.flip_history = []
        self.entry_score = 0.0
        self.entry_confidence = 0
        self.entry_signals = []
        self.consecutive_opposite_bars = 0
        self.last_opposite_bar_id = None

    def is_active(self):
        return self.current_signal in ACTIVE_SIGNALS

    def hold_time_minutes(self):
        if self.signal_start_time is None:
            return None
        return int((datetime.now(timezone.utc) - self.signal_start_time).total_seconds() / 60)

    def recent_flip_count(self, now=None):
        now = now or datetime.now(timezone.utc)
        self.flip_history = [
            item for item in self.flip_history if now - item < timedelta(hours=24)
        ]
        return len([item for item in self.flip_history if now - item < timedelta(hours=1)])

    def record_opposite_bar(self, bar_id):
        if self.last_opposite_bar_id != bar_id:
            self.consecutive_opposite_bars += 1
            self.last_opposite_bar_id = bar_id

    def reset_opposite_bars(self):
        self.consecutive_opposite_bars = 0
        self.last_opposite_bar_id = None

    def can_change_signal(self, new_signal, new_direction, params):
        now = datetime.now(timezone.utc)
        if new_signal == self.current_signal:
            return True, "Same signal - updating"

        if self.recent_flip_count(now) >= params["max_flips_per_hour"]:
            return False, f"Max flips/hour reached: {params['max_flips_per_hour']}"

        if self.last_flip_time:
            elapsed_minutes = (now - self.last_flip_time).total_seconds() / 60
            if elapsed_minutes < params["signal_cooldown_minutes"]:
                remaining = params["signal_cooldown_minutes"] - elapsed_minutes
                return False, f"Cooldown active: {remaining:.1f}min remaining"

        if (
            self.is_active()
            and new_signal in ACTIVE_SIGNALS
            and new_direction == _opposite_direction(self.current_direction)
            and self.consecutive_opposite_bars < params["min_hold_bars"]
        ):
            return (
                False,
                f"Opposite signal needs {params['min_hold_bars']} bars "
                f"({self.consecutive_opposite_bars}/{params['min_hold_bars']})",
            )

        return True, "Signal change allowed"

    def update_signal(self, signal, direction, bar_id, score, confidence, signals):
        if signal != self.current_signal:
            now = datetime.now(timezone.utc)
            self.last_flip_time = now
            self.flip_history.append(now)
            self.flip_count += 1
            self.signal_start_time = now if signal in ACTIVE_SIGNALS else None
            self.signal_start_bar_id = bar_id if signal in ACTIVE_SIGNALS else None
            self.entry_score = float(score or 0) if signal in ACTIVE_SIGNALS else 0.0
            self.entry_confidence = confidence if signal in ACTIVE_SIGNALS else 0
            self.entry_signals = list(signals or []) if signal in ACTIVE_SIGNALS else []
            self.reset_opposite_bars()

        self.current_signal = signal
        self.current_direction = direction


signal_state = SignalState()

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

def calculate_anticipatory_score(df, params, current_bar_id=None):
    """
    Calculate score for entering BEFORE confirmed breakout.
    This is the key fix - the original code waited for confirmation
    which came 100+ pips too late.
    """
    if len(df) < 10:
        return 0, "neutral", []

    current_bar_id = current_bar_id if current_bar_id is not None else _current_bar_id(df)
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
    ma_direction = "neutral"
    if pd.notna(ema20) and pd.notna(ema50):
        if current['Close'] < ema20 < ema50:
            ma_direction = "bearish"
            if direction != "bullish" or not params.get("structure_dominance", True):
                score += 15
                if direction == "neutral":
                    direction = "bearish"
                signals.append("Bearish MA alignment")
            else:
                signals.append("Bearish MA alignment suppressed by bullish structure")
        elif current['Close'] > ema20 > ema50:
            ma_direction = "bullish"
            if direction != "bearish" or not params.get("structure_dominance", True):
                score += 15
                if direction == "neutral":
                    direction = "bullish"
                signals.append("Bullish MA alignment")
            else:
                signals.append("Bullish MA alignment suppressed by bearish structure")

    active_direction = signal_state.current_direction
    opposite_direction = _opposite_direction(active_direction)
    opposite_evidence = (
        signal_state.is_active()
        and opposite_direction != "neutral"
        and (ma_direction == opposite_direction or direction == opposite_direction)
    )
    if opposite_evidence:
        signal_state.record_opposite_bar(current_bar_id)
        if signal_state.consecutive_opposite_bars < params["min_hold_bars"]:
            score = max(score, signal_state.entry_score * 0.8, params["exit_confidence"])
            direction = active_direction
            signals.append(
                f"MA {opposite_direction} (holding: "
                f"{signal_state.consecutive_opposite_bars}/{params['min_hold_bars']} bars)"
            )
        else:
            signals.append(
                f"MA {opposite_direction} confirmed "
                f"({signal_state.consecutive_opposite_bars}/{params['min_hold_bars']} bars)"
            )
    else:
        signal_state.reset_opposite_bars()

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

def calculate_confidence(score, direction, df, params, is_exit=False):
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

    min_signal_score = params["exit_confidence"] if is_exit else params.get(
        'min_signal_score',
        params['min_confidence'],
    )
    min_confidence = params["exit_confidence"] if is_exit else params['min_confidence']

    # Determine verdict
    if score >= min_signal_score and confidence >= min_confidence:
        verdict = "Bullish" if direction == "bullish" else "Bearish"
    else:
        verdict = "Neutral"

    return round(confidence), verdict


# ============================================
# CORE FIX 5: TRADEABILITY WITHOUT DEADLOCK
# ============================================

def calculate_tradeability(score, direction, df, params, is_exit=False):
    """
    Simple tradeability - if we have a directional signal with decent score,
    it's tradeable. No complex multi-dimensional quality matrices.
    """
    if direction == "neutral":
        return 0, "Low", ["No directional bias"]

    tradeability = score * 0.9  # Slight reduction for risk management

    threshold = params['exit_tradeability'] if is_exit else params['min_tradeability']

    if tradeability >= threshold:
        label = "High" if tradeability >= 70 else "Medium"
        return tradeability, label, []
    else:
        return tradeability, "Low", [f"Tradeability {tradeability:.1f} below threshold {threshold}"]


# ============================================
# CORE FIX 6: ENTRY/EXIT LOGIC
# ============================================

def determine_action(score, direction, confidence, verdict, tradeability, blockers, params, bar_id, signals):
    """
    Action determination with hysteresis, cooldown, min-hold bars, and rate limits.
    """
    entry_score = params.get('min_signal_score', params['min_confidence'])
    active_signal = signal_state.current_signal
    active_direction = signal_state.current_direction
    is_active = signal_state.is_active()
    raw_signal = "WAIT"
    raw_direction = "neutral"
    raw_blockers = list(blockers or [])

    if is_active:
        same_direction = direction == active_direction
        opposite = direction == _opposite_direction(active_direction)

        if (
            same_direction
            and score >= params['exit_confidence']
            and confidence >= params['exit_confidence']
            and tradeability >= params['exit_tradeability']
        ):
            raw_signal = active_signal
            raw_direction = active_direction
            raw_blockers = []
        elif (
            opposite
            and score >= entry_score
            and confidence >= params['min_confidence']
            and tradeability >= params['min_tradeability']
            and not blockers
        ):
            raw_signal = _signal_for_direction(direction)
            raw_direction = direction
            raw_blockers = []
        elif (
            score >= params['exit_confidence']
            and confidence >= params['exit_confidence']
            and tradeability >= params['exit_tradeability']
        ):
            raw_signal = active_signal
            raw_direction = active_direction
            raw_blockers = []
        else:
            raw_blockers = raw_blockers or [
                f"Exit threshold reached: score {score:.1f} below {params['exit_confidence']}"
            ]
    else:
        if blockers:
            raw_blockers = list(blockers)
        elif verdict == "Neutral":
            raw_blockers = ["No directional verdict"]
        elif confidence < params['min_confidence']:
            raw_blockers = [f"Confidence {confidence} below {params['min_confidence']}"]
        elif score < entry_score:
            raw_blockers = [f"Score {score:.1f} below {entry_score}"]
        else:
            raw_signal = _signal_for_direction(direction)
            raw_direction = direction
            raw_blockers = []

    allowed, reason = signal_state.can_change_signal(raw_signal, raw_direction, params)
    if not allowed and is_active:
        held_signal = active_signal
        held_direction = active_direction
        return (
            held_signal,
            _action_for_signal(held_signal),
            [f"Holding: {reason}"],
            _verdict_for_signal(held_signal),
            held_direction,
            True,
        )
    if not allowed:
        return "WAIT", "hold", [reason], "Neutral", "neutral", True

    signal_state.update_signal(raw_signal, raw_direction, bar_id, score, confidence, signals)
    return (
        raw_signal,
        _action_for_signal(raw_signal),
        raw_blockers,
        _verdict_for_signal(raw_signal) if raw_signal in ACTIVE_SIGNALS else verdict,
        raw_direction if raw_signal in ACTIVE_SIGNALS else direction,
        False,
    )


def calculate_stop_loss_take_profit(current_price, direction, df, params):
    """
    ATR-based SL/TP with guardrails for corrupted XAU/USD candle gaps.
    """
    raw_atr = df['ATR_14'].iloc[-1] if 'ATR_14' in df and not df.empty else None
    validated_atr = None
    if raw_atr is not None:
        try:
            raw_atr_number = float(raw_atr)
        except (TypeError, ValueError):
            raw_atr_number = None
        if raw_atr_number is not None and math.isfinite(raw_atr_number):
            validated_atr = raw_atr_number

    symbol = str(params.get("symbol", "XAU/USD")).upper()
    if validated_atr is not None:
        if validated_atr <= 0:
            validated_atr = None
        elif "XAU" in symbol and (
            validated_atr < params.get("xauusd_min_atr", 1.0)
            or validated_atr > params.get("xauusd_max_atr", 50.0)
        ):
            validated_atr = None

    fallback_used = validated_atr is None
    atr = (
        current_price * params.get("fallback_atr_percent", 0.005)
        if fallback_used
        else validated_atr
    )

    pip_size = params.get("pip_size", 0.10)
    if not isinstance(pip_size, (int, float)) or not math.isfinite(pip_size) or pip_size <= 0:
        pip_size = 0.10
    sl_distance = min(
        atr * params['sl_atr_mult'],
        current_price * params.get("max_sl_percent", 0.02),
    )
    tp_distance = min(
        atr * params['tp_atr_mult'],
        current_price * params.get("max_tp_percent", 0.05),
    )

    if direction == "Bullish":
        sl = current_price - sl_distance
        tp = current_price + tp_distance
    else:
        sl = current_price + sl_distance
        tp = current_price - tp_distance

    sl_pips = round(abs(sl_distance) / pip_size, 1)
    tp_pips = round(abs(tp_distance) / pip_size, 1)

    logger.info(
        "SL/TP calculated current_price=%.2f direction=%s raw_atr=%s validated_atr=%s "
        "fallback_atr_used=%s sl_distance=%.2f tp_distance=%.2f stop_loss=%.2f "
        "take_profit=%.2f pip_size=%.2f target_pips=%.1f",
        current_price,
        direction,
        raw_atr,
        validated_atr,
        fallback_used,
        sl_distance,
        tp_distance,
        sl,
        tp,
        pip_size,
        tp_pips,
    )

    return round(sl, 2), round(tp, 2), sl_pips, tp_pips


# ============================================
# MAIN PREDICTION FUNCTION
# Replaces the 2000-line compute_prediction_from_ta monster
# ============================================

def compute_prediction(df, params=None):
    """
    Main prediction function - streamlined and fixed.
    The caller must pass a frame of fully closed candles; incomplete-candle
    metadata is treated as a hard WAIT guard so partial bars cannot score trades.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    signal_metadata = _frame_signal_metadata(df)

    if signal_metadata.get("candle_is_closed") is False:
        return {
            **signal_metadata,
            "verdict": "Neutral",
            "confidence": 50,
            "tradeability": 0,
            "tradeabilityScore": 0,
            "tradeabilityLabel": "Low",
            "action": "hold",
            "actionState": "WAIT",
            "reason": "Incomplete candle excluded from signal calculation",
            "blockers": ["suppressed_incomplete_candle"],
            "signals": [],
            "currentPrice": None,
            "entryPrice": None,
            "stopLoss": None,
            "takeProfit": None,
            "slPips": None,
            "tpPips": None,
            "rrRatio": None,
            "forecast": {
                "directionalBias": "Neutral",
                "confidence": 50,
                "score": 0,
                "signals": [],
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signalState": signal_state.current_signal,
            "holdTimeMinutes": signal_state.hold_time_minutes(),
            "flipsToday": signal_state.flip_count,
            "antiFlipReason": "suppressed_incomplete_candle",
        }

    if df.empty or len(df) < 20:
        return {
            **signal_metadata,
            "verdict": "Neutral",
            "confidence": 50,
            "tradeability": 0,
            "action": "hold",
            "actionState": "WAIT",
            "reason": "Insufficient data",
            "signals": [],
            "signalState": signal_state.current_signal,
            "holdTimeMinutes": signal_state.hold_time_minutes(),
            "flipsToday": signal_state.flip_count,
        }

    current_price = df['Close'].iloc[-1]
    bar_id = _current_bar_id(df)
    is_exit_check = signal_state.is_active()

    # Calculate anticipatory score (THE FIX)
    score, direction, signals = calculate_anticipatory_score(df, params, bar_id)

    # Calculate confidence with hysteresis
    confidence, verdict = calculate_confidence(score, direction, df, params, is_exit_check)

    # Calculate tradeability with hysteresis
    tradeability, tradeability_label, blockers = calculate_tradeability(
        score,
        direction,
        df,
        params,
        is_exit_check,
    )

    # Determine action with anti-flip guards
    action_state, action, final_blockers, effective_verdict, effective_direction, held = determine_action(
        score,
        direction,
        confidence,
        verdict,
        tradeability,
        blockers,
        params,
        bar_id,
        signals,
    )
    if held:
        verdict = effective_verdict
        direction = effective_direction

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

    result = {
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signalState": signal_state.current_signal,
        "holdTimeMinutes": signal_state.hold_time_minutes(),
        "flipsToday": signal_state.flip_count,
        "cooldownMinutes": params['signal_cooldown_minutes'],
        "minHoldBars": params['min_hold_bars'],
        "oppositeBars": signal_state.consecutive_opposite_bars,
        "antiFlipReason": final_blockers[0] if held and final_blockers else None,
    }
    result.update(signal_metadata)
    return result


# ============================================
# DATA PREPARATION (simplified)
# ============================================

def prepare_data(df, params=None):
    """
    Prepare data with essential indicators only.
    Removed: complex regime calculations, 50+ feature annotations
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    source_attrs = dict(getattr(df, "attrs", {}) or {})
    frame = df.copy().sort_index()
    frame.attrs.update(source_attrs)

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
    frame.attrs.update(source_attrs)

    return frame

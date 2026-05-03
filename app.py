
"""Fixed XAUUSD predictor using Twelve Data as the market data source."""

import atexit
import base64
import json
import logging
import math
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory
from twelvedata import TDClient

try:
    from pywebpush import WebPushException, webpush
except ImportError:  # pragma: no cover - handled by configuration route
    WebPushException = Exception
    webpush = None

from signal_engine import DEFAULT_PARAMS, compute_prediction, prepare_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def _read_prediction_refresh_seconds():
    configured_seconds = os.getenv("PREDICTION_REFRESH_SECONDS", "").strip()
    legacy_seconds = os.getenv("PREDICTION_REFRESH_MINUTES", "").strip()
    raw_value = configured_seconds or legacy_seconds or "5"

    try:
        return max(1, int(raw_value))
    except ValueError:
        logger.warning(
            "Invalid prediction refresh interval %r; defaulting to 5 seconds.",
            raw_value,
        )
        return 5


def _read_bar_open_grace_seconds():
    raw_value = os.getenv("SIGNAL_BAR_OPEN_GRACE_SECONDS", "300").strip()
    try:
        return max(0, int(raw_value))
    except ValueError:
        logger.warning(
            "Invalid bar-open grace period %r; defaulting to 300 seconds.",
            raw_value,
        )
        return 300

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


def _first_env_value(*names, default=""):
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def _runtime_path(env_name, filename):
    configured_path = _first_env_value(env_name)
    if not configured_path:
        return os.path.join(BASE_DIR, filename)
    if os.path.isabs(configured_path):
        return configured_path
    return os.path.join(BASE_DIR, configured_path)


def _web_push_subject():
    return _first_env_value(
        "WEB_PUSH_SUBJECT",
        "VAPID_CLAIMS_SUBJECT",
        default="mailto:notifications@xauusd.local",
    )

TD_OUTPUT_TIMEZONE = "UTC"
DEFAULT_SYMBOL = os.getenv("TWELVE_DATA_SYMBOL", "XAU/USD").strip() or "XAU/USD"
DEFAULT_PERIOD = os.getenv("PREDICTOR_PERIOD", "5d").strip() or "5d"
DEFAULT_INTERVAL = os.getenv("PREDICTOR_INTERVAL", "1h").strip() or "1h"
REFRESH_SECONDS = _read_prediction_refresh_seconds()
MAX_PREDICTION_STALENESS = timedelta(seconds=max(REFRESH_SECONDS * 4, 20))
WEB_PUSH_SUBSCRIPTIONS_PATH = _runtime_path(
    "WEB_PUSH_SUBSCRIPTIONS_PATH",
    "runtime_webpush_subscriptions.json",
)
WEB_PUSH_VAPID_PATH = _runtime_path("WEB_PUSH_VAPID_PATH", "runtime_webpush_vapid.json")
WEB_PUSH_VAPID_PEM_PATH = _runtime_path(
    "WEB_PUSH_VAPID_PEM_PATH",
    "runtime_webpush_vapid_private.pem",
)
WEB_PUSH_SIGNAL_SNAPSHOT_PATH = _runtime_path(
    "WEB_PUSH_SIGNAL_SNAPSHOT_PATH",
    "runtime_webpush_signal_snapshot.json",
)
MAX_NOTIFICATION_PAYLOAD_BYTES = 64 * 1024
MAX_PUSH_ENDPOINT_BYTES = 2048
MAX_PUSH_KEY_BYTES = 512
MAX_PUSH_CLIENT_ID_BYTES = 128
MAX_PUSH_SUBSCRIPTIONS = 200
ACTIVE_ACTION_STATES = {"LONG_ACTIVE", "SHORT_ACTIVE"}
OPEN_TRADE_IDENTITY_FIELDS = (
    "entryPrice",
    "stopLoss",
    "takeProfit",
    "slPips",
    "tpPips",
    "rrRatio",
    "createdAt",
)
SIGNAL_SNAPSHOT_METADATA_FIELDS = (
    "signal_snapshot_id",
    "calculation_time",
    "latest_provider_candle_time",
    "last_closed_candle_time",
    "candle_used_for_signal",
    "candle_is_closed",
    "grace_period_active",
)
SIGNAL_PUSH_PAYLOAD_VERSION = "authoritative-signal-v2"
SIGNAL_PUSH_MAX_AGE_SECONDS = 300
MAX_OHLC_RANGE_RATIO = 0.20
BAR_OPEN_GRACE_SECONDS = _read_bar_open_grace_seconds()
SIGNAL_SCORE_THRESHOLD = float(DEFAULT_PARAMS.get("min_tradeability", 45))


def _empty_risk_state(**overrides):
    state = {
        "slHit": False,
        "tpHit": False,
        "exitReason": None,
        "exitPrice": None,
        "exitTime": None,
        "closedSignalKey": None,
        "closedSignalAction": None,
        "closedSignalCandle": None,
        "closedSignalSnapshotId": None,
    }
    state.update(overrides)
    return state


latest_prediction = None
last_update = None
last_price_update = None
error_state = None
last_push_snapshot = None
last_notification_snapshot = None
signal_snapshot_state = {}
active_trade_state = None
risk_state = _empty_risk_state()
bar_state = {}
prediction_refresh_lock = threading.RLock()
push_lock = threading.Lock()
snapshot_state_lock = threading.Lock()
risk_lock = threading.RLock()


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _no_store_json(payload):
    response = jsonify(_json_safe(payload))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def _urlsafe_b64encode(raw_bytes):
    return base64.urlsafe_b64encode(raw_bytes).rstrip(b"=").decode("ascii")


def _urlsafe_b64decode(value):
    normalized = str(value or "").strip()
    if not normalized:
        return b""
    padding = "=" * ((4 - len(normalized) % 4) % 4)
    return base64.urlsafe_b64decode(normalized + padding)


def _read_json_file(path, default):
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except FileNotFoundError:
        return default
    except Exception as exc:
        logger.warning("Failed to read JSON file %s: %s", path, exc)
        return default


def _write_json_file(path, payload, mode=None):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    if mode is not None:
        os.chmod(path, mode)


def _build_vapid_private_key(private_key_b64):
    raw_private_key = _urlsafe_b64decode(private_key_b64)
    if len(raw_private_key) != 32:
        raise ValueError("WEB_PUSH_VAPID_PRIVATE_KEY must decode to 32 bytes")
    private_value = int.from_bytes(raw_private_key, "big")
    return ec.derive_private_key(private_value, ec.SECP256R1())


def _public_key_from_private_key(private_key):
    return _urlsafe_b64encode(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint,
        )
    )


def _write_vapid_pem(private_key_b64):
    private_key = _build_vapid_private_key(private_key_b64)
    pem_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    directory = os.path.dirname(WEB_PUSH_VAPID_PEM_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(WEB_PUSH_VAPID_PEM_PATH, "wb") as file_handle:
        file_handle.write(pem_bytes)
    os.chmod(WEB_PUSH_VAPID_PEM_PATH, 0o600)
    return WEB_PUSH_VAPID_PEM_PATH


def _generate_runtime_vapid_keys():
    private_key = ec.generate_private_key(ec.SECP256R1())
    raw_private_key = private_key.private_numbers().private_value.to_bytes(32, "big")
    payload = {
        "privateKey": _urlsafe_b64encode(raw_private_key),
        "publicKey": _public_key_from_private_key(private_key),
        "subject": _web_push_subject(),
        "source": "runtime",
    }
    _write_json_file(WEB_PUSH_VAPID_PATH, payload, mode=0o600)
    return payload


def _ensure_web_push_keys():
    env_private_key = _first_env_value("WEB_PUSH_VAPID_PRIVATE_KEY", "VAPID_PRIVATE_KEY")
    env_public_key = _first_env_value("WEB_PUSH_VAPID_PUBLIC_KEY", "VAPID_PUBLIC_KEY")
    subject = _web_push_subject()
    if env_private_key:
        private_key = _build_vapid_private_key(env_private_key)
        derived_public_key = _public_key_from_private_key(private_key)
        if env_public_key and env_public_key != derived_public_key:
            logger.warning(
                "Configured VAPID public key does not match the private key; "
                "using the derived public key."
            )
        payload = {
            "privateKey": env_private_key,
            "publicKey": derived_public_key,
            "subject": subject,
            "source": "env",
        }
        _write_vapid_pem(payload["privateKey"])
        return payload

    payload = _read_json_file(WEB_PUSH_VAPID_PATH, default={})
    if not isinstance(payload, dict) or not payload.get("privateKey") or not payload.get("publicKey"):
        payload = _generate_runtime_vapid_keys()
    else:
        payload["subject"] = subject

    _write_vapid_pem(payload["privateKey"])
    return payload


@lru_cache(maxsize=1)
def get_web_push_config():
    if webpush is None:
        return {
            "available": False,
            "workerPath": "/notification-sw.js",
            "warning": "pywebpush is not installed on the server.",
        }

    try:
        payload = _ensure_web_push_keys()
    except Exception as exc:
        logger.warning("Web push keys unavailable: %s", exc)
        return {
            "available": False,
            "workerPath": "/notification-sw.js",
            "warning": str(exc),
        }

    warning = None
    if payload.get("source") != "env":
        warning = (
            "Server-generated push keys are active. Background alerts stay working after you subscribe, "
            "but stable VAPID environment variables are recommended so subscriptions survive redeploys."
        )

    return {
        "available": True,
        "workerPath": "/notification-sw.js",
        "vapidPublicKey": payload["publicKey"],
        "subject": payload.get("subject") or _web_push_subject(),
        "privateKeyPath": WEB_PUSH_VAPID_PEM_PATH,
        "warning": warning,
    }


def _load_push_subscriptions():
    payload = _read_json_file(WEB_PUSH_SUBSCRIPTIONS_PATH, default=[])
    if not isinstance(payload, list):
        return []
    return [subscription for subscription in payload if isinstance(subscription, dict) and subscription.get("endpoint")]


def _save_push_subscriptions(subscriptions):
    _write_json_file(WEB_PUSH_SUBSCRIPTIONS_PATH, subscriptions, mode=0o600)


def _sanitize_push_client_id(client_id):
    client_id = str(client_id or "").strip()
    if not client_id:
        return None
    if len(client_id.encode("utf-8")) > MAX_PUSH_CLIENT_ID_BYTES:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", client_id):
        return None
    return client_id


def _normalize_push_subscription(subscription, client_id=None):
    if not isinstance(subscription, dict):
        return None

    endpoint = str(subscription.get("endpoint", "")).strip()
    keys = subscription.get("keys") or {}
    p256dh = str(keys.get("p256dh", "")).strip()
    auth = str(keys.get("auth", "")).strip()
    if not endpoint or not p256dh or not auth:
        return None
    if len(endpoint.encode("utf-8")) > MAX_PUSH_ENDPOINT_BYTES:
        return None
    if (
        len(p256dh.encode("utf-8")) > MAX_PUSH_KEY_BYTES
        or len(auth.encode("utf-8")) > MAX_PUSH_KEY_BYTES
    ):
        return None

    parsed_endpoint = urlparse(endpoint)
    if parsed_endpoint.scheme != "https" or not parsed_endpoint.netloc:
        return None

    expiration_time = subscription.get("expirationTime")
    if expiration_time is not None and not isinstance(expiration_time, (int, float)):
        expiration_time = None

    normalized = {
        "endpoint": endpoint,
        "expirationTime": expiration_time,
        "keys": {
            "p256dh": p256dh,
            "auth": auth,
        },
    }
    sanitized_client_id = _sanitize_push_client_id(
        client_id or subscription.get("clientId") or subscription.get("client_id")
    )
    if sanitized_client_id:
        normalized["clientId"] = sanitized_client_id
    return normalized


def _upsert_push_subscription(subscription, client_id=None):
    normalized = _normalize_push_subscription(subscription, client_id=client_id)
    if normalized is None:
        raise ValueError("Invalid push subscription payload")

    with push_lock:
        client_id = normalized.get("clientId")
        subscriptions = [
            item
            for item in _load_push_subscriptions()
            if item.get("endpoint") != normalized["endpoint"]
            and (not client_id or item.get("clientId") != client_id)
        ]
        if len(subscriptions) >= MAX_PUSH_SUBSCRIPTIONS:
            raise ValueError("Push subscription limit reached")
        subscriptions.append(normalized)
        _save_push_subscriptions(subscriptions)
        return len(subscriptions)


def _remove_push_subscription(endpoint):
    endpoint = str(endpoint or "").strip()
    if not endpoint:
        return len(_load_push_subscriptions())

    with push_lock:
        subscriptions = [
            item for item in _load_push_subscriptions() if item.get("endpoint") != endpoint
        ]
        _save_push_subscriptions(subscriptions)
        return len(subscriptions)


def _normalize_signal_text(signal):
    return re.sub(r"\s+", " ", re.sub(r"\([^)]*\d[^)]*\)", "", str(signal or ""))).strip()


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _signal_score_threshold():
    return SIGNAL_SCORE_THRESHOLD


def _format_signal_number(value):
    number = _finite_float(value)
    if number is None:
        return "--"
    return str(int(number)) if number.is_integer() else f"{number:.1f}"


def _identity_number(value):
    number = _finite_float(value)
    if number is None:
        return ""
    return f"{number:.4f}"


def _signal_candle_identity(signal):
    return (
        signal.get("candle_used_for_signal")
        or signal.get("candleUsedForSignal")
        or signal.get("last_closed_candle_time")
        or signal.get("lastClosedCandleTime")
        or signal.get("candleTimestamp")
        or signal.get("signalsKey")
        or ""
    )


def _active_signal_identity_key(signal):
    if not isinstance(signal, dict):
        return None
    action_state = signal.get("actionState") or signal.get("action")
    if action_state not in ACTIVE_ACTION_STATES:
        return None

    entry_price = signal.get("entryPrice")
    if entry_price is None:
        entry_price = signal.get("signalPrice")
    stop_loss = signal.get("stopLoss")
    take_profit = signal.get("takeProfit")
    if (
        _finite_float(entry_price) is None
        or _finite_float(stop_loss) is None
        or _finite_float(take_profit) is None
    ):
        return None

    return "|".join(
        [
            str(signal.get("symbol") or DEFAULT_SYMBOL),
            str(action_state),
            str(_signal_candle_identity(signal)),
            _identity_number(entry_price),
            _identity_number(stop_loss),
            _identity_number(take_profit),
        ]
    )


def _closed_signal_exit_reason(signal):
    signal_key = _active_signal_identity_key(signal)
    if not signal_key:
        return None
    with risk_lock:
        closed_signal_key = risk_state.get("closedSignalKey")
        exit_reason = risk_state.get("exitReason")
    if signal_key and closed_signal_key and signal_key == closed_signal_key:
        return f"closed signal already exited: {exit_reason or 'risk exit'}"
    return None


def _prediction_score(prediction):
    forecast = prediction.get("forecast") if isinstance(prediction.get("forecast"), dict) else {}
    return _finite_float(forecast.get("score")) or 0.0


def _valid_signal_risk(action_state, signal_price, stop_loss, take_profit):
    if action_state == "LONG_ACTIVE":
        return stop_loss < signal_price < take_profit
    if action_state == "SHORT_ACTIVE":
        return take_profit < signal_price < stop_loss
    return False


def _direction_for_action_state(action_state):
    if action_state == "LONG_ACTIVE":
        return "bullish"
    if action_state == "SHORT_ACTIVE":
        return "bearish"
    return "neutral"


def _action_for_signal_state(action_state):
    if action_state == "LONG_ACTIVE":
        return "buy"
    if action_state == "SHORT_ACTIVE":
        return "sell"
    return "hold"


def _verdict_for_signal_state(action_state):
    if action_state == "LONG_ACTIVE":
        return "Bullish"
    if action_state == "SHORT_ACTIVE":
        return "Bearish"
    return "Neutral"


def _parse_datetime(value):
    if not value:
        return None
    try:
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _prediction_timestamp(prediction):
    return _parse_datetime(prediction.get("lastUpdate") or prediction.get("timestamp"))


def _prediction_candle_timestamp(prediction):
    chart_data = prediction.get("chartData") if isinstance(prediction.get("chartData"), dict) else {}
    timestamps = chart_data.get("timestamps") if isinstance(chart_data.get("timestamps"), list) else []
    return _parse_datetime(timestamps[-1]) if timestamps else None


def _prediction_symbol(prediction):
    return str(prediction.get("symbol") or DEFAULT_SYMBOL or "XAU/USD")


def _build_authoritative_signal_state(prediction, status=None):
    """Single validated signal contract shared by dashboard rendering and push alerts."""
    raw_blockers = prediction.get("blockers") if isinstance(prediction.get("blockers"), list) else []
    blockers = [str(blocker) for blocker in raw_blockers if str(blocker).strip()]
    score = _prediction_score(prediction)
    threshold = _signal_score_threshold()
    raw_action_state = prediction.get("actionState") or "WAIT"
    raw_action = prediction.get("action") or "hold"
    raw_verdict = prediction.get("verdict") or "Neutral"
    tradeability_label = prediction.get("tradeabilityLabel") or "Low"
    app_status = status or prediction.get("status") or "active"
    signal_price = _finite_float(prediction.get("entryPrice"))
    stop_loss = _finite_float(prediction.get("stopLoss"))
    take_profit = _finite_float(prediction.get("takeProfit"))
    candle_is_closed = prediction.get("candle_is_closed", True) is not False

    suppression_reasons = []
    if raw_action_state in ACTIVE_ACTION_STATES:
        expected_verdict = "Bullish" if raw_action_state == "LONG_ACTIVE" else "Bearish"
        expected_action = "buy" if raw_action_state == "LONG_ACTIVE" else "sell"

        if blockers:
            suppression_reasons.append("blockers present")
        if raw_verdict != expected_verdict:
            suppression_reasons.append(f"verdict {raw_verdict} is not {expected_verdict}")
        if raw_action != expected_action:
            suppression_reasons.append("dashboard would render Waiting for Signal")
        if score < threshold:
            suppression_reasons.append(
                f"signal score {_format_signal_number(score)} below threshold {_format_signal_number(threshold)}"
            )
        if tradeability_label == "Low":
            suppression_reasons.append("tradeability is Low")
        if signal_price is None or signal_price <= 0:
            suppression_reasons.append("signal price is unavailable")
        if stop_loss is None or take_profit is None:
            suppression_reasons.append("risk management is incomplete")
        elif signal_price is not None and not _valid_signal_risk(
            raw_action_state,
            signal_price,
            stop_loss,
            take_profit,
        ):
            suppression_reasons.append("risk management values are invalid")
        if app_status != "active":
            suppression_reasons.append(f"app status is {app_status}")

    active_trade = raw_action_state in ACTIVE_ACTION_STATES and not suppression_reasons
    if active_trade:
        action_state = raw_action_state
        action = raw_action
        verdict = raw_verdict
        canonical_blockers = []
        display_status = "Signal Active"
    else:
        action_state = "WAIT" if raw_action_state in ACTIVE_ACTION_STATES else raw_action_state
        action = "hold" if raw_action_state in ACTIVE_ACTION_STATES else raw_action
        verdict = "Neutral" if raw_action_state in ACTIVE_ACTION_STATES else raw_verdict
        canonical_blockers = list(blockers)
        for reason in suppression_reasons:
            if reason not in canonical_blockers:
                canonical_blockers.append(reason)
        display_status = "Waiting for Signal" if action == "hold" else "Signal Blocked"

    return {
        "rawActionState": raw_action_state,
        "rawAction": raw_action,
        "rawVerdict": raw_verdict,
        "actionState": action_state,
        "action": action,
        "verdict": verdict,
        "tradeabilityLabel": tradeability_label,
        "score": round(score, 2),
        "threshold": threshold,
        "blockers": canonical_blockers,
        "activeTrade": active_trade,
        "displayStatus": display_status,
        "signalPrice": signal_price,
        "stopLoss": stop_loss,
        "takeProfit": take_profit,
        "suppressionReasons": suppression_reasons,
        "dataStatus": app_status,
        "candleIsClosed": candle_is_closed,
        "gracePeriodActive": bool(prediction.get("grace_period_active")),
    }


def _candidate_signal_fields(prediction):
    validation = prediction.get("signalValidation") if isinstance(prediction.get("signalValidation"), dict) else {}
    signals = prediction.get("signals") if isinstance(prediction.get("signals"), list) else []
    blockers = validation.get("blockers") if isinstance(validation.get("blockers"), list) else []
    action_state = validation.get("actionState") or prediction.get("actionState") or "WAIT"
    score = _finite_float(validation.get("score"))
    if score is None:
        score = _prediction_score(prediction)
    confidence = _finite_float(prediction.get("confidence")) or 50.0
    candle_is_closed = validation.get("candleIsClosed")
    if candle_is_closed is None:
        candle_is_closed = prediction.get("candle_is_closed", True) is not False
    grace_period_active = validation.get("gracePeriodActive")
    if grace_period_active is None:
        grace_period_active = bool(prediction.get("grace_period_active"))
    created_at = prediction.get("createdAt")
    if not created_at and action_state in ACTIVE_ACTION_STATES:
        created_at = prediction.get("lastUpdate") or prediction.get("timestamp")

    fields = {
        "symbol": _prediction_symbol(prediction),
        "actionState": action_state,
        "direction": _direction_for_action_state(action_state),
        "action": validation.get("action") or prediction.get("action") or _action_for_signal_state(action_state),
        "verdict": validation.get("verdict") or prediction.get("verdict") or _verdict_for_signal_state(action_state),
        "tradeabilityLabel": validation.get("tradeabilityLabel") or prediction.get("tradeabilityLabel") or "Low",
        "confidence": confidence,
        "score": round(score, 2),
        "threshold": _finite_float(validation.get("threshold")) or _signal_score_threshold(),
        "blockers": list(blockers),
        "activeTrade": bool(validation.get("activeTrade")),
        "displayStatus": validation.get("displayStatus") or ("Signal Active" if action_state in ACTIVE_ACTION_STATES else "Waiting for Signal"),
        "entryPrice": prediction.get("entryPrice"),
        "stopLoss": prediction.get("stopLoss"),
        "takeProfit": prediction.get("takeProfit"),
        "slPips": prediction.get("slPips"),
        "tpPips": prediction.get("tpPips"),
        "rrRatio": prediction.get("rrRatio"),
        "createdAt": created_at,
        "signals": signals,
        "signalsKey": "|".join(_normalize_signal_text(signal) for signal in signals[:4]),
        "timestamp": prediction.get("lastUpdate") or prediction.get("timestamp"),
        "snapshotTimestamp": _prediction_timestamp(prediction),
        "candleTimestamp": _prediction_candle_timestamp(prediction),
        "dataStatus": validation.get("dataStatus") or prediction.get("status") or "active",
        "candleIsClosed": bool(candle_is_closed),
        "gracePeriodActive": bool(grace_period_active),
    }
    for field in SIGNAL_SNAPSHOT_METADATA_FIELDS:
        fields[field] = prediction.get(field)
    return fields


def _snapshot_is_stale(candidate, now):
    snapshot_timestamp = candidate.get("snapshotTimestamp")
    if snapshot_timestamp is None:
        return False
    max_age = timedelta(seconds=SIGNAL_PUSH_MAX_AGE_SECONDS)
    return now - snapshot_timestamp > max_age


def _blocked_wait_fields(candidate, reason):
    blocked = dict(candidate)
    blockers = list(candidate.get("blockers") or [])
    if reason and reason not in blockers:
        blockers.append(reason)
    blocked.update(
        {
            "actionState": "WAIT",
            "direction": "neutral",
            "action": "hold",
            "verdict": "Neutral",
            "tradeabilityLabel": "Low",
            "blockers": blockers,
            "activeTrade": False,
            "displayStatus": "Waiting for Signal",
            "entryPrice": None,
            "stopLoss": None,
            "takeProfit": None,
            "slPips": None,
            "tpPips": None,
            "createdAt": None,
        }
    )
    return blocked


def _active_runtime_signal_fields(candidate):
    with risk_lock:
        active_signal = dict(active_trade_state) if active_trade_state else None
    if not active_signal or active_signal.get("status") != "OPEN":
        return None

    action_state = active_signal.get("action")
    if action_state not in ACTIVE_ACTION_STATES:
        return None

    is_long = action_state == "LONG_ACTIVE"
    preserved = dict(candidate)
    preserved.update(
        {
            "actionState": action_state,
            "direction": "bullish" if is_long else "bearish",
            "action": "buy" if is_long else "sell",
            "verdict": "Bullish" if is_long else "Bearish",
            "tradeabilityLabel": active_signal.get("lastConfirmedTradeability")
            or candidate.get("tradeabilityLabel")
            or "Medium",
            "confidence": _finite_float(active_signal.get("lastConfirmedConfidence"))
            or candidate.get("confidence")
            or 50.0,
            "score": _finite_float(active_signal.get("lastConfirmedScore"))
            or candidate.get("score")
            or 0.0,
            "blockers": [],
            "activeTrade": True,
            "displayStatus": "Signal Active",
            "entryPrice": active_signal.get("entryPrice"),
            "stopLoss": active_signal.get("stopLoss"),
            "takeProfit": active_signal.get("takeProfit"),
            "slPips": active_signal.get("slPips"),
            "tpPips": active_signal.get("tpPips"),
            "rrRatio": active_signal.get("rrRatio"),
            "createdAt": active_signal.get("createdAt"),
        }
    )
    return preserved


def _preserve_runtime_risk_fields(signal):
    with risk_lock:
        active_signal = dict(active_trade_state) if active_trade_state else None
    if (
        not active_signal
        or active_signal.get("status") != "OPEN"
        or active_signal.get("action") != signal.get("actionState")
    ):
        return signal

    preserved = dict(signal)
    preserved.update(
        {
            "entryPrice": active_signal.get("entryPrice"),
            "stopLoss": active_signal.get("stopLoss"),
            "takeProfit": active_signal.get("takeProfit"),
            "slPips": active_signal.get("slPips"),
            "tpPips": active_signal.get("tpPips"),
            "rrRatio": active_signal.get("rrRatio"),
            "createdAt": active_signal.get("createdAt"),
        }
    )
    return preserved


def _compose_validated_signal_prediction(prediction, signal, candidate, notification_allowed, reason):
    validated = dict(prediction)
    forecast = validated.get("forecast") if isinstance(validated.get("forecast"), dict) else {}
    validated["forecast"] = dict(forecast)

    validated["verdict"] = signal["verdict"]
    validated["action"] = signal["action"]
    validated["actionState"] = signal["actionState"]
    validated["tradeabilityLabel"] = signal["tradeabilityLabel"]
    validated["confidence"] = round(signal["confidence"])
    validated["blockers"] = list(signal.get("blockers") or [])
    validated["entryPrice"] = signal.get("entryPrice")
    validated["stopLoss"] = signal.get("stopLoss")
    validated["takeProfit"] = signal.get("takeProfit")
    validated["slPips"] = signal.get("slPips")
    validated["tpPips"] = signal.get("tpPips")
    if signal.get("rrRatio") is not None:
        validated["rrRatio"] = signal.get("rrRatio")
    validated["createdAt"] = signal.get("createdAt")
    validated["signals"] = list(signal.get("signals") or [])
    validated["forecast"]["score"] = signal["score"]
    validated["forecast"]["scoreThreshold"] = signal["threshold"]
    validated["signalScoreThreshold"] = signal["threshold"]
    for field in SIGNAL_SNAPSHOT_METADATA_FIELDS:
        if field in candidate:
            validated[field] = candidate.get(field)

    suppression_reasons = [] if notification_allowed else ([reason] if reason else [])
    validated["signalValidation"] = {
        "validated": True,
        "rawActionState": candidate["actionState"],
        "rawAction": candidate["action"],
        "rawVerdict": candidate["verdict"],
        "actionState": signal["actionState"],
        "action": signal["action"],
        "verdict": signal["verdict"],
        "tradeabilityLabel": signal["tradeabilityLabel"],
        "score": signal["score"],
        "threshold": signal["threshold"],
        "blockers": list(signal.get("blockers") or []),
        "activeTrade": signal["actionState"] in ACTIVE_ACTION_STATES and bool(signal.get("activeTrade")),
        "displayStatus": signal.get("displayStatus") or "Waiting for Signal",
        "signalPrice": signal.get("entryPrice"),
        "stopLoss": signal.get("stopLoss"),
        "takeProfit": signal.get("takeProfit"),
        "suppressionReasons": suppression_reasons,
        "dataStatus": candidate.get("dataStatus"),
        "candleIsClosed": candidate.get("candleIsClosed", True),
        "gracePeriodActive": candidate.get("gracePeriodActive", False),
        "candidateState": candidate["actionState"],
        "candidateSignal": candidate["actionState"],
        "candidateDirection": candidate["direction"],
        "candidateBias": candidate["verdict"],
        "candidateScore": candidate["score"],
        "candidateConfidence": candidate["confidence"],
        "candidateBlockers": list(candidate.get("blockers") or []),
        "candidateCandleIsClosed": candidate.get("candleIsClosed", True),
        "candidateGracePeriodActive": candidate.get("gracePeriodActive", False),
        "transitionAllowed": notification_allowed,
        "notificationAllowed": notification_allowed,
        "suppressionOrWaitReason": reason,
        "snapshotTimestamp": candidate.get("timestamp"),
        "candleTimestamp": candidate.get("candleTimestamp").isoformat() if candidate.get("candleTimestamp") else None,
        "signalSnapshotId": candidate.get("signal_snapshot_id"),
        "calculationTime": candidate.get("calculation_time"),
        "latestProviderCandleTime": candidate.get("latest_provider_candle_time"),
        "lastClosedCandleTime": candidate.get("last_closed_candle_time"),
        "candleUsedForSignal": candidate.get("candle_used_for_signal"),
    }
    return validated


def _log_signal_snapshot_decision(symbol, candidate, signal, notification_allowed, reason):
    logger.info(
        "Signal snapshot symbol=%s raw_candidate=%s resolved_state=%s candidate_direction=%s "
        "resolved_direction=%s candidate_bias=%s resolved_bias=%s candidate_score=%.2f "
        "resolved_score=%.2f candidate_confidence=%.1f resolved_confidence=%.1f blockers=%s "
        "notification_allowed=%s suppression_reason=%s snapshot_timestamp=%s candle_timestamp=%s",
        symbol,
        candidate["actionState"],
        signal["actionState"],
        candidate["direction"],
        signal["direction"],
        candidate["verdict"],
        signal["verdict"],
        candidate["score"],
        signal["score"],
        candidate["confidence"],
        signal["confidence"],
        candidate.get("blockers", []),
        notification_allowed,
        reason,
        candidate.get("timestamp"),
        candidate.get("candleTimestamp").isoformat() if candidate.get("candleTimestamp") else None,
    )


def _resolve_signal_snapshot(prediction, now=None, advance=True):
    now = now or datetime.now(timezone.utc)
    candidate = _candidate_signal_fields(prediction)
    symbol = candidate["symbol"]

    notification_allowed = True
    reason = "accepted latest valid signal snapshot"
    signal = _preserve_runtime_risk_fields(candidate)

    with snapshot_state_lock:
        state = signal_snapshot_state.setdefault(symbol, {})
        snapshot_timestamp = candidate.get("snapshotTimestamp")
        last_snapshot_timestamp = state.get("lastSnapshotTimestamp")

        if candidate.get("dataStatus") != "active":
            reason = f"data status {candidate.get('dataStatus')} blocks active signal"
            signal = _blocked_wait_fields(candidate, reason)
            notification_allowed = False
        elif candidate.get("candleIsClosed") is False:
            preserved_signal = _active_runtime_signal_fields(candidate)
            reason = "suppressed_incomplete_candle"
            if preserved_signal is not None:
                signal = preserved_signal
            else:
                signal = _blocked_wait_fields(candidate, reason)
            notification_allowed = False
        elif snapshot_timestamp and last_snapshot_timestamp and snapshot_timestamp <= last_snapshot_timestamp:
            reason = "duplicate or out-of-order snapshot ignored"
            notification_allowed = False
        elif _snapshot_is_stale(candidate, now):
            reason = "stale snapshot ignored"
            signal = _blocked_wait_fields(candidate, reason)
            notification_allowed = False
        elif candidate.get("gracePeriodActive") and candidate["actionState"] == "WAIT":
            preserved_signal = _active_runtime_signal_fields(candidate)
            if preserved_signal is not None:
                reason = "suppressed_bar_open_instability"
                signal = preserved_signal
                notification_allowed = False
        elif candidate["actionState"] in ACTIVE_ACTION_STATES:
            closed_exit_reason = _closed_signal_exit_reason(candidate)
            if closed_exit_reason:
                reason = closed_exit_reason
                signal = _blocked_wait_fields(candidate, reason)
                notification_allowed = False

        if advance and (snapshot_timestamp is not None) and reason != "duplicate or out-of-order snapshot ignored":
            state["lastSnapshotTimestamp"] = snapshot_timestamp

    _log_signal_snapshot_decision(symbol, candidate, signal, notification_allowed, reason)
    return _compose_validated_signal_prediction(
        prediction,
        signal,
        candidate,
        notification_allowed,
        reason,
    )


def _is_validated_signal_prediction(prediction):
    validation = prediction.get("signalValidation") if isinstance(prediction, dict) else None
    return isinstance(validation, dict) and validation.get("validated") is True


def _ensure_validated_signal_prediction(prediction, status=None, now=None, advance=False):
    if not isinstance(prediction, dict) or prediction.get("error"):
        return prediction
    if _is_validated_signal_prediction(prediction) and status is None:
        return prediction
    validated = _apply_authoritative_signal_state(prediction, status=status)
    return _resolve_signal_snapshot(validated, now=now, advance=advance)


def _apply_authoritative_signal_state(prediction, status=None):
    if not isinstance(prediction, dict) or prediction.get("error"):
        return prediction

    normalized = dict(prediction)
    forecast = normalized.get("forecast") if isinstance(normalized.get("forecast"), dict) else {}
    normalized["forecast"] = dict(forecast)
    state = _build_authoritative_signal_state(normalized, status=status)

    normalized["verdict"] = state["verdict"]
    normalized["action"] = state["action"]
    normalized["actionState"] = state["actionState"]
    normalized["blockers"] = state["blockers"]
    normalized["forecast"]["score"] = state["score"]
    normalized["forecast"]["scoreThreshold"] = state["threshold"]
    normalized["signalScoreThreshold"] = state["threshold"]
    normalized["signalValidation"] = state

    if state["rawActionState"] in ACTIVE_ACTION_STATES and not state["activeTrade"]:
        normalized["entryPrice"] = None
        normalized["stopLoss"] = None
        normalized["takeProfit"] = None
        normalized["slPips"] = None
        normalized["tpPips"] = None

    return normalized


def _build_server_signal_snapshot(prediction):
    if not isinstance(prediction, dict) or prediction.get("error"):
        return None

    normalized = _ensure_validated_signal_prediction(prediction, advance=False)
    validation = normalized.get("signalValidation") or {}
    signals = normalized.get("signals") if isinstance(normalized.get("signals"), list) else []
    blockers = validation.get("blockers") if isinstance(validation.get("blockers"), list) else []

    confidence = _finite_float(normalized.get("confidence")) or 0.0
    score = _finite_float(validation.get("score")) or _prediction_score(normalized)

    return {
        "verdict": validation.get("verdict") or normalized.get("verdict") or "Neutral",
        "action": validation.get("action") or normalized.get("action") or "hold",
        "actionState": validation.get("actionState") or normalized.get("actionState") or "WAIT",
        "tradeabilityLabel": validation.get("tradeabilityLabel") or normalized.get("tradeabilityLabel") or "Low",
        "confidence": confidence,
        "score": score,
        "threshold": _finite_float(validation.get("threshold")) or _signal_score_threshold(),
        "signals": signals[:4],
        "signalsKey": "|".join(_normalize_signal_text(signal) for signal in signals[:4]),
        "hasBlockers": bool(blockers),
        "blockers": blockers,
        "isActionable": bool(validation.get("activeTrade")),
        "notificationAllowed": bool(validation.get("notificationAllowed", True)),
        "suppressionReasons": list(validation.get("suppressionReasons") or []),
        "displayStatus": validation.get("displayStatus") or "Waiting for Signal",
        "dataStatus": validation.get("dataStatus") or "active",
        "signalSnapshotId": validation.get("signalSnapshotId") or normalized.get("signal_snapshot_id"),
        "calculationTime": validation.get("calculationTime") or normalized.get("calculation_time"),
        "latestProviderCandleTime": validation.get("latestProviderCandleTime") or normalized.get("latest_provider_candle_time"),
        "lastClosedCandleTime": validation.get("lastClosedCandleTime") or normalized.get("last_closed_candle_time"),
        "candleUsedForSignal": validation.get("candleUsedForSignal") or normalized.get("candle_used_for_signal"),
        "candleIsClosed": bool(validation.get("candleIsClosed", normalized.get("candle_is_closed", True))),
        "gracePeriodActive": bool(validation.get("gracePeriodActive", normalized.get("grace_period_active", False))),
        "timestamp": normalized.get("lastUpdate") or normalized.get("timestamp"),
    }


def _build_server_alert_title(previous_snapshot, current_snapshot):
    if previous_snapshot["actionState"] != current_snapshot["actionState"] and current_snapshot["actionState"] in {"LONG_ACTIVE", "SHORT_ACTIVE"}:
        return "XAU/USD long signal active" if current_snapshot["action"] == "buy" else "XAU/USD short signal active"
    if previous_snapshot["actionState"] in {"LONG_ACTIVE", "SHORT_ACTIVE"} and current_snapshot["actionState"] == "WAIT":
        return "XAU/USD signal paused"
    if previous_snapshot["verdict"] != current_snapshot["verdict"]:
        return f"XAU/USD bias changed to {current_snapshot['verdict']}"
    if previous_snapshot["hasBlockers"] and not current_snapshot["hasBlockers"]:
        return "XAU/USD blockers cleared"
    return "XAU/USD signal update"


def _is_actionable_signal_snapshot(snapshot):
    if not snapshot.get("isActionable"):
        return False
    if snapshot["actionState"] == "LONG_ACTIVE":
        return snapshot["action"] == "buy" and snapshot["verdict"] == "Bullish"
    if snapshot["actionState"] == "SHORT_ACTIVE":
        return snapshot["action"] == "sell" and snapshot["verdict"] == "Bearish"
    return False


def _is_pushworthy_signal_change(previous_snapshot, current_snapshot):
    if not current_snapshot.get("notificationAllowed", True):
        return False

    previous_actionable = _is_actionable_signal_snapshot(previous_snapshot)
    current_actionable = _is_actionable_signal_snapshot(current_snapshot)

    if current_actionable:
        return (
            not previous_actionable
            or previous_snapshot["actionState"] != current_snapshot["actionState"]
        )

    return previous_actionable and current_snapshot["actionState"] == "WAIT"


def _notification_suppression_reason(previous_snapshot, current_snapshot):
    if previous_snapshot is None:
        return "initial snapshot"
    if not current_snapshot.get("notificationAllowed", True):
        reasons = current_snapshot.get("suppressionReasons") or []
        return "; ".join(reasons) or "notification suppressed by signal validator"
    if current_snapshot["actionState"] in ACTIVE_ACTION_STATES and not current_snapshot.get("isActionable"):
        reasons = current_snapshot.get("suppressionReasons") or []
        return "; ".join(reasons) or "active signal failed validation"
    if current_snapshot.get("hasBlockers"):
        return "blockers present"
    if current_snapshot.get("score", 0) < current_snapshot.get("threshold", _signal_score_threshold()):
        return "score below threshold"
    if current_snapshot.get("tradeabilityLabel") == "Low":
        return "tradeability is Low"
    if current_snapshot.get("displayStatus") == "Waiting for Signal":
        return "dashboard would render Waiting for Signal"
    return "not a push-worthy transition"


def _log_signal_transition(previous_snapshot, current_snapshot, decision, reason):
    if current_snapshot is None:
        return

    previous_state = previous_snapshot["actionState"] if previous_snapshot else "None"
    previous_verdict = previous_snapshot["verdict"] if previous_snapshot else "None"
    logger.info(
        "Signal transition previous=%s next=%s bias=%s->%s score=%.2f threshold=%.2f "
        "blockers=%s tradeability=%s notification=%s reason=%s",
        previous_state,
        current_snapshot["actionState"],
        previous_verdict,
        current_snapshot["verdict"],
        current_snapshot.get("score", 0.0),
        current_snapshot.get("threshold", _signal_score_threshold()),
        current_snapshot.get("blockers", []),
        current_snapshot.get("tradeabilityLabel"),
        decision,
        reason,
    )


def _describe_server_signal_change(previous_snapshot, current_snapshot):
    if previous_snapshot is None or current_snapshot is None:
        return None
    if not _is_pushworthy_signal_change(previous_snapshot, current_snapshot):
        return None

    changes = []
    severity = "info"

    if previous_snapshot["actionState"] != current_snapshot["actionState"]:
        changes.append(f"Action {previous_snapshot['actionState']} -> {current_snapshot['actionState']}")
        severity = "success" if current_snapshot["actionState"] in {"LONG_ACTIVE", "SHORT_ACTIVE"} else "warn"
    if previous_snapshot["verdict"] != current_snapshot["verdict"]:
        changes.append(f"Bias {previous_snapshot['verdict']} -> {current_snapshot['verdict']}")
    if previous_snapshot["tradeabilityLabel"] != current_snapshot["tradeabilityLabel"]:
        changes.append(
            f"Tradeability {previous_snapshot['tradeabilityLabel']} -> {current_snapshot['tradeabilityLabel']}"
        )
    if previous_snapshot["signalsKey"] != current_snapshot["signalsKey"] and current_snapshot["signals"]:
        changes.append(f"Lead signal: {current_snapshot['signals'][0]}")
    if previous_snapshot["hasBlockers"] != current_snapshot["hasBlockers"]:
        changes.append("New blockers detected" if current_snapshot["hasBlockers"] else "Blockers cleared")
    if abs(current_snapshot["confidence"] - previous_snapshot["confidence"]) >= 10:
        changes.append(
            f"Confidence {round(previous_snapshot['confidence'])}% -> {round(current_snapshot['confidence'])}%"
        )
    if abs(current_snapshot["score"] - previous_snapshot["score"]) >= 10:
        changes.append(
            f"Signal score {round(previous_snapshot['score'])} -> {round(current_snapshot['score'])}"
        )

    if not changes:
        return None

    return {
        "title": _build_server_alert_title(previous_snapshot, current_snapshot),
        "body": " | ".join(changes),
        "severity": severity,
    }


def _load_last_notification_snapshot():
    payload = _read_json_file(WEB_PUSH_SIGNAL_SNAPSHOT_PATH, default=None)
    return payload if isinstance(payload, dict) else None


def _save_last_notification_snapshot(snapshot):
    if not isinstance(snapshot, dict):
        return
    try:
        _write_json_file(WEB_PUSH_SIGNAL_SNAPSHOT_PATH, _json_safe(snapshot), mode=0o600)
    except Exception as exc:
        logger.warning("Failed to persist notification snapshot state: %s", exc)


def _previous_notification_snapshot():
    if isinstance(last_notification_snapshot, dict):
        return last_notification_snapshot
    persisted = _load_last_notification_snapshot()
    if isinstance(persisted, dict):
        return persisted
    return last_push_snapshot if isinstance(last_push_snapshot, dict) else None


def _set_notification_snapshot(snapshot):
    global last_notification_snapshot

    if not isinstance(snapshot, dict):
        return
    last_notification_snapshot = dict(snapshot)
    _save_last_notification_snapshot(last_notification_snapshot)


def _web_push_error_text(exc):
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)

    text = getattr(response, "text", None)
    if text:
        return str(text)

    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    if content:
        return str(content)

    return str(exc)


def _is_stale_push_failure(exc):
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in {404, 410}:
        return True
    if status_code != 403:
        return False

    normalized_text = re.sub(r"[^a-z0-9]+", "", _web_push_error_text(exc).lower())
    return "badjwttoken" in normalized_text


def _send_web_push_notification(title, body, severity):
    subscriptions = _load_push_subscriptions()
    if not subscriptions:
        return

    config = get_web_push_config()
    if not config.get("available") or webpush is None:
        return

    payload = json.dumps(
        {
            "version": SIGNAL_PUSH_PAYLOAD_VERSION,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "maxAgeSeconds": SIGNAL_PUSH_MAX_AGE_SECONDS,
            "title": title,
            "body": body,
            "dedupeKey": f"{title}|{body}",
            "url": "/",
            "tag": "xauusd-important-signal",
            "requireInteraction": severity in {"success", "error"},
        }
    )
    stale_endpoints = []

    for subscription in subscriptions:
        endpoint = subscription.get("endpoint")
        try:
            webpush(
                subscription_info=subscription,
                data=payload,
                vapid_private_key=config["privateKeyPath"],
                vapid_claims={"sub": config["subject"]},
                ttl=300,
            )
        except WebPushException as exc:
            logger.warning("Web push failed for %s: %s", endpoint, exc)
            if _is_stale_push_failure(exc):
                stale_endpoints.append(endpoint)
        except Exception as exc:
            logger.warning("Unexpected web push failure for %s: %s", endpoint, exc)

    for endpoint in stale_endpoints:
        _remove_push_subscription(endpoint)


def _runtime_state_snapshot():
    with risk_lock:
        return {
            "activeSignal": _json_safe(active_trade_state),
            "riskState": _json_safe(dict(risk_state)),
            "barState": _json_safe(dict(bar_state)),
        }


def _attach_runtime_state(prediction):
    if not isinstance(prediction, dict):
        return prediction
    enriched = dict(prediction)
    enriched.update(_runtime_state_snapshot())
    return enriched


def _active_signal_from_prediction(prediction, now=None):
    if not isinstance(prediction, dict):
        return None

    validation = prediction.get("signalValidation") if isinstance(prediction.get("signalValidation"), dict) else {}
    action_state = validation.get("actionState") or prediction.get("actionState")
    active_trade = validation.get("activeTrade")
    if active_trade is None:
        active_trade = action_state in ACTIVE_ACTION_STATES
    if action_state not in ACTIVE_ACTION_STATES or not active_trade:
        return None

    signal_price = _finite_float(prediction.get("entryPrice") or validation.get("signalPrice"))
    stop_loss = _finite_float(prediction.get("stopLoss") or validation.get("stopLoss"))
    take_profit = _finite_float(prediction.get("takeProfit") or validation.get("takeProfit"))
    if signal_price is None or stop_loss is None or take_profit is None:
        return None

    if not _valid_signal_risk(action_state, signal_price, stop_loss, take_profit):
        logger.warning(
            "Refusing to arm risk engine for invalid active signal action=%s entry=%s sl=%s tp=%s",
            action_state,
            signal_price,
            stop_loss,
            take_profit,
        )
        return None

    created_at = (
        prediction.get("createdAt")
        or prediction.get("timestamp")
        or prediction.get("lastUpdate")
        or (now or datetime.now(timezone.utc)).isoformat()
    )
    signals = prediction.get("signals") if isinstance(prediction.get("signals"), list) else []
    is_long = action_state == "LONG_ACTIVE"
    active_signal = {
        "symbol": _prediction_symbol(prediction),
        "direction": "LONG" if is_long else "SHORT",
        "actionState": action_state,
        "action": action_state,
        "entryPrice": signal_price,
        "stopLoss": stop_loss,
        "takeProfit": take_profit,
        "slPips": prediction.get("slPips"),
        "tpPips": prediction.get("tpPips"),
        "rrRatio": prediction.get("rrRatio"),
        "signalPrice": signal_price,
        "createdAt": created_at,
        "signalsKey": "|".join(_normalize_signal_text(signal) for signal in signals[:4]),
        "lastConfirmedScore": validation.get("score") or _prediction_score(prediction),
        "lastConfirmedConfidence": prediction.get("confidence"),
        "lastConfirmedTradeability": validation.get("tradeabilityLabel")
        or prediction.get("tradeabilityLabel"),
        "signalSnapshotId": validation.get("signalSnapshotId") or prediction.get("signal_snapshot_id"),
        "candleUsedForSignal": validation.get("candleUsedForSignal") or prediction.get("candle_used_for_signal"),
        "lastClosedCandleTime": validation.get("lastClosedCandleTime") or prediction.get("last_closed_candle_time"),
        "calculationTime": validation.get("calculationTime") or prediction.get("calculation_time"),
        "status": "OPEN",
    }
    active_signal["signalIdentityKey"] = _active_signal_identity_key(active_signal)
    return active_signal


def _sync_active_trade_state(prediction, now=None):
    global active_trade_state, risk_state

    now = now or datetime.now(timezone.utc)
    next_active_signal = _active_signal_from_prediction(prediction, now=now)
    with risk_lock:
        if next_active_signal is None:
            if risk_state.get("exitReason") is None:
                active_trade_state = None
            return None

        next_signal_key = next_active_signal.get("signalIdentityKey") or _active_signal_identity_key(next_active_signal)
        if next_signal_key and next_signal_key == risk_state.get("closedSignalKey"):
            logger.info(
                "Suppressing re-arm of closed signal symbol=%s action=%s candle=%s reason=%s",
                next_active_signal.get("symbol") or DEFAULT_SYMBOL,
                next_active_signal.get("action"),
                next_active_signal.get("candleUsedForSignal"),
                risk_state.get("exitReason"),
            )
            active_trade_state = None
            return None

        if (
            active_trade_state
            and active_trade_state.get("status") == "OPEN"
            and active_trade_state.get("action") == next_active_signal.get("action")
        ):
            preserved_active_signal = dict(active_trade_state)
            preserved_active_signal.update(
                {
                    "lastConfirmedScore": next_active_signal.get("lastConfirmedScore"),
                    "lastConfirmedConfidence": next_active_signal.get("lastConfirmedConfidence"),
                    "lastConfirmedTradeability": next_active_signal.get("lastConfirmedTradeability"),
                    "signalSnapshotId": next_active_signal.get("signalSnapshotId"),
                    "calculationTime": next_active_signal.get("calculationTime"),
                    "status": "OPEN",
                }
            )
            active_trade_state = preserved_active_signal
            return dict(active_trade_state)

        active_trade_state = next_active_signal
        risk_state = _empty_risk_state()
        return dict(active_trade_state)


def _risk_exit_for_price(active_signal, current_price):
    if not active_signal:
        return None

    direction = active_signal.get("direction")
    stop_loss = _finite_float(active_signal.get("stopLoss"))
    take_profit = _finite_float(active_signal.get("takeProfit"))
    current_price = _finite_float(current_price)
    if current_price is None or stop_loss is None or take_profit is None:
        return None

    if direction == "SHORT":
        if current_price >= stop_loss:
            return "SL_HIT"
        if current_price <= take_profit:
            return "TP_HIT"
    if direction == "LONG":
        if current_price <= stop_loss:
            return "SL_HIT"
        if current_price >= take_profit:
            return "TP_HIT"
    return None


def _risk_exit_notification(reason, active_signal, current_price):
    direction = active_signal.get("direction", "TRADE")
    exit_label = "stop loss" if reason == "SL_HIT" else "take profit"
    title = f"XAU/USD {exit_label} hit"
    body = (
        f"{direction} {reason.replace('_', ' ')} at {_format_signal_number(current_price)} | "
        f"Entry {_format_signal_number(active_signal.get('entryPrice'))} | "
        f"SL {_format_signal_number(active_signal.get('stopLoss'))} | "
        f"TP {_format_signal_number(active_signal.get('takeProfit'))}"
    )
    severity = "error" if reason == "SL_HIT" else "success"
    return title, body, severity


def _set_wait_after_risk_exit(active_signal, current_price, reason, timestamp):
    symbol = DEFAULT_SYMBOL
    reason_text = "Stop loss hit" if reason == "SL_HIT" else "Take profit hit"
    timestamp_iso = timestamp.isoformat()
    base_prediction = dict(latest_prediction) if isinstance(latest_prediction, dict) else {}
    base_prediction.update(
        {
            "verdict": "Neutral",
            "confidence": base_prediction.get("confidence", 50),
            "tradeabilityLabel": "Low",
            "action": "hold",
            "actionState": "WAIT",
            "blockers": [],
            "signals": [f"{reason_text} at {_format_signal_number(current_price)}"],
            "currentPrice": round(current_price, 2),
            "entryPrice": active_signal.get("entryPrice"),
            "stopLoss": active_signal.get("stopLoss"),
            "takeProfit": active_signal.get("takeProfit"),
            "timestamp": timestamp_iso,
            "lastUpdate": timestamp_iso,
            "dataSource": "Twelve Data",
            "symbol": symbol,
            "reason": reason_text,
            "forecast": {
                **(base_prediction.get("forecast") if isinstance(base_prediction.get("forecast"), dict) else {}),
                "score": 0,
                "scoreThreshold": _signal_score_threshold(),
            },
        }
    )
    return _apply_authoritative_signal_state(base_prediction, status="active")


def _commit_risk_exit(active_signal, current_price, reason, timestamp, notify=True):
    global latest_prediction, last_update, last_push_snapshot, active_trade_state, risk_state

    current_price = _finite_float(current_price)
    if active_signal is None or current_price is None:
        return False

    with risk_lock:
        if active_trade_state is None:
            return False

        closed_active_signal = dict(active_signal)
        closed_signal_key = closed_active_signal.get("signalIdentityKey") or _active_signal_identity_key(closed_active_signal)
        closed_active_signal["signalIdentityKey"] = closed_signal_key
        risk_state = _empty_risk_state(
            slHit=reason == "SL_HIT",
            tpHit=reason == "TP_HIT",
            exitReason=reason,
            exitPrice=round(current_price, 2),
            exitTime=timestamp.isoformat(),
            closedSignalKey=closed_signal_key,
            closedSignalAction=closed_active_signal.get("action"),
            closedSignalCandle=closed_active_signal.get("candleUsedForSignal")
            or closed_active_signal.get("lastClosedCandleTime"),
            closedSignalSnapshotId=closed_active_signal.get("signalSnapshotId"),
        )
        closed_active_signal.update(
            {
                "status": "CLOSED",
                "exitReason": reason,
                "exitPrice": round(current_price, 2),
                "exitTime": timestamp.isoformat(),
            }
        )
        active_trade_state = None

        signal_snapshot_state.pop(DEFAULT_SYMBOL, None)
        latest_prediction = _json_safe(
            _attach_runtime_state(
                _set_wait_after_risk_exit(closed_active_signal, current_price, reason, timestamp)
            )
        )
        latest_prediction["activeSignal"] = _json_safe(closed_active_signal)
        last_update = timestamp
        last_push_snapshot = _build_server_signal_snapshot(latest_prediction)
        _set_notification_snapshot(last_push_snapshot)

    title, body, severity = _risk_exit_notification(reason, closed_active_signal, current_price)
    logger.info(
        "Risk exit committed symbol=%s direction=%s reason=%s price=%s sl=%s tp=%s notification=%s",
        DEFAULT_SYMBOL,
        closed_active_signal.get("direction"),
        reason,
        current_price,
        closed_active_signal.get("stopLoss"),
        closed_active_signal.get("takeProfit"),
        "sent" if notify else "suppressed",
    )
    if notify:
        _send_web_push_notification(title, body, severity)
    return True


def _update_latest_live_price(current_price, timestamp):
    global latest_prediction, last_price_update

    current_price = _finite_float(current_price)
    if current_price is None or not _has_usable_prediction():
        return

    latest_prediction = dict(latest_prediction)
    latest_prediction["currentPrice"] = round(current_price, 2)
    latest_prediction["priceUpdatedAt"] = timestamp.isoformat()
    latest_prediction = _json_safe(_attach_runtime_state(latest_prediction))
    last_price_update = timestamp


def _process_live_price_tick(current_price, timestamp=None, notify=True):
    timestamp = _coerce_utc_datetime(timestamp or datetime.now(timezone.utc))
    current_price = _finite_float(current_price)
    if current_price is None:
        return False

    _update_latest_live_price(current_price, timestamp)
    with risk_lock:
        active_signal = dict(active_trade_state) if active_trade_state else None

    reason = _risk_exit_for_price(active_signal, current_price)
    logger.info(
        "Risk tick checked symbol=%s price=%s active_signal=%s exit_reason=%s",
        DEFAULT_SYMBOL,
        current_price,
        active_signal.get("action") if active_signal else None,
        reason,
    )
    if reason is None:
        return False

    return _commit_risk_exit(active_signal, current_price, reason, timestamp, notify=notify)


def _remember_signal_snapshot(prediction):
    global last_push_snapshot

    validated_prediction = _ensure_validated_signal_prediction(prediction, advance=True)
    current_snapshot = _build_server_signal_snapshot(validated_prediction)
    if current_snapshot is None:
        return None
    last_push_snapshot = current_snapshot
    if current_snapshot.get("notificationAllowed", True) or _previous_notification_snapshot() is None:
        _set_notification_snapshot(current_snapshot)
    _sync_active_trade_state(validated_prediction)
    return current_snapshot


def _notify_signal_change(prediction):
    global last_push_snapshot

    validated_prediction = _ensure_validated_signal_prediction(prediction, advance=True)
    current_snapshot = _build_server_signal_snapshot(validated_prediction)
    if current_snapshot is None:
        return

    previous_snapshot = _previous_notification_snapshot()
    last_push_snapshot = current_snapshot
    _sync_active_trade_state(validated_prediction)
    if previous_snapshot is None:
        _set_notification_snapshot(current_snapshot)
        _log_signal_transition(previous_snapshot, current_snapshot, "suppressed", "initial snapshot")
        return

    if not current_snapshot.get("notificationAllowed", True):
        _log_signal_transition(
            previous_snapshot,
            current_snapshot,
            "suppressed",
            _notification_suppression_reason(previous_snapshot, current_snapshot),
        )
        return

    change = _describe_server_signal_change(previous_snapshot, current_snapshot)
    if change is None:
        _set_notification_snapshot(current_snapshot)
        _log_signal_transition(
            previous_snapshot,
            current_snapshot,
            "suppressed",
            _notification_suppression_reason(previous_snapshot, current_snapshot),
        )
        return
    _set_notification_snapshot(current_snapshot)
    _log_signal_transition(previous_snapshot, current_snapshot, "sent", "push-worthy transition")
    _send_web_push_notification(change["title"], change["body"], change["severity"])


def _prediction_is_stale(max_staleness=MAX_PREDICTION_STALENESS):
    if latest_prediction is None or last_update is None:
        return True
    return datetime.now(timezone.utc) - last_update > max_staleness


def _ensure_prediction_fresh():
    if not _prediction_is_stale():
        return False

    with prediction_refresh_lock:
        if not _prediction_is_stale():
            return False

        logger.warning(
            "Cached prediction is stale by more than %s; refreshing on demand.",
            MAX_PREDICTION_STALENESS,
        )
        generate_prediction(notify=True)
        return True


def _twelve_data_api_key():
    return (
        os.getenv("TWELVE_DATA_API_KEY", "").strip()
        or os.getenv("TWELVEDATA_API_KEY", "").strip()
    )


def _parse_positive_int(value, default, label):
    try:
        parsed = int(str(value or "").strip())
    except ValueError:
        logger.warning("Invalid %s %r; defaulting to %s.", label, value, default)
        return default
    return parsed if parsed > 0 else default


@lru_cache(maxsize=1)
def get_td_client():
    api_key = _twelve_data_api_key()
    if not api_key:
        raise RuntimeError("TWELVE_DATA_API_KEY is missing. Set it in Render environment variables.")
    return TDClient(apikey=api_key)


def interval_to_twelvedata(interval):
    raw = str(interval or "1h").strip().lower()
    mapping = {
        "15m": "15min",
        "15min": "15min",
        "1h": "1h",
        "60m": "1h",
        "60min": "1h",
        "4h": "4h",
        "240m": "4h",
        "1d": "1day",
        "1day": "1day",
        "1w": "1week",
        "1wk": "1week",
        "1week": "1week",
    }
    return mapping.get(raw, raw)


def bars_for_period(period, interval):
    raw_period = str(period or "5d").strip().lower()
    td_interval = interval_to_twelvedata(interval)

    if raw_period.endswith("d"):
        days = _parse_positive_int(raw_period[:-1], 5, "PREDICTOR_PERIOD")
    elif raw_period.endswith("mo"):
        days = _parse_positive_int(raw_period[:-2], 1, "PREDICTOR_PERIOD") * 30
    elif raw_period.endswith("y"):
        days = _parse_positive_int(raw_period[:-1], 1, "PREDICTOR_PERIOD") * 365
    else:
        days = 5

    bars_per_day = {
        "15min": 96,
        "1h": 24,
        "4h": 6,
        "1day": 1,
        "1week": 1 / 7,
    }.get(td_interval, 24)
    bars = int(math.ceil(days * bars_per_day))
    return max(120, min(5000, bars))


def _to_utc_datetime_index(values):
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if not isinstance(parsed, pd.DatetimeIndex):
        parsed = pd.DatetimeIndex(parsed)
    if parsed.tz is None:
        return parsed.tz_localize(TD_OUTPUT_TIMEZONE)
    return parsed.tz_convert(TD_OUTPUT_TIMEZONE)


def _validate_ohlcv_quality(df):
    required_columns = ["Open", "High", "Low", "Close"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing OHLC columns from Twelve Data: {', '.join(missing_columns)}")
    if df.empty:
        raise ValueError("No OHLC candles returned by Twelve Data")

    if df.index.has_duplicates:
        raise ValueError("Duplicate candle timestamps returned by Twelve Data")
    if not (df.index.is_monotonic_increasing or df.index.is_monotonic_decreasing):
        raise ValueError("Out-of-order candle timestamps returned by Twelve Data")

    ohlc = df[required_columns]
    finite_mask = np.isfinite(ohlc.to_numpy(dtype=float)).all(axis=1)
    positive_mask = (ohlc > 0).all(axis=1)
    range_mask = (
        (df["High"] >= df["Low"])
        & (df["High"] >= df["Open"])
        & (df["High"] >= df["Close"])
        & (df["Low"] <= df["Open"])
        & (df["Low"] <= df["Close"])
    )
    candle_range = df["High"] - df["Low"]
    sane_range_mask = candle_range <= df["Close"].abs() * MAX_OHLC_RANGE_RATIO
    valid_mask = finite_mask & positive_mask & range_mask & sane_range_mask

    rejected_count = int((~valid_mask).sum())
    if rejected_count:
        logger.warning(
            "Rejected %s malformed Twelve Data OHLC candles before signal calculation.",
            rejected_count,
        )

    filtered = df.loc[valid_mask].copy()
    if filtered.empty:
        raise ValueError("No valid OHLC candles remained after Twelve Data quality checks")
    return filtered


def _interval_duration(interval):
    raw = str(interval or "").strip().lower()
    mapping = {
        "15min": timedelta(minutes=15),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "60min": timedelta(hours=1),
        "60m": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "240min": timedelta(hours=4),
        "240m": timedelta(hours=4),
        "1day": timedelta(days=1),
        "1d": timedelta(days=1),
        "1week": timedelta(weeks=1),
        "1w": timedelta(weeks=1),
    }
    return mapping.get(raw)


def _validate_latest_candle_freshness(df, interval):
    if df.empty:
        return

    duration = _interval_duration(interval)
    max_age = max(duration * 4, timedelta(minutes=20)) if duration else timedelta(hours=4)
    latest_timestamp = df.index[-1]
    latest_age = datetime.now(timezone.utc) - latest_timestamp.to_pydatetime()
    if latest_age > max_age:
        raise ValueError(
            f"Latest Twelve Data candle is stale: age {latest_age} exceeds {max_age}"
        )


def _coerce_utc_datetime(value):
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _current_candle_open(timeframe, now=None):
    now = _coerce_utc_datetime(now or datetime.now(timezone.utc))
    duration = _interval_duration(interval_to_twelvedata(timeframe))
    if duration is None:
        return None

    duration_seconds = int(duration.total_seconds())
    if duration_seconds <= 0:
        return None

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    elapsed_seconds = int((now - epoch).total_seconds())
    open_seconds = elapsed_seconds - (elapsed_seconds % duration_seconds)
    return epoch + timedelta(seconds=open_seconds)


def get_last_closed_candle(df, timeframe, now=None):
    """Return the latest candle-open timestamp that is fully closed for now/timeframe."""
    if df is None or df.empty:
        return None

    current_open = _current_candle_open(timeframe, now)
    if current_open is None:
        return df.index[-1]

    current_open = pd.Timestamp(current_open)
    closed_index = df.index[df.index < current_open]
    return closed_index[-1] if len(closed_index) else None


def _single_tick_candle_shape(row):
    try:
        open_price = float(row["Open"])
        high_price = float(row["High"])
        low_price = float(row["Low"])
        close_price = float(row["Close"])
    except (KeyError, TypeError, ValueError):
        return False

    if not all(math.isfinite(value) for value in [open_price, high_price, low_price, close_price]):
        return False
    identical_ohlc = open_price == high_price == low_price == close_price
    volume = _finite_float(row.get("Volume"))
    return identical_ohlc and (volume is None or volume <= 1)


def build_intrabar_signal_metadata(df, timeframe=DEFAULT_INTERVAL, now=None):
    """Return metadata for bar-aware signal calculation while keeping live price separate."""
    global bar_state

    if df is None or df.empty:
        raise ValueError("No candles available for signal calculation")

    now = _coerce_utc_datetime(now or datetime.now(timezone.utc))
    latest_provider_timestamp = df.index[-1]
    current_open = _current_candle_open(timeframe, now)
    last_closed_timestamp = get_last_closed_candle(df, timeframe, now)
    bar_age_seconds = None
    grace_period_active = False
    grace_remaining_seconds = 0

    if current_open is not None:
        bar_age_seconds = max(0, int((now - current_open).total_seconds()))
        grace_period_active = (
            BAR_OPEN_GRACE_SECONDS > 0
            and 0 <= bar_age_seconds < BAR_OPEN_GRACE_SECONDS
        )
        grace_remaining_seconds = max(0, BAR_OPEN_GRACE_SECONDS - bar_age_seconds)

    provider_candle_is_closed = (
        current_open is None
        or pd.Timestamp(latest_provider_timestamp) < pd.Timestamp(current_open)
    )
    if current_open is not None and last_closed_timestamp is None:
        raise ValueError("No fully closed candle is available for signal calculation")

    signal_candle_timestamp = last_closed_timestamp if current_open is not None else latest_provider_timestamp
    signal_candle_is_closed = (
        current_open is None
        or pd.Timestamp(signal_candle_timestamp) < pd.Timestamp(current_open)
    )
    calculation_time = now.isoformat()
    latest_provider_iso = latest_provider_timestamp.isoformat()
    signal_candle_iso = signal_candle_timestamp.isoformat()
    last_closed_iso = last_closed_timestamp.isoformat() if last_closed_timestamp is not None else None
    current_bar_iso = current_open.isoformat() if current_open is not None else None
    snapshot_id = "|".join(
        [
            str(DEFAULT_SYMBOL),
            str(timeframe),
            calculation_time,
            latest_provider_iso,
        ]
    )

    previous_current_bar = bar_state.get("currentBarTime")
    new_bar_detected = bool(current_bar_iso and current_bar_iso != previous_current_bar)
    last_bar_time = previous_current_bar if new_bar_detected else bar_state.get("lastBarTime")
    new_bar_detected_at = calculation_time if new_bar_detected else bar_state.get("newBarDetectedAt")
    if new_bar_detected:
        logger.info(
            "New %s bar detected symbol=%s last_bar=%s current_bar=%s grace_period_seconds=%s",
            timeframe,
            DEFAULT_SYMBOL,
            previous_current_bar,
            current_bar_iso,
            BAR_OPEN_GRACE_SECONDS,
        )

    if grace_period_active:
        logger.info(
            "Bar-open grace active symbol=%s timeframe=%s current_bar=%s remaining_seconds=%s "
            "latest_provider_candle=%s",
            DEFAULT_SYMBOL,
            timeframe,
            current_bar_iso,
            grace_remaining_seconds,
            latest_provider_iso,
        )
    elif bar_state.get("gracePeriodActive"):
        logger.info(
            "Bar-open grace ended symbol=%s timeframe=%s current_bar=%s; signal expiry can be evaluated.",
            DEFAULT_SYMBOL,
            timeframe,
            current_bar_iso,
        )

    excluded_single_tick_count = 0
    if not provider_candle_is_closed:
        try:
            excluded_single_tick_count = 1 if _single_tick_candle_shape(df.iloc[-1]) else 0
        except Exception:
            excluded_single_tick_count = 0

    bar_state = {
        "currentBarTime": current_bar_iso,
        "lastBarTime": last_bar_time,
        "newBarDetectedAt": new_bar_detected_at,
        "barAgeSeconds": bar_age_seconds,
        "gracePeriodActive": bool(grace_period_active),
        "gracePeriodSeconds": BAR_OPEN_GRACE_SECONDS,
        "graceRemainingSeconds": grace_remaining_seconds,
    }

    logger.info(
        "Intrabar signal metadata now=%s timeframe=%s latest_provider_candle=%s "
        "current_candle_open=%s last_closed_candle=%s candle_used_for_signal=%s "
        "candle_is_closed=%s grace_period_active=%s rows=%s",
        calculation_time,
        timeframe,
        latest_provider_iso,
        current_bar_iso,
        last_closed_iso,
        signal_candle_iso,
        signal_candle_is_closed,
        grace_period_active,
        len(df),
    )

    metadata = {
        "signal_snapshot_id": snapshot_id,
        "calculation_time": calculation_time,
        "latest_provider_candle_time": latest_provider_iso,
        "last_closed_candle_time": last_closed_iso,
        "candle_used_for_signal": signal_candle_iso,
        "candle_is_closed": bool(signal_candle_is_closed),
        "grace_period_active": bool(grace_period_active),
        "excluded_current_candle": bool(latest_provider_timestamp != signal_candle_timestamp),
        "excluded_single_tick_count": excluded_single_tick_count,
    }
    df.attrs.update(metadata)
    return metadata


def _closed_signal_frame(df, metadata):
    if df is None or df.empty:
        raise ValueError("No candles available for signal calculation")

    candle_used = _parse_datetime(metadata.get("candle_used_for_signal"))
    if candle_used is None:
        raise ValueError("No closed candle is available for signal calculation")

    candle_used = pd.Timestamp(candle_used)
    closed_df = df.loc[df.index <= candle_used].copy()
    if closed_df.empty:
        raise ValueError("No closed candles remained for signal calculation")
    closed_df.attrs.update(metadata)
    return closed_df


def normalize_ohlcv_frame(frame):
    if frame is None or frame.empty:
        return pd.DataFrame()

    df = frame.copy()
    df.columns = [str(column).capitalize() for column in df.columns]

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "Datetime" in df.columns:
        df["Datetime"] = _to_utc_datetime_index(df["Datetime"])
        df = df.set_index("Datetime")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = _to_utc_datetime_index(df.index)
    elif df.index.tz is None:
        df.index = df.index.tz_localize(TD_OUTPUT_TIMEZONE)
    else:
        df.index = df.index.tz_convert(TD_OUTPUT_TIMEZONE)

    df = df[~df.index.isna()]
    df = _validate_ohlcv_quality(df)

    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    else:
        df["Volume"] = df["Volume"].fillna(0.0)

    df = df.sort_index()
    return df


def fetch_xauusd_data(period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    """Fetch XAU/USD candles from Twelve Data."""
    try:
        client = get_td_client()
        symbol = DEFAULT_SYMBOL
        td_interval = interval_to_twelvedata(interval)
        outputsize = bars_for_period(period, td_interval)

        series = client.time_series(
            symbol=symbol,
            interval=td_interval,
            outputsize=outputsize,
            timezone=TD_OUTPUT_TIMEZONE,
        )
        df = normalize_ohlcv_frame(series.as_pandas())
        if df.empty:
            raise ValueError(f"No data returned from Twelve Data for {symbol}")
        _validate_latest_candle_freshness(df, td_interval)

        logger.info("Fetched %s %s bars of %s from Twelve Data", len(df), td_interval, symbol)
        return df
    except Exception as exc:
        logger.error("Error fetching Twelve Data candles: %s", exc)
        raise


def fetch_live_price():
    """Fetch the latest spot price from Twelve Data when available."""
    try:
        payload = get_td_client().price(symbol=DEFAULT_SYMBOL).as_json()
        if isinstance(payload, dict) and payload.get("price") is not None:
            return float(payload["price"])
    except Exception as exc:
        logger.warning("Falling back to last candle close; live price fetch failed: %s", exc)
    return None


# ============================================
# PREDICTION PIPELINE
# ============================================

def _has_usable_prediction():
    return isinstance(latest_prediction, dict) and not latest_prediction.get("error")


def generate_prediction(notify=True):
    """Generate a prediction using the fixed signal engine and Twelve Data."""
    global latest_prediction, last_update, error_state

    with prediction_refresh_lock:
        try:
            now = datetime.now(timezone.utc)
            live_price = fetch_live_price()
            if live_price is not None and _process_live_price_tick(live_price, now, notify=notify):
                return

            provider_df = fetch_xauusd_data(period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL)
            signal_metadata = build_intrabar_signal_metadata(provider_df, DEFAULT_INTERVAL, now=now)
            signal_source_df = _closed_signal_frame(provider_df, signal_metadata)
            df = prepare_data(signal_source_df, params=DEFAULT_PARAMS)
            df.attrs.update(signal_metadata)

            prediction = compute_prediction(df, params=DEFAULT_PARAMS)
            now_iso = now.isoformat()
            recent_frame = df.tail(50)

            if live_price is not None:
                prediction["currentPrice"] = round(live_price, 2)

            prediction["timestamp"] = now_iso
            prediction["lastUpdate"] = now_iso
            prediction["predictionUpdatedAt"] = now_iso
            if live_price is not None:
                prediction["priceUpdatedAt"] = now_iso
            prediction["dataPoints"] = len(df)
            prediction["providerDataPoints"] = len(provider_df)
            prediction["timeframe"] = DEFAULT_INTERVAL
            prediction["dataSource"] = "Twelve Data"
            prediction["symbol"] = DEFAULT_SYMBOL
            prediction.update(signal_metadata)
            prediction["chartData"] = {
                "prices": recent_frame["Close"].tolist(),
                "highs": recent_frame["High"].tolist(),
                "lows": recent_frame["Low"].tolist(),
                "ema20": recent_frame["EMA_20"].tolist(),
                "ema50": recent_frame["EMA_50"].tolist(),
                "vwap": recent_frame["VWAP"].tolist(),
                "timestamps": [timestamp.isoformat() for timestamp in recent_frame.index],
            }

            prediction = _ensure_validated_signal_prediction(
                prediction,
                status="active",
                now=now,
                advance=True,
            )
            _sync_active_trade_state(prediction, now=now)
            latest_prediction = _json_safe(_attach_runtime_state(prediction))
            last_update = now
            error_state = None
            if notify:
                _notify_signal_change(latest_prediction)
            else:
                _remember_signal_snapshot(latest_prediction)

            logger.info(
                "Prediction generated from Twelve Data: %s @ %s%% snapshot=%s "
                "candle_used_for_signal=%s last_closed_candle=%s grace_period_active=%s",
                prediction["verdict"],
                prediction["confidence"],
                prediction.get("signal_snapshot_id"),
                prediction.get("candle_used_for_signal"),
                prediction.get("last_closed_candle_time"),
                prediction.get("grace_period_active"),
            )

        except Exception as exc:
            logger.error("Prediction error: %s", exc)
            error_state = str(exc)
            if _has_usable_prediction():
                logger.warning(
                    "Keeping last successful prediction after refresh failure; prediction timestamp unchanged."
                )
                return
            last_update = datetime.now(timezone.utc)
            latest_prediction = {
                "verdict": "Neutral",
                "confidence": 50,
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "predictionUpdatedAt": None,
                "dataSource": "Twelve Data",
                "symbol": DEFAULT_SYMBOL,
            }


# ============================================
# FLASK ROUTES
# ============================================

@app.route("/notification-sw.js")
def notification_service_worker():
    response = send_from_directory(
        os.path.join(BASE_DIR, "static"),
        "notification-sw.js",
        mimetype="application/javascript",
    )
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.route("/api/notifications/config")
def notification_config():
    config = get_web_push_config()
    return jsonify(
        _json_safe(
            {
                "pushAvailable": bool(config.get("available")),
                "vapidPublicKey": config.get("vapidPublicKey"),
                "workerPath": config.get("workerPath", "/notification-sw.js"),
                "warning": config.get("warning"),
                "subscriberCount": len(_load_push_subscriptions()),
            }
        )
    )


@app.route("/api/notifications/subscribe", methods=["POST"])
def notification_subscribe():
    config = get_web_push_config()
    if not config.get("available"):
        return jsonify({"ok": False, "error": config.get("warning") or "Web push is unavailable."}), 503
    if request.content_length and request.content_length > MAX_NOTIFICATION_PAYLOAD_BYTES:
        return jsonify({"ok": False, "error": "Notification payload is too large."}), 413

    payload = request.get_json(silent=True) or {}
    subscription = payload.get("subscription") or payload
    client_id = payload.get("clientId") or payload.get("client_id")
    try:
        subscriber_count = _upsert_push_subscription(subscription, client_id=client_id)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, "subscriberCount": subscriber_count})


@app.route("/api/notifications/unsubscribe", methods=["POST"])
def notification_unsubscribe():
    if request.content_length and request.content_length > MAX_NOTIFICATION_PAYLOAD_BYTES:
        return jsonify({"ok": False, "error": "Notification payload is too large."}), 413

    payload = request.get_json(silent=True) or {}
    endpoint = payload.get("endpoint") or (payload.get("subscription") or {}).get("endpoint")
    subscriber_count = _remove_push_subscription(endpoint)
    return jsonify({"ok": True, "subscriberCount": subscriber_count})

@app.route("/")
def dashboard():
    """Main dashboard page."""
    return render_template("dashboard.html")


@app.route("/api/prediction")
def api_prediction():
    """API endpoint for current prediction."""
    _ensure_prediction_fresh()

    if latest_prediction is None:
        return _no_store_json({
            "verdict": "Neutral",
            "confidence": 50,
            "status": "initializing",
            "message": "Predictor is warming up...",
            "dataSource": "Twelve Data",
            "symbol": DEFAULT_SYMBOL,
        })

    has_usable_prediction = _has_usable_prediction()
    status = (
        "error"
        if error_state and not has_usable_prediction
        else "stale"
        if error_state or _prediction_is_stale()
        else "active"
    )
    response = {
        **latest_prediction,
        "status": status,
        "error": error_state if not has_usable_prediction else None,
        "warning": error_state if has_usable_prediction and error_state else None,
    }
    if last_update is not None:
        response["predictionUpdatedAt"] = response.get("predictionUpdatedAt") or last_update.isoformat()
    if last_price_update is not None:
        response["priceUpdatedAt"] = last_price_update.isoformat()
    response = _ensure_validated_signal_prediction(response, status=status, advance=False)
    response = _attach_runtime_state(response)

    return _no_store_json(response)


@app.route("/api/health")
def health_check():
    """Health check endpoint."""
    has_usable_prediction = _has_usable_prediction()
    status = (
        "error"
        if error_state and not has_usable_prediction
        else "stale"
        if error_state or _prediction_is_stale()
        else "healthy"
    )
    return jsonify(_json_safe({
        "status": status,
        "lastUpdate": last_update.isoformat() if last_update else None,
        "error": error_state,
        "dataSource": "Twelve Data",
        "symbol": DEFAULT_SYMBOL,
        "timeframe": DEFAULT_INTERVAL,
        "priceUpdatedAt": last_price_update.isoformat() if last_price_update else None,
    }))


# ============================================
# BACKGROUND SCHEDULER
# ============================================

scheduler = None


def start_prediction_scheduler():
    global scheduler
    if os.getenv("DISABLE_PREDICTION_SCHEDULER", "0") == "1":
        logger.info("Prediction scheduler disabled by environment flag.")
        return
    if scheduler is not None and scheduler.running:
        return

    scheduler = BackgroundScheduler(timezone=TD_OUTPUT_TIMEZONE)
    scheduler.add_job(
        generate_prediction,
        "interval",
        seconds=REFRESH_SECONDS,
        id="prediction_job",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown(wait=False) if scheduler and scheduler.running else None)

    with app.app_context():
        generate_prediction()


start_prediction_scheduler()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

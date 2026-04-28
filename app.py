
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
MAX_NOTIFICATION_PAYLOAD_BYTES = 64 * 1024
MAX_PUSH_ENDPOINT_BYTES = 2048
MAX_PUSH_KEY_BYTES = 512
MAX_PUSH_SUBSCRIPTIONS = 200
NOTIFICATION_DEDUPE_SECONDS = 300

latest_prediction = None
last_update = None
error_state = None
last_push_snapshot = None
last_notification = {"key": None, "time": None}
prediction_refresh_lock = threading.RLock()
push_lock = threading.Lock()


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


def _normalize_push_subscription(subscription):
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

    return {
        "endpoint": endpoint,
        "expirationTime": expiration_time,
        "keys": {
            "p256dh": p256dh,
            "auth": auth,
        },
    }


def _upsert_push_subscription(subscription):
    normalized = _normalize_push_subscription(subscription)
    if normalized is None:
        raise ValueError("Invalid push subscription payload")

    with push_lock:
        subscriptions = [
            item
            for item in _load_push_subscriptions()
            if item.get("endpoint") != normalized["endpoint"]
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


def _build_server_signal_snapshot(prediction):
    if not isinstance(prediction, dict) or prediction.get("error"):
        return None

    signals = prediction.get("signals") if isinstance(prediction.get("signals"), list) else []
    blockers = prediction.get("blockers") if isinstance(prediction.get("blockers"), list) else []
    forecast = prediction.get("forecast") if isinstance(prediction.get("forecast"), dict) else {}

    confidence = float(prediction.get("confidence") or 0)
    score = float(forecast.get("score") or 0)

    return {
        "verdict": prediction.get("verdict") or "Neutral",
        "action": prediction.get("action") or "hold",
        "actionState": prediction.get("actionState") or "WAIT",
        "tradeabilityLabel": prediction.get("tradeabilityLabel") or "Low",
        "confidence": confidence,
        "score": score,
        "signals": signals[:4],
        "signalsKey": "|".join(_normalize_signal_text(signal) for signal in signals[:4]),
        "hasBlockers": bool(blockers),
        "timestamp": prediction.get("lastUpdate") or prediction.get("timestamp"),
    }


def _build_server_alert_title(previous_snapshot, current_snapshot):
    if previous_snapshot["actionState"] != current_snapshot["actionState"] and current_snapshot["actionState"] in {"LONG_ACTIVE", "SHORT_ACTIVE"}:
        return "XAU/USD long signal active" if current_snapshot["action"] == "buy" else "XAU/USD short signal active"
    if previous_snapshot["verdict"] != current_snapshot["verdict"]:
        return f"XAU/USD bias changed to {current_snapshot['verdict']}"
    if previous_snapshot["hasBlockers"] and not current_snapshot["hasBlockers"]:
        return "XAU/USD blockers cleared"
    return "XAU/USD signal update"


def _describe_server_signal_change(previous_snapshot, current_snapshot):
    if previous_snapshot is None or current_snapshot is None:
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


def _notification_dedupe_key(snapshot):
    return "|".join(
        [
            str(snapshot.get("actionState") or "WAIT"),
            str(snapshot.get("verdict") or "Neutral"),
        ]
    )


def _should_send_signal_notification(current_snapshot, now=None):
    global last_notification

    now = now or datetime.now(timezone.utc)
    key = _notification_dedupe_key(current_snapshot)
    last_key = last_notification.get("key")
    last_time = last_notification.get("time")

    if (
        key == last_key
        and last_time is not None
        and (now - last_time).total_seconds() < NOTIFICATION_DEDUPE_SECONDS
    ):
        logger.info("Skipping duplicate push alert for %s inside dedupe window.", key)
        return False

    last_notification = {"key": key, "time": now}
    return True


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
            "title": title,
            "body": body,
            "url": "/",
            "tag": "xauusd-important-signal",
            "requireInteraction": severity == "success",
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


def _remember_signal_snapshot(prediction):
    global last_push_snapshot

    current_snapshot = _build_server_signal_snapshot(prediction)
    if current_snapshot is None:
        return None
    last_push_snapshot = current_snapshot
    return current_snapshot


def _notify_signal_change(prediction):
    global last_push_snapshot

    current_snapshot = _build_server_signal_snapshot(prediction)
    if current_snapshot is None:
        return

    previous_snapshot = last_push_snapshot
    last_push_snapshot = current_snapshot
    if previous_snapshot is None:
        return

    change = _describe_server_signal_change(previous_snapshot, current_snapshot)
    if change is None:
        return
    if not _should_send_signal_notification(current_snapshot):
        return

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
        generate_prediction(notify=False)
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

    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    else:
        df["Volume"] = df["Volume"].fillna(0.0)

    df = df.sort_index()
    return df.dropna(subset=[column for column in ["Open", "High", "Low", "Close"] if column in df.columns])


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
            df = fetch_xauusd_data(period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL)
            df = prepare_data(df, params=DEFAULT_PARAMS)

            prediction = compute_prediction(df, params=DEFAULT_PARAMS)
            live_price = fetch_live_price()
            now_iso = datetime.now(timezone.utc).isoformat()
            recent_frame = df.tail(50)

            if live_price is not None:
                prediction["currentPrice"] = round(live_price, 2)

            prediction["timestamp"] = now_iso
            prediction["lastUpdate"] = now_iso
            prediction["dataPoints"] = len(df)
            prediction["timeframe"] = DEFAULT_INTERVAL
            prediction["dataSource"] = "Twelve Data"
            prediction["symbol"] = DEFAULT_SYMBOL
            prediction["chartData"] = {
                "prices": recent_frame["Close"].tolist(),
                "highs": recent_frame["High"].tolist(),
                "lows": recent_frame["Low"].tolist(),
                "ema20": recent_frame["EMA_20"].tolist(),
                "ema50": recent_frame["EMA_50"].tolist(),
                "vwap": recent_frame["VWAP"].tolist(),
                "timestamps": [timestamp.isoformat() for timestamp in recent_frame.index],
            }

            latest_prediction = _json_safe(prediction)
            last_update = datetime.now(timezone.utc)
            error_state = None
            if notify:
                _notify_signal_change(latest_prediction)
            else:
                _remember_signal_snapshot(latest_prediction)

            logger.info(
                "Prediction generated from Twelve Data: %s @ %s%%",
                prediction["verdict"],
                prediction["confidence"],
            )

        except Exception as exc:
            logger.error("Prediction error: %s", exc)
            error_state = str(exc)
            last_update = datetime.now(timezone.utc)
            if _has_usable_prediction():
                logger.warning("Keeping last successful prediction after refresh failure.")
                return
            latest_prediction = {
                "verdict": "Neutral",
                "confidence": 50,
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
    try:
        subscriber_count = _upsert_push_subscription(subscription)
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
        return jsonify({
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

    return jsonify(_json_safe(response))


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

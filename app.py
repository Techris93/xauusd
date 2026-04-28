
"""Fixed XAUUSD predictor using Twelve Data as the market data source."""

import atexit
import logging
import math
import os
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template
from twelvedata import TDClient

from signal_engine import DEFAULT_PARAMS, compute_prediction, prepare_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

TD_OUTPUT_TIMEZONE = "UTC"
DEFAULT_SYMBOL = os.getenv("TWELVE_DATA_SYMBOL", "XAU/USD").strip() or "XAU/USD"
DEFAULT_PERIOD = os.getenv("PREDICTOR_PERIOD", "5d").strip() or "5d"
DEFAULT_INTERVAL = os.getenv("PREDICTOR_INTERVAL", "1h").strip() or "1h"
REFRESH_MINUTES = max(1, int(os.getenv("PREDICTION_REFRESH_MINUTES", "5")))

latest_prediction = None
last_update = None
error_state = None


def _twelve_data_api_key():
    return (
        os.getenv("TWELVE_DATA_API_KEY", "").strip()
        or os.getenv("TWELVEDATA_API_KEY", "").strip()
    )


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
        days = int(raw_period[:-1] or "0")
    elif raw_period.endswith("mo"):
        days = int(raw_period[:-2] or "0") * 30
    elif raw_period.endswith("y"):
        days = int(raw_period[:-1] or "0") * 365
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

def generate_prediction():
    """Generate a prediction using the fixed signal engine and Twelve Data."""
    global latest_prediction, last_update, error_state

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

        latest_prediction = prediction
        last_update = datetime.now(timezone.utc)
        error_state = None

        logger.info(
            "Prediction generated from Twelve Data: %s @ %s%%",
            prediction["verdict"],
            prediction["confidence"],
        )

    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        error_state = str(exc)
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

@app.route("/")
def dashboard():
    """Main dashboard page."""
    return render_template("dashboard.html")


@app.route("/api/prediction")
def api_prediction():
    """API endpoint for current prediction."""
    if latest_prediction is None:
        return jsonify({
            "verdict": "Neutral",
            "confidence": 50,
            "status": "initializing",
            "message": "Predictor is warming up...",
            "dataSource": "Twelve Data",
            "symbol": DEFAULT_SYMBOL,
        })

    response = {
        **latest_prediction,
        "status": "active" if not error_state else "error",
        "error": error_state
    }

    return jsonify(response)


@app.route("/api/health")
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if not error_state else "error",
        "lastUpdate": last_update.isoformat() if last_update else None,
        "error": error_state,
        "dataSource": "Twelve Data",
        "symbol": DEFAULT_SYMBOL,
        "timeframe": DEFAULT_INTERVAL,
    })


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
        minutes=REFRESH_MINUTES,
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

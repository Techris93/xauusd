import math
import os
import sys
import tempfile
import unittest
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DISABLE_PREDICTION_SCHEDULER", "1")

import app as app_module
import signal_engine as signal_module
from signal_engine import (
    calculate_anticipatory_score,
    calculate_confidence,
    calculate_stop_loss_take_profit,
    detect_structure_break,
    prepare_data,
    signal_state,
)


class PredictorRegressionTests(unittest.TestCase):
    def setUp(self):
        app_module.latest_prediction = None
        app_module.last_update = None
        app_module.error_state = None
        app_module.last_push_snapshot = None
        app_module.last_notification = {"key": None, "time": None}
        app_module.signal_stabilizer_state = {}
        app_module.active_trade_state = None
        app_module.risk_state = {
            "slHit": False,
            "tpHit": False,
            "exitReason": None,
            "exitPrice": None,
            "exitTime": None,
        }
        app_module.bar_state = {}
        app_module.get_web_push_config.cache_clear()
        signal_state.reset()
        self.client = app_module.app.test_client()

    def tearDown(self):
        app_module.latest_prediction = None
        app_module.last_update = None
        app_module.error_state = None
        app_module.last_push_snapshot = None
        app_module.last_notification = {"key": None, "time": None}
        app_module.signal_stabilizer_state = {}
        app_module.active_trade_state = None
        app_module.risk_state = {
            "slHit": False,
            "tpHit": False,
            "exitReason": None,
            "exitPrice": None,
            "exitTime": None,
        }
        app_module.bar_state = {}
        app_module.get_web_push_config.cache_clear()
        signal_state.reset()

    @staticmethod
    def build_frame(index=None, volume=1000.0):
        if index is None:
            index = pd.date_range(
                start=pd.Timestamp("2024-01-01T00:00:00Z"),
                periods=120,
                freq="h",
                tz="UTC",
            )

        rows = []
        base_price = 2400.0
        for offset in range(len(index)):
            open_price = base_price + offset * 0.5
            rows.append(
                {
                    "Open": open_price,
                    "High": open_price + 0.2,
                    "Low": open_price - 0.2,
                    "Close": open_price + 0.1,
                    "Volume": volume,
                }
            )

        frame = pd.DataFrame(rows, index=index)
        frame.index.name = "Datetime"
        return frame

    @staticmethod
    def build_signal_frame():
        index = pd.date_range(
            start=pd.Timestamp("2024-01-01T00:00:00Z"),
            periods=30,
            freq="h",
            tz="UTC",
        )
        close = [100.0 + offset * 0.1 for offset in range(len(index))]
        frame = pd.DataFrame(
            {
                "Open": close,
                "High": [value + 0.3 for value in close],
                "Low": [value - 0.3 for value in close],
                "Close": close,
                "Volume": [1000.0] * len(index),
                "EMA_20": [value - 0.5 for value in close],
                "EMA_50": [value - 1.0 for value in close],
                "VWAP": [value - 0.2 for value in close],
                "ADX_14": [25.0] * len(index),
                "ATR_14": [1.0] * len(index),
                "VOLUME_SPIKE": [0] * len(index),
                "RECENT_SWING_LOW": [value - 2.0 for value in close],
                "RECENT_SWING_HIGH": [value + 2.0 for value in close],
            },
            index=index,
        )
        frame["nearest_support"] = [{"label": "Round Number", "price": 99.0}] * len(frame)
        frame["nearest_resistance"] = [{"label": "Round Number", "price": 105.0}] * len(frame)
        return frame

    @staticmethod
    def wait_snapshot(score=22.5, has_blockers=True):
        return {
            "verdict": "Neutral",
            "action": "hold",
            "actionState": "WAIT",
            "tradeabilityLabel": "Low",
            "confidence": 50.0,
            "score": score,
            "threshold": 45.0,
            "signals": [],
            "signalsKey": "",
            "hasBlockers": has_blockers,
            "blockers": ["Tradeability 22.5 below threshold 45"] if has_blockers else [],
            "isActionable": False,
            "suppressionReasons": [],
            "displayStatus": "Waiting for Signal",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def active_prediction(action_state="SHORT_ACTIVE", score=60, blockers=None, timestamp=None, candle_timestamp=None):
        is_long = action_state == "LONG_ACTIVE"
        entry_price = 2400.0
        timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        candle_timestamp = candle_timestamp or timestamp
        return {
            "verdict": "Bullish" if is_long else "Bearish",
            "confidence": 82,
            "action": "buy" if is_long else "sell",
            "actionState": action_state,
            "tradeabilityLabel": "High",
            "blockers": list(blockers or []),
            "signals": ["Bullish structure break" if is_long else "Bearish structure break"],
            "forecast": {"score": score},
            "entryPrice": entry_price,
            "stopLoss": entry_price - 3.0 if is_long else entry_price + 3.0,
            "takeProfit": entry_price + 6.0 if is_long else entry_price - 6.0,
            "timestamp": timestamp,
            "lastUpdate": timestamp,
            "chartData": {"timestamps": [candle_timestamp]},
        }

    @staticmethod
    def wait_prediction(score=35, blockers=None, timestamp=None, candle_timestamp=None):
        timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        candle_timestamp = candle_timestamp or timestamp
        if blockers is None:
            blockers = [f"signal score {score} below threshold 45"]
        return {
            "verdict": "Neutral",
            "confidence": 70,
            "action": "hold",
            "actionState": "WAIT",
            "tradeabilityLabel": "Low",
            "blockers": list(blockers),
            "signals": ["VWAP rejection faded"],
            "forecast": {"score": score},
            "timestamp": timestamp,
            "lastUpdate": timestamp,
            "chartData": {"timestamps": [candle_timestamp]},
        }

    @staticmethod
    def risk_frame(atr_value=None):
        payload = {"Close": [4577.0]}
        if atr_value is not None:
            payload["ATR_14"] = [atr_value]
        return pd.DataFrame(payload)

    @staticmethod
    def ohlcv_frame(index=None, **overrides):
        if index is None:
            index = pd.date_range(
                start=pd.Timestamp("2026-04-30T08:00:00Z"),
                periods=3,
                freq="h",
                tz="UTC",
            )
        payload = {
            "Open": [2400.0, 2401.0, 2402.0],
            "High": [2401.0, 2402.0, 2403.0],
            "Low": [2399.0, 2400.0, 2401.0],
            "Close": [2400.5, 2401.5, 2402.5],
            "Volume": [1000.0, 1000.0, 1000.0],
        }
        payload.update(overrides)
        return pd.DataFrame(payload, index=index)

    @staticmethod
    def iso_at(offset_seconds):
        return (
            datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc)
            + timedelta(seconds=offset_seconds)
        ).isoformat()

    @staticmethod
    def fixed_datetime(fixed_now):
        class FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)

            @classmethod
            def fromisoformat(cls, value):
                return datetime.fromisoformat(value)

        return FixedDatetime

    def test_prepare_data_assigns_nearest_levels_without_existing_columns(self):
        prepared = prepare_data(self.build_frame())

        self.assertIn("nearest_support", prepared.columns)
        self.assertIn("nearest_resistance", prepared.columns)
        self.assertIsInstance(prepared.iloc[-1]["nearest_support"], dict)
        self.assertIsInstance(prepared.iloc[-1]["nearest_resistance"], dict)

    def test_prepare_data_stays_stable_with_duplicate_last_timestamp(self):
        base_index = pd.date_range(
            start=pd.Timestamp("2024-01-01T00:00:00Z"),
            periods=120,
            freq="h",
            tz="UTC",
        )
        duplicate_index = base_index[:-1].append(pd.DatetimeIndex([base_index[-2]]))

        prepared = prepare_data(self.build_frame(index=duplicate_index))

        self.assertFalse(prepared.index.is_unique)
        self.assertIsInstance(prepared.iloc[-1]["nearest_support"], dict)
        self.assertIsInstance(prepared.iloc[-1]["nearest_resistance"], dict)

    def test_prepare_data_uses_finite_vwap_when_volume_is_zero(self):
        prepared = prepare_data(self.build_frame(volume=0.0))
        tail_values = prepared["VWAP"].tail(10).tolist()

        self.assertFalse(any(pd.isna(value) for value in tail_values))
        self.assertTrue(all(math.isfinite(value) for value in tail_values))

    def test_prepare_data_adds_missing_volume_column(self):
        frame = self.build_frame().drop(columns=["Volume"])

        prepared = prepare_data(frame)

        self.assertIn("Volume", prepared.columns)
        self.assertTrue((prepared["Volume"] == 0.0).all())
        self.assertFalse(prepared["VWAP"].tail(10).isna().any())

    def test_normalize_ohlcv_frame_sorts_descending_timestamps(self):
        frame = self.ohlcv_frame().iloc[::-1]

        normalized = app_module.normalize_ohlcv_frame(frame)

        self.assertTrue(normalized.index.is_monotonic_increasing)

    def test_normalize_ohlcv_frame_rejects_duplicate_timestamps(self):
        index = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-04-30T08:00:00Z"),
                pd.Timestamp("2026-04-30T09:00:00Z"),
                pd.Timestamp("2026-04-30T09:00:00Z"),
            ]
        )

        with self.assertRaisesRegex(ValueError, "Duplicate candle timestamps"):
            app_module.normalize_ohlcv_frame(self.ohlcv_frame(index=index))

    def test_normalize_ohlcv_frame_rejects_mixed_out_of_order_timestamps(self):
        index = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-04-30T08:00:00Z"),
                pd.Timestamp("2026-04-30T10:00:00Z"),
                pd.Timestamp("2026-04-30T09:00:00Z"),
            ]
        )

        with self.assertRaisesRegex(ValueError, "Out-of-order candle timestamps"):
            app_module.normalize_ohlcv_frame(self.ohlcv_frame(index=index))

    def test_normalize_ohlcv_frame_drops_non_finite_ohlc_values(self):
        frame = self.ohlcv_frame(Close=[2400.5, float("nan"), 2402.5])

        normalized = app_module.normalize_ohlcv_frame(frame)

        self.assertEqual(len(normalized), 2)
        self.assertFalse(normalized["Close"].isna().any())

    def test_normalize_ohlcv_frame_drops_invalid_ohlc_range(self):
        frame = self.ohlcv_frame(High=[2401.0, 2399.0, 2403.0])

        normalized = app_module.normalize_ohlcv_frame(frame)

        self.assertEqual(len(normalized), 2)
        self.assertTrue((normalized["High"] >= normalized["Close"]).all())

    def test_normalize_ohlcv_frame_drops_absurd_ohlc_range(self):
        frame = self.ohlcv_frame(High=[2401.0, 3500.0, 2403.0], Low=[2399.0, 1000.0, 2401.0])

        normalized = app_module.normalize_ohlcv_frame(frame)

        self.assertEqual(len(normalized), 2)
        self.assertLessEqual(
            (normalized["High"] - normalized["Low"]).max(),
            normalized["Close"].abs().max() * app_module.MAX_OHLC_RANGE_RATIO,
        )

    def test_normalize_ohlcv_frame_rejects_all_malformed_ohlc_rows(self):
        frame = self.ohlcv_frame(High=[3500.0, 3500.0, 3500.0], Low=[1000.0, 1000.0, 1000.0])

        with self.assertRaisesRegex(ValueError, "No valid OHLC candles remained"):
            app_module.normalize_ohlcv_frame(frame)

    def test_latest_candle_freshness_rejects_stale_hourly_candle(self):
        index = pd.date_range(
            end=pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=8),
            periods=3,
            freq="h",
        )
        normalized = app_module.normalize_ohlcv_frame(self.ohlcv_frame(index=index))

        with self.assertRaisesRegex(ValueError, "Latest Twelve Data candle is stale"):
            app_module._validate_latest_candle_freshness(normalized, "1h")

    def test_intrabar_metadata_uses_current_candle_at_0901_but_marks_grace(self):
        index = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-04-30T07:00:00Z"),
                pd.Timestamp("2026-04-30T08:00:00Z"),
                pd.Timestamp("2026-04-30T09:00:00Z"),
            ]
        )
        frame = app_module.normalize_ohlcv_frame(self.ohlcv_frame(index=index))
        now = datetime(2026, 4, 30, 9, 1, tzinfo=timezone.utc)

        last_closed = app_module.get_last_closed_candle(frame, "1h", now)
        metadata = app_module.build_intrabar_signal_metadata(frame, "1h", now=now)

        self.assertEqual(last_closed, pd.Timestamp("2026-04-30T08:00:00Z"))
        self.assertEqual(frame.index[-1], pd.Timestamp("2026-04-30T09:00:00Z"))
        self.assertFalse(metadata["candle_is_closed"])
        self.assertTrue(metadata["grace_period_active"])
        self.assertEqual(metadata["last_closed_candle_time"], "2026-04-30T08:00:00+00:00")
        self.assertEqual(metadata["candle_used_for_signal"], "2026-04-30T09:00:00+00:00")
        self.assertEqual(app_module.bar_state["gracePeriodSeconds"], 300)

    def test_single_tick_current_candle_stays_intrabar_with_grace_metadata(self):
        index = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-04-30T07:00:00Z"),
                pd.Timestamp("2026-04-30T08:00:00Z"),
                pd.Timestamp("2026-04-30T09:00:00Z"),
            ]
        )
        frame = app_module.normalize_ohlcv_frame(
            self.ohlcv_frame(
                index=index,
                Open=[2400.0, 2401.0, 2402.0],
                High=[2401.0, 2402.0, 2402.0],
                Low=[2399.0, 2400.0, 2402.0],
                Close=[2400.5, 2401.5, 2402.0],
                Volume=[1000.0, 1000.0, 1.0],
            )
        )

        metadata = app_module.build_intrabar_signal_metadata(
            frame,
            "1h",
            now=datetime(2026, 4, 30, 9, 1, tzinfo=timezone.utc),
        )

        self.assertEqual(frame.index[-1], pd.Timestamp("2026-04-30T09:00:00Z"))
        self.assertEqual(metadata["candle_used_for_signal"], "2026-04-30T09:00:00+00:00")
        self.assertFalse(metadata["excluded_current_candle"])
        self.assertEqual(metadata["excluded_single_tick_count"], 1)
        self.assertTrue(metadata["grace_period_active"])

    def test_stop_loss_take_profit_uses_fallback_for_corrupted_high_atr(self):
        sl, tp, sl_pips, tp_pips = calculate_stop_loss_take_profit(
            4577.0,
            "Bullish",
            self.risk_frame(207.0),
            app_module.DEFAULT_PARAMS,
        )

        self.assertAlmostEqual(sl, 4542.67, places=2)
        self.assertAlmostEqual(tp, 4645.65, places=2)
        self.assertAlmostEqual(sl_pips, 343.3, places=1)
        self.assertAlmostEqual(tp_pips, 686.5, places=1)
        self.assertNotAlmostEqual(sl, 4266.5, places=1)
        self.assertNotAlmostEqual(tp, 5198.0, places=1)

    def test_stop_loss_take_profit_uses_fallback_for_missing_atr(self):
        sl, tp, sl_pips, tp_pips = calculate_stop_loss_take_profit(
            4577.0,
            "Bullish",
            self.risk_frame(),
            app_module.DEFAULT_PARAMS,
        )

        self.assertAlmostEqual(sl, 4542.67, places=2)
        self.assertAlmostEqual(tp, 4645.65, places=2)
        self.assertAlmostEqual(sl_pips, 343.3, places=1)
        self.assertAlmostEqual(tp_pips, 686.5, places=1)

    def test_stop_loss_take_profit_uses_fallback_for_nan_atr(self):
        sl, tp, sl_pips, tp_pips = calculate_stop_loss_take_profit(
            4577.0,
            "Bullish",
            self.risk_frame(float("nan")),
            app_module.DEFAULT_PARAMS,
        )

        self.assertAlmostEqual(sl, 4542.67, places=2)
        self.assertAlmostEqual(tp, 4645.65, places=2)
        self.assertAlmostEqual(sl_pips, 343.3, places=1)
        self.assertAlmostEqual(tp_pips, 686.5, places=1)

    def test_stop_loss_take_profit_uses_fallback_for_low_invalid_atr(self):
        sl, tp, sl_pips, tp_pips = calculate_stop_loss_take_profit(
            4577.0,
            "Bullish",
            self.risk_frame(0.2),
            app_module.DEFAULT_PARAMS,
        )

        self.assertAlmostEqual(sl, 4542.67, places=2)
        self.assertAlmostEqual(tp, 4645.65, places=2)
        self.assertAlmostEqual(sl_pips, 343.3, places=1)
        self.assertAlmostEqual(tp_pips, 686.5, places=1)

    def test_stop_loss_take_profit_uses_valid_atr_for_long_and_short(self):
        long_sl, long_tp, long_sl_pips, long_tp_pips = calculate_stop_loss_take_profit(
            4577.0,
            "Bullish",
            self.risk_frame(10.0),
            app_module.DEFAULT_PARAMS,
        )
        short_sl, short_tp, short_sl_pips, short_tp_pips = calculate_stop_loss_take_profit(
            4577.0,
            "Bearish",
            self.risk_frame(10.0),
            app_module.DEFAULT_PARAMS,
        )

        self.assertEqual(long_sl, 4562.0)
        self.assertEqual(long_tp, 4607.0)
        self.assertEqual(long_sl_pips, 150.0)
        self.assertEqual(long_tp_pips, 300.0)
        self.assertEqual(short_sl, 4592.0)
        self.assertEqual(short_tp, 4547.0)
        self.assertEqual(short_sl_pips, 150.0)
        self.assertEqual(short_tp_pips, 300.0)

    def test_stop_loss_take_profit_caps_extreme_distances(self):
        sl, tp, sl_pips, tp_pips = calculate_stop_loss_take_profit(
            1000.0,
            "Bullish",
            self.risk_frame(49.0),
            app_module.DEFAULT_PARAMS,
        )

        self.assertEqual(sl, 980.0)
        self.assertEqual(tp, 1050.0)
        self.assertLessEqual(abs(1000.0 - sl), 20.0)
        self.assertLessEqual(abs(tp - 1000.0), 50.0)
        self.assertEqual(sl_pips, 200.0)
        self.assertEqual(tp_pips, 500.0)

    def test_stop_loss_take_profit_pips_match_backend_distance(self):
        sl, tp, sl_pips, tp_pips = calculate_stop_loss_take_profit(
            4577.0,
            "Bullish",
            self.risk_frame(207.0),
            app_module.DEFAULT_PARAMS,
        )

        self.assertAlmostEqual((4577.0 - sl) / 0.10, sl_pips, places=1)
        self.assertAlmostEqual((tp - 4577.0) / 0.10, tp_pips, places=1)

    def test_detect_structure_break_uses_prior_range_without_current_candle(self):
        rows = []
        for offset in range(10):
            base_price = 110.0 - offset
            rows.append(
                {
                    "Open": base_price,
                    "High": base_price + 1.0,
                    "Low": base_price - 1.0,
                    "Close": base_price - 0.5,
                }
            )
        rows.append({"Open": 96.0, "High": 96.5, "Low": 93.0, "Close": 94.0})

        frame = pd.DataFrame(rows)

        structure, strength = detect_structure_break(frame, lookback=5)

        self.assertEqual(structure, "bearish_break")
        self.assertGreaterEqual(strength, 0.7)

    def test_verdict_requires_planned_minimum_signal_score(self):
        frame = pd.DataFrame([{"Close": 100.0}])

        confidence, verdict = calculate_confidence(
            50,
            "bullish",
            frame,
            app_module.DEFAULT_PARAMS,
        )
        self.assertEqual(confidence, 75)
        self.assertEqual(verdict, "Neutral")

        confidence, verdict = calculate_confidence(
            55,
            "bullish",
            frame,
            app_module.DEFAULT_PARAMS,
        )
        self.assertEqual(confidence, 78)
        self.assertEqual(verdict, "Bullish")

    def test_sr_bonus_requires_actual_break_not_nearby_level(self):
        frame = self.build_frame().tail(10).copy()
        frame["Close"] = 100.0
        frame["High"] = 100.4
        frame["Low"] = 99.6
        frame["EMA_20"] = 100.5
        frame["EMA_50"] = 101.0
        frame["ADX_14"] = 0
        frame["VOLUME_SPIKE"] = 0
        frame["VWAP"] = frame["Close"]
        frame["RECENT_SWING_LOW"] = 99.5
        frame["RECENT_SWING_HIGH"] = 101.5
        frame["nearest_support"] = [{"label": "Round Number", "price": 99.9}] * len(frame)
        frame["nearest_resistance"] = [{"label": "Round Number", "price": 100.1}] * len(frame)

        score, direction, signals = calculate_anticipatory_score(
            frame,
            app_module.DEFAULT_PARAMS,
        )

        self.assertEqual(direction, "bearish")
        self.assertEqual(score, 15)
        self.assertNotIn("Breaking key support", signals)

        frame.iloc[-1, frame.columns.get_loc("Close")] = 99.4
        score, direction, signals = calculate_anticipatory_score(
            frame,
            app_module.DEFAULT_PARAMS,
        )

        self.assertEqual(direction, "bearish")
        self.assertIn("Breaking key support", signals)
        self.assertGreaterEqual(score, 25)

    def test_signal_hysteresis_keeps_active_signal_above_exit_score(self):
        frame = self.build_signal_frame()

        with mock.patch.object(
            signal_module,
            "calculate_anticipatory_score",
            side_effect=[
                (80, "bullish", ["Bullish structure break"]),
                (50, "bullish", ["Bullish structure cooling"]),
            ],
        ):
            first_prediction = signal_module.compute_prediction(frame, app_module.DEFAULT_PARAMS)
            second_prediction = signal_module.compute_prediction(frame, app_module.DEFAULT_PARAMS)

        self.assertEqual(first_prediction["actionState"], "LONG_ACTIVE")
        self.assertEqual(second_prediction["actionState"], "LONG_ACTIVE")
        self.assertEqual(second_prediction["verdict"], "Bullish")
        self.assertIsNone(second_prediction["antiFlipReason"])

    def test_signal_engine_accepts_intrabar_frame_metadata(self):
        frame = self.build_signal_frame()
        frame.attrs["candle_is_closed"] = False
        frame.attrs["signal_snapshot_id"] = "incomplete-engine-frame"

        with mock.patch.object(
            signal_module,
            "calculate_anticipatory_score",
            return_value=(80, "bearish", ["Bearish structure break"]),
        ):
            prediction = signal_module.compute_prediction(frame, app_module.DEFAULT_PARAMS)

        self.assertEqual(prediction["actionState"], "SHORT_ACTIVE")
        self.assertEqual(prediction["verdict"], "Bearish")
        self.assertNotIn("suppressed_incomplete_candle", prediction["blockers"])
        self.assertEqual(prediction["signal_snapshot_id"], "incomplete-engine-frame")

    def test_signal_cooldown_blocks_immediate_opposite_flip(self):
        frame = self.build_signal_frame()

        with mock.patch.object(
            signal_module,
            "calculate_anticipatory_score",
            side_effect=[
                (80, "bullish", ["Bullish structure break"]),
                (80, "bearish", ["Bearish MA alignment"]),
            ],
        ):
            first_prediction = signal_module.compute_prediction(frame, app_module.DEFAULT_PARAMS)
            second_prediction = signal_module.compute_prediction(frame, app_module.DEFAULT_PARAMS)

        self.assertEqual(first_prediction["actionState"], "LONG_ACTIVE")
        self.assertEqual(second_prediction["actionState"], "LONG_ACTIVE")
        self.assertIn("Cooldown active", second_prediction["antiFlipReason"])

    def test_min_hold_requires_confirmed_opposite_bars(self):
        params = app_module.DEFAULT_PARAMS
        signal_state.current_signal = "LONG_ACTIVE"
        signal_state.current_direction = "bullish"
        signal_state.last_flip_time = datetime.now(timezone.utc) - timedelta(minutes=16)
        signal_state.consecutive_opposite_bars = params["min_hold_bars"] - 1

        allowed, reason = signal_state.can_change_signal("SHORT_ACTIVE", "bearish", params)
        self.assertFalse(allowed)
        self.assertIn("Opposite signal needs", reason)

        signal_state.consecutive_opposite_bars = params["min_hold_bars"]
        allowed, reason = signal_state.can_change_signal("SHORT_ACTIVE", "bearish", params)
        self.assertTrue(allowed)

    def test_refresh_seconds_prefers_new_setting_and_supports_legacy_alias(self):
        with mock.patch.dict(
            os.environ,
            {
                "PREDICTION_REFRESH_SECONDS": "7",
                "PREDICTION_REFRESH_MINUTES": "5",
            },
            clear=False,
        ):
            self.assertEqual(app_module._read_prediction_refresh_seconds(), 7)

        with mock.patch.dict(
            os.environ,
            {
                "PREDICTION_REFRESH_SECONDS": "",
                "PREDICTION_REFRESH_MINUTES": "5",
            },
            clear=False,
        ):
            self.assertEqual(app_module._read_prediction_refresh_seconds(), 5)

    def test_bars_for_period_falls_back_for_invalid_period_values(self):
        self.assertEqual(app_module.bars_for_period("bad", "1h"), 120)
        self.assertEqual(app_module.bars_for_period("abcmo", "1h"), 720)

    def test_prediction_endpoint_returns_json_safe_chart_payload(self):
        frame = self.build_frame(volume=0.0)

        with mock.patch.object(app_module, "fetch_xauusd_data", return_value=frame.copy()):
            with mock.patch.object(app_module, "fetch_live_price", return_value=2500.12):
                app_module.generate_prediction()

        response = self.client.get("/api/prediction")
        payload = response.get_json()
        chart_data = payload.get("chartData")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("NaN", response.get_data(as_text=True))
        self.assertEqual(payload.get("status"), "active")
        self.assertIsNotNone(chart_data)
        self.assertEqual(len(chart_data.get("prices", [])), 50)
        self.assertEqual(len(chart_data.get("timestamps", [])), 50)
        self.assertFalse(any(value is None for value in chart_data.get("vwap", [])))
        for field in [
            "signal_snapshot_id",
            "calculation_time",
            "last_closed_candle_time",
            "candle_used_for_signal",
            "candle_is_closed",
            "grace_period_active",
        ]:
            self.assertIn(field, payload)
        self.assertTrue(payload["candle_is_closed"])

    def test_prediction_endpoint_refreshes_stale_cache_on_request(self):
        stale_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        fresh_time = datetime.now(timezone.utc)
        fresh_iso = fresh_time.isoformat()

        app_module.latest_prediction = {
            "verdict": "Bearish",
            "confidence": 88,
            "timestamp": stale_time.isoformat(),
            "lastUpdate": stale_time.isoformat(),
            "dataSource": "Twelve Data",
            "symbol": app_module.DEFAULT_SYMBOL,
        }
        app_module.last_update = stale_time
        app_module.error_state = None

        def fake_generate_prediction(notify=True):
            self.assertFalse(notify)
            app_module.latest_prediction = {
                "verdict": "Bullish",
                "confidence": 91,
                "timestamp": fresh_iso,
                "lastUpdate": fresh_iso,
                "dataSource": "Twelve Data",
                "symbol": app_module.DEFAULT_SYMBOL,
            }
            app_module.last_update = fresh_time
            app_module.error_state = None

        with mock.patch.object(app_module, "generate_prediction", side_effect=fake_generate_prediction) as mocked_generate:
            response = self.client.get("/api/prediction")

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload.get("status"), "active")
        self.assertEqual(payload.get("timestamp"), fresh_iso)
        self.assertEqual(payload.get("verdict"), "Bullish")
        mocked_generate.assert_called_once_with(notify=False)

    def test_refresh_failure_keeps_last_good_prediction_visible(self):
        fresh_time = datetime.now(timezone.utc)
        fresh_iso = fresh_time.isoformat()
        app_module.latest_prediction = {
            "verdict": "Bullish",
            "confidence": 88,
            "currentPrice": 2500.12,
            "timestamp": fresh_iso,
            "lastUpdate": fresh_iso,
            "dataSource": "Twelve Data",
            "symbol": app_module.DEFAULT_SYMBOL,
            "forecast": {"score": 72},
        }
        app_module.last_update = fresh_time
        app_module.error_state = None

        with mock.patch.object(app_module, "fetch_xauusd_data", side_effect=RuntimeError("rate limited")):
            app_module.generate_prediction()

        response = self.client.get("/api/prediction")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload.get("status"), "stale")
        self.assertIsNone(payload.get("error"))
        self.assertIn("rate limited", payload.get("warning"))
        self.assertEqual(payload.get("currentPrice"), 2500.12)

    def test_request_driven_prediction_refresh_updates_snapshot_without_push(self):
        frame = self.build_frame(volume=0.0)

        with mock.patch.object(app_module, "fetch_xauusd_data", return_value=frame.copy()):
            with mock.patch.object(app_module, "fetch_live_price", return_value=2500.12):
                with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
                    app_module.generate_prediction(notify=False)

        self.assertIsNotNone(app_module.last_push_snapshot)
        mocked_push.assert_not_called()

    def test_bar_open_generate_uses_intrabar_candle_but_preserves_active_signal_during_grace(self):
        fixed_now = datetime(2026, 4, 30, 9, 1, tzinfo=timezone.utc)
        index = pd.date_range(
            end=pd.Timestamp("2026-04-30T09:00:00Z"),
            periods=120,
            freq="h",
            tz="UTC",
        )
        frame = self.build_frame(index=index)
        frame.iloc[-1, frame.columns.get_loc("Open")] = 2402.0
        frame.iloc[-1, frame.columns.get_loc("High")] = 2402.0
        frame.iloc[-1, frame.columns.get_loc("Low")] = 2402.0
        frame.iloc[-1, frame.columns.get_loc("Close")] = 2402.0
        frame.iloc[-1, frame.columns.get_loc("Volume")] = 1.0

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(
                self.active_prediction(
                    "SHORT_ACTIVE",
                    score=72,
                    timestamp="2026-04-30T08:58:00+00:00",
                    candle_timestamp="2026-04-30T07:00:00+00:00",
                )
            )
            app_module._notify_signal_change(
                self.active_prediction(
                    "SHORT_ACTIVE",
                    score=72,
                    timestamp="2026-04-30T08:59:00+00:00",
                    candle_timestamp="2026-04-30T07:00:00+00:00",
                )
            )
            self.assertEqual(app_module.last_push_snapshot["actionState"], "SHORT_ACTIVE")
            mocked_push.reset_mock()

            def fake_compute(signal_df, params=None):
                self.assertEqual(signal_df.index[-1], pd.Timestamp("2026-04-30T09:00:00Z"))
                self.assertEqual(
                    signal_df.attrs.get("candle_used_for_signal"),
                    "2026-04-30T09:00:00+00:00",
                )
                return self.wait_prediction(
                    score=25,
                    timestamp=fixed_now.isoformat(),
                    candle_timestamp="2026-04-30T09:00:00+00:00",
                )

            with mock.patch.object(app_module, "datetime", self.fixed_datetime(fixed_now)):
                with mock.patch.object(app_module, "fetch_xauusd_data", return_value=frame.copy()):
                    with mock.patch.object(app_module, "fetch_live_price", return_value=None):
                        with mock.patch.object(app_module, "compute_prediction", side_effect=fake_compute):
                            app_module.generate_prediction()

        mocked_push.assert_not_called()
        self.assertEqual(app_module.latest_prediction["actionState"], "SHORT_ACTIVE")
        self.assertEqual(app_module.latest_prediction["candle_used_for_signal"], "2026-04-30T09:00:00+00:00")
        self.assertTrue(app_module.latest_prediction["grace_period_active"])
        self.assertIn(
            "suppressed_bar_open_instability",
            app_module.latest_prediction["signalValidation"]["suppressionReasons"],
        )

    def test_short_stop_loss_triggers_immediately_during_grace_period(self):
        timestamp = datetime(2026, 4, 30, 9, 2, tzinfo=timezone.utc)
        app_module.active_trade_state = {
            "direction": "SHORT",
            "action": "SHORT_ACTIVE",
            "entryPrice": 4560.0,
            "stopLoss": 4568.08,
            "takeProfit": 4544.0,
            "signalPrice": 4560.0,
            "createdAt": "2026-04-30T08:59:00+00:00",
            "lastConfirmedScore": 72,
            "lastConfirmedConfidence": 95,
            "lastConfirmedTradeability": "High",
            "status": "OPEN",
        }
        app_module.bar_state = {
            "currentBarTime": "2026-04-30T09:00:00+00:00",
            "gracePeriodActive": True,
            "gracePeriodSeconds": 300,
            "graceRemainingSeconds": 180,
        }

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            triggered = app_module._process_live_price_tick(4568.09, timestamp, notify=True)

        self.assertTrue(triggered)
        self.assertTrue(app_module.risk_state["slHit"])
        self.assertEqual(app_module.risk_state["exitReason"], "SL_HIT")
        self.assertIsNone(app_module.active_trade_state)
        self.assertEqual(app_module.latest_prediction["actionState"], "WAIT")
        self.assertEqual(app_module.latest_prediction["activeSignal"]["status"], "CLOSED")
        mocked_push.assert_called_once()
        self.assertEqual(mocked_push.call_args.args[0], "XAU/USD stop loss hit")

    def test_short_take_profit_triggers_immediately_during_grace_period(self):
        timestamp = datetime(2026, 4, 30, 9, 3, tzinfo=timezone.utc)
        app_module.active_trade_state = {
            "direction": "SHORT",
            "action": "SHORT_ACTIVE",
            "entryPrice": 4560.0,
            "stopLoss": 4568.08,
            "takeProfit": 4544.0,
            "signalPrice": 4560.0,
            "createdAt": "2026-04-30T08:59:00+00:00",
            "lastConfirmedScore": 72,
            "lastConfirmedConfidence": 95,
            "lastConfirmedTradeability": "High",
            "status": "OPEN",
        }
        app_module.bar_state = {
            "currentBarTime": "2026-04-30T09:00:00+00:00",
            "gracePeriodActive": True,
            "gracePeriodSeconds": 300,
            "graceRemainingSeconds": 120,
        }

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            triggered = app_module._process_live_price_tick(4543.99, timestamp, notify=True)

        self.assertTrue(triggered)
        self.assertTrue(app_module.risk_state["tpHit"])
        self.assertEqual(app_module.risk_state["exitReason"], "TP_HIT")
        self.assertIsNone(app_module.active_trade_state)
        mocked_push.assert_called_once()
        self.assertEqual(mocked_push.call_args.args[0], "XAU/USD take profit hit")

    def test_live_risk_exit_runs_before_candle_fetch_or_signal_recalculation(self):
        timestamp = datetime(2026, 4, 30, 9, 2, tzinfo=timezone.utc)
        app_module.active_trade_state = {
            "direction": "LONG",
            "action": "LONG_ACTIVE",
            "entryPrice": 4560.0,
            "stopLoss": 4550.0,
            "takeProfit": 4580.0,
            "signalPrice": 4560.0,
            "createdAt": "2026-04-30T08:59:00+00:00",
            "lastConfirmedScore": 72,
            "lastConfirmedConfidence": 95,
            "lastConfirmedTradeability": "High",
            "status": "OPEN",
        }

        with mock.patch.object(app_module, "datetime", self.fixed_datetime(timestamp)):
            with mock.patch.object(app_module, "fetch_live_price", return_value=4549.99):
                with mock.patch.object(app_module, "fetch_xauusd_data") as mocked_candles:
                    with mock.patch.object(app_module, "compute_prediction") as mocked_compute:
                        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
                            app_module.generate_prediction()

        mocked_candles.assert_not_called()
        mocked_compute.assert_not_called()
        mocked_push.assert_called_once()
        self.assertEqual(app_module.risk_state["exitReason"], "SL_HIT")

    def test_notification_service_worker_route_is_available(self):
        response = self.client.get("/notification-sw.js")
        response_body = response.get_data(as_text=True)
        response.close()

        self.assertEqual(response.status_code, 200)
        self.assertIn("application/javascript", response.content_type)
        self.assertIn('self.addEventListener("push"', response_body)
        self.assertIn(
            'self.addEventListener("notificationclick"',
            response_body,
        )
        self.assertIn("showNotification", response_body)
        self.assertNotIn("hasVisibleClient", response_body)
        self.assertIn("SIGNAL_PUSH_PAYLOAD_VERSION", response_body)
        self.assertIn("Ignoring stale signal push payload", response_body)
        self.assertIn("Ignoring expired signal push payload", response_body)

    def test_dashboard_does_not_show_duplicate_in_app_signal_toasts(self):
        response = self.client.get("/")
        response_body = response.get_data(as_text=True)
        response.close()

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("showToast(`${change.title}", response_body)

    def test_dashboard_declares_favicon_and_manifest(self):
        response = self.client.get("/")
        response_body = response.get_data(as_text=True)
        response.close()

        self.assertEqual(response.status_code, 200)
        self.assertIn('rel="icon"', response_body)
        self.assertIn('href="/static/favicon.svg"', response_body)
        self.assertIn('rel="manifest"', response_body)
        self.assertIn('href="/static/site.webmanifest"', response_body)

    def test_favicon_and_manifest_assets_are_available(self):
        favicon_response = self.client.get("/static/favicon.svg")
        manifest_response = self.client.get("/static/site.webmanifest")
        manifest_payload = json.loads(manifest_response.get_data(as_text=True))
        favicon_response.close()
        manifest_response.close()

        self.assertEqual(favicon_response.status_code, 200)
        self.assertIn("image/svg+xml", favicon_response.content_type)
        self.assertEqual(manifest_response.status_code, 200)
        self.assertEqual(manifest_payload.get("name"), "XAUUSD Predictor")
        self.assertEqual(manifest_payload["icons"][0]["src"], "/static/favicon.svg")

    def test_notification_config_route_returns_metadata(self):
        response = self.client.get("/api/notifications/config")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("pushAvailable", payload)
        self.assertEqual(payload.get("workerPath"), "/notification-sw.js")
        if payload.get("pushAvailable"):
            self.assertTrue(payload.get("vapidPublicKey"))

    def test_notification_test_route_is_not_exposed(self):
        response = self.client.post("/api/notifications/test", json={})

        self.assertEqual(response.status_code, 404)

    def test_signal_notifications_dedupe_same_signal_inside_window(self):
        def prediction(signals, score, confidence=85):
            return {
                "verdict": "Bullish",
                "confidence": confidence,
                "action": "buy",
                "actionState": "LONG_ACTIVE",
                "tradeabilityLabel": "High",
                "blockers": [],
                "signals": signals,
                "forecast": {"score": score},
                "entryPrice": 2400.0,
                "stopLoss": 2397.0,
                "takeProfit": 2406.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(prediction(["Bullish structure break"], 70))
            app_module._notify_signal_change(prediction(["Bullish structure break confirmed"], 82))
            app_module._notify_signal_change(prediction(["Bullish continuation"], 95, confidence=96))

        self.assertEqual(mocked_push.call_count, 1)

    def test_cooldown_blocked_wait_prediction_does_not_send_active_push(self):
        app_module.last_push_snapshot = self.wait_snapshot(score=80.0, has_blockers=False)
        blocked_prediction = {
            "verdict": "Neutral",
            "confidence": 95,
            "action": "hold",
            "actionState": "WAIT",
            "tradeabilityLabel": "High",
            "blockers": ["Warning: Cooldown active: 4.7min remaining"],
            "signals": [
                "Bearish structure break (strength: 1.00)",
                "VWAP bearish rejection (strength: 1.00)",
                "Bearish MA alignment",
            ],
            "forecast": {"score": 90},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(blocked_prediction)

        mocked_push.assert_not_called()
        self.assertEqual(app_module.last_push_snapshot["actionState"], "WAIT")
        self.assertTrue(app_module.last_push_snapshot["hasBlockers"])

    def test_active_push_requires_unblocked_actionable_prediction(self):
        app_module.last_push_snapshot = self.wait_snapshot(score=80.0)
        first_prediction = self.active_prediction("SHORT_ACTIVE", score=90, timestamp=self.iso_at(1))
        second_prediction = self.active_prediction("SHORT_ACTIVE", score=90, timestamp=self.iso_at(2))

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(first_prediction)
            app_module._notify_signal_change(second_prediction)

        mocked_push.assert_called_once()
        self.assertEqual(mocked_push.call_args.args[0], "XAU/USD short signal active")

    def test_neutral_blocked_low_score_cannot_send_short_active_push(self):
        app_module.last_push_snapshot = self.wait_snapshot()
        bad_prediction = self.active_prediction(
            "SHORT_ACTIVE",
            score=25,
            blockers=["Tradeability 22.5 below threshold 45"],
        )

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(bad_prediction)

        mocked_push.assert_not_called()
        self.assertEqual(app_module.last_push_snapshot["actionState"], "WAIT")
        self.assertEqual(app_module.last_push_snapshot["verdict"], "Neutral")
        self.assertTrue(app_module.last_push_snapshot["hasBlockers"])

    def test_bearish_ma_alignment_alone_cannot_send_short_active_below_threshold(self):
        app_module.last_push_snapshot = self.wait_snapshot()
        bad_prediction = self.active_prediction("SHORT_ACTIVE", score=25)
        bad_prediction["tradeabilityLabel"] = "Medium"
        bad_prediction["signals"] = ["Bearish MA alignment"]

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(bad_prediction)

        mocked_push.assert_not_called()
        self.assertEqual(app_module.last_push_snapshot["actionState"], "WAIT")
        self.assertIn(
            "signal score 25 below threshold 45",
            app_module.last_push_snapshot["blockers"],
        )

    def test_blockers_cleared_requires_empty_validated_blockers(self):
        app_module.last_push_snapshot = self.wait_snapshot()
        bad_prediction = self.active_prediction("SHORT_ACTIVE", score=25)

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(bad_prediction)

        mocked_push.assert_not_called()
        self.assertTrue(app_module.last_push_snapshot["blockers"])
        self.assertNotIn("Blockers cleared", " | ".join(app_module.last_push_snapshot["blockers"]))

    def test_dashboard_and_notification_use_same_authoritative_signal_state(self):
        raw_prediction = self.active_prediction("SHORT_ACTIVE", score=25)
        dashboard_prediction = app_module._apply_authoritative_signal_state(raw_prediction)
        notification_snapshot = app_module._build_server_signal_snapshot(raw_prediction)

        self.assertEqual(dashboard_prediction["actionState"], notification_snapshot["actionState"])
        self.assertEqual(dashboard_prediction["verdict"], notification_snapshot["verdict"])
        self.assertEqual(dashboard_prediction["blockers"], notification_snapshot["blockers"])
        self.assertEqual(dashboard_prediction["actionState"], "WAIT")
        self.assertEqual(dashboard_prediction["verdict"], "Neutral")

    def test_valid_short_active_push_requires_score_blockers_signal_price_and_risk(self):
        app_module.last_push_snapshot = self.wait_snapshot()
        first_prediction = self.active_prediction("SHORT_ACTIVE", score=60, timestamp=self.iso_at(1))
        second_prediction = self.active_prediction("SHORT_ACTIVE", score=60, timestamp=self.iso_at(2))

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(first_prediction)
            app_module._notify_signal_change(second_prediction)

        mocked_push.assert_called_once()
        self.assertEqual(mocked_push.call_args.args[0], "XAU/USD short signal active")

    def test_valid_long_active_push_requires_score_blockers_signal_price_and_risk(self):
        app_module.last_push_snapshot = self.wait_snapshot()
        first_prediction = self.active_prediction("LONG_ACTIVE", score=60, timestamp=self.iso_at(1))
        second_prediction = self.active_prediction("LONG_ACTIVE", score=60, timestamp=self.iso_at(2))

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(first_prediction)
            app_module._notify_signal_change(second_prediction)

        mocked_push.assert_called_once()
        self.assertEqual(mocked_push.call_args.args[0], "XAU/USD long signal active")

    def test_active_trade_push_is_suppressed_without_valid_risk_values(self):
        app_module.last_push_snapshot = self.wait_snapshot()
        active_prediction = self.active_prediction("SHORT_ACTIVE", score=60)
        active_prediction["takeProfit"] = None

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(active_prediction)

        mocked_push.assert_not_called()
        self.assertEqual(app_module.last_push_snapshot["actionState"], "WAIT")
        self.assertIn("risk management is incomplete", app_module.last_push_snapshot["blockers"])

    def test_stale_cached_signal_cannot_trigger_active_notification(self):
        app_module.last_push_snapshot = self.wait_snapshot()
        active_prediction = self.active_prediction("SHORT_ACTIVE", score=90)
        active_prediction["status"] = "stale"

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(active_prediction)

        mocked_push.assert_not_called()
        self.assertEqual(app_module.last_push_snapshot["actionState"], "WAIT")
        self.assertEqual(app_module.last_push_snapshot["verdict"], "Neutral")
        self.assertIn("app status is stale", app_module.last_push_snapshot["blockers"])

    def test_stale_status_blocks_committed_active_dashboard_without_pause_spam(self):
        app_module.last_push_snapshot = self.wait_snapshot()

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=72, timestamp=self.iso_at(1)))
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=72, timestamp=self.iso_at(2)))
            self.assertEqual(app_module.last_push_snapshot["actionState"], "LONG_ACTIVE")
            mocked_push.reset_mock()

            stale_prediction = self.active_prediction("LONG_ACTIVE", score=72, timestamp=self.iso_at(3))
            stale_prediction["status"] = "stale"
            dashboard_prediction = app_module._ensure_stabilized_signal_prediction(
                stale_prediction,
                status="stale",
                advance=False,
            )
            app_module._notify_signal_change(stale_prediction)

        self.assertEqual(dashboard_prediction["actionState"], "WAIT")
        self.assertEqual(dashboard_prediction["verdict"], "Neutral")
        self.assertIn("app status is stale", dashboard_prediction["blockers"])
        self.assertFalse(dashboard_prediction["signalValidation"]["notificationAllowed"])
        mocked_push.assert_not_called()
        self.assertEqual(app_module.last_push_snapshot["actionState"], "WAIT")
        self.assertFalse(app_module.last_push_snapshot["notificationAllowed"])

    def test_low_tradeability_suppresses_active_alert(self):
        app_module.last_push_snapshot = self.wait_snapshot()
        active_prediction = self.active_prediction("LONG_ACTIVE", score=90)
        active_prediction["tradeabilityLabel"] = "Low"

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(active_prediction)

        mocked_push.assert_not_called()
        self.assertEqual(app_module.last_push_snapshot["actionState"], "WAIT")
        self.assertIn("tradeability is Low", app_module.last_push_snapshot["blockers"])

    def test_displayed_score_matches_score_used_by_validation_and_blockers(self):
        raw_prediction = self.active_prediction("SHORT_ACTIVE", score=25)

        dashboard_prediction = app_module._apply_authoritative_signal_state(raw_prediction)
        notification_snapshot = app_module._build_server_signal_snapshot(raw_prediction)

        self.assertEqual(dashboard_prediction["forecast"]["score"], 25.0)
        self.assertEqual(dashboard_prediction["signalValidation"]["score"], 25.0)
        self.assertEqual(notification_snapshot["score"], 25.0)
        self.assertIn("signal score 25 below threshold 45", dashboard_prediction["blockers"])

    def test_unstable_wait_long_wait_long_does_not_commit_or_spam(self):
        app_module.last_push_snapshot = self.wait_snapshot()

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=72, timestamp=self.iso_at(1)))
            app_module._notify_signal_change(self.wait_prediction(score=40, timestamp=self.iso_at(2)))
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=72, timestamp=self.iso_at(3)))

        mocked_push.assert_not_called()
        self.assertEqual(app_module.last_push_snapshot["actionState"], "WAIT")

    def test_unstable_wait_short_wait_short_does_not_commit_or_spam(self):
        app_module.last_push_snapshot = self.wait_snapshot()

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=72, timestamp=self.iso_at(1)))
            app_module._notify_signal_change(self.wait_prediction(score=40, timestamp=self.iso_at(2)))
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=72, timestamp=self.iso_at(3)))

        mocked_push.assert_not_called()
        self.assertEqual(app_module.last_push_snapshot["actionState"], "WAIT")

    def test_stable_long_activation_commits_once(self):
        app_module.last_push_snapshot = self.wait_snapshot()

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=72, timestamp=self.iso_at(1)))
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=72, timestamp=self.iso_at(2)))
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=74, timestamp=self.iso_at(3)))

        self.assertEqual(mocked_push.call_count, 1)
        self.assertEqual(mocked_push.call_args.args[0], "XAU/USD long signal active")
        self.assertEqual(app_module.last_push_snapshot["actionState"], "LONG_ACTIVE")

    def test_stable_short_activation_commits_once(self):
        app_module.last_push_snapshot = self.wait_snapshot()

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=72, timestamp=self.iso_at(1)))
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=72, timestamp=self.iso_at(2)))
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=74, timestamp=self.iso_at(3)))

        self.assertEqual(mocked_push.call_count, 1)
        self.assertEqual(mocked_push.call_args.args[0], "XAU/USD short signal active")
        self.assertEqual(app_module.last_push_snapshot["actionState"], "SHORT_ACTIVE")

    def test_stable_long_pause_commits_once(self):
        app_module.last_push_snapshot = self.wait_snapshot()

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=72, timestamp=self.iso_at(1)))
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=72, timestamp=self.iso_at(2)))
            mocked_push.reset_mock()
            app_module._notify_signal_change(self.wait_prediction(score=35, timestamp=self.iso_at(3)))
            app_module._notify_signal_change(self.wait_prediction(score=35, timestamp=self.iso_at(4)))

        self.assertEqual(mocked_push.call_count, 1)
        self.assertEqual(mocked_push.call_args.args[0], "XAU/USD signal paused")
        self.assertEqual(app_module.last_push_snapshot["actionState"], "WAIT")

    def test_stable_short_pause_commits_once(self):
        app_module.last_push_snapshot = self.wait_snapshot()

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=72, timestamp=self.iso_at(1)))
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=72, timestamp=self.iso_at(2)))
            mocked_push.reset_mock()
            app_module._notify_signal_change(self.wait_prediction(score=35, timestamp=self.iso_at(3)))
            app_module._notify_signal_change(self.wait_prediction(score=35, timestamp=self.iso_at(4)))

        self.assertEqual(mocked_push.call_count, 1)
        self.assertEqual(mocked_push.call_args.args[0], "XAU/USD signal paused")
        self.assertEqual(app_module.last_push_snapshot["actionState"], "WAIT")

    def test_long_to_short_reversal_requires_confirmation(self):
        app_module.last_push_snapshot = self.wait_snapshot()

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=72, timestamp=self.iso_at(1)))
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=72, timestamp=self.iso_at(2)))
            mocked_push.reset_mock()
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=74, timestamp=self.iso_at(3)))
            self.assertEqual(app_module.last_push_snapshot["actionState"], "LONG_ACTIVE")
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=74, timestamp=self.iso_at(4)))

        self.assertEqual(mocked_push.call_count, 1)
        self.assertEqual(mocked_push.call_args.args[0], "XAU/USD short signal active")
        self.assertEqual(app_module.last_push_snapshot["actionState"], "SHORT_ACTIVE")

    def test_short_to_long_reversal_requires_confirmation(self):
        app_module.last_push_snapshot = self.wait_snapshot()

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=72, timestamp=self.iso_at(1)))
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=72, timestamp=self.iso_at(2)))
            mocked_push.reset_mock()
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=74, timestamp=self.iso_at(3)))
            self.assertEqual(app_module.last_push_snapshot["actionState"], "SHORT_ACTIVE")
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=74, timestamp=self.iso_at(4)))

        self.assertEqual(mocked_push.call_count, 1)
        self.assertEqual(mocked_push.call_args.args[0], "XAU/USD long signal active")
        self.assertEqual(app_module.last_push_snapshot["actionState"], "LONG_ACTIVE")

    def test_score_hysteresis_prevents_threshold_chatter(self):
        app_module.last_push_snapshot = self.wait_snapshot()

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=46, timestamp=self.iso_at(1)))
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=46, timestamp=self.iso_at(2)))
            mocked_push.reset_mock()
            app_module._notify_signal_change(self.wait_prediction(score=44, blockers=[], timestamp=self.iso_at(3)))
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=46, timestamp=self.iso_at(4)))

        mocked_push.assert_not_called()
        self.assertEqual(app_module.last_push_snapshot["actionState"], "LONG_ACTIVE")

    def test_bar_open_grace_suppresses_active_pause_for_both_directions(self):
        for action_state in ("LONG_ACTIVE", "SHORT_ACTIVE"):
            with self.subTest(action_state=action_state):
                self.tearDown()
                self.setUp()
                app_module.last_push_snapshot = self.wait_snapshot()

                with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
                    app_module._notify_signal_change(
                        self.active_prediction(action_state, score=72, timestamp=self.iso_at(1))
                    )
                    app_module._notify_signal_change(
                        self.active_prediction(action_state, score=72, timestamp=self.iso_at(2))
                    )
                    self.assertEqual(app_module.last_push_snapshot["actionState"], action_state)
                    mocked_push.reset_mock()

                    weak_prediction = self.wait_prediction(
                        score=25,
                        timestamp=self.iso_at(61),
                        candle_timestamp=self.iso_at(0),
                    )
                    weak_prediction.update(
                        {
                            "signal_snapshot_id": f"{action_state}-bar-open",
                            "calculation_time": self.iso_at(61),
                            "latest_provider_candle_time": self.iso_at(60),
                            "last_closed_candle_time": self.iso_at(0),
                            "candle_used_for_signal": self.iso_at(0),
                            "candle_is_closed": True,
                            "grace_period_active": True,
                        }
                    )
                    app_module._notify_signal_change(weak_prediction)

                mocked_push.assert_not_called()
                self.assertEqual(app_module.last_push_snapshot["actionState"], action_state)
                self.assertFalse(app_module.last_push_snapshot["notificationAllowed"])
                self.assertIn(
                    "suppressed_bar_open_instability",
                    app_module.last_push_snapshot["suppressionReasons"],
                )

    def test_grace_expiry_allows_confirmed_signal_pause_after_five_minutes(self):
        app_module.last_push_snapshot = self.wait_snapshot()

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=72, timestamp=self.iso_at(1)))
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=72, timestamp=self.iso_at(2)))
            mocked_push.reset_mock()

            for offset in (301, 302):
                weak_prediction = self.wait_prediction(score=25, timestamp=self.iso_at(offset), candle_timestamp=self.iso_at(offset))
                weak_prediction.update(
                    {
                        "signal_snapshot_id": f"post-grace-weak-{offset}",
                        "calculation_time": self.iso_at(offset),
                        "latest_provider_candle_time": self.iso_at(offset),
                        "last_closed_candle_time": self.iso_at(0),
                        "candle_used_for_signal": self.iso_at(offset),
                        "candle_is_closed": False,
                        "grace_period_active": False,
                    }
                )
                app_module._notify_signal_change(weak_prediction)

        self.assertEqual(mocked_push.call_count, 1)
        self.assertEqual(mocked_push.call_args.args[0], "XAU/USD signal paused")
        self.assertEqual(app_module.last_push_snapshot["actionState"], "WAIT")

    def test_signal_survives_after_grace_when_score_remains_valid(self):
        app_module.last_push_snapshot = self.wait_snapshot()

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=72, timestamp=self.iso_at(1)))
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=72, timestamp=self.iso_at(2)))
            mocked_push.reset_mock()

            valid_prediction = self.active_prediction("SHORT_ACTIVE", score=68, timestamp=self.iso_at(301))
            valid_prediction.update(
                {
                    "signal_snapshot_id": "post-grace-valid",
                    "calculation_time": self.iso_at(301),
                    "latest_provider_candle_time": self.iso_at(301),
                    "last_closed_candle_time": self.iso_at(0),
                    "candle_used_for_signal": self.iso_at(301),
                    "candle_is_closed": False,
                    "grace_period_active": False,
                }
            )
            app_module._notify_signal_change(valid_prediction)

        mocked_push.assert_not_called()
        self.assertEqual(app_module.last_push_snapshot["actionState"], "SHORT_ACTIVE")

    def test_intrabar_snapshot_after_grace_can_pause_after_confirmation(self):
        app_module.last_push_snapshot = self.wait_snapshot()

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=72, timestamp=self.iso_at(1)))
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=72, timestamp=self.iso_at(2)))
            mocked_push.reset_mock()

            weak_prediction = self.wait_prediction(score=25, timestamp=self.iso_at(3), candle_timestamp=self.iso_at(3))
            weak_prediction.update(
                {
                    "signal_snapshot_id": "incomplete-candle",
                    "calculation_time": self.iso_at(3),
                    "latest_provider_candle_time": self.iso_at(3),
                    "last_closed_candle_time": self.iso_at(2),
                    "candle_used_for_signal": self.iso_at(3),
                    "candle_is_closed": False,
                    "grace_period_active": False,
                }
            )
            app_module._notify_signal_change(weak_prediction)
            self.assertEqual(app_module.last_push_snapshot["actionState"], "SHORT_ACTIVE")
            self.assertNotIn("suppressed_incomplete_candle", app_module.last_push_snapshot["suppressionReasons"])

            confirmed_weak_prediction = self.wait_prediction(score=25, timestamp=self.iso_at(4), candle_timestamp=self.iso_at(4))
            confirmed_weak_prediction.update(
                {
                    "signal_snapshot_id": "incomplete-candle-confirmed",
                    "calculation_time": self.iso_at(4),
                    "latest_provider_candle_time": self.iso_at(4),
                    "last_closed_candle_time": self.iso_at(2),
                    "candle_used_for_signal": self.iso_at(4),
                    "candle_is_closed": False,
                    "grace_period_active": False,
                }
            )
            app_module._notify_signal_change(confirmed_weak_prediction)

        self.assertEqual(mocked_push.call_count, 1)
        self.assertEqual(mocked_push.call_args.args[0], "XAU/USD signal paused")
        self.assertEqual(app_module.last_push_snapshot["actionState"], "WAIT")

    def test_single_snapshot_blocker_does_not_pause_active_trade(self):
        app_module.last_push_snapshot = self.wait_snapshot()

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=72, timestamp=self.iso_at(1)))
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=72, timestamp=self.iso_at(2)))
            mocked_push.reset_mock()
            app_module._notify_signal_change(self.wait_prediction(score=35, blockers=["temporary blocker"], timestamp=self.iso_at(3)))

        mocked_push.assert_not_called()
        self.assertEqual(app_module.last_push_snapshot["actionState"], "SHORT_ACTIVE")

    def test_blockers_cleared_requires_confirmation_window(self):
        app_module.last_push_snapshot = self.wait_snapshot()

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(self.wait_prediction(score=35, blockers=["confirmed blocker"], timestamp=self.iso_at(1)))
            app_module._notify_signal_change(self.wait_prediction(score=35, blockers=["confirmed blocker"], timestamp=self.iso_at(2)))
            mocked_push.reset_mock()
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=72, timestamp=self.iso_at(3)))

        mocked_push.assert_not_called()
        self.assertEqual(app_module.last_push_snapshot["actionState"], "WAIT")
        self.assertTrue(app_module.last_push_snapshot["hasBlockers"])

    def test_duplicate_snapshot_is_ignored_for_confirmation(self):
        app_module.last_push_snapshot = self.wait_snapshot()
        timestamp = self.iso_at(1)

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=72, timestamp=timestamp))
            app_module._notify_signal_change(self.active_prediction("LONG_ACTIVE", score=72, timestamp=timestamp))

        mocked_push.assert_not_called()
        self.assertEqual(app_module.last_push_snapshot["actionState"], "WAIT")

    def test_stale_snapshot_is_ignored_for_confirmation(self):
        app_module.last_push_snapshot = self.wait_snapshot()
        stale_timestamp = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()

        with mock.patch.object(app_module, "_send_web_push_notification") as mocked_push:
            app_module._notify_signal_change(self.active_prediction("SHORT_ACTIVE", score=72, timestamp=stale_timestamp))

        mocked_push.assert_not_called()
        self.assertEqual(app_module.last_push_snapshot["actionState"], "WAIT")

    def test_web_push_config_derives_public_key_when_env_key_is_mismatched(self):
        private_key = app_module.ec.generate_private_key(app_module.ec.SECP256R1())
        private_key_b64 = app_module._urlsafe_b64encode(
            private_key.private_numbers().private_value.to_bytes(32, "big")
        )
        expected_public_key = app_module._public_key_from_private_key(private_key)

        with tempfile.TemporaryDirectory() as temp_dir:
            pem_path = str(Path(temp_dir) / "vapid_private.pem")
            with mock.patch.dict(
                os.environ,
                {
                    "WEB_PUSH_VAPID_PRIVATE_KEY": private_key_b64,
                    "WEB_PUSH_VAPID_PUBLIC_KEY": "mismatched-public-key",
                },
                clear=False,
            ):
                with mock.patch.object(app_module, "WEB_PUSH_VAPID_PEM_PATH", pem_path):
                    payload = app_module._ensure_web_push_keys()

            self.assertEqual(payload["publicKey"], expected_public_key)
            self.assertEqual(Path(pem_path).stat().st_mode & 0o777, 0o600)

    def test_web_push_config_accepts_legacy_vapid_env_names(self):
        private_key = app_module.ec.generate_private_key(app_module.ec.SECP256R1())
        private_key_b64 = app_module._urlsafe_b64encode(
            private_key.private_numbers().private_value.to_bytes(32, "big")
        )
        expected_public_key = app_module._public_key_from_private_key(private_key)

        with tempfile.TemporaryDirectory() as temp_dir:
            pem_path = str(Path(temp_dir) / "vapid_private.pem")
            with mock.patch.dict(
                os.environ,
                {
                    "WEB_PUSH_SUBJECT": "",
                    "WEB_PUSH_VAPID_PRIVATE_KEY": "",
                    "WEB_PUSH_VAPID_PUBLIC_KEY": "",
                    "VAPID_CLAIMS_SUBJECT": "mailto:legacy@example.com",
                    "VAPID_PRIVATE_KEY": private_key_b64,
                    "VAPID_PUBLIC_KEY": expected_public_key,
                },
                clear=False,
            ):
                with mock.patch.object(app_module, "WEB_PUSH_VAPID_PEM_PATH", pem_path):
                    payload = app_module._ensure_web_push_keys()

            self.assertEqual(payload["source"], "env")
            self.assertEqual(payload["publicKey"], expected_public_key)
            self.assertEqual(payload["subject"], "mailto:legacy@example.com")

    def test_bad_jwt_push_failure_is_treated_as_stale_subscription(self):
        class Response:
            status_code = 403
            text = '{"reason": "Bad JwtToken"}'

        exc = Exception("Push failed")
        exc.response = Response()

        self.assertTrue(app_module._is_stale_push_failure(exc))

    def test_web_push_payload_includes_authoritative_signal_version(self):
        sample_subscription = {
            "endpoint": "https://example.com/subscriptions/device-1",
            "expirationTime": None,
            "keys": {
                "p256dh": "sample-p256dh-key",
                "auth": "sample-auth-key",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            subscriptions_path = str(Path(temp_dir) / "push_subscriptions.json")
            push_config = {
                "available": True,
                "workerPath": "/notification-sw.js",
                "vapidPublicKey": "sample-public-key",
                "subject": "mailto:test@example.com",
                "privateKeyPath": str(Path(temp_dir) / "test-private.pem"),
            }
            with mock.patch.object(app_module, "WEB_PUSH_SUBSCRIPTIONS_PATH", subscriptions_path):
                app_module._save_push_subscriptions([sample_subscription])
                with mock.patch.object(app_module, "get_web_push_config", return_value=push_config):
                    with mock.patch.object(app_module, "webpush") as mocked_webpush:
                        app_module._send_web_push_notification("title", "body", "success")

        payload = json.loads(mocked_webpush.call_args.kwargs["data"])
        self.assertEqual(payload["version"], app_module.SIGNAL_PUSH_PAYLOAD_VERSION)
        self.assertEqual(payload["maxAgeSeconds"], app_module.SIGNAL_PUSH_MAX_AGE_SECONDS)
        self.assertIsNotNone(datetime.fromisoformat(payload["createdAt"]))

    def test_notification_subscription_routes_roundtrip(self):
        sample_subscription = {
            "endpoint": "https://example.com/subscriptions/device-1",
            "expirationTime": None,
            "keys": {
                "p256dh": "sample-p256dh-key",
                "auth": "sample-auth-key",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            subscriptions_path = str(Path(temp_dir) / "push_subscriptions.json")
            push_config = {
                "available": True,
                "workerPath": "/notification-sw.js",
                "vapidPublicKey": "sample-public-key",
                "subject": "mailto:test@example.com",
                "privateKeyPath": str(Path(temp_dir) / "test-private.pem"),
            }

            with mock.patch.object(
                app_module,
                "WEB_PUSH_SUBSCRIPTIONS_PATH",
                subscriptions_path,
            ):
                with mock.patch.object(
                    app_module,
                    "get_web_push_config",
                    return_value=push_config,
                ):
                    subscribe_response = self.client.post(
                        "/api/notifications/subscribe",
                        json={"subscription": sample_subscription},
                    )
                    duplicate_response = self.client.post(
                        "/api/notifications/subscribe",
                        json={"subscription": sample_subscription},
                    )
                    unsubscribe_response = self.client.post(
                        "/api/notifications/unsubscribe",
                        json={"endpoint": sample_subscription["endpoint"]},
                    )

        self.assertEqual(subscribe_response.status_code, 200)
        self.assertTrue(subscribe_response.get_json().get("ok"))
        self.assertEqual(subscribe_response.get_json().get("subscriberCount"), 1)
        self.assertEqual(duplicate_response.status_code, 200)
        self.assertEqual(duplicate_response.get_json().get("subscriberCount"), 1)
        self.assertEqual(unsubscribe_response.status_code, 200)
        self.assertEqual(unsubscribe_response.get_json().get("subscriberCount"), 0)

    def test_notification_subscription_rejects_non_https_endpoint(self):
        sample_subscription = {
            "endpoint": "javascript:alert(1)",
            "expirationTime": None,
            "keys": {
                "p256dh": "sample-p256dh-key",
                "auth": "sample-auth-key",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            subscriptions_path = str(Path(temp_dir) / "push_subscriptions.json")
            push_config = {
                "available": True,
                "workerPath": "/notification-sw.js",
                "vapidPublicKey": "sample-public-key",
                "subject": "mailto:test@example.com",
                "privateKeyPath": str(Path(temp_dir) / "test-private.pem"),
            }

            with mock.patch.object(
                app_module,
                "WEB_PUSH_SUBSCRIPTIONS_PATH",
                subscriptions_path,
            ):
                with mock.patch.object(
                    app_module,
                    "get_web_push_config",
                    return_value=push_config,
                ):
                    response = self.client.post(
                        "/api/notifications/subscribe",
                        json={"subscription": sample_subscription},
                    )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(Path(subscriptions_path).exists())

    def test_health_endpoint_reports_healthy_after_prediction(self):
        frame = self.build_frame(volume=0.0)

        with mock.patch.object(app_module, "fetch_xauusd_data", return_value=frame.copy()):
            with mock.patch.object(app_module, "fetch_live_price", return_value=2500.12):
                app_module.generate_prediction()

        response = self.client.get("/api/health")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload.get("status"), "healthy")
        self.assertIsNone(payload.get("error"))

    def test_health_endpoint_does_not_refresh_stale_cache_on_request(self):
        stale_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        app_module.latest_prediction = {
            "verdict": "Bearish",
            "confidence": 88,
            "timestamp": stale_time.isoformat(),
            "lastUpdate": stale_time.isoformat(),
            "dataSource": "Twelve Data",
            "symbol": app_module.DEFAULT_SYMBOL,
        }
        app_module.last_update = stale_time
        app_module.error_state = None

        with mock.patch.object(app_module, "generate_prediction") as mocked_generate:
            response = self.client.get("/api/health")

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload.get("status"), "stale")
        mocked_generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()

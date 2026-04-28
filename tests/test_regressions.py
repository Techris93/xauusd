import math
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DISABLE_PREDICTION_SCHEDULER", "1")

import app as app_module
from signal_engine import prepare_data


class PredictorRegressionTests(unittest.TestCase):
    def setUp(self):
        app_module.latest_prediction = None
        app_module.last_update = None
        app_module.error_state = None
        app_module.last_push_snapshot = None
        self.client = app_module.app.test_client()

    def tearDown(self):
        app_module.latest_prediction = None
        app_module.last_update = None
        app_module.error_state = None
        app_module.last_push_snapshot = None

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

        def fake_generate_prediction():
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
        mocked_generate.assert_called_once()

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

    def test_notification_config_route_returns_metadata(self):
        response = self.client.get("/api/notifications/config")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("pushAvailable", payload)
        self.assertEqual(payload.get("workerPath"), "/notification-sw.js")
        if payload.get("pushAvailable"):
            self.assertTrue(payload.get("vapidPublicKey"))

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


if __name__ == "__main__":
    unittest.main()
# XAUUSD Fixed Predictor

This repository is the separate Render deployment for the streamlined fixed XAUUSD predictor. It keeps the simplified signal engine from the fixed bundle, but the market data source is now Twelve Data instead of Yahoo Finance.

## What This Repo Does

- Runs the fixed predictor as a standalone Flask web app
- Pulls XAU/USD candles from Twelve Data
- Refreshes predictions on a background schedule
- Exposes:
  - `/` for the dashboard
  - `/api/prediction` for the current signal payload
  - `/api/health` for Render health checks

## Data Source

The app uses Twelve Data for:

- Historical candles via `time_series`
- Optional live spot price via `price`

Default symbol:

```text
XAU/USD
```

## Environment Variables

Required:

- `TWELVE_DATA_API_KEY`

Recommended:

- `TWELVE_DATA_SYMBOL=XAU/USD`
- `PREDICTOR_INTERVAL=1h`
- `PREDICTOR_PERIOD=5d`
- `PREDICTION_REFRESH_SECONDS=5`
- `TZ=UTC`
- `PYTHON_VERSION=3.11.11`
- `WEB_PUSH_SUBJECT=mailto:alerts@example.com`
- `WEB_PUSH_VAPID_PUBLIC_KEY` and `WEB_PUSH_VAPID_PRIVATE_KEY` for stable push subscriptions across redeploys

Compatibility note:

- `PREDICTION_REFRESH_MINUTES` is still accepted as a legacy alias, but it is now interpreted in seconds.

## Background Push Alerts

The dashboard now supports true Web Push alerts for important signal changes.

- Open the dashboard once and click `Enable Push Alerts`.
- After the browser grants permission, the server stores a device subscription and can push alerts even when the page is closed.
- If `WEB_PUSH_VAPID_*` is not set, the app generates runtime keys automatically. That is enough to get started, but subscriptions may need to be re-enabled after a redeploy.
- Push subscriptions are accepted only for HTTPS push-service endpoints, and runtime push key/subscription files are written with owner-only permissions.
- Background pushes are sent only when the server detects an important signal change. They are not sent on every 5-second refresh.
- The prediction scheduler runs inside the web service process. If your host sleeps inactive web services, the scheduler cannot poll Twelve Data or send push alerts until the service wakes up again.

Browser note:

- Background push depends on browser support for service workers, push, and site notifications.

## Local Run

```bash
pip install -r requirements.txt
export TWELVE_DATA_API_KEY=your_key_here
python app.py
```

For stable production push alerts, also set:

```bash
export WEB_PUSH_SUBJECT=mailto:alerts@example.com
export WEB_PUSH_VAPID_PUBLIC_KEY=your_public_vapid_key
export WEB_PUSH_VAPID_PRIVATE_KEY=your_private_vapid_key
```

If the public VAPID key is omitted or does not match the private key, the app derives the public key from the private key so browser subscriptions and server pushes stay paired.

## Render Setup

Use the GitHub repo you created:

```text
https://github.com/Techris93/xauusd
```

### Option A: Manual Render Setup

1. Push this repo to GitHub.
2. Sign in to Render.
3. Click `New +`.
4. Click `Web Service`.
5. Connect your GitHub account if it is not already connected.
6. Select the repo `Techris93/xauusd`.
7. Set `Name` to something like `xauusd-fixed`.
8. Set `Branch` to `main`.
9. Set `Runtime` to `Python 3`.
10. Set `Region` to the one closest to you.
11. Set `Build Command` to:

```bash
pip install -r requirements.txt
```

12. Set `Start Command` to:

```bash
gunicorn --workers 1 --threads 4 --timeout 120 app:app
```

13. In `Environment Variables`, add:

```text
TWELVE_DATA_API_KEY = your real Twelve Data API key
TWELVE_DATA_SYMBOL = XAU/USD
PREDICTOR_INTERVAL = 1h
PREDICTOR_PERIOD = 5d
PREDICTION_REFRESH_SECONDS = 5
TZ = UTC
PYTHON_VERSION = 3.11.11
WEB_PUSH_SUBJECT = mailto:alerts@example.com
WEB_PUSH_VAPID_PUBLIC_KEY = optional but recommended stable public VAPID key
WEB_PUSH_VAPID_PRIVATE_KEY = optional but recommended stable private VAPID key
```

14. Set the health check path to:

```text
/api/health
```

15. Click `Create Web Service`.
16. Wait for the first deploy to finish.
17. Open the live URL and verify:

```text
/
/api/health
/api/prediction
```

### Option B: Blueprint Setup

This repo also includes `render.yaml`.

1. In Render, click `New +`.
2. Click `Blueprint`.
3. Choose `Techris93/xauusd`.
4. Render will read `render.yaml` and prefill the service settings.
5. Create the service.
6. After the service is created, add `TWELVE_DATA_API_KEY` manually in the service environment tab.
7. Redeploy once after adding the API key.

## Files

- `app.py` - standalone Flask app using Twelve Data
- `signal_engine.py` - streamlined fixed predictor logic
- `templates/dashboard.html` - dashboard UI
- `static/notification-sw.js` - service worker for background push delivery
- `requirements.txt` - Python dependencies
- `Procfile` - Render-compatible process command
- `render.yaml` - optional Blueprint config

## Notes

- This is intended to be a second, separate predictor from the main `gold-predictor` app.
- The app will stay up even if the API key is missing, but `/api/prediction` and `/api/health` will show the error until the key is added.
- The dashboard refreshes from the cached prediction payload; it does not hit Twelve Data on every browser refresh.
- `/api/health` reports current cached state for Render and does not force an on-demand Twelve Data refresh. `/api/prediction` refreshes stale predictions on demand.

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
- `PREDICTION_REFRESH_MINUTES=5`
- `TZ=UTC`
- `PYTHON_VERSION=3.11.11`

## Local Run

```bash
pip install -r requirements.txt
export TWELVE_DATA_API_KEY=your_key_here
python app.py
```

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
PREDICTION_REFRESH_MINUTES = 5
TZ = UTC
PYTHON_VERSION = 3.11.11
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
- `requirements.txt` - Python dependencies
- `Procfile` - Render-compatible process command
- `render.yaml` - optional Blueprint config

## Notes

- This is intended to be a second, separate predictor from the main `gold-predictor` app.
- The app will stay up even if the API key is missing, but `/api/prediction` and `/api/health` will show the error until the key is added.
- The dashboard refreshes from the cached prediction payload; it does not hit Twelve Data on every browser refresh.

# TradePatterns

![Python](https://img.shields.io/badge/python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009485)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Machine-learning candlestick pattern detector and backtesting platform. A Random Forest classifier identifies five classical reversal/indecision patterns from OHLC data, and a browser-based simulator turns those signals into a tradeable strategy with full performance attribution against a buy-and-hold benchmark.

## Screenshot

![TradePatterns UI](docs/screenshot.png)

> _Replace `docs/screenshot.png` with a capture of the running UI._

## Features

- **5 classical candlestick patterns**: Hammer, Doji, Bullish Engulfing, Bearish Engulfing, Shooting Star
- **Random Forest classifier**: 200 trees, 11 engineered features (body/wick ratios, momentum, prior-candle context), balanced class weights
- **FastAPI backend**: typed Pydantic request/response, JSON endpoints, OpenAPI docs at `/docs`
- **Live market data**: ticker fetched on-demand via [yfinance](https://github.com/ranaroussi/yfinance)
- **Plotly candlestick UI**: interactive chart with pattern markers and a sortable detection table
- **Backtesting simulator**: pattern-driven strategy vs buy-and-hold, with KPIs (AUM, ROI, Sharpe, Tracking Error, Active Share, Max Drawdown, Win Rate), equity curve, drawdown chart, and full trade log

## Project Structure

```
tradepatterns/
├── app/
│   ├── main.py             # FastAPI application entrypoint
│   ├── detector.py         # Feature engineering + RF inference
│   ├── model.pkl           # Trained Random Forest (joblib)
│   ├── model_meta.json     # Feature columns + pattern label map
│   ├── __init__.py
│   └── static/
│       └── index.html      # Single-page UI (Plotly + vanilla JS)
├── notebook/               # Training notebooks (exploration, evaluation)
├── requirements.txt
├── README.md
├── CHANGELOG.md
└── .gitignore
```

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```

Then open `http://127.0.0.1:8000/`.

## API Endpoints

| Method | Endpoint       | Description                                                      |
|--------|----------------|------------------------------------------------------------------|
| GET    | `/`            | Serves the single-page web UI                                    |
| POST   | `/detect`      | Detect patterns in OHLC data — body: `{ticker, start_date, end_date}` |
| GET    | `/health`      | Liveness probe — returns `{status: "ok"}`                        |
| GET    | `/model-info`  | Returns model metadata (feature columns + pattern label map)     |

OpenAPI/Swagger docs are auto-generated at `/docs`.

## Version History

| Version | Description                                                | Accuracy           |
|---------|------------------------------------------------------------|--------------------|
| v1.0    | Baseline Random Forest, 5 patterns, FastAPI + Plotly UI    | 93%                |
| v1.1    | 19 patterns, HGB model, MA features                        | 78% acc, 0.65 F1   |

## Tech Stack

- **ML / Data**: scikit-learn, NumPy, pandas, joblib
- **Market data**: yfinance
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: vanilla JS, Plotly.js, single-file HTML/CSS (Share Tech Mono / Barlow)
- **Tooling**: Python 3.13, Jupyter for exploration

## Author

**Axel Peluso** — Atlantis University, Winter 2nd Term 2026
Final project, Machine Learning course.

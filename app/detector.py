"""
Inference path for the v1.1 candlestick pattern model.

The classifier itself is loaded by joblib and treated as opaque — no class
import is needed at inference time. The trained model is a
HistGradientBoostingClassifier (v1.1, replacing v1.0's RandomForestClassifier);
the switch was driven by RF being unable to satisfy the v1.1 recall targets
on minority pattern classes.

Input  : ticker, start_date, end_date (user-facing window)
Output : per-day candle records and detected pattern events with class,
         family, and typical candle count.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
META_PATH  = os.path.join(BASE_DIR, "model_meta.json")

# Trained model (HistGradientBoostingClassifier in v1.1).
rf_model = joblib.load(MODEL_PATH)

with open(META_PATH, "r") as f:
    model_meta = json.load(f)

FEATURE_COLS  = model_meta["feature_cols"]
PATTERN_NAMES = {int(k): v for k, v in model_meta["pattern_names"].items()}
PATTERN_DEFS  = model_meta.get("pattern_defs", [])

# Per-class metadata derived from PATTERN_DEFS: first-defined wins.
_CLASS_FAMILY  = {}
_CLASS_CANDLES = {}
for _pd in PATTERN_DEFS:
    _cid = int(_pd["class_id"])
    if _cid not in _CLASS_FAMILY:
        _CLASS_FAMILY[_cid]  = _pd["family"]
        _CLASS_CANDLES[_cid] = int(_pd["candle_count"])

# 200MA needs ~200 trading days of warmup; 300 calendar days covers it.
MA_WARMUP_DAYS = 300


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars with a 300-calendar-day warmup *before* the user's
    start so that 200-day MA features are populated for the very first
    requested day. The full fetched range is returned; trimming back to the
    user range happens in detect_patterns().
    """
    start_dt    = pd.to_datetime(start)
    fetch_start = (start_dt - pd.Timedelta(days=MA_WARMUP_DAYS)).strftime("%Y-%m-%d")

    df = yf.download(ticker, start=fetch_start, end=end,
                     progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["Ticker"] = ticker.upper()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 23 v1.1 features used by the trained model.
    Mirrors the notebook training pipeline exactly; column names and order
    must match model_meta['feature_cols'].
    """
    df = df.copy()

    # --- v1.0 candle anatomy + ratios (11 features) ---
    df["body"]         = abs(df["Close"] - df["Open"])
    df["upper_wick"]   = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["lower_wick"]   = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["candle_range"] = df["High"] - df["Low"]

    df["body_ratio"]       = df["body"]       / df["candle_range"].replace(0, np.nan)
    df["upper_wick_ratio"] = df["upper_wick"] / df["candle_range"].replace(0, np.nan)
    df["lower_wick_ratio"] = df["lower_wick"] / df["candle_range"].replace(0, np.nan)

    df["bullish"]       = (df["Close"] > df["Open"]).astype(int)
    df["body_position"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) \
                          / df["candle_range"].replace(0, np.nan)

    df["pct_change_1"] = df["Close"].pct_change(1)
    df["pct_change_3"] = df["Close"].pct_change(3)

    df["prev_bullish"]      = df["bullish"].shift(1)
    df["prev_body_ratio"]   = df["body_ratio"].shift(1)
    df["prev_body"]         = df["body"].shift(1)
    df["prev_candle_range"] = df["candle_range"].shift(1)

    # --- v1.1: normalized MA features (8 features) ---
    ma20  = df["Close"].rolling(20).mean()
    ma50  = df["Close"].rolling(50).mean()
    ma200 = df["Close"].rolling(200).mean()
    df["close_to_ma20"]  = (df["Close"] / ma20)  - 1.0
    df["close_to_ma50"]  = (df["Close"] / ma50)  - 1.0
    df["close_to_ma200"] = (df["Close"] / ma200) - 1.0
    df["above_ma20"]  = (df["Close"] > ma20).astype(int)
    df["above_ma50"]  = (df["Close"] > ma50).astype(int)
    df["above_ma200"] = (df["Close"] > ma200).astype(int)
    df["ma20_slope"]  = ma20.pct_change(5)
    df["ma50_slope"]  = ma50.pct_change(5)

    # --- v1.1: lag-2 features for 3-candle pattern context (4 features) ---
    df["prev2_bullish"]      = df["bullish"].shift(2)
    df["prev2_body_ratio"]   = df["body_ratio"].shift(2)
    df["prev2_body"]         = df["body"].shift(2)
    df["prev2_candle_range"] = df["candle_range"].shift(2)

    df = df.dropna()
    return df


def detect_patterns(ticker: str, start: str, end: str) -> dict:
    raw   = fetch_data(ticker, start, end)
    feats = engineer_features(raw)

    # Trim back to the user's requested window (drops MA warmup rows).
    start_dt   = pd.to_datetime(start)
    raw_user   = raw[raw.index >= start_dt]
    feats_user = feats[feats.index >= start_dt]

    if feats_user.empty:
        return {
            "ticker": ticker.upper(), "start": start, "end": end,
            "candles": [], "patterns": [], "summary": {},
            "total_candles": 0, "total_patterns": 0,
            "model_version": model_meta.get("version", "unknown"),
        }

    X = feats_user[FEATURE_COLS]
    predictions = rf_model.predict(X)

    candles = []
    for date, row in raw_user.iterrows():
        candles.append({
            "date":   date.strftime("%Y-%m-%d"),
            "open":   round(float(row["Open"]),  4),
            "high":   round(float(row["High"]),  4),
            "low":    round(float(row["Low"]),   4),
            "close":  round(float(row["Close"]), 4),
            "volume": int(row["Volume"]),
        })

    patterns = []
    for date, pred in zip(feats_user.index, predictions):
        cid = int(pred)
        if cid != 0:
            patterns.append({
                "date":           date.strftime("%Y-%m-%d"),
                "pattern_id":     cid,
                "pattern_name":   PATTERN_NAMES.get(cid, f"class_{cid}"),
                "pattern_family": _CLASS_FAMILY.get(cid, "Unknown"),
                "candle_count":   _CLASS_CANDLES.get(cid, 1),
            })

    summary = {}
    for p in patterns:
        summary[p["pattern_name"]] = summary.get(p["pattern_name"], 0) + 1

    return {
        "ticker":         ticker.upper(),
        "start":          start,
        "end":            end,
        "candles":        candles,
        "patterns":       patterns,
        "summary":        summary,
        "total_candles":  len(candles),
        "total_patterns": len(patterns),
        "model_version":  model_meta.get("version", "unknown"),
    }

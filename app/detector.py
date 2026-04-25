import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import json
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
META_PATH  = os.path.join(BASE_DIR, "model_meta.json")

rf_model = joblib.load(MODEL_PATH)

with open(META_PATH, "r") as f:
    model_meta = json.load(f)

FEATURE_COLS  = model_meta["feature_cols"]
PATTERN_NAMES = {int(k): v for k, v in model_meta["pattern_names"].items()}

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end,
                     progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["Ticker"] = ticker.upper()
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    df = df.dropna()
    return df

def detect_patterns(ticker: str, start: str, end: str) -> dict:
    raw   = fetch_data(ticker, start, end)
    feats = engineer_features(raw)
    X     = feats[FEATURE_COLS]
    predictions = rf_model.predict(X)

    candles = []
    for date, row in raw.iterrows():
        candles.append({
            "date":   date.strftime("%Y-%m-%d"),
            "open":   round(float(row["Open"]),  4),
            "high":   round(float(row["High"]),  4),
            "low":    round(float(row["Low"]),   4),
            "close":  round(float(row["Close"]), 4),
            "volume": int(row["Volume"])
        })

    patterns = []
    for date, pred in zip(feats.index, predictions):
        if pred != 0:
            patterns.append({
                "date":         date.strftime("%Y-%m-%d"),
                "pattern_id":   int(pred),
                "pattern_name": PATTERN_NAMES[int(pred)]
            })

    summary = {}
    for p in patterns:
        name = p["pattern_name"]
        summary[name] = summary.get(name, 0) + 1

    return {
        "ticker":          ticker.upper(),
        "start":           start,
        "end":             end,
        "candles":         candles,
        "patterns":        patterns,
        "summary":         summary,
        "total_candles":   len(candles),
        "total_patterns":  len(patterns)
    }
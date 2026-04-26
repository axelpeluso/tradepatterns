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
import pandas_ta as ta
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


def compute_indicators(df: pd.DataFrame) -> dict:
    """
    v2.0 inference-layer context. Computes momentum + volume indicators
    on the engineered DataFrame and returns the most recent values plus
    human-readable signal labels. These are NOT classifier inputs — adding
    them to the model regressed metrics in the v2.0 experiment, so they
    live here as supporting context only.
    """
    result = {}

    # Stochastics %K / %D
    stoch = ta.stoch(df["High"], df["Low"], df["Close"])
    k = float(stoch.iloc[-1, 0])
    d = float(stoch.iloc[-1, 1])
    result["stoch_k"] = round(k, 2)
    result["stoch_d"] = round(d, 2)
    result["stoch_signal"] = (
        "Oversold"   if k < 20 else
        "Overbought" if k > 80 else
        "Neutral"
    )

    # ADX (trend strength, direction-agnostic)
    adx = ta.adx(df["High"], df["Low"], df["Close"])
    a = float(adx.iloc[-1, 0])
    result["adx"] = round(a, 2)
    result["adx_signal"] = "Strong trend" if a > 25 else "Weak trend"

    # Volume vs 20-day average
    vol_ratio = float(df["Volume"].iloc[-1] /
                      df["Volume"].rolling(20).mean().iloc[-1])
    result["volume_ratio"] = round(vol_ratio, 2)
    result["volume_signal"] = (
        "High volume"   if vol_ratio > 1.5 else
        "Low volume"    if vol_ratio < 0.7 else
        "Normal volume"
    )

    # Squeeze momentum — pandas-ta returns SQZ_ON / SQZ_OFF / SQZ_NO booleans
    # plus the momentum value column SQZ_20_2.0_20_1.5. Look up SQZ_ON
    # explicitly because the column-name filter in the original spec would
    # match the momentum value, not the on/off flag.
    try:
        sq = ta.squeeze(df["High"], df["Low"], df["Close"])
        if sq is None or "SQZ_ON" not in sq.columns:
            raise ValueError("squeeze unavailable")
        result["squeeze_active"] = bool(int(sq["SQZ_ON"].iloc[-1]) == 1)
        result["squeeze_signal"] = (
            "Squeeze ON — breakout imminent"
            if result["squeeze_active"]
            else "No squeeze"
        )
    except Exception:
        result["squeeze_active"] = False
        result["squeeze_signal"] = "N/A"

    return result


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

    # v2.0: indicator context — computed on the full warmed-up `feats` so
    # rolling windows are valid even when the user requests a narrow range.
    try:
        indicators = compute_indicators(feats)
    except Exception as exc:
        indicators = {"error": f"indicator computation failed: {exc}"}

    return {
        "ticker":         ticker.upper(),
        "start":          start,
        "end":            end,
        "candles":        candles,
        "patterns":       patterns,
        "summary":        summary,
        "indicators":     indicators,
        "total_candles":  len(candles),
        "total_patterns": len(patterns),
        "model_version":  model_meta.get("version", "unknown"),
    }

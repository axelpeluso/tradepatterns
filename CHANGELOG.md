# Changelog

All notable changes to this project are documented in this file.

## [v1.1] - 2026-04-25

### Model
- Switched from RandomForestClassifier to HistGradientBoostingClassifier
- Reason: RF with custom weights could not simultaneously hit None recall > 0.80 AND multi-candle recall > 0.50
- HGB passed all four targets cleanly
- Accuracy: 0.7833, Macro F1: 0.6460
- Expanded from 5 to 19 candlestick patterns
- Labels now generated via pandas-ta CDL functions backed by TA-Lib (ground truth)
- Features expanded from 11 to 23
- Added normalized 20/50/200 MA features (`close_to_ma*`, `above_ma*`, `ma*_slope`)
- Added lag-2 features for 3-candle pattern context
- 19 patterns grouped into 6 classification classes (None, Doji family, Bullish/Bearish single-candle, Bullish/Bearish multi-candle); pattern_name and candle_count preserved in metadata

### Patterns Added
- Dragonfly Doji, Gravestone Doji
- Inverted Hammer, Hanging Man, Bullish/Bearish Marubozu
- Bullish/Bearish Harami, Piercing Line, Dark Cloud Cover
- Morning Star, Evening Star
- Three White Soldiers, Three Black Crows

### Results vs v1.0
- Macro F1: 0.9346 → 0.6460 (raw drop, but v1.0 only had 5 easy classes; v1.1 has 6 harder classes with imbalance)
- Multi-candle recall: collapsed in attempt 1, recovered to 0.56-0.60 with HGB
- "None" class precision 0.84, recall 0.87 — model rarely hallucinates patterns

### API
- `/detect` response now includes `pattern_family`, `candle_count`, and `model_version`

## [v1.0] - 2026-04-21

### Model
- Random Forest, 200 trees, max_depth=10, balanced class weights
- 11 engineered features: body ratios, wick ratios, momentum, previous candle context
- Time-ordered train/test split (2020-2023 train, 2024 test)

### Patterns Detected
- Hammer, Doji, Bullish Engulfing, Bearish Engulfing, Shooting Star

### Results
- Overall accuracy: 93%
- Doji: 100% precision/recall
- Hammer: 100% precision, 94% recall

### API
- FastAPI with `/detect`, `/health`, `/model-info` endpoints

### UI
- Interactive Plotly candlestick chart with pattern markers
- Backtesting simulator with AUM, ROI, Sharpe Ratio, Tracking Error, Active Share KPIs

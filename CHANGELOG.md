# Changelog

All notable changes to this project are documented in this file.

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

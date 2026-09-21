# MarketLens-ML

Machine-learning market research platform for probabilistic trend forecasting, regime detection, walk-forward validation, and backtesting.

## Initial scope

MarketLens starts with SPY and VOO and predicts the probability of a positive forward 5-trading-day return. The research pipeline is designed to avoid look-ahead leakage and compare ML results against buy-and-hold.

## Models

- Logistic Regression baseline
- Random Forest
- Gradient boosting (XGBoost when installed)

## Features

Returns, moving-average distance, momentum, RSI, realized volatility, drawdown, volume changes, and related lagged indicators.

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.train
```

This project is for research and decision support, not guaranteed investment returns.

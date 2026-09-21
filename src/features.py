from __future__ import annotations
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features using only information available at or before each row."""
    x = df.copy()
    x["ret_1d"] = x["Close"].pct_change()
    for n in (5, 10, 20, 60):
        x[f"ret_{n}d"] = x["Close"].pct_change(n)
    for n in (10, 20, 50, 200):
        ma = x["Close"].rolling(n).mean()
        x[f"ma_dist_{n}"] = x["Close"] / ma - 1
    x["rsi_14"] = _rsi(x["Close"], 14)
    x["vol_20"] = x["ret_1d"].rolling(20).std() * np.sqrt(252)
    x["vol_60"] = x["ret_1d"].rolling(60).std() * np.sqrt(252)
    x["volume_change_5"] = x["Volume"].pct_change(5)
    x["drawdown_252"] = x["Close"] / x["Close"].rolling(252).max() - 1
    return x.replace([np.inf, -np.inf], np.nan)


def add_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    out = df.copy()
    out["forward_return"] = out["Close"].shift(-horizon) / out["Close"] - 1
    out["target"] = (out["forward_return"] > 0).astype(int)
    # Last horizon rows have unknown outcomes and must not enter training.
    out.loc[out["forward_return"].isna(), "target"] = np.nan
    return out

from __future__ import annotations
import pandas as pd


def classify_regime(df: pd.DataFrame) -> pd.Series:
    """Simple interpretable regime label based on trend and realized volatility."""
    close = df["Close"]
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    ret = close.pct_change()
    vol20 = ret.rolling(20).std() * (252 ** 0.5)
    vol252 = ret.rolling(252).std() * (252 ** 0.5)
    trend = (close > ma200) & (ma50 > ma200)
    high_vol = vol20 > vol252
    labels = pd.Series("Unclassified", index=df.index)
    labels[trend & ~high_vol] = "Uptrend / Normal Vol"
    labels[trend & high_vol] = "Uptrend / High Vol"
    labels[~trend & ~high_vol] = "Downtrend / Normal Vol"
    labels[~trend & high_vol] = "Downtrend / High Vol"
    return labels

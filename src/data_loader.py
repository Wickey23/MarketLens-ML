from __future__ import annotations
import pandas as pd
import yfinance as yf


def download_prices(ticker: str, start: str = "2010-01-01") -> pd.DataFrame:
    """Download adjusted daily OHLCV data and return a clean DataFrame."""
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No market data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()

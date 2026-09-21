from __future__ import annotations
import numpy as np
import pandas as pd


def backtest_probabilities(df: pd.DataFrame, threshold: float = 0.55, cost_bps: float = 5.0) -> pd.DataFrame:
    """Backtest precomputed out-of-sample probabilities. Never pass in-sample predictions."""
    out = df.copy()
    out["position"] = (out["probability"] >= threshold).astype(float)
    turnover = out["position"].diff().abs().fillna(out["position"])
    out["strategy_return"] = out["position"].shift(1).fillna(0) * out["asset_return"] - turnover * cost_bps / 10000
    out["strategy_equity"] = (1 + out["strategy_return"]).cumprod()
    out["buy_hold_equity"] = (1 + out["asset_return"]).cumprod()
    return out


def summary(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    equity = (1 + returns).cumprod()
    dd = equity / equity.cummax() - 1
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() else np.nan
    return {"total_return": float(equity.iloc[-1] - 1), "sharpe": float(sharpe), "max_drawdown": float(dd.min())}

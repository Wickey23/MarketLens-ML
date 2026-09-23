from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

CORE_TICKERS=["SPY","VOO","QQQ","VXUS"]

# Broad, liquid U.S. optionable universe. This stage is deliberately cheap:
# it screens underlyings first, then MarketLens performs expensive option-chain
# work only on the small selected set.
LIQUID_OPTION_UNIVERSE=[
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","AVGO","NFLX",
    "PLTR","ORCL","CRM","INTC","MU","QCOM","SOFI","JPM","BAC","GS","MS","C",
    "WFC","V","MA","XOM","CVX","COP","OXY","UNH","LLY","JNJ","PFE","MRK",
    "ABBV","WMT","COST","HD","LOW","DIS","NKE","BA","CAT","GE","F","GM",
    "UBER","COIN","IWM","DIA","XLF","XLK","XLE","SMH","TLT","GLD",
]


def _finite(value):
    try:
        x=float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _frame_for_symbol(downloaded: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    if downloaded is None or downloaded.empty:
        return None
    if isinstance(downloaded.columns,pd.MultiIndex):
        if symbol in downloaded.columns.get_level_values(0):
            out=downloaded[symbol].copy()
        elif symbol in downloaded.columns.get_level_values(-1):
            out=downloaded.xs(symbol,axis=1,level=-1).copy()
        else:
            return None
    else:
        out=downloaded.copy()
    if "Close" not in out or "Volume" not in out:
        return None
    return out.dropna(subset=["Close"]).copy()


def rank_market_universe(symbols=None, limit=12):
    """Rank underlyings for deeper options research, not for buying.

    Score rewards dollar liquidity plus current movement/volatility so the
    expensive options stage spends its budget where contracts are more likely
    to be active and informative.
    """
    symbols=list(dict.fromkeys(symbols or LIQUID_OPTION_UNIVERSE))
    try:
        data=yf.download(
            symbols,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return []

    rows=[]
    for symbol in symbols:
        frame=_frame_for_symbol(data,symbol)
        if frame is None or len(frame)<30:
            continue
        close=frame["Close"].astype(float)
        volume=frame["Volume"].fillna(0).astype(float)
        price=_finite(close.iloc[-1])
        if price is None or price<5:
            continue
        ret=close.pct_change()
        dollar_volume=_finite((close*volume).tail(20).mean()) or 0.0
        rv20=_finite(ret.tail(20).std()*math.sqrt(252)) or 0.0
        ret20=_finite(close.pct_change(20).iloc[-1]) or 0.0
        ret5=_finite(close.pct_change(5).iloc[-1]) or 0.0
        # Minimum underlying liquidity avoids wasting option-chain calls on
        # thin symbols. The downstream chain screen remains the real gate.
        if dollar_volume<100_000_000:
            continue
        score=math.log10(max(dollar_volume,1.0))+2.0*rv20+2.0*abs(ret20)+abs(ret5)
        rows.append({
            "ticker":symbol,
            "scan_score":round(score,4),
            "price":price,
            "avg_dollar_volume_20d":dollar_volume,
            "realized_vol_20d":rv20,
            "return_20d":ret20,
            "return_5d":ret5,
        })
    rows.sort(key=lambda x:x["scan_score"],reverse=True)
    return rows[:limit]


def choose_research_universe(core=None, manual=None, max_total=12):
    core=list(dict.fromkeys(core or CORE_TICKERS))
    manual=[x for x in list(dict.fromkeys(manual or [])) if x not in core]
    base=(core+manual)[:max_total]
    remaining=max(0,max_total-len(base))
    ranked=rank_market_universe(limit=max(remaining*3,remaining)) if remaining else []
    selected=list(base)
    for row in ranked:
        if row["ticker"] not in selected:
            selected.append(row["ticker"])
        if len(selected)>=max_total:
            break
    return {
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "selected":selected,
        "core":core,
        "manual":manual,
        "ranked_candidates":ranked,
        "method":"Underlying liquidity + recent movement/volatility screen; full options quality and historical evidence are evaluated later.",
    }

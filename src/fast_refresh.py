from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from math import sqrt
import numpy as np

from src.company_context import company_context
from src.data_loader import download_prices
from src.features import build_features
from src.options_data import option_snapshot
from src.regime import classify_regime

DATA_PATH=Path("data/dashboard.json")
DEFAULT_TICKERS=["SPY","VOO","QQQ","VXUS"]


def sf(x):
    try:
        v=float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def load_existing():
    if DATA_PATH.exists():
        try:
            return json.loads(DATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"generated_at":None,"horizon_days":5,"tickers":[],"errors":[]}


def options_summary(chain,spot):
    expiries=chain.get("expirations") or []
    contracts=chain.get("contracts") or []
    out={"nearest_expiration":expiries[0] if expiries else None,"atm_straddle_implied_move":None}
    if expiries and contracts and spot:
        exp=expiries[0]
        calls=[x for x in contracts if x.get("expiration")==exp and x.get("type")=="call" and x.get("mid")]
        puts=[x for x in contracts if x.get("expiration")==exp and x.get("type")=="put" and x.get("mid")]
        if calls and puts:
            c=min(calls,key=lambda x:abs((x.get("strike") or spot)-spot))
            p=min(puts,key=lambda x:abs((x.get("strike") or spot)-spot))
            out.update({"atm_straddle_implied_move":sf((c["mid"]+p["mid"])/spot),"atm_reference_strikes":[c["strike"],p["strike"]],"atm_expiration":exp})
    return out


def quick_snapshot(ticker):
    raw=download_prices(ticker,"2010-01-01")
    feat=build_features(raw)
    latest=feat.iloc[-1]
    close=raw["Close"]
    daily=close.pct_change()
    annual_rv=sf(daily.rolling(20).std().iloc[-1]*sqrt(252))
    horizons={}
    for h in [5,10,20,30,60]:
        rr=close.pct_change(h).dropna()
        horizons[str(h)]={
            "mean_return":sf(rr.mean()),"median_return":sf(rr.median()),
            "positive_rate":sf((rr>0).mean()),"p10":sf(rr.quantile(.10)),
            "p25":sf(rr.quantile(.25)),"p75":sf(rr.quantile(.75)),
            "p90":sf(rr.quantile(.90)),
            "realized_move_1sd":sf(daily.rolling(252).std().iloc[-1]*sqrt(h)),
        }
    chain=option_snapshot(ticker,float(close.iloc[-1]),annual_rv)
    try:
        ctx=company_context(ticker)
    except Exception:
        ctx={"news":[],"earnings":{},"catalyst_flags":[],"news_note":"Company context unavailable for this refresh."}
    return {
        "ticker":ticker,
        "as_of":str(raw.index[-1].date()),
        "price":sf(close.iloc[-1]),
        "change_1d":sf(daily.iloc[-1]),
        "change_5d":sf(close.pct_change(5).iloc[-1]),
        "regime":str(classify_regime(raw).iloc[-1]),
        "volatility_20d":annual_rv,
        "rsi_14":sf(latest.get("rsi_14")),
        "drawdown_252":sf(latest.get("drawdown_252")),
        "company_context":ctx,
        "options":{
            "status":"Fast market/options snapshot active",
            "horizons":horizons,
            "realized_vol_20d":annual_rv,
            "chain":chain,
            "summary":options_summary(chain,float(close.iloc[-1])),
        },
        "history":[{"date":str(i.date()),"close":sf(v)} for i,v in close.tail(260).items()],
        "fast_refreshed_at":datetime.now(timezone.utc).isoformat(),
    }


def merge(old,new):
    # Preserve expensive walk-forward evidence while replacing time-sensitive market/event/options fields.
    keep=["probability_5d_up","model_probabilities","base_up_rate","similar_setups","evidence","relative_strength","plain_language","calibration_buckets","metrics","research_refreshed_at"]
    for k in keep:
        if k in old:
            new[k]=old[k]
    return new


def main():
    requested=(os.getenv("MARKETLENS_TICKER") or "").strip().upper()
    tickers=[requested] if requested else DEFAULT_TICKERS
    p=load_existing()
    by={x.get("ticker"):x for x in p.get("tickers",[]) if x.get("ticker")}
    errors=[]
    for t in tickers:
        try:
            by[t]=merge(by.get(t,{}),quick_snapshot(t))
        except Exception as e:
            errors.append({"ticker":t,"error":str(e)})
    p["tickers"]=list(by.values())
    p["generated_at"]=datetime.now(timezone.utc).isoformat()
    p["fast_generated_at"]=p["generated_at"]
    p["errors"]=errors
    DATA_PATH.parent.mkdir(exist_ok=True)
    DATA_PATH.write_text(json.dumps(p,indent=2),encoding="utf-8")
    print(json.dumps({"refreshed":tickers,"errors":errors},indent=2))


if __name__=="__main__":
    main()

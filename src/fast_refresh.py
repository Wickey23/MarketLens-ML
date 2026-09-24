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
from src.options_data import option_snapshot, tradier_market_clock
from src.regime import classify_regime
from src.opportunity_radar import build_opportunity_radar, build_market_guidance
from src.ai_paper_trader import load_state, save_state, run_ai_paper_portfolio, performance_summary, performance_attribution, learning_profile, paper_to_real_readiness, strategy_performance_summary, forward_validation_summary, CURRENT_STRATEGY_VERSION
from src.universe_scanner import choose_research_universe

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
    contracts=chain.get("contracts") or []
    expiries=chain.get("expirations") or []
    out={
        "nearest_expiration":expiries[0] if expiries else None,
        "atm_straddle_implied_move":None,
        "implied_moves_by_expiration":{},
    }
    if not contracts or not spot:
        return out
    for exp in expiries:
        calls=[x for x in contracts if x.get("expiration")==exp and x.get("type")=="call" and x.get("mid")]
        puts=[x for x in contracts if x.get("expiration")==exp and x.get("type")=="put" and x.get("mid")]
        if not calls or not puts:
            continue
        call=min(calls,key=lambda x:abs((x.get("strike") or spot)-spot))
        put=min(puts,key=lambda x:abs((x.get("strike") or spot)-spot))
        move=sf(((call.get("mid") or 0)+(put.get("mid") or 0))/spot)
        if move is None:
            continue
        out["implied_moves_by_expiration"][exp]={
            "move":move,
            "call_strike":call.get("strike"),
            "put_strike":put.get("strike"),
        }
        if exp==out["nearest_expiration"]:
            out["atm_straddle_implied_move"]=move
            out["atm_reference_strikes"]=[call.get("strike"),put.get("strike")]
            out["atm_expiration"]=exp
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
    research_close=float(close.iloc[-1])
    chain=option_snapshot(ticker,research_close,annual_rv)
    market_spot=sf(chain.get("underlying_price")) or research_close
    underlying_quote=chain.get("underlying_quote") or {}
    previous_close=sf(underlying_quote.get("previous_close"))
    market_change_1d=(market_spot/previous_close-1) if previous_close and previous_close>0 else sf(daily.iloc[-1])
    regime_series=classify_regime(raw)
    current_regime=str(regime_series.iloc[-1])
    try:
        ctx=company_context(ticker)
    except Exception:
        ctx={"news":[],"earnings":{},"catalyst_flags":[],"news_note":"Company context unavailable for this refresh."}
    return {
        "ticker":ticker,
        "as_of":str(raw.index[-1].date()),
        "price":sf(market_spot),
        "research_close":sf(research_close),
        "change_1d":sf(market_change_1d),
        "change_5d":sf(close.pct_change(5).iloc[-1]),
        "regime":current_regime,
        "volatility_20d":annual_rv,
        "rsi_14":sf(latest.get("rsi_14")),
        "drawdown_252":sf(latest.get("drawdown_252")),
        "company_context":ctx,
        "options":{
            "status":("Execution-grade multi-provider option chain active" if chain.get("realtime") is True else "Fast market/options snapshot active"),
            "horizons":horizons,
            "realized_vol_20d":annual_rv,
            "chain":chain,
            "summary":options_summary(chain,float(market_spot)),
            "_radar_inputs":{"current_regime":current_regime},
        },
        "history":[{"date":str(i.date()),"close":sf(v)} for i,v in close.tail(260).items()],
        "fast_refreshed_at":datetime.now(timezone.utc).isoformat(),
    }


def merge(old,new,learning=None):
    # Preserve expensive walk-forward evidence while replacing time-sensitive market/event/options fields.
    keep=["probability_5d_up","model_probabilities","base_up_rate","similar_setups","evidence","relative_strength","plain_language","calibration_buckets","metrics","research_refreshed_at"]
    for k in keep:
        if k in old:
            new[k]=old[k]
    # Preserve deep-research earnings history across lightweight refreshes.
    old_earn=((old.get("company_context") or {}).get("earnings") or {})
    new_ctx=new.setdefault("company_context",{})
    new_earn=new_ctx.setdefault("earnings",{})
    for k in ("historical_moves","avg_abs_1d_move","median_abs_1d_move"):
        if new_earn.get(k) is None and old_earn.get(k) is not None:
            new_earn[k]=old_earn[k]
    try:
        raw=download_prices(new["ticker"],"2010-01-01")
        regimes=classify_regime(raw)
        chain=(new["options"].get("chain") or {})
        radar_spot=float(chain.get("underlying_price") or new["price"])
        new["options"]["opportunity_radar"]=build_opportunity_radar(raw,(chain.get("contracts") or []),regimes,new.get("regime"),new.get("evidence") or {},radar_spot,learning=learning,context=new.get("company_context") or {},relative_strength=new.get("relative_strength") or {},model_probability=new.get("probability_5d_up"),options_summary=(new.get("options") or {}).get("summary") or {})
    except Exception as exc:
        new["options"]["opportunity_radar"]={"state":"unavailable","opportunities":[],"watchlist":[],"error":str(exc)}
    return new


def scan_is_fresh(scan,max_age_hours=2.0):
    ts=(scan or {}).get("generated_at")
    if not ts:
        return False
    try:
        dt=datetime.fromisoformat(str(ts).replace("Z","+00:00"))
        if dt.tzinfo is None:
            dt=dt.replace(tzinfo=timezone.utc)
        age=(datetime.now(timezone.utc)-dt.astimezone(timezone.utc)).total_seconds()/3600.0
        return 0<=age<=max_age_hours
    except Exception:
        return False


def main():
    current_learning=learning_profile(load_state())
    requested=(os.getenv("MARKETLENS_TICKER") or "").strip().upper()
    p=load_existing()
    tracked=[x.get("ticker") for x in p.get("tickers",[]) if x.get("ticker")]
    prior_scan=set(((p.get("universe_scan") or {}).get("selected") or []))
    manual=list(dict.fromkeys(p.get("manual_tickers") or [
        t for t in tracked if t not in DEFAULT_TICKERS and t not in prior_scan
    ]))
    if requested and requested not in DEFAULT_TICKERS and requested not in manual:
        manual.append(requested)
    p["manual_tickers"]=manual[:8]
    if requested:
        tickers=[requested]
    else:
        prior_scan=p.get("universe_scan") or {}
        same_manual=list(prior_scan.get("manual") or [])==list(p["manual_tickers"])
        if scan_is_fresh(prior_scan,2.0) and same_manual and prior_scan.get("selected"):
            scan=prior_scan
            scan["reused_at"]=datetime.now(timezone.utc).isoformat()
        else:
            scan=choose_research_universe(DEFAULT_TICKERS,p["manual_tickers"],max_total=12)
        p["universe_scan"]=scan
        tickers=scan["selected"]

    by={x.get("ticker"):x for x in p.get("tickers",[]) if x.get("ticker")}
    errors=[]
    for t in tickers:
        try:
            by[t]=merge(by.get(t,{}),quick_snapshot(t),current_learning)
        except Exception as e:
            errors.append({"ticker":t,"error":str(e)})
    if requested:
        p["tickers"]=list(by.values())
    else:
        p["tickers"]=[by[t] for t in tickers if t in by]
    p["market_guidance"]=build_market_guidance(p.get("tickers") or [])
    p["generated_at"]=datetime.now(timezone.utc).isoformat()
    p["market_clock"]=tradier_market_clock()
    p["fast_generated_at"]=p["generated_at"]
    p["errors"]=errors
    ai_state=run_ai_paper_portfolio(p,load_state())
    save_state(ai_state)
    p["ai_portfolio"]={"summary":performance_summary(ai_state),"strategy_summary":strategy_performance_summary(ai_state),"readiness":paper_to_real_readiness(ai_state),"forward_validation":forward_validation_summary(ai_state),"attribution":performance_attribution(ai_state),"learning":learning_profile(ai_state),"strategy_version":CURRENT_STRATEGY_VERSION,"updated_at":ai_state.get("updated_at"),"paper_market_session_open":ai_state.get("paper_market_session_open"),"open":ai_state.get("open",[]),"closed":ai_state.get("closed",[])[-100:],"decisions":ai_state.get("decisions",[])[-100:],"equity_history":ai_state.get("equity_history",[])[-300:]}
    DATA_PATH.parent.mkdir(exist_ok=True)
    DATA_PATH.write_text(json.dumps(p,indent=2),encoding="utf-8")
    print(json.dumps({"refreshed":tickers,"universe_scan":p.get("universe_scan"),"errors":errors,"ai_portfolio":p["ai_portfolio"]["summary"]},indent=2))


if __name__=="__main__":
    main()

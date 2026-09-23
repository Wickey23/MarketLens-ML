from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from math import sqrt

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

from src.company_context import company_context
from src.data_loader import download_prices
from src.features import add_target, build_features
from src.options_data import option_snapshot
from src.regime import classify_regime
from src.opportunity_radar import build_opportunity_radar
from src.ai_paper_trader import load_state, save_state, run_ai_paper_portfolio, performance_summary, performance_attribution, learning_profile
from src.train import FEATURES
from src.walk_forward import expanding_predictions, model_library

DEFAULT_TICKERS=["SPY","VOO","QQQ","VXUS"]
HORIZON=5
DATA_PATH=Path("data/dashboard.json")


def sf(x):
    try:
        v=float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def bucket_stats(p,y,r):
    out=[]
    for lo,hi in [(0,.45),(.45,.5),(.5,.55),(.55,.6),(.6,.65),(.65,.7),(.7,1.01)]:
        m=(p>=lo)&(p<hi)
        n=int(m.sum())
        if n:
            out.append({
                "bucket":f"{int(lo*100)}-{int(min(hi,1)*100)}%",
                "n":n,
                "actual_up_rate":sf(y[m].mean()),
                "mean_forward_return":sf(r[m].mean()),
            })
    return out


def wilson(k,n,z=1.96):
    if not n:
        return [None,None]
    p=k/n
    d=1+z*z/n
    center=(p+z*z/(2*n))/d
    half=z*sqrt((p*(1-p)+z*z/(4*n))/n)/d
    return [sf(center-half),sf(center+half)]


def relative_strength(raw,ticker):
    if ticker=="SPY":
        return {"vs_spy_20d":0.0,"vs_spy_60d":0.0}
    try:
        spy=download_prices("SPY","2010-01-01")
        a=raw["Close"]
        b=spy["Close"].reindex(a.index).ffill()
        return {
            "vs_spy_20d":sf(a.pct_change(20).iloc[-1]-b.pct_change(20).iloc[-1]),
            "vs_spy_60d":sf(a.pct_change(60).iloc[-1]-b.pct_change(60).iloc[-1]),
        }
    except Exception:
        return {"vs_spy_20d":None,"vs_spy_60d":None}


def enrich_earnings_history(ctx, raw):
    earnings=(ctx.get("earnings") or {})
    close=raw["Close"]
    moves=[]
    for row in earnings.get("recent_and_upcoming") or []:
        ds=row.get("date")
        if not ds:
            continue
        try:
            d=np.datetime64(str(ds)[:10])
            idx=np.where(close.index.normalize().values.astype("datetime64[D]")>=d)[0]
            if not len(idx):
                continue
            i=int(idx[0])
            if i<1 or i>=len(close):
                continue
            before=float(close.iloc[i-1])
            after=float(close.iloc[i])
            move=(after/before)-1
            moves.append({"date":str(close.index[i].date()),"move_1d":sf(move),"abs_move_1d":sf(abs(move))})
        except Exception:
            continue
    if moves:
        earnings["historical_moves"]=moves[:8]
        earnings["avg_abs_1d_move"]=sf(np.mean([x["abs_move_1d"] for x in moves if x["abs_move_1d"] is not None]))
        earnings["median_abs_1d_move"]=sf(np.median([x["abs_move_1d"] for x in moves if x["abs_move_1d"] is not None]))
    ctx["earnings"]=earnings
    return ctx


def options_summary(chain,spot):
    contracts=chain.get("contracts") or []
    expiries=chain.get("expirations") or []
    out={"nearest_expiration":expiries[0] if expiries else None,"atm_straddle_implied_move":None}
    if not contracts or not spot:
        return out
    if expiries:
        exp=expiries[0]
        calls=[x for x in contracts if x["expiration"]==exp and x["type"]=="call" and x.get("mid")]
        puts=[x for x in contracts if x["expiration"]==exp and x["type"]=="put" and x.get("mid")]
        if calls and puts:
            c=min(calls,key=lambda x:abs((x.get("strike") or spot)-spot))
            p=min(puts,key=lambda x:abs((x.get("strike") or spot)-spot))
            if c.get("strike") is not None and p.get("strike") is not None:
                # Use the closest common/nearby strikes as a practical ATM straddle estimate.
                out["atm_straddle_implied_move"]=sf(((c["mid"] or 0)+(p["mid"] or 0))/spot)
                out["atm_reference_strikes"]=[c["strike"],p["strike"]]
                out["atm_expiration"]=exp
    return out


def plain_language(ticker,evidence,ctx,options,rel):
    notes=[]
    auc=evidence.get("mean_roc_auc")
    lift=evidence.get("historical_lift")
    if auc is None or auc<.53:
        notes.append("The directional model has not shown strong out-of-sample discrimination, so its probability should receive limited weight.")
    elif auc<.58:
        notes.append("The directional model has shown modest historical discrimination, but the signal should be confirmed by other evidence.")
    else:
        notes.append("The directional model has shown stronger historical discrimination than the current baseline models.")

    if lift is not None:
        if lift>.03:
            notes.append("Comparable historical signals finished higher more often than the ticker's normal base rate.")
        elif lift<-.03:
            notes.append("Comparable historical signals underperformed the ticker's normal base rate.")
        else:
            notes.append("Comparable historical signals were close to the ticker's normal base rate, so the current signal adds little directional evidence.")

    dte=(ctx.get("earnings") or {}).get("days_to_earnings")
    if dte is not None:
        if dte<=7:
            notes.append(f"Earnings are about {dte} days away, creating substantial event and implied-volatility risk.")
        elif dte<=30:
            notes.append(f"Earnings are about {dte} days away and should be considered when choosing an expiration.")

    rs20=rel.get("vs_spy_20d")
    if rs20 is not None and ticker!="SPY":
        direction="outperformed" if rs20>0 else "underperformed"
        notes.append(f"Over the last 20 trading days, {ticker} has {direction} SPY by about {abs(rs20)*100:.1f} percentage points.")

    if (ctx.get("catalyst_flags") or []):
        notes.append("Recent headlines contain potential catalyst themes: "+", ".join(ctx["catalyst_flags"])+". Review the underlying headlines rather than treating this as a sentiment score.")

    hist_move=(ctx.get("earnings") or {}).get("avg_abs_1d_move")
    if hist_move is not None:
        notes.append(f"Recent earnings dates in the available history produced an average absolute next-session move of about {hist_move*100:.1f}%.")

    if options.get("summary",{}).get("atm_straddle_implied_move") is not None:
        mv=options["summary"]["atm_straddle_implied_move"]*100
        notes.append(f"The nearest-expiration at-the-money straddle is pricing a move of roughly {mv:.1f}% of the stock price, before transaction costs.")

    return notes


def analyze(ticker,learning=None):
    ticker=ticker.upper().strip()
    raw=download_prices(ticker,"2010-01-01")
    feat=build_features(raw)
    lab=add_target(feat,HORIZON).dropna(subset=FEATURES+["target"])
    if len(lab)<500:
        raise ValueError(f"Not enough usable history for {ticker}")
    latest=feat.dropna(subset=FEATURES).iloc[-1]
    probs=[]
    mp={}
    metrics=[]

    for name,model in model_library().items():
        p=expanding_predictions(lab,FEATURES,model,horizon=HORIZON)
        probs.append(p)
        v=p.notna()
        y=lab.loc[v,"target"].astype(int)
        pv=p[v]
        pred=(pv>=.5).astype(int)
        metrics.append({
            "model":name,
            "accuracy":sf(accuracy_score(y,pred)),
            "f1":sf(f1_score(y,pred,zero_division=0)),
            "roc_auc":sf(roc_auc_score(y,pv)) if y.nunique()>1 else None,
            "brier":sf(brier_score_loss(y,pv)),
            "observations":int(len(y)),
        })
        final=clone(model)
        final.fit(lab[FEATURES],lab["target"].astype(int))
        mp[name]=sf(final.predict_proba(feat.dropna(subset=FEATURES).iloc[[-1]][FEATURES])[:,1][0])

    ens=np.nanmean(np.vstack([x.values for x in probs]),axis=0)
    ens=np.asarray(ens)
    valid=np.isfinite(ens)
    ep=ens[valid]
    ey=lab["target"].to_numpy()[valid].astype(int)
    er=lab["forward_return"].to_numpy()[valid]

    base=sf(ey.mean())
    current=sf(np.mean(list(mp.values())))
    sm=np.abs(ep-current)<=.025
    sn=int(sm.sum())
    similar={
        "observations":sn,
        "actual_up_rate":sf(ey[sm].mean()) if sn else None,
        "mean_forward_return":sf(er[sm].mean()) if sn else None,
        "median_forward_return":sf(np.median(er[sm])) if sn else None,
        "base_up_rate":base,
    }
    similar["up_rate_ci95"]=wilson(int(ey[sm].sum()),sn) if sn else [None,None]
    if sn:
        wins=er[sm][er[sm]>0]
        losses=er[sm][er[sm]<0]
        similar["avg_gain"]=sf(wins.mean()) if len(wins) else None
        similar["avg_loss"]=sf(losses.mean()) if len(losses) else None
        similar["best_return"]=sf(er[sm].max())
        similar["worst_return"]=sf(er[sm].min())

    aucs=[m["roc_auc"] for m in metrics if m["roc_auc"] is not None]
    mean_auc=sf(np.mean(aucs)) if aucs else None
    agreement=sf(1-(max(mp.values())-min(mp.values()))) if len(mp)>1 else None
    lift=sf(similar["actual_up_rate"]-base) if similar["actual_up_rate"] is not None and base is not None else None

    if mean_auc is None:
        quality="Not available"
    elif mean_auc>=.58:
        quality="Stronger historical discrimination"
    elif mean_auc>=.53:
        quality="Modest historical discrimination"
    else:
        quality="Weak historical discrimination"

    evidence={
        "validation_quality":quality,
        "mean_roc_auc":mean_auc,
        "model_agreement":agreement,
        "historical_lift":lift,
        "similar_sample_size":sn,
        "notes":[
            "Model output is not a calibrated real-world probability.",
            "Compare signal lift with the unconditional base rate.",
            "Give more weight to signals only when validation and sample size support them.",
        ],
    }
    if mean_auc is None or mean_auc<.53:
        evidence["state"]="Insufficient validated edge"
    elif sn<100:
        evidence["state"]="Limited comparable history"
    elif lift is not None and lift>0:
        evidence["state"]="Historically favorable setup"
    elif lift is not None and lift<0:
        evidence["state"]="Historically unfavorable setup"
    else:
        evidence["state"]="Historically neutral setup"

    close=raw["Close"]
    daily=close.pct_change()
    horizons={}
    for h in [5,10,20,30,60]:
        rr=close.pct_change(h).dropna()
        horizons[str(h)]={
            "mean_return":sf(rr.mean()),
            "median_return":sf(rr.median()),
            "positive_rate":sf((rr>0).mean()),
            "p10":sf(rr.quantile(.10)),
            "p25":sf(rr.quantile(.25)),
            "p75":sf(rr.quantile(.75)),
            "p90":sf(rr.quantile(.90)),
            "realized_move_1sd":sf(daily.rolling(252).std().iloc[-1]*sqrt(h)),
        }

    annual_rv=sf(daily.rolling(20).std().iloc[-1]*sqrt(252))
    options={
        "status":"Historical distribution + delayed option chain active",
        "horizons":horizons,
        "realized_vol_20d":annual_rv,
    }
    try:
        options["chain"]=option_snapshot(ticker,float(close.iloc[-1]),annual_rv)
        options["summary"]=options_summary(options["chain"],float(close.iloc[-1]))
    except Exception as e:
        options["chain"]={"error":str(e),"contracts":[],"quote_note":"Option chain unavailable for this run."}
        options["summary"]={}

    try:
        ctx=enrich_earnings_history(company_context(ticker), raw)
    except Exception:
        ctx={"news":[],"earnings":{},"catalyst_flags":[],"news_note":"Company context unavailable for this run."}

    rel=relative_strength(raw,ticker)
    regime_series=classify_regime(raw)
    current_regime=str(regime_series.iloc[-1])
    options["opportunity_radar"]=build_opportunity_radar(raw,(options.get("chain") or {}).get("contracts") or [],regime_series,current_regime,evidence,float(close.iloc[-1]),learning=learning)
    explanation=plain_language(ticker,evidence,ctx,options,rel)

    return {
        "ticker":ticker,
        "research_refreshed_at":datetime.now(timezone.utc).isoformat(),
        "as_of":str(raw.index[-1].date()),
        "price":sf(close.iloc[-1]),
        "change_1d":sf(daily.iloc[-1]),
        "change_5d":sf(close.pct_change(5).iloc[-1]),
        "probability_5d_up":current,
        "model_probabilities":mp,
        "regime":current_regime,
        "volatility_20d":annual_rv,
        "rsi_14":sf(latest["rsi_14"]),
        "drawdown_252":sf(latest["drawdown_252"]),
        "base_up_rate":base,
        "similar_setups":similar,
        "evidence":evidence,
        "relative_strength":rel,
        "company_context":ctx,
        "plain_language":explanation,
        "options":options,
        "calibration_buckets":bucket_stats(ep,ey,er),
        "metrics":metrics,
        "history":[{"date":str(i.date()),"close":sf(v)} for i,v in close.tail(260).items()],
    }


def load_existing():
    if not DATA_PATH.exists():
        return {"generated_at":None,"horizon_days":HORIZON,"tickers":[],"errors":[]}
    try:
        return json.loads(DATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"generated_at":None,"horizon_days":HORIZON,"tickers":[],"errors":[]}


def main():
    current_learning=learning_profile(load_state())
    requested=(os.getenv("MARKETLENS_TICKER") or "").strip().upper()
    if requested:
        tickers=[requested]
        p=load_existing()
        p["errors"]=[e for e in p.get("errors",[]) if e.get("ticker")!=requested]
    else:
        tickers=DEFAULT_TICKERS
        p={"generated_at":None,"horizon_days":HORIZON,"tickers":[],"errors":[]}

    by_ticker={x["ticker"]:x for x in p.get("tickers",[]) if x.get("ticker")}
    for t in tickers:
        try:
            by_ticker[t]=analyze(t,current_learning)
        except Exception as e:
            p.setdefault("errors",[]).append({"ticker":t,"error":str(e)})

    p["tickers"]=list(by_ticker.values())
    p["generated_at"]=datetime.now(timezone.utc).isoformat()
    p["horizon_days"]=HORIZON
    ai_state=run_ai_paper_portfolio(p,load_state())
    save_state(ai_state)
    p["ai_portfolio"]={"summary":performance_summary(ai_state),"attribution":performance_attribution(ai_state),"learning":learning_profile(ai_state),"updated_at":ai_state.get("updated_at"),"open":ai_state.get("open",[]),"closed":ai_state.get("closed",[])[-100:],"decisions":ai_state.get("decisions",[])[-100:],"equity_history":ai_state.get("equity_history",[])[-300:]}
    DATA_PATH.parent.mkdir(exist_ok=True)
    DATA_PATH.write_text(json.dumps(p,indent=2),encoding="utf-8")
    print(json.dumps({"tickers":[x["ticker"] for x in p["tickers"]],"errors":p["errors"]},indent=2))


if __name__=="__main__":
    main()

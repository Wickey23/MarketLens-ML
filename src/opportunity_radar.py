from __future__ import annotations

import math
import numpy as np
import pandas as pd

def _f(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def _historical_outcomes(contract, spot, entry, returns):
    if entry <= 0 or returns.empty:
        return {}
    terminal=spot*(1.0+returns.to_numpy(dtype=float))
    strike=float(contract["strike"])
    if contract["type"]=="call":
        intrinsic=np.maximum(terminal-strike,0.0)
    else:
        intrinsic=np.maximum(strike-terminal,0.0)
    pnl=(intrinsic-entry)*100.0
    ret=(intrinsic-entry)/entry
    return {
        "samples":int(len(pnl)),
        "prob_profit":_f(np.mean(pnl>0)),
        "prob_total_premium_loss":_f(np.mean(intrinsic<=0)),
        "expected_pnl_per_contract":_f(np.mean(pnl)),
        "median_pnl_per_contract":_f(np.median(pnl)),
        "p10_pnl_per_contract":_f(np.quantile(pnl,.10)),
        "p90_pnl_per_contract":_f(np.quantile(pnl,.90)),
        "expected_return_on_debit":_f(np.mean(ret)),
        "median_return_on_debit":_f(np.median(ret)),
    }

def build_opportunity_radar(raw, contracts, regime_series, current_regime, evidence, spot, limit=12):
    """Historical contract screen.

    Replays today's strike/premium economics across historical underlying moves.
    This is a research screen, not a forecast: today's IV, premium and market
    structure did not exist in those historical periods.
    """
    close=raw["Close"].astype(float)
    rows=[]
    for c in contracts or []:
        entry=(c.get("ask") if c.get("ask") and c.get("ask")>0 else c.get("mid")) or 0
        if entry<=0 or not c.get("strike"):
            continue
        # Convert calendar DTE to an approximate trading-day horizon.
        h=max(1,min(60,int(round(max(c.get("dte") or 1,1)*252/365))))
        fwd=close.shift(-h)/close-1
        frame=pd.DataFrame({"r":fwd,"regime":regime_series}).dropna()
        same=frame.loc[frame["regime"]==current_regime,"r"]
        sample=same if len(same)>=80 else frame["r"]
        stats=_historical_outcomes(c,float(spot),float(entry),sample)
        if not stats:
            continue

        spread=c.get("spread_pct")
        theta=c.get("theta_cost_pct_per_day")
        oi=c.get("open_interest") or 0
        vol=c.get("volume") or 0
        pp=stats.get("prob_profit") or 0
        ev=stats.get("expected_return_on_debit")
        loss=stats.get("prob_total_premium_loss") or 0

        # Transparent 0-100 research score. Positive EV/history alone is not
        # enough: execution quality, decay burden and sample size also matter.
        score=50.0
        score += max(-20,min(20,(pp-.50)*80))
        if ev is not None:
            score += max(-15,min(15,ev*20))
        score -= max(0,(loss-.50)*20)
        if spread is not None:
            score += 8 if spread<=.08 else (3 if spread<=.15 else (-8 if spread>.25 else 0))
        if theta is not None:
            score += 6 if theta<=.01 else (-8 if theta>.03 else 0)
        score += 5 if (oi>=500 or vol>=100) else (2 if (oi>=100 or vol>=20) else -4)
        if stats["samples"]<100:
            score-=8
        score=max(0,min(100,score))

        reasons=[]
        risks=[]
        if pp>=.60: reasons.append(f"Historical replay cleared breakeven {pp*100:.0f}% of the time")
        if ev is not None and ev>0: reasons.append("Historical payoff replay had positive mean P/L")
        if spread is not None and spread<=.10: reasons.append("Quoted spread is relatively tight")
        if oi>=500 or vol>=100: reasons.append("Higher open interest or activity")
        if theta is not None and theta>.025: risks.append("Daily theta is high relative to premium")
        if spread is not None and spread>.20: risks.append("Wide spread can materially reduce realized returns")
        if loss>=.50: risks.append(f"Historical replay lost the full premium about {loss*100:.0f}% of the time")
        if stats["samples"]<100: risks.append("Historical sample is limited")
        if (evidence or {}).get("mean_roc_auc") is None or (evidence or {}).get("mean_roc_auc",0)<.53:
            risks.append("Directional ML model has not demonstrated strong out-of-sample discrimination")

        state="watch"
        if score>=72 and pp>=.58 and ev is not None and ev>0 and stats["samples"]>=100 and (spread is None or spread<=.20):
            state="investigate"
        elif score<55 or ev is None or ev<=0:
            state="pass"

        rows.append({
            "contract_symbol":c.get("contract_symbol"),"type":c.get("type"),
            "expiration":c.get("expiration"),"dte":c.get("dte"),"strike":c.get("strike"),
            "entry_quote":_f(entry),"score":round(score,1),"state":state,
            "historical_scope":"same regime" if len(same)>=80 else "all regimes",
            "trading_day_horizon":h,**stats,"reasons":reasons,"risks":risks,
        })

    rows.sort(key=lambda x:(x["state"]=="investigate",x["score"],x.get("expected_pnl_per_contract") or -1e9),reverse=True)
    surfaced=[x for x in rows if x["state"]=="investigate"][:limit]
    watch=[x for x in rows if x["state"]=="watch"][:limit]
    return {
        "state":"opportunities_detected" if surfaced else "no_strong_setup",
        "opportunities":surfaced,
        "watchlist":watch,
        "contracts_evaluated":len(rows),
        "method_note":"Today's contract economics replayed across historical underlying moves. Results are hypothetical, exclude changing historical IV/Greeks and are not a profitability guarantee.",
    }

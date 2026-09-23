from __future__ import annotations

from datetime import datetime, timezone, date
from pathlib import Path
from collections import defaultdict
import json

STARTING_CASH = 10000.0
STATE_PATH = Path("data/ai_paper_portfolio.json")

def load_state(path=STATE_PATH):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"starting_cash":STARTING_CASH,"cash":STARTING_CASH,"open":[],"closed":[],"equity_history":[],"decisions":[]}

def save_state(state,path=STATE_PATH):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(state,indent=2),encoding="utf-8")

def contract_key(x):
    return x.get("contract_symbol") or f'{x.get("ticker","")}:{x.get("expiration")}:{x.get("strike")}:{x.get("type")}'

def _expiration_spot(ticker_payload, expiration):
    rows=ticker_payload.get("history") or []
    eligible=[x for x in rows if x.get("date") and x.get("date")<=expiration and x.get("close") is not None]
    if eligible:
        row=max(eligible,key=lambda x:x["date"])
        return float(row["close"])
    spot=ticker_payload.get("price")
    return float(spot) if spot is not None else None

def current_contract(pos,tickers):
    t=tickers.get(pos["ticker"])
    if not t:
        return None
    contracts=(((t.get("options") or {}).get("chain") or {}).get("contracts") or [])
    return next((x for x in contracts if contract_key(x)==pos["contract_key"]),None)

def mark_position(pos,tickers):
    """Executable long-option exit mark.

    For realism, use the bid only. A midpoint without a usable bid is not an
    executable long-option exit and should not create paper profits.
    """
    q=current_contract(pos,tickers)
    if q:
        bid=q.get("bid")
        return float(bid) if bid is not None and float(bid)>0 else None
    return None

def valuation_mark(pos,tickers):
    """Conservative equity mark for an open long option."""
    q=current_contract(pos,tickers)
    if q is not None:
        bid=q.get("bid")
        try:
            return max(0.0,float(bid or 0.0))
        except Exception:
            return 0.0
    if pos.get("last_mark") is not None:
        return float(pos["last_mark"])
    return float(pos.get("entry_price") or 0.0)

def quote_age_hours(contract, now_dt):
    raw=contract.get("last_trade")
    if not raw:
        return None
    try:
        dt=datetime.fromisoformat(str(raw).replace("Z","+00:00"))
        if dt.tzinfo is None:
            dt=dt.replace(tzinfo=timezone.utc)
        return max(0.0,(now_dt-dt.astimezone(timezone.utc)).total_seconds()/3600.0)
    except Exception:
        return None

def days_to_expiry(expiration, today):
    try:
        return (date.fromisoformat(expiration)-date.fromisoformat(today)).days
    except Exception:
        return None

def run_ai_paper_portfolio(snapshot,state=None,max_positions=3,risk_per_trade=.02,min_score=75.0,min_dte=3,max_dte=45,max_spread=.20,max_theta_pct=.03):
    """Rule-based autonomous paper portfolio driven only by MarketLens research.

    It never places a brokerage order. Entries/exits are recorded against delayed
    research quotes so the strategy can be evaluated prospectively.
    """
    state=state or load_state()
    now_dt=datetime.now(timezone.utc)
    now=now_dt.isoformat()
    snapshot_id=snapshot.get("generated_at") or snapshot.get("fast_generated_at")
    if snapshot_id and state.get("last_processed_snapshot")==snapshot_id:
        return state
    tickers={x["ticker"]:x for x in snapshot.get("tickers",[]) if x.get("ticker")}

    # Mark/exit first. Exit on expiry, deteriorating quote, +50% gain, or -35% loss.
    still=[]
    for p in state["open"]:
        mark=mark_position(p,tickers)
        pnl_pct=((mark/p["entry_price"])-1) if mark is not None and p["entry_price"]>0 else None
        today=now[:10]
        remaining=days_to_expiry(p["expiration"],today)
        reason=None
        if p["expiration"]<today:
            reason="expiration"
            if mark is None:
                t=tickers.get(p["ticker"]) or {}
                spot=_expiration_spot(t,p["expiration"])
                if spot is not None:
                    intrinsic=max(0.0,float(spot)-float(p["strike"])) if p["type"]=="call" else max(0.0,float(p["strike"])-float(spot))
                    mark=intrinsic
                    pnl_pct=((mark/p["entry_price"])-1) if p["entry_price"]>0 else None
        elif pnl_pct is not None and pnl_pct>=.50: reason="profit_target"
        elif pnl_pct is not None and pnl_pct<=-.35: reason="risk_limit"
        elif remaining is not None and remaining<=0:
            # Apply this to legacy positions too so an older strategy cannot
            # remain stuck in a same-day-expiry contract indefinitely.
            reason="time_risk"
        elif p.get("strategy_version")=="v2_conservative":
            if remaining is not None and remaining<=1:
                reason="time_risk"
            elif pnl_pct is not None and pnl_pct>=.25 and remaining is not None and remaining<=3:
                reason="profit_protection"
        p["exit_signal"]=reason or "hold"
        if reason and mark is not None:
            proceeds=mark*100*p["qty"]
            state["cash"]+=proceeds
            p.update({"exit_price":mark,"closed_at":now,"exit_reason":reason,
                      "pnl":(mark-p["entry_price"])*100*p["qty"]})
            state["closed"].append(p)
            state["decisions"].append({
                "at":now,"ticker":p["ticker"],"contract":p["contract_key"],
                "action":"paper_exit","qty":p["qty"],"price":mark,
                "pnl":p["pnl"],"reason":reason,
                "strategy_version":p.get("strategy_version","legacy"),
            })
        else:
            p["last_mark"]=mark
            p["unrealized_pnl"]=((mark-p["entry_price"])*100*p["qty"]) if mark is not None else None
            still.append(p)
    state["open"]=still

    held={p["contract_key"] for p in state["open"]}
    candidates=[]
    for ticker,t in tickers.items():
        radar=((t.get("options") or {}).get("opportunity_radar") or {})
        chain=(((t.get("options") or {}).get("chain") or {}).get("contracts") or [])
        cmap={contract_key(x):x for x in chain}
        guidance=(radar.get("guidance") or {})
        guided=guidance.get("best_overall") if guidance.get("state")=="strong_candidates" else None
        candidate_rows=[guided] if guided else (radar.get("opportunities") or [])
        for r in candidate_rows:
            key=contract_key(r)
            q=cmap.get(key)
            candidate_score=float(r.get("combined_evidence_score") or r.get("score") or 0)
            if not q or key in held or candidate_score<min_score:
                continue
            dte=q.get("dte")
            spread=q.get("spread_pct")
            theta=q.get("theta_cost_pct_per_day")
            iv=q.get("iv")
            prob=r.get("prob_profit")
            ev=r.get("expected_pnl_per_contract")
            bid=q.get("bid")
            ask=q.get("ask")
            age=quote_age_hours(q,now_dt)
            guard_reasons=[]
            if dte is None or int(dte)<min_dte or int(dte)>max_dte: guard_reasons.append("DTE outside autonomous policy")
            if bid is None or ask is None or float(bid)<=0 or float(ask)<=0 or float(ask)<float(bid): guard_reasons.append("two-sided executable quote unavailable")
            if age is None or age>96: guard_reasons.append("option quote is stale or timestamp unavailable")
            if spread is not None and float(spread)>max_spread: guard_reasons.append("spread above autonomous policy")
            if theta is not None and float(theta)>max_theta_pct: guard_reasons.append("theta burden above autonomous policy")
            if iv is not None and (float(iv)<.03 or float(iv)>5.0): guard_reasons.append("implausible IV for autonomous entry")
            if prob is None or float(prob)<.58: guard_reasons.append("historical breakeven rate below policy")
            if ev is None or float(ev)<=0: guard_reasons.append("historical expected payoff is not positive")
            if guard_reasons:
                state["decisions"].append({"at":now,"ticker":ticker,"contract":key,"action":"skip","score":r.get("score"),
                                           "reason":"; ".join(guard_reasons),"strategy_version":"v2_conservative"})
                continue
            candidates.append((candidate_score,ticker,r,q))
    candidates.sort(reverse=True,key=lambda z:z[0])

    # Fixed fractional premium-at-risk sizing, capped at one new contract group per ticker.
    active_tickers={p["ticker"] for p in state["open"]}
    for score,ticker,r,q in candidates:
        if len(state["open"])>=max_positions or ticker in active_tickers:
            continue
        entry=(q.get("ask") if q.get("ask") and q.get("ask")>0 else q.get("mid"))
        if not entry or entry<=0:
            continue
        equity=state["cash"]+sum(valuation_mark(p,tickers)*100*p["qty"] for p in state["open"])
        budget=min(state["cash"],equity*risk_per_trade)
        qty=int(budget//(entry*100))
        key=contract_key(q)
        if qty<1:
            state["decisions"].append({"at":now,"ticker":ticker,"contract":key,"action":"skip","reason":"risk budget below one contract","strategy_version":"v2_conservative"})
            continue
        cost=entry*100*qty
        state["cash"]-=cost
        pos={"id":f"{now}:{key}","ticker":ticker,"contract_key":key,"type":q["type"],
             "strike":q["strike"],"expiration":q["expiration"],"qty":qty,
             "entry_price":entry,"entry_cost":cost,"opened_at":now,"entry_score":score,"entry_combined_evidence_score":r.get("combined_evidence_score"),
             "entry_dte":q.get("dte"),"entry_iv":q.get("iv"),"entry_iv_rv_ratio":q.get("iv_rv_ratio"),
             "entry_spread_pct":q.get("spread_pct"),"entry_theta_cost_pct_per_day":q.get("theta_cost_pct_per_day"),
             "entry_regime":t.get("regime"),"entry_model_auc":(t.get("evidence") or {}).get("mean_roc_auc"),
             "entry_prob_profit":r.get("prob_profit"),"entry_expected_pnl":r.get("expected_pnl_per_contract"),
             "entry_scope":r.get("historical_scope"),"strategy_version":"v2_conservative","entry_quote_age_hours":quote_age_hours(q,now_dt),"entry_reasons":r.get("reasons") or [],
             "entry_risks":r.get("risks") or [],"entry_research_generated_at":snapshot.get("generated_at")}
        state["open"].append(pos); active_tickers.add(ticker)
        state["decisions"].append({"at":now,"ticker":ticker,"contract":key,"action":"paper_buy",
                                   "qty":qty,"price":entry,"score":score,"strategy_version":"v2_conservative"})

    open_value=sum(valuation_mark(p,tickers)*100*p["qty"] for p in state["open"])
    equity=state["cash"]+open_value
    state["equity_history"].append({"at":now,"equity":equity,"cash":state["cash"],"open_value":open_value})
    state["equity_history"]=state["equity_history"][-1000:]
    state["decisions"]=state.get("decisions",[])[-2000:]
    state["updated_at"]=now
    state["last_processed_snapshot"]=snapshot_id
    return state

def performance_summary(state):
    closed=state.get("closed") or []
    wins=[p for p in closed if (p.get("pnl") or 0)>0]
    losses=[p for p in closed if (p.get("pnl") or 0)<0]
    realized_pnl=sum(p.get("pnl") or 0 for p in closed)
    hist=state.get("equity_history") or []
    equity=hist[-1]["equity"] if hist else state.get("cash",STARTING_CASH)
    pnl=equity-float(state.get("starting_cash",STARTING_CASH))
    peak=STARTING_CASH; max_dd=0.0
    for x in hist:
        peak=max(peak,x["equity"])
        if peak: max_dd=min(max_dd,x["equity"]/peak-1)
    return {"equity":equity,"total_pnl":pnl,"realized_pnl":realized_pnl,"return":equity/STARTING_CASH-1,
            "closed_trades":len(closed),"win_rate":len(wins)/len(closed) if closed else None,
            "avg_win":sum(p["pnl"] for p in wins)/len(wins) if wins else None,
            "avg_loss":sum(p["pnl"] for p in losses)/len(losses) if losses else None,
            "max_drawdown":max_dd,"open_positions":len(state.get("open") or [])}


def _group_stats(closed,key_fn):
    groups=defaultdict(list)
    for p in closed:
        groups[str(key_fn(p))].append(p)
    out=[]
    for name,rows in groups.items():
        pnl=[float(x.get("pnl") or 0) for x in rows]
        wins=sum(x>0 for x in pnl)
        cost=sum(float(x.get("entry_cost") or 0) for x in rows)
        out.append({"group":name,"trades":len(rows),"win_rate":wins/len(rows) if rows else None,
                    "total_pnl":sum(pnl),"avg_pnl":sum(pnl)/len(pnl) if pnl else None,
                    "return_on_premium":sum(pnl)/cost if cost else None})
    return sorted(out,key=lambda x:(x["trades"],x["total_pnl"]),reverse=True)

def performance_attribution(state):
    """Describe where forward paper results came from; never backfills entry facts."""
    closed=state.get("closed") or []
    def score_band(p):
        s=float(p.get("entry_score") or 0)
        return "80+" if s>=80 else ("75-79" if s>=75 else ("72-74" if s>=72 else "<72"))
    def dte_band(p):
        d=p.get("entry_dte")
        if d is None: return "unknown"
        d=int(d)
        return "0-7" if d<=7 else ("8-21" if d<=21 else ("22-45" if d<=45 else "46+"))
    def prob_band(p):
        q=p.get("entry_prob_profit")
        if q is None: return "unknown"
        q=float(q)
        return "65%+" if q>=.65 else ("60-65%" if q>=.60 else "<60%")
    def iv_band(p):
        x=p.get("entry_iv_rv_ratio")
        if x is None:return "unknown"
        x=float(x)
        return "IV<0.9xRV" if x<.9 else ("0.9-1.2x" if x<=1.2 else "IV>1.2xRV")
    return {
        "by_type":_group_stats(closed,lambda p:p.get("type","unknown")),
        "by_regime":_group_stats(closed,lambda p:p.get("entry_regime","unknown")),
        "by_dte":_group_stats(closed,dte_band),
        "by_score":_group_stats(closed,score_band),
        "by_historical_probability":_group_stats(closed,prob_band),
        "by_iv_environment":_group_stats(closed,iv_band),
        "note":"Attribution uses facts captured at entry. Small samples should not be treated as evidence of a durable edge.",
    }


def learning_profile(state,min_trades=20,min_group_trades=8):
    """Conservative forward-only learning profile.

    It does not rewrite historical scores and it does not optimize thresholds.
    A factor is only marked positive/negative after enough closed paper trades.
    """
    attr=performance_attribution(state)
    total=len(state.get("closed") or [])
    profile={"enabled":total>=min_trades,"closed_trades":total,"minimum_closed_trades":min_trades,
             "minimum_group_trades":min_group_trades,"factors":{}}
    mapping={"type":"by_type","regime":"by_regime","dte":"by_dte","score":"by_score",
             "historical_probability":"by_historical_probability","iv_environment":"by_iv_environment"}
    for factor,key in mapping.items():
        rows=[]
        for g in attr.get(key,[]):
            n=int(g.get("trades") or 0); rp=g.get("return_on_premium"); wr=g.get("win_rate")
            status="insufficient_sample"
            weight=1.0
            if profile["enabled"] and n>=min_group_trades and rp is not None:
                # Deliberately tiny adjustment: at most +/-10%. This is evidence
                # annotation, not an optimizer that chases recent winners.
                if rp>.10 and (wr is None or wr>=.50): status="positive_forward_evidence"; weight=1.10
                elif rp<-.10: status="negative_forward_evidence"; weight=.90
                else: status="mixed_forward_evidence"
            rows.append({"group":g.get("group"),"trades":n,"win_rate":wr,"return_on_premium":rp,
                         "status":status,"weight_multiplier":weight})
        profile["factors"][factor]=rows
    profile["note"]="Learning remains disabled until the minimum forward sample is reached. Eligible adjustments are capped at ±10% and never rewrite prior decisions."
    return profile

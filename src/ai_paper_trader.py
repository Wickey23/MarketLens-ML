from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
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

def mark_position(pos,tickers):
    t=tickers.get(pos["ticker"])
    if not t:
        return None
    contracts=(((t.get("options") or {}).get("chain") or {}).get("contracts") or [])
    q=next((x for x in contracts if contract_key(x)==pos["contract_key"]),None)
    if q:
        return (q.get("bid") if q.get("bid") and q.get("bid")>0 else q.get("mid"))
    return None

def run_ai_paper_portfolio(snapshot,state=None,max_positions=3,risk_per_trade=.02,min_score=72.0):
    """Rule-based autonomous paper portfolio driven only by MarketLens research.

    It never places a brokerage order. Entries/exits are recorded against delayed
    research quotes so the strategy can be evaluated prospectively.
    """
    state=state or load_state()
    now=datetime.now(timezone.utc).isoformat()
    tickers={x["ticker"]:x for x in snapshot.get("tickers",[]) if x.get("ticker")}

    # Mark/exit first. Exit on expiry, deteriorating quote, +50% gain, or -35% loss.
    still=[]
    for p in state["open"]:
        mark=mark_position(p,tickers)
        pnl_pct=((mark/p["entry_price"])-1) if mark is not None and p["entry_price"]>0 else None
        today=now[:10]
        reason=None
        if p["expiration"]<=today: reason="expiration"
        elif pnl_pct is not None and pnl_pct>=.50: reason="profit_target"
        elif pnl_pct is not None and pnl_pct<=-.35: reason="risk_limit"
        if reason and mark is not None:
            proceeds=mark*100*p["qty"]
            state["cash"]+=proceeds
            p.update({"exit_price":mark,"closed_at":now,"exit_reason":reason,
                      "pnl":(mark-p["entry_price"])*100*p["qty"]})
            state["closed"].append(p)
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
        for r in radar.get("opportunities") or []:
            key=contract_key(r)
            q=cmap.get(key)
            if not q or key in held or (r.get("score") or 0)<min_score:
                continue
            candidates.append((float(r["score"]),ticker,r,q))
    candidates.sort(reverse=True,key=lambda z:z[0])

    # Fixed fractional premium-at-risk sizing, capped at one new contract group per ticker.
    active_tickers={p["ticker"] for p in state["open"]}
    for score,ticker,r,q in candidates:
        if len(state["open"])>=max_positions or ticker in active_tickers:
            continue
        entry=(q.get("ask") if q.get("ask") and q.get("ask")>0 else q.get("mid"))
        if not entry or entry<=0:
            continue
        equity=state["cash"]+sum((p.get("last_mark") or p["entry_price"])*100*p["qty"] for p in state["open"])
        budget=min(state["cash"],equity*risk_per_trade)
        qty=int(budget//(entry*100))
        if qty<1:
            state["decisions"].append({"at":now,"ticker":ticker,"contract":key,"action":"skip","reason":"risk budget below one contract"})
            continue
        key=contract_key(q)
        cost=entry*100*qty
        state["cash"]-=cost
        pos={"id":f"{now}:{key}","ticker":ticker,"contract_key":key,"type":q["type"],
             "strike":q["strike"],"expiration":q["expiration"],"qty":qty,
             "entry_price":entry,"entry_cost":cost,"opened_at":now,"entry_score":score,\n             "entry_dte":q.get("dte"),"entry_iv":q.get("iv"),"entry_iv_rv_ratio":q.get("iv_rv_ratio"),\n             "entry_spread_pct":q.get("spread_pct"),"entry_theta_cost_pct_per_day":q.get("theta_cost_pct_per_day"),\n             "entry_regime":t.get("regime"),"entry_model_auc":(t.get("evidence") or {}).get("mean_roc_auc"),
             "entry_prob_profit":r.get("prob_profit"),"entry_expected_pnl":r.get("expected_pnl_per_contract"),
             "entry_scope":r.get("historical_scope"),"entry_reasons":r.get("reasons") or [],
             "entry_risks":r.get("risks") or [],"entry_research_generated_at":snapshot.get("generated_at")}
        state["open"].append(pos); active_tickers.add(ticker)
        state["decisions"].append({"at":now,"ticker":ticker,"contract":key,"action":"paper_buy",
                                   "qty":qty,"price":entry,"score":score})

    open_value=sum((mark_position(p,tickers) or p["entry_price"])*100*p["qty"] for p in state["open"])
    equity=state["cash"]+open_value
    state["equity_history"].append({"at":now,"equity":equity,"cash":state["cash"],"open_value":open_value})
    state["equity_history"]=state["equity_history"][-1000:]
    state["updated_at"]=now
    return state

def performance_summary(state):
    closed=state.get("closed") or []
    wins=[p for p in closed if (p.get("pnl") or 0)>0]
    losses=[p for p in closed if (p.get("pnl") or 0)<0]
    pnl=sum(p.get("pnl") or 0 for p in closed)
    hist=state.get("equity_history") or []
    equity=hist[-1]["equity"] if hist else state.get("cash",STARTING_CASH)
    peak=STARTING_CASH; max_dd=0.0
    for x in hist:
        peak=max(peak,x["equity"])
        if peak: max_dd=min(max_dd,x["equity"]/peak-1)
    return {"equity":equity,"total_pnl":pnl,"return":equity/STARTING_CASH-1,
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

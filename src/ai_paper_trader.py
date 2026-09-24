from __future__ import annotations

from datetime import datetime, timezone, date, time as dt_time
from pathlib import Path
from collections import defaultdict
from zoneinfo import ZoneInfo
import json

STARTING_CASH = 10000.0
STATE_PATH = Path("data/ai_paper_portfolio.json")
CURRENT_STRATEGY_VERSION = "v3_realtime_session"
MARKET_TZ = ZoneInfo("America/New_York")

def load_state(path=STATE_PATH):
    if not path.exists():
        return {"starting_cash":STARTING_CASH,"cash":STARTING_CASH,"open":[],"closed":[],"equity_history":[],"decisions":[]}
    try:
        state=json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"AI paper state is unreadable at {path}; refusing to reset history") from exc
    required={"starting_cash","cash","open","closed","equity_history","decisions"}
    if not isinstance(state,dict) or not required.issubset(state):
        raise RuntimeError(f"AI paper state is invalid at {path}; refusing to reset history")
    return state

def save_state(state,path=STATE_PATH):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(state,indent=2),encoding="utf-8")
    tmp.replace(path)

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

def timestamp_age_hours(value, now_dt):
    if not value:
        return None
    try:
        dt=datetime.fromisoformat(str(value).replace("Z","+00:00"))
        if dt.tzinfo is None:
            dt=dt.replace(tzinfo=timezone.utc)
        return max(0.0,(now_dt-dt.astimezone(timezone.utc)).total_seconds()/3600.0)
    except Exception:
        return None


def quote_age_hours(contract, now_dt):
    # Prefer the provider's quote timestamp. A last-trade timestamp is only a
    # fallback because a recent trade does not prove the current bid/ask is fresh.
    provided=contract.get("quote_age_hours")
    if provided is not None:
        try:
            value=float(provided)
            if value>=0:
                return value
        except Exception:
            pass
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

def is_regular_market_session(now_dt):
    """Conservative U.S. options execution window for autonomous paper fills."""
    if now_dt.tzinfo is None:
        now_dt=now_dt.replace(tzinfo=timezone.utc)
    local=now_dt.astimezone(MARKET_TZ)
    if local.weekday()>=5:
        return False
    current=local.time().replace(tzinfo=None)
    return dt_time(9,35)<=current<=dt_time(15,55)

def days_to_expiry(expiration, today):
    try:
        return (date.fromisoformat(expiration)-date.fromisoformat(today)).days
    except Exception:
        return None

def run_ai_paper_portfolio(snapshot,state=None,max_positions=3,risk_per_trade=.02,min_score=75.0,min_dte=3,max_dte=45,max_spread=.20,max_theta_pct=.03,max_quote_age_hours=1.0,now_dt=None):
    """Rule-based autonomous PAPER portfolio driven only by MarketLens research.

    The current strategy never places a brokerage order, requires a real-time
    option-chain source for new entries, and only simulates normal entries/exits
    during a conservative regular-market execution window.
    """
    state=state or load_state()
    now_dt=now_dt or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt=now_dt.replace(tzinfo=timezone.utc)
    local_session_open=is_regular_market_session(now_dt)
    clock=(snapshot.get("market_clock") or {})
    clock_state=str(clock.get("state") or "").lower()
    if clock_state in ("open","premarket","postmarket","closed"):
        market_session_open=local_session_open and clock_state=="open"
    else:
        market_session_open=local_session_open
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
            # Expired options settle from intrinsic value. Never use a stale
            # post-expiry option bid if an old contract remains in the snapshot.
            reason="expiration"
            t=tickers.get(p["ticker"]) or {}
            spot=_expiration_spot(t,p["expiration"])
            if spot is not None:
                intrinsic=max(0.0,float(spot)-float(p["strike"])) if p["type"]=="call" else max(0.0,float(p["strike"])-float(spot))
                mark=intrinsic
                pnl_pct=((mark/p["entry_price"])-1) if p["entry_price"]>0 else None
            else:
                mark=None
        elif pnl_pct is not None and pnl_pct>=.50: reason="profit_target"
        elif pnl_pct is not None and pnl_pct<=-.35: reason="risk_limit"
        elif remaining is not None and remaining<=0:
            # Apply this to legacy positions too so an older strategy cannot
            # remain stuck in a same-day-expiry contract indefinitely.
            reason="time_risk"
        elif p.get("strategy_version") in ("v2_conservative",CURRENT_STRATEGY_VERSION):
            if remaining is not None and remaining<=1:
                reason="time_risk"
            elif pnl_pct is not None and pnl_pct>=.25 and remaining is not None and remaining<=3:
                reason="profit_protection"
        can_execute=reason=="expiration" or market_session_open
        p["exit_signal"]=(reason if can_execute else f"{reason}_waiting_market") if reason else "hold"
        if reason and mark is not None and can_execute:
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
        if not market_session_open:
            continue
        radar=((t.get("options") or {}).get("opportunity_radar") or {})
        chain_obj=((t.get("options") or {}).get("chain") or {})
        chain=chain_obj.get("contracts") or []
        chain_source=chain_obj.get("source")
        chain_realtime=chain_obj.get("realtime") is True
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
            research_age=timestamp_age_hours(t.get("research_refreshed_at"),now_dt)
            evidence=t.get("evidence") or {}
            if research_age is None or research_age>48 or evidence.get("mean_roc_auc") is None:
                guard_reasons.append("recent deep research evidence required")
            if not chain_realtime: guard_reasons.append("real-time option data required for current strategy")
            if dte is None or int(dte)<min_dte or int(dte)>max_dte: guard_reasons.append("DTE outside autonomous policy")
            if bid is None or ask is None or float(bid)<=0 or float(ask)<=0 or float(ask)<float(bid): guard_reasons.append("two-sided executable quote unavailable")
            if age is None or age>max_quote_age_hours: guard_reasons.append("option quote is stale or timestamp unavailable")
            if spread is not None and float(spread)>max_spread: guard_reasons.append("spread above autonomous policy")
            if theta is not None and float(theta)>max_theta_pct: guard_reasons.append("theta burden above autonomous policy")
            if iv is not None and (float(iv)<.03 or float(iv)>5.0): guard_reasons.append("implausible IV for autonomous entry")
            if prob is None or float(prob)<.58: guard_reasons.append("historical breakeven rate below policy")
            if ev is None or float(ev)<=0: guard_reasons.append("historical expected payoff is not positive")
            if guard_reasons:
                state["decisions"].append({"at":now,"ticker":ticker,"contract":key,"action":"skip","score":r.get("score"),
                                           "reason":"; ".join(guard_reasons),"strategy_version":CURRENT_STRATEGY_VERSION})
                continue
            candidates.append((candidate_score,ticker,r,q,chain_source,chain_realtime))
    candidates.sort(reverse=True,key=lambda z:z[0])

    # Fixed fractional premium-at-risk sizing, capped at one new contract group per ticker.
    active_tickers={p["ticker"] for p in state["open"]}
    for score,ticker,r,q,chain_source,chain_realtime in candidates:
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
            state["decisions"].append({"at":now,"ticker":ticker,"contract":key,"action":"skip","reason":"risk budget below one contract","strategy_version":CURRENT_STRATEGY_VERSION})
            continue
        cost=entry*100*qty
        state["cash"]-=cost
        pos={"id":f"{now}:{key}","ticker":ticker,"contract_key":key,"type":q["type"],
             "strike":q["strike"],"expiration":q["expiration"],"qty":qty,
             "entry_price":entry,"entry_cost":cost,"opened_at":now,"entry_score":score,"entry_combined_evidence_score":r.get("combined_evidence_score"),
             "entry_dte":q.get("dte"),"entry_iv":q.get("iv"),"entry_iv_rv_ratio":q.get("iv_rv_ratio"),
             "entry_spread_pct":q.get("spread_pct"),"entry_theta_cost_pct_per_day":q.get("theta_cost_pct_per_day"),
             "entry_regime":t.get("regime"),"entry_model_auc":(t.get("evidence") or {}).get("mean_roc_auc"),
             "entry_research_age_hours":timestamp_age_hours(t.get("research_refreshed_at"),now_dt),
             "entry_prob_profit":r.get("prob_profit"),"entry_expected_pnl":r.get("expected_pnl_per_contract"),
             "entry_scope":r.get("historical_scope"),"strategy_version":CURRENT_STRATEGY_VERSION,
             "entry_market_data_source":chain_source,"entry_market_data_realtime":chain_realtime,
             "entry_quote_age_hours":quote_age_hours(q,now_dt),"entry_reasons":r.get("reasons") or [],
             "entry_risks":r.get("risks") or [],"entry_research_generated_at":snapshot.get("generated_at")}
        state["open"].append(pos); active_tickers.add(ticker)
        state["decisions"].append({"at":now,"ticker":ticker,"contract":key,"action":"paper_buy",
                                   "qty":qty,"price":entry,"score":score,"strategy_version":CURRENT_STRATEGY_VERSION})

    open_value=sum(valuation_mark(p,tickers)*100*p["qty"] for p in state["open"])
    equity=state["cash"]+open_value
    current_closed=eligible_strategy_trades(state,CURRENT_STRATEGY_VERSION,True)
    current_realized=sum(float(p.get("pnl") or 0.0) for p in current_closed)
    current_open=[
        p for p in state["open"]
        if p.get("strategy_version")==CURRENT_STRATEGY_VERSION
        and p.get("entry_market_data_realtime") is True
    ]
    current_unrealized=sum(
        (valuation_mark(p,tickers)-float(p.get("entry_price") or 0.0))*100*int(p.get("qty") or 1)
        for p in current_open
    )
    strategy_equity=STARTING_CASH+current_realized+current_unrealized
    state["equity_history"].append({"at":now,"equity":equity,"cash":state["cash"],"open_value":open_value,
                                    "strategy_version":CURRENT_STRATEGY_VERSION,
                                    "strategy_equity":strategy_equity,
                                    "strategy_realized_pnl":current_realized,
                                    "strategy_unrealized_pnl":current_unrealized,
                                    "market_session_open":market_session_open})
    state["equity_history"]=state["equity_history"][-1000:]
    state["decisions"]=state.get("decisions",[])[-2000:]
    state["updated_at"]=now
    state["strategy_version"]=CURRENT_STRATEGY_VERSION
    state["paper_market_session_open"]=market_session_open
    state["market_clock_state"]=clock_state or "local_fallback"
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


def _parse_timestamp(value):
    if not value:
        return None
    try:
        dt=datetime.fromisoformat(str(value).replace("Z","+00:00"))
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def eligible_strategy_trades(state,strategy_version=CURRENT_STRATEGY_VERSION,require_realtime=True):
    """Closed trades eligible for current-strategy forward evidence."""
    rows=[]
    for p in state.get("closed") or []:
        if p.get("strategy_version")!=strategy_version:
            continue
        if require_realtime and p.get("entry_market_data_realtime") is not True:
            continue
        rows.append(p)
    return rows


def strategy_performance_summary(state,strategy_version=CURRENT_STRATEGY_VERSION,require_realtime=True):
    """Performance isolated to the current forward-test strategy generation."""
    closed=eligible_strategy_trades(state,strategy_version,require_realtime)
    open_rows=[
        p for p in (state.get("open") or [])
        if p.get("strategy_version")==strategy_version
        and (not require_realtime or p.get("entry_market_data_realtime") is True)
    ]
    pnls=[float(p.get("pnl") or 0.0) for p in closed]
    realized=sum(pnls)
    unrealized=sum(float(p.get("unrealized_pnl") or 0.0) for p in open_rows)
    total=realized+unrealized
    wins=[x for x in pnls if x>0]
    losses=[x for x in pnls if x<0]

    hist=[
        x for x in (state.get("equity_history") or [])
        if x.get("strategy_version")==strategy_version and x.get("strategy_equity") is not None
    ]
    max_dd=0.0
    if hist:
        peak=float(hist[0]["strategy_equity"])
        for x in hist:
            eq=float(x["strategy_equity"])
            peak=max(peak,eq)
            if peak>0:
                max_dd=min(max_dd,eq/peak-1)
    elif closed:
        eq=STARTING_CASH
        peak=eq
        for p in sorted(closed,key=lambda z:str(z.get("closed_at") or "")):
            eq+=float(p.get("pnl") or 0.0)
            peak=max(peak,eq)
            if peak>0:
                max_dd=min(max_dd,eq/peak-1)

    return {
        "strategy_version":strategy_version,
        "realtime_evidence_only":require_realtime,
        "equity":STARTING_CASH+total,
        "total_pnl":total,
        "realized_pnl":realized,
        "unrealized_pnl":unrealized,
        "return":total/STARTING_CASH,
        "closed_trades":len(closed),
        "win_rate":len(wins)/len(closed) if closed else None,
        "avg_win":sum(wins)/len(wins) if wins else None,
        "avg_loss":sum(losses)/len(losses) if losses else None,
        "max_drawdown":max_dd,
        "open_positions":len(open_rows),
    }


def forward_validation_summary(state,strategy_version=CURRENT_STRATEGY_VERSION):
    """Compare forward realized outcomes with evidence captured before entry."""
    closed=eligible_strategy_trades(state,strategy_version,True)
    rows=[]
    for p in closed:
        qty=max(1,int(p.get("qty") or 1))
        actual=float(p.get("pnl") or 0.0)/qty
        prob=p.get("entry_prob_profit")
        expected=p.get("entry_expected_pnl")
        rows.append({
            "actual_pnl_per_contract":actual,
            "profitable":actual>0,
            "reference_profit_frequency":float(prob) if prob is not None else None,
            "reference_expected_pnl":float(expected) if expected is not None else None,
        })
    probs=[x["reference_profit_frequency"] for x in rows if x["reference_profit_frequency"] is not None]
    expected=[x["reference_expected_pnl"] for x in rows if x["reference_expected_pnl"] is not None]
    actual=[x["actual_pnl_per_contract"] for x in rows]
    calibration=[]
    for label,lo,hi in [("<60%",0,.60),("60-65%",.60,.65),("65%+",.65,1.01)]:
        group=[x for x in rows if x["reference_profit_frequency"] is not None and lo<=x["reference_profit_frequency"]<hi]
        if group:
            calibration.append({
                "group":label,
                "trades":len(group),
                "mean_reference_profit_frequency":sum(x["reference_profit_frequency"] for x in group)/len(group),
                "realized_win_rate":sum(1 for x in group if x["profitable"])/len(group),
            })
    paired=[
        x["actual_pnl_per_contract"]-x["reference_expected_pnl"]
        for x in rows if x["reference_expected_pnl"] is not None
    ]
    return {
        "strategy_version":strategy_version,
        "closed_trades":len(rows),
        "realized_win_rate":sum(1 for x in rows if x["profitable"])/len(rows) if rows else None,
        "mean_reference_profit_frequency":sum(probs)/len(probs) if probs else None,
        "profit_frequency_gap":(
            (sum(1 for x in rows if x["profitable"])/len(rows))-(sum(probs)/len(probs))
            if rows and probs else None
        ),
        "mean_actual_pnl_per_contract":sum(actual)/len(actual) if actual else None,
        "mean_reference_expected_pnl_per_contract":sum(expected)/len(expected) if expected else None,
        "mean_expected_pnl_error_per_contract":sum(paired)/len(paired) if paired else None,
        "calibration_buckets":calibration,
        "note":"Reference profit frequency and expected P/L come from the historical payoff replay captured at entry. They are diagnostics, not calibrated forecasts.",
    }


def paper_to_real_readiness(state,min_closed_trades=30,min_observation_days=21,strategy_version=CURRENT_STRATEGY_VERSION):
    """Forward-only evidence gate for a later human review.

    Only the current strategy generation and real-time-source entries count.
    Legacy experiments and delayed-data trades remain in history but cannot
    make the current strategy look ready.
    """
    closed=eligible_strategy_trades(state,strategy_version,True)
    raw_current=[
        p for p in (state.get("closed") or [])
        if p.get("strategy_version")==strategy_version
    ]
    n=len(closed)
    pnls=[float(p.get("pnl") or 0.0) for p in closed]
    net_pnl=sum(pnls)
    avg_pnl=(net_pnl/n) if n else None
    winners=[x for x in pnls if x>0]
    winner_sum=sum(winners)
    largest_winner=max(winners) if winners else 0.0
    winner_concentration=(largest_winner/winner_sum) if winner_sum>0 else None
    ticker_count=len({p.get("ticker") for p in closed if p.get("ticker")})
    summary=strategy_performance_summary(state,strategy_version,True)

    opened=[_parse_timestamp(p.get("opened_at")) for p in closed]
    ended=[_parse_timestamp(p.get("closed_at")) for p in closed]
    opened=[x for x in opened if x is not None]
    ended=[x for x in ended if x is not None]
    observation_days=0.0
    if opened and ended:
        observation_days=max(0.0,(max(ended)-min(opened)).total_seconds()/86400.0)

    checks=[
        {"id":"forward_sample","label":"Forward sample","pass":n>=min_closed_trades,
         "value":n,"target":min_closed_trades},
        {"id":"observation_span","label":"Observation span","pass":observation_days>=min_observation_days,
         "value":round(observation_days,1),"target":f">= {min_observation_days} days"},
        {"id":"realtime_data","label":"Real-time entry data","pass":n>0 and len(raw_current)==n,
         "value":n,"target":"All counted trades"},
        {"id":"net_pnl","label":"Net paper P/L","pass":n>0 and net_pnl>0,
         "value":net_pnl,"target":"> 0"},
        {"id":"avg_trade","label":"Average trade","pass":n>0 and avg_pnl is not None and avg_pnl>0,
         "value":avg_pnl,"target":"> 0"},
        {"id":"max_drawdown","label":"Max drawdown","pass":n>0 and abs(float(summary.get("max_drawdown") or 0))<=.15,
         "value":summary.get("max_drawdown"),"target":"<= 15%"},
        {"id":"winner_concentration","label":"Winner concentration","pass":n>=10 and winner_concentration is not None and winner_concentration<=.50,
         "value":winner_concentration,"target":"<= 50% of gross winning P/L"},
        {"id":"ticker_breadth","label":"Ticker breadth","pass":n>=15 and ticker_count>=3,
         "value":ticker_count,"target":">= 3 tickers"},
    ]
    enough_sample=n>=min_closed_trades and observation_days>=min_observation_days
    all_pass=all(bool(x["pass"]) for x in checks)
    if not enough_sample:
        state_name="collecting_forward_data"
    elif all_pass:
        state_name="paper_results_ready_for_review"
    else:
        state_name="paper_results_not_ready"
    return {
        "state":state_name,
        "strategy_version":strategy_version,
        "closed_trades":n,
        "minimum_closed_trades":min_closed_trades,
        "observation_days":round(observation_days,1),
        "minimum_observation_days":min_observation_days,
        "all_checks_pass":all_pass,
        "checks":checks,
        "note":"Only forward paper trades from the current strategy using real-time entry data count. Passing these checks does not guarantee future profitability or authorize real-money trading.",
    }

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

def performance_attribution(state,strategy_version=CURRENT_STRATEGY_VERSION):
    """Describe current-strategy forward results without backfilling entry facts."""
    closed=eligible_strategy_trades(state,strategy_version,True)
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
        "strategy_version":strategy_version,
        "note":"Attribution uses current-strategy, real-time-source trades and facts captured at entry. Small samples should not be treated as evidence of a durable edge.",
    }


def learning_profile(state,min_trades=20,min_group_trades=8,strategy_version=CURRENT_STRATEGY_VERSION):
    """Conservative forward-only learning profile.

    It does not rewrite historical scores and it does not optimize thresholds.
    A factor is only marked positive/negative after enough closed paper trades.
    """
    attr=performance_attribution(state,strategy_version)
    total=len(eligible_strategy_trades(state,strategy_version,True))
    profile={"enabled":total>=min_trades,"closed_trades":total,"minimum_closed_trades":min_trades,
             "strategy_version":strategy_version,"realtime_evidence_only":True,
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
    profile["note"]="Learning uses only current-strategy trades entered from real-time data. It remains disabled until the minimum forward sample is reached; eligible adjustments are capped at ±10% and never rewrite prior decisions."
    return profile

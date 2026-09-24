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

def _wilson(k,n,z=1.96):
    if not n:
        return [None,None]
    p=k/n
    d=1+z*z/n
    center=(p+z*z/(2*n))/d
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d
    return [_f(center-half),_f(center+half)]

def _historical_outcomes(contract, spot, entry, returns, overlap_horizon=1):
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
    n=int(len(pnl))
    wins=int(np.sum(pnl>0))
    prob=wins/n
    ordered_ret=np.sort(ret)
    trim=max(0,int(math.floor(n*.10)))
    trimmed_ret=ordered_ret[trim:n-trim] if trim>0 and n>2*trim else ordered_ret
    p10_pnl=float(np.quantile(pnl,.10))
    worst_tail=pnl[pnl<=p10_pnl]
    # h-day forward returns overlap when sampled daily. Use a conservative
    # overlap-adjusted effective count for uncertainty and qualification gates.
    effective_n=max(1,int(n/max(1,int(overlap_horizon))))
    return {
        "samples":n,
        "effective_samples":effective_n,
        "prob_profit":_f(prob),
        "prob_profit_ci95":_wilson(prob*effective_n,effective_n),
        "prob_total_premium_loss":_f(np.mean(intrinsic<=0)),
        "expected_pnl_per_contract":_f(np.mean(pnl)),
        "median_pnl_per_contract":_f(np.median(pnl)),
        "p10_pnl_per_contract":_f(p10_pnl),
        "expected_shortfall_10_pnl":_f(np.mean(worst_tail)) if len(worst_tail) else None,
        "p25_pnl_per_contract":_f(np.quantile(pnl,.25)),
        "p75_pnl_per_contract":_f(np.quantile(pnl,.75)),
        "p90_pnl_per_contract":_f(np.quantile(pnl,.90)),
        "expected_return_on_debit":_f(np.mean(ret)),
        "trimmed_mean_return_on_debit":_f(np.mean(trimmed_ret)) if len(trimmed_ret) else None,
        "median_return_on_debit":_f(np.median(ret)),
        "prob_return_ge_50pct":_f(np.mean(ret>=.50)),
        "prob_return_ge_100pct":_f(np.mean(ret>=1.0)),
        "prob_loss_ge_50pct":_f(np.mean(ret<=-.50)),
    }

def _guidance_payload(rows, evidence, current_regime=None, context=None, relative_strength=None, model_probability=None, options_summary=None):
    """Build neutral decision-support lenses from already-screened contracts."""
    eligible=[r for r in rows if r.get("state")=="investigate"]
    if not eligible:
        return {
            "state":"no_strong_contract",
            "best_overall":None,
            "highest_upside":None,
            "higher_probability":None,
            "note":"No contract currently clears the full quality, execution and historical-evidence gates.",
        }

    def downside_return(r):
        entry=float(r.get("entry_quote") or 0)
        p10=r.get("p10_pnl_per_contract")
        return (float(p10)/(entry*100.0)) if entry>0 and p10 is not None else -1.0

    def reward_downside(r):
        ev=float(r.get("expected_return_on_debit") or 0)
        d=abs(min(0.0,downside_return(r)))
        return ev/(d if d>1e-9 else .01)

    auc=(evidence or {}).get("mean_roc_auc")
    lift_ci=(evidence or {}).get("historical_lift_ci95") or [None,None]
    ctx=context or {}
    earnings=(ctx.get("earnings") or {})
    days_to_earnings=earnings.get("days_to_earnings")
    catalyst_flags=list(ctx.get("catalyst_flags") or [])
    rel20=(relative_strength or {}).get("vs_spy_20d")
    implied_moves=((options_summary or {}).get("implied_moves_by_expiration") or {})
    historical_earnings_move=earnings.get("avg_abs_1d_move")

    # User-facing guidance is stricter than the raw research screen: avoid
    # presenting ultra-short contracts as the default "best" choice.
    guidance_eligible=[r for r in eligible if int(r.get("dte") or 0)>=3]
    if guidance_eligible:
        eligible=guidance_eligible

    enriched=[]
    for r in eligible:
        q=dict(r)
        p=float(q.get("prob_profit") or 0)
        ci=q.get("prob_profit_ci95") or [None,None]
        ci_floor=float(ci[0]) if ci and ci[0] is not None else 0.0
        raw_ev=float(q.get("expected_return_on_debit") or 0)
        robust_ev=q.get("trimmed_mean_return_on_debit")
        ev=float(robust_ev) if robust_ev is not None else raw_ev
        loss=float(q.get("prob_total_premium_loss") or 0)
        rr=reward_downside(q)
        combined=float(q.get("score") or 0)

        # Historical option payoff and execution remain the dominant inputs.
        combined += max(-6,min(6,(ci_floor-.50)*30))
        combined += max(-6,min(6,ev*8))
        combined += max(-5,min(5,(rr-.25)*4))
        combined -= max(0,min(6,(loss-.35)*12))

        # Directional evidence is intentionally low weight and only activates
        # when walk-forward validation is at least modest.
        direction_points=0.0
        side=str(q.get("type") or "").lower()
        if auc is not None and float(auc)>=.53:
            validation_points=max(-2.0,min(2.0,(float(auc)-.50)*20))
            combined += validation_points
            if lift_ci[0] is not None and float(lift_ci[0])>0:
                direction_points += 3.0 if side=="call" else (-3.0 if side=="put" else 0.0)
            elif lift_ci[1] is not None and float(lift_ci[1])<0:
                direction_points += 3.0 if side=="put" else (-3.0 if side=="call" else 0.0)
            if model_probability is not None:
                raw=max(-3.0,min(3.0,(float(model_probability)-.50)*12))
                direction_points += raw if side=="call" else (-raw if side=="put" else 0.0)
        combined += max(-5.0,min(5.0,direction_points))

        # Regime and relative strength are small contextual modifiers rather
        # than primary signals.
        context_points=0.0
        regime=str(current_regime or "")
        if "Uptrend" in regime:
            context_points += 1.5 if side=="call" else (-1.0 if side=="put" else 0.0)
        elif "Downtrend" in regime:
            context_points += 1.5 if side=="put" else (-1.0 if side=="call" else 0.0)
        if rel20 is not None and abs(float(rel20))>=.02:
            rs=1.0 if float(rel20)>0 else -1.0
            context_points += rs if side=="call" else (-rs if side=="put" else 0.0)
        combined += max(-2.5,min(2.5,context_points))

        # Earnings inside the contract life is treated as event risk unless a
        # dedicated event model exists; headlines are surfaced but not scored
        # directionally from text alone.
        event_points=0.0
        event_notes=[]
        dte=q.get("dte")
        if days_to_earnings is not None and dte is not None and 0<=int(days_to_earnings)<=int(dte):
            event_points=-2.5 if int(days_to_earnings)<=7 else -1.0
            event_notes.append(f"Earnings falls inside this contract's life ({int(days_to_earnings)} days)")
            exp_move=((implied_moves.get(str(q.get("expiration"))) or {}).get("move"))
            if historical_earnings_move is not None and exp_move is not None and float(historical_earnings_move)>0:
                earnings_move_ratio=float(exp_move)/float(historical_earnings_move)
                if earnings_move_ratio>=1.25:
                    event_points-=2.0
                    event_notes.append(f"Expiration implied move {float(exp_move)*100:.1f}% is above the historical average earnings move {float(historical_earnings_move)*100:.1f}%")
                elif earnings_move_ratio<=.80:
                    event_points+=0.5
                    event_notes.append(f"Expiration implied move {float(exp_move)*100:.1f}% is below the historical average earnings move {float(historical_earnings_move)*100:.1f}%")
                else:
                    event_notes.append(f"Expiration implied move {float(exp_move)*100:.1f}% is near the historical average earnings move {float(historical_earnings_move)*100:.1f}%")
            else:
                earnings_move_ratio=None
        else:
            exp_move=None
            earnings_move_ratio=None
        if catalyst_flags:
            event_notes.append("Current catalyst themes: "+", ".join(map(str,catalyst_flags[:4])))
        combined += event_points

        quote_confidence=str(q.get("data_confidence") or "unknown").lower()
        execution_live=q.get("execution_realtime") is True
        quote_age_seconds=q.get("quote_age_seconds")
        if quote_age_seconds is None and q.get("quote_age_hours") is not None:
            quote_age_seconds=float(q.get("quote_age_hours"))*3600.0
        execution_points=0.0
        if quote_confidence=="conflict":
            execution_points=-10.0
        elif execution_live:
            execution_points=4.0 if quote_confidence=="high" else (2.5 if quote_confidence=="medium" else 1.0)
            if quote_age_seconds is not None:
                execution_points += 1.0 if float(quote_age_seconds)<=15 else (0.0 if float(quote_age_seconds)<=60 else -3.0)
        else:
            execution_points=-2.0
        combined += execution_points

        q["combined_evidence_score"]=round(max(0,min(100,combined)),1)
        q["execution_status"]=("conflict" if quote_confidence=="conflict" else ("verified_live" if execution_live else "research_snapshot"))
        q["execution_points"]=_f(execution_points)
        # Explain the practical risk profile separately from the score.
        dte_i=int(q.get("dte") or 0)
        theta_abs=abs(float(q.get("theta_cost_pct_per_day") or 0))
        spread_abs=abs(float(q.get("spread_pct") or 0))
        full_loss=float(q.get("prob_total_premium_loss") or 0)
        if dte_i<=5 or theta_abs>.035 or full_loss>.30:
            q["risk_tier"]="aggressive"
        elif dte_i<=14 or theta_abs>.02 or spread_abs>.12 or full_loss>.18:
            q["risk_tier"]="moderate"
        else:
            q["risk_tier"]="lower_relative_risk"
        ci_width=(float(ci[1])-float(ci[0])) if ci[0] is not None and ci[1] is not None else None
        effective_n=int(q.get("effective_samples") if q.get("effective_samples") is not None else (q.get("samples") or 0))
        if effective_n>=300 and ci_width is not None and ci_width<=.10:
            q["evidence_confidence"]="higher"
        elif effective_n>=150 and ci_width is not None and ci_width<=.16:
            q["evidence_confidence"]="moderate"
        else:
            q["evidence_confidence"]="limited"
        q["simulator_priority"]=bool(
            dte_i>=3 and q["combined_evidence_score"]>=72 and p>=.58 and ev>0
            and effective_n>=60 and q["execution_status"]!="conflict"
        )
        q["execution_verified_for_forward_test"]=bool(
            execution_live and quote_confidence!="conflict"
            and quote_age_seconds is not None and float(quote_age_seconds)<=60
        )
        q["historical_downside_return_p10"]=_f(downside_return(q))
        q["reward_to_p10_downside"]=_f(rr)
        q["all_data_components"]={
            "contract_quality_score":_f(q.get("score")),
            "historical_profit_frequency":_f(q.get("prob_profit")),
            "historical_profit_frequency_ci95":q.get("prob_profit_ci95"),
            "historical_raw_samples":q.get("samples"),
            "historical_overlap_adjusted_samples":q.get("effective_samples"),
            "historical_mean_return_on_debit":_f(q.get("expected_return_on_debit")),
            "historical_trimmed_mean_return_on_debit":_f(q.get("trimmed_mean_return_on_debit")),
            "historical_median_return_on_debit":_f(q.get("median_return_on_debit")),
            "historical_prob_return_ge_50pct":_f(q.get("prob_return_ge_50pct")),
            "historical_prob_return_ge_100pct":_f(q.get("prob_return_ge_100pct")),
            "historical_prob_loss_ge_50pct":_f(q.get("prob_loss_ge_50pct")),
            "historical_expected_shortfall_10_pnl":_f(q.get("expected_shortfall_10_pnl")),
            "historical_p10_return_on_debit":_f(downside_return(q)),
            "full_premium_loss_frequency":_f(q.get("prob_total_premium_loss")),
            "reward_to_p10_downside":_f(rr),
            "walk_forward_auc":_f(auc),
            "model_probability_5d_up":_f(model_probability),
            "directional_alignment_points":_f(direction_points),
            "context_alignment_points":_f(context_points),
            "event_risk_points":_f(event_points),
            "execution_quality_points":_f(execution_points),
            "execution_status":q.get("execution_status"),
            "execution_verified_for_forward_test":q.get("execution_verified_for_forward_test"),
            "market_data_provider":q.get("selected_quote_provider"),
            "market_data_provider_key":q.get("selected_quote_provider_key"),
            "market_data_confidence":q.get("data_confidence"),
            "provider_agreement_pct":_f(q.get("provider_agreement_pct")),
            "quote_age_seconds":_f(quote_age_seconds),
            "regime":current_regime,
            "relative_strength_vs_spy_20d":_f(rel20),
            "days_to_earnings":days_to_earnings,
            "historical_avg_abs_earnings_move":_f(historical_earnings_move),
            "expiration_implied_move":_f(exp_move),
            "earnings_implied_vs_historical_ratio":_f(earnings_move_ratio),
            "catalyst_flags":catalyst_flags[:6],
        }
        q["guidance_explanation"]=[
            f"Historical replay profit frequency {p*100:.0f}%",
            f"Robust historical mean return on debit {ev*100:.0f}% (raw mean {raw_ev*100:.0f}%)",
            f"Full-premium-loss frequency {loss*100:.0f}%",
            f"Contract quality score {float(q.get('score') or 0):.1f}/100",
            ("Directional ML remains weak and receives no directional boost"
             if auc is None or float(auc)<.53 else
             f"Validated directional evidence contributes {direction_points:+.1f} points"),
            f"Regime/relative-strength context contributes {context_points:+.1f} points",
            ("Execution quote is verified live" if q.get("execution_status")=="verified_live" else
             ("Market-data providers conflict; forward execution is blocked" if q.get("execution_status")=="conflict" else
              "Current ranking uses a research snapshot; verify the live quote before forward testing")),
            *event_notes,
        ]
        if event_notes:
            q["risks"]=list(dict.fromkeys((q.get("risks") or [])+event_notes))
        enriched.append(q)

    best=max(enriched,key=lambda r:(r["combined_evidence_score"],r.get("score") or 0))
    upside=max(enriched,key=lambda r:(r.get("trimmed_mean_return_on_debit") if r.get("trimmed_mean_return_on_debit") is not None else (r.get("expected_return_on_debit") if r.get("expected_return_on_debit") is not None else -1e9),
                                     r.get("combined_evidence_score") or 0))
    probability=max(enriched,key=lambda r:((r.get("prob_profit_ci95") or [0])[0] or 0,
                                          r.get("prob_profit") or 0,
                                          r.get("combined_evidence_score") or 0))
    return {
        "state":"strong_candidates",
        "best_overall":best,
        "highest_upside":upside,
        "higher_probability":probability,
        "note":"Candidates are ranked from current execution quality plus historical replay and validated evidence. They are not guaranteed future-return estimates.",
    }


def build_opportunity_radar(raw, contracts, regime_series, current_regime, evidence, spot, limit=12, learning=None, context=None, relative_strength=None, model_probability=None, options_summary=None):
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
        stats=_historical_outcomes(c,float(spot),float(entry),sample,overlap_horizon=h)
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
        if stats.get("effective_samples",0)<60:
            score-=8
        base_score=max(0,min(100,score))
        learned_multiplier=1.0
        learned_components=[]
        lp=learning or {}
        if lp.get("enabled"):
            factors=lp.get("factors") or {}
            def apply_factor(name, group):
                nonlocal learned_multiplier
                row=next((z for z in factors.get(name,[]) if str(z.get("group"))==str(group)),None)
                if row and row.get("status") in ("positive_forward_evidence","negative_forward_evidence"):
                    m=float(row.get("weight_multiplier") or 1.0)
                    learned_multiplier*=m
                    learned_components.append({"factor":name,"group":str(group),"multiplier":m,"trades":row.get("trades"),"status":row.get("status")})
            apply_factor("type",c.get("type","unknown"))
            apply_factor("regime",current_regime)
            d=c.get("dte")
            dgroup="unknown" if d is None else ("0-7" if int(d)<=7 else ("8-21" if int(d)<=21 else ("22-45" if int(d)<=45 else "46+")))
            apply_factor("dte",dgroup)
            pgroup="65%+" if pp>=.65 else ("60-65%" if pp>=.60 else "<60%")
            apply_factor("historical_probability",pgroup)
            ivr=c.get("iv_rv_ratio")
            ivgroup="unknown" if ivr is None else ("IV<0.9xRV" if float(ivr)<.9 else ("0.9-1.2x" if float(ivr)<=1.2 else "IV>1.2xRV"))
            apply_factor("iv_environment",ivgroup)
        # Keep adaptation bounded even if several eligible factors align.
        learned_multiplier=max(.85,min(1.15,learned_multiplier))
        score=max(0,min(100,base_score*learned_multiplier))

        reasons=[]
        risks=[]
        if pp>=.60: reasons.append(f"Historical replay cleared breakeven {pp*100:.0f}% of the time")
        if ev is not None and ev>0: reasons.append("Historical payoff replay had positive mean P/L")
        if spread is not None and spread<=.10: reasons.append("Quoted spread is relatively tight")
        if oi>=500 or vol>=100: reasons.append("Higher open interest or activity")
        if theta is not None and theta>.025: risks.append("Daily theta is high relative to premium")
        if spread is not None and spread>.20: risks.append("Wide spread can materially reduce realized returns")
        if loss>=.50: risks.append(f"Historical replay lost the full premium about {loss*100:.0f}% of the time")
        if stats.get("effective_samples",0)<60: risks.append("Overlap-adjusted historical sample is limited")
        dte=c.get("dte")
        iv=c.get("iv")
        bid=c.get("bid")
        ask=c.get("ask")
        if dte is None or int(dte)<2: risks.append("Near-expiry contracts are excluded from strong-opportunity status")
        if iv is not None and (float(iv)<.03 or float(iv)>5.0): risks.append("Implied volatility input appears unreliable for screening")
        if bid is None or ask is None or float(bid)<=0 or float(ask)<=0 or float(ask)<float(bid): risks.append("Two-sided executable quote is unavailable")
        if (evidence or {}).get("mean_roc_auc") is None or (evidence or {}).get("mean_roc_auc",0)<.53:
            risks.append("Directional ML model has not demonstrated strong out-of-sample discrimination")

        state="watch"
        hard_quality_gate=(dte is None or int(dte)<2 or (spread is not None and spread>.20)
                           or (theta is not None and theta>.05)
                           or (iv is not None and (float(iv)<.03 or float(iv)>5.0))
                           or bid is None or ask is None or float(bid)<=0 or float(ask)<=0 or float(ask)<float(bid))
        if not hard_quality_gate and score>=72 and pp>=.58 and ev is not None and ev>0 and stats.get("effective_samples",0)>=60:
            state="investigate"
        elif score<55 or ev is None or ev<=0:
            state="pass"

        rows.append({
            "contract_symbol":c.get("contract_symbol"),"type":c.get("type"),
            "expiration":c.get("expiration"),"dte":c.get("dte"),"strike":c.get("strike"),
            "entry_quote":_f(entry),
            "bid":_f(c.get("bid")),"ask":_f(c.get("ask")),"spread_pct":_f(spread),
            "quote_age_hours":_f(c.get("quote_age_hours")),"quote_age_seconds":_f(c.get("quote_age_seconds")),
            "selected_quote_provider":c.get("selected_quote_provider") or c.get("provider"),
            "selected_quote_provider_key":c.get("selected_quote_provider_key") or c.get("provider_key"),
            "selected_quote_feed":c.get("selected_quote_feed") or c.get("feed"),
            "data_confidence":c.get("data_confidence"),
            "provider_agreement_pct":_f(c.get("provider_agreement_pct")),
            "execution_realtime":bool(c.get("execution_realtime")),
            "realtime":bool(c.get("realtime")),"consolidated":bool(c.get("consolidated")),
            "iv":_f(iv),"iv_rv_ratio":_f(c.get("iv_rv_ratio")),
            "theta_cost_pct_per_day":_f(theta),"open_interest":int(oi),"volume":int(vol),
            "breakeven":_f(c.get("breakeven")),"breakeven_move":_f(c.get("breakeven_move")),
            "base_score":round(base_score,1),"learned_multiplier":round(learned_multiplier,3),"learned_adjustment_points":round(score-base_score,1),"learning_components":learned_components,"score":round(score,1),"state":state,
            "historical_scope":"same regime" if len(same)>=80 else "all regimes",
            "trading_day_horizon":h,**stats,"reasons":reasons,"risks":risks,
        })

    rows.sort(key=lambda x:(x["state"]=="investigate",x["score"],x.get("expected_pnl_per_contract") or -1e9),reverse=True)
    surfaced=[x for x in rows if x["state"]=="investigate"][:limit]
    watch=[x for x in rows if x["state"]=="watch"][:limit]
    guidance=_guidance_payload(rows,evidence,current_regime=current_regime,context=context,relative_strength=relative_strength,model_probability=model_probability,options_summary=options_summary)
    return {
        "state":"opportunities_detected" if surfaced else "no_strong_setup",
        "opportunities":surfaced,
        "watchlist":watch,
        "guidance":guidance,
        "contracts_evaluated":len(rows),
        "learning_enabled":bool((learning or {}).get("enabled")),
        "method_note":"Today's contract economics replayed across historical underlying moves. Profit-frequency uncertainty and qualification use an overlap-adjusted effective sample count because multi-day forward returns overlap. Results are hypothetical, exclude changing historical IV/Greeks and are not a profitability guarantee.",
    }



def build_market_guidance(tickers,limit=10):
    """Combine per-ticker guidance into one market-wide research shortlist."""
    overall=[];upside=[];probability=[]
    for t in tickers or []:
        ticker=t.get("ticker")
        guidance=(((t.get("options") or {}).get("opportunity_radar") or {}).get("guidance") or {})
        if guidance.get("state")!="strong_candidates":
            continue
        for key,dest in (("best_overall",overall),("highest_upside",upside),("higher_probability",probability)):
            row=guidance.get(key)
            if row:
                dest.append({**row,"ticker":ticker})

    overall.sort(key=lambda r:(float(r.get("combined_evidence_score") or r.get("score") or 0),
                               float(r.get("expected_return_on_debit") or -1e9)),reverse=True)
    upside.sort(key=lambda r:(float(r.get("trimmed_mean_return_on_debit") if r.get("trimmed_mean_return_on_debit") is not None else (r.get("expected_return_on_debit") or -1e9)),
                              float(r.get("combined_evidence_score") or r.get("score") or 0)),reverse=True)
    probability.sort(key=lambda r:(
        float(((r.get("prob_profit_ci95") or [0,None])[0]) or 0),
        float(r.get("prob_profit") or 0),
        float(r.get("combined_evidence_score") or r.get("score") or 0),
    ),reverse=True)

    return {
        "state":"strong_candidates" if overall else "no_strong_contract",
        "strongest_overall":overall[0] if overall else None,
        "highest_historical_upside":upside[0] if upside else None,
        "highest_historical_profit_frequency":probability[0] if probability else None,
        "top_overall":overall[:limit],
        "tickers_with_strong_candidates":len({r.get("ticker") for r in overall if r.get("ticker")}),
        "note":"Market-wide lenses compare only deeply analyzed contracts that already cleared per-ticker quality gates. Historical payoff statistics are not forecasts.",
    }

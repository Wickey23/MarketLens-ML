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
    n=int(len(pnl))
    wins=int(np.sum(pnl>0))
    return {
        "samples":n,
        "prob_profit":_f(wins/n),
        "prob_profit_ci95":_wilson(wins,n),
        "prob_total_premium_loss":_f(np.mean(intrinsic<=0)),
        "expected_pnl_per_contract":_f(np.mean(pnl)),
        "median_pnl_per_contract":_f(np.median(pnl)),
        "p10_pnl_per_contract":_f(np.quantile(pnl,.10)),
        "p25_pnl_per_contract":_f(np.quantile(pnl,.25)),
        "p75_pnl_per_contract":_f(np.quantile(pnl,.75)),
        "p90_pnl_per_contract":_f(np.quantile(pnl,.90)),
        "expected_return_on_debit":_f(np.mean(ret)),
        "median_return_on_debit":_f(np.median(ret)),
    }

def _guidance_payload(rows, evidence, current_regime=None, context=None, relative_strength=None, model_probability=None):
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
        ev=float(q.get("expected_return_on_debit") or 0)
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
        if catalyst_flags:
            event_notes.append("Current catalyst themes: "+", ".join(map(str,catalyst_flags[:4])))
        combined += event_points

        q["combined_evidence_score"]=round(max(0,min(100,combined)),1)
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
        if q.get("samples",0)>=300 and ci_width is not None and ci_width<=.10:
            q["evidence_confidence"]="higher"
        elif q.get("samples",0)>=150 and ci_width is not None and ci_width<=.16:
            q["evidence_confidence"]="moderate"
        else:
            q["evidence_confidence"]="limited"
        q["simulator_priority"]=bool(
            dte_i>=3 and q["combined_evidence_score"]>=72 and p>=.58 and ev>0
            and q.get("samples",0)>=100
        )
        q["historical_downside_return_p10"]=_f(downside_return(q))
        q["reward_to_p10_downside"]=_f(rr)
        q["all_data_components"]={
            "contract_quality_score":_f(q.get("score")),
            "historical_profit_frequency":_f(q.get("prob_profit")),
            "historical_profit_frequency_ci95":q.get("prob_profit_ci95"),
            "historical_mean_return_on_debit":_f(q.get("expected_return_on_debit")),
            "historical_median_return_on_debit":_f(q.get("median_return_on_debit")),
            "historical_p10_return_on_debit":_f(downside_return(q)),
            "full_premium_loss_frequency":_f(q.get("prob_total_premium_loss")),
            "reward_to_p10_downside":_f(rr),
            "walk_forward_auc":_f(auc),
            "model_probability_5d_up":_f(model_probability),
            "directional_alignment_points":_f(direction_points),
            "context_alignment_points":_f(context_points),
            "event_risk_points":_f(event_points),
            "regime":current_regime,
            "relative_strength_vs_spy_20d":_f(rel20),
            "days_to_earnings":days_to_earnings,
            "catalyst_flags":catalyst_flags[:6],
        }
        q["guidance_explanation"]=[
            f"Historical replay profit frequency {p*100:.0f}%",
            f"Historical mean return on debit {ev*100:.0f}%",
            f"Full-premium-loss frequency {loss*100:.0f}%",
            f"Contract quality score {float(q.get('score') or 0):.1f}/100",
            ("Directional ML remains weak and receives no directional boost"
             if auc is None or float(auc)<.53 else
             f"Validated directional evidence contributes {direction_points:+.1f} points"),
            f"Regime/relative-strength context contributes {context_points:+.1f} points",
            *event_notes,
        ]
        if event_notes:
            q["risks"]=list(dict.fromkeys((q.get("risks") or [])+event_notes))
        enriched.append(q)

    best=max(enriched,key=lambda r:(r["combined_evidence_score"],r.get("score") or 0))
    upside=max(enriched,key=lambda r:(r.get("expected_return_on_debit") if r.get("expected_return_on_debit") is not None else -1e9,
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


def build_opportunity_radar(raw, contracts, regime_series, current_regime, evidence, spot, limit=12, learning=None, context=None, relative_strength=None, model_probability=None):
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
        if stats["samples"]<100: risks.append("Historical sample is limited")
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
        if not hard_quality_gate and score>=72 and pp>=.58 and ev is not None and ev>0 and stats["samples"]>=100:
            state="investigate"
        elif score<55 or ev is None or ev<=0:
            state="pass"

        rows.append({
            "contract_symbol":c.get("contract_symbol"),"type":c.get("type"),
            "expiration":c.get("expiration"),"dte":c.get("dte"),"strike":c.get("strike"),
            "entry_quote":_f(entry),
            "bid":_f(c.get("bid")),"ask":_f(c.get("ask")),"spread_pct":_f(spread),
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
    guidance=_guidance_payload(rows,evidence,current_regime=current_regime,context=context,relative_strength=relative_strength,model_probability=model_probability)
    return {
        "state":"opportunities_detected" if surfaced else "no_strong_setup",
        "opportunities":surfaced,
        "watchlist":watch,
        "guidance":guidance,
        "contracts_evaluated":len(rows),
        "learning_enabled":bool((learning or {}).get("enabled")),
        "method_note":"Today's contract economics replayed across historical underlying moves. Results are hypothetical, exclude changing historical IV/Greeks and are not a profitability guarantee.",
    }

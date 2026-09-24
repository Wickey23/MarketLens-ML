import pandas as pd
from src.opportunity_radar import build_opportunity_radar, _guidance_payload, build_market_guidance, _historical_outcomes

def test_radar_surfaces_and_reports_risk():
    idx=pd.bdate_range("2020-01-01",periods=320)
    close=pd.Series([100*(1.001**i) for i in range(len(idx))],index=idx)
    raw=pd.DataFrame({"Close":close})
    regimes=pd.Series(["Uptrend"]*len(idx),index=idx)
    contracts=[{
        "contract_symbol":"TEST","type":"call","expiration":"2030-01-01","dte":14,
        "strike":float(close.iloc[-1]),"ask":1.0,"mid":.95,"spread_pct":.05,
        "theta_cost_pct_per_day":.008,"open_interest":1000,"volume":200,
    }]
    r=build_opportunity_radar(raw,contracts,regimes,"Uptrend",{"mean_roc_auc":.55},float(close.iloc[-1]))
    assert r["contracts_evaluated"]==1
    row=(r["opportunities"] or r["watchlist"])[0]
    assert 0<=row["score"]<=100
    assert row["samples"]>=100
    assert row["effective_samples"] < row["samples"]
    assert "prob_profit" in row
    assert "prob_total_premium_loss" in row

def test_radar_returns_no_setup_for_empty_chain():
    idx=pd.bdate_range("2024-01-01",periods=100)
    raw=pd.DataFrame({"Close":[100.0]*100},index=idx)
    regimes=pd.Series(["Sideways"]*100,index=idx)
    r=build_opportunity_radar(raw,[],regimes,"Sideways",{},100)
    assert r["state"]=="no_strong_setup"


def test_radar_does_not_surface_zero_dte_as_strong_opportunity():
    idx=pd.bdate_range("2020-01-01",periods=320)
    close=pd.Series([100*(1.001**i) for i in range(len(idx))],index=idx)
    raw=pd.DataFrame({"Close":close})
    regimes=pd.Series(["Uptrend"]*len(idx),index=idx)
    contracts=[{
        "contract_symbol":"ZERO","type":"call","expiration":"2030-01-01","dte":0,
        "strike":float(close.iloc[-1]),"ask":1.0,"mid":.99,"spread_pct":.01,
        "theta_cost_pct_per_day":.01,"open_interest":1000,"volume":500,"iv":.25,
    }]
    out=build_opportunity_radar(raw,contracts,regimes,"Uptrend",{"mean_roc_auc":.55},float(close.iloc[-1]))
    assert out["opportunities"]==[]


def test_guidance_exposes_three_decision_lenses():
    rows=[
        {
            "contract_symbol":"A","state":"investigate","score":82.0,
            "entry_quote":2.0,"prob_profit":0.66,"prob_profit_ci95":[0.60,0.71],
            "prob_total_premium_loss":0.18,"expected_return_on_debit":0.22,
            "p10_pnl_per_contract":-80.0,
        },
        {
            "contract_symbol":"B","state":"investigate","score":77.0,
            "entry_quote":1.0,"prob_profit":0.60,"prob_profit_ci95":[0.54,0.66],
            "prob_total_premium_loss":0.25,"expected_return_on_debit":0.70,
            "p10_pnl_per_contract":-90.0,
        },
    ]
    g=_guidance_payload(rows,{"mean_roc_auc":0.56,"historical_lift_ci95":[0.01,0.08]})
    assert g["state"]=="strong_candidates"
    assert g["best_overall"] is not None
    assert g["highest_upside"]["contract_symbol"]=="B"
    assert g["higher_probability"]["contract_symbol"]=="A"
    assert "combined_evidence_score" in g["best_overall"]


def test_guidance_returns_explicit_no_trade_state():
    g=_guidance_payload([
        {"contract_symbol":"WATCH","state":"watch","score":90.0}
    ],{"mean_roc_auc":0.60})
    assert g["state"]=="no_strong_contract"
    assert g["best_overall"] is None


def test_guidance_uses_validated_direction_and_context():
    base={
        "state":"investigate","score":78.0,"entry_quote":2.0,
        "prob_profit":0.63,"prob_profit_ci95":[0.58,0.68],
        "prob_total_premium_loss":0.20,"expected_return_on_debit":0.25,
        "median_return_on_debit":0.10,"p10_pnl_per_contract":-90.0,
        "dte":21,"risks":[],"reasons":[],
    }
    rows=[
        {**base,"contract_symbol":"CALL","type":"call"},
        {**base,"contract_symbol":"PUT","type":"put"},
    ]
    g=_guidance_payload(
        rows,
        {"mean_roc_auc":0.58,"historical_lift_ci95":[0.02,0.08]},
        current_regime="Uptrend / Normal Vol",
        context={"earnings":{"days_to_earnings":10},"catalyst_flags":["earnings"]},
        relative_strength={"vs_spy_20d":0.05},
        model_probability=0.64,
    )
    call=next(x for x in (g["best_overall"],g["highest_upside"],g["higher_probability"]) if x and x["contract_symbol"]=="CALL")
    put=[x for x in [g["best_overall"],g["highest_upside"],g["higher_probability"]] if x and x["contract_symbol"]=="PUT"]
    assert call["combined_evidence_score"] > 78.0
    assert call["all_data_components"]["directional_alignment_points"] > 0
    assert call["all_data_components"]["context_alignment_points"] > 0
    assert call["all_data_components"]["event_risk_points"] < 0
    assert any("Earnings falls inside" in x for x in call["risks"])


def test_guidance_prefers_three_plus_dte_for_default_choices():
    base={
        "state":"investigate","score":90.0,"entry_quote":2.0,
        "prob_profit":0.64,"prob_profit_ci95":[0.59,0.69],
        "prob_total_premium_loss":0.20,"expected_return_on_debit":0.30,
        "median_return_on_debit":0.12,"p10_pnl_per_contract":-80.0,
        "samples":250,"theta_cost_pct_per_day":0.02,"spread_pct":0.05,
        "risks":[],"reasons":[],
    }
    rows=[
        {**base,"contract_symbol":"TWO","type":"call","dte":2,"score":99.0,"expected_return_on_debit":0.80},
        {**base,"contract_symbol":"SEVEN","type":"call","dte":7,"score":86.0,"expected_return_on_debit":0.35},
    ]
    g=_guidance_payload(rows,{"mean_roc_auc":0.56,"historical_lift_ci95":[0.01,0.05]})
    assert g["best_overall"]["contract_symbol"]=="SEVEN"
    assert g["highest_upside"]["contract_symbol"]=="SEVEN"
    assert g["best_overall"]["risk_tier"] in ("moderate","lower_relative_risk","aggressive")
    assert g["best_overall"]["evidence_confidence"] in ("higher","moderate","limited")
    assert g["best_overall"]["simulator_priority"] is True


def test_guidance_tracks_live_execution_quality():
    base={
        "state":"investigate","score":80.0,"entry_quote":2.0,
        "prob_profit":0.64,"prob_profit_ci95":[0.58,0.69],
        "prob_total_premium_loss":0.18,"expected_return_on_debit":0.28,
        "median_return_on_debit":0.12,"p10_pnl_per_contract":-75.0,
        "dte":14,"effective_samples":120,"theta_cost_pct_per_day":0.015,
        "spread_pct":0.05,"risks":[],"reasons":[],"type":"call",
    }
    live={**base,"contract_symbol":"LIVE","execution_realtime":True,
          "data_confidence":"high","quote_age_seconds":8,
          "selected_quote_provider":"Tradier Brokerage API"}
    snap={**base,"contract_symbol":"SNAP","execution_realtime":False,
          "data_confidence":"low","quote_age_seconds":300,
          "selected_quote_provider":"Yahoo Finance via yfinance"}
    g=_guidance_payload([live,snap],{"mean_roc_auc":0.55,"historical_lift_ci95":[0.0,0.04]})
    assert g["best_overall"]["contract_symbol"]=="LIVE"
    assert g["best_overall"]["execution_status"]=="verified_live"
    assert g["best_overall"]["execution_verified_for_forward_test"] is True
    assert g["best_overall"]["all_data_components"]["market_data_confidence"]=="high"


def test_guidance_marks_provider_conflict_and_blocks_forward_test():
    row={
        "contract_symbol":"CONFLICT","state":"investigate","score":90.0,
        "entry_quote":1.5,"prob_profit":0.66,"prob_profit_ci95":[0.59,0.72],
        "prob_total_premium_loss":0.16,"expected_return_on_debit":0.35,
        "median_return_on_debit":0.15,"p10_pnl_per_contract":-60.0,
        "dte":21,"effective_samples":150,"theta_cost_pct_per_day":0.01,
        "spread_pct":0.04,"risks":[],"reasons":[],"type":"call",
        "execution_realtime":True,"data_confidence":"conflict","quote_age_seconds":5,
    }
    g=_guidance_payload([row],{"mean_roc_auc":0.57,"historical_lift_ci95":[0.01,0.05]})
    x=g["best_overall"]
    assert x["execution_status"]=="conflict"
    assert x["execution_verified_for_forward_test"] is False
    assert x["simulator_priority"] is False
    assert x["execution_points"] < 0


def test_market_guidance_combines_ticker_leaders():
    def g(sym,score,ev,prob,ci):
        row={
            "contract_symbol":sym,"combined_evidence_score":score,"score":score,
            "expected_return_on_debit":ev,"prob_profit":prob,"prob_profit_ci95":ci,
        }
        return {"state":"strong_candidates","best_overall":row,"highest_upside":row,"higher_probability":row}
    tickers=[
        {"ticker":"AAA","options":{"opportunity_radar":{"guidance":g("AAA1",82,0.20,0.62,[0.56,0.68])}}},
        {"ticker":"BBB","options":{"opportunity_radar":{"guidance":g("BBB1",78,0.45,0.66,[0.60,0.71])}}},
    ]
    out=build_market_guidance(tickers)
    assert out["state"]=="strong_candidates"
    assert out["strongest_overall"]["ticker"]=="AAA"
    assert out["highest_historical_upside"]["ticker"]=="BBB"
    assert out["highest_historical_profit_frequency"]["ticker"]=="BBB"
    assert len(out["top_overall"])==2


def test_market_guidance_has_explicit_no_contract_state():
    out=build_market_guidance([{"ticker":"AAA","options":{"opportunity_radar":{"guidance":{"state":"no_strong_contract"}}}}])
    assert out["state"]=="no_strong_contract"
    assert out["strongest_overall"] is None


def test_historical_outcomes_reports_robust_return_and_tail_risk():
    contract={"type":"call","strike":100.0}
    returns=pd.Series([-0.20,-0.10,0.0,0.05,0.10,0.20,0.30,0.40,0.50,2.00])
    out=_historical_outcomes(contract,100.0,5.0,returns,overlap_horizon=1)
    assert out["trimmed_mean_return_on_debit"] is not None
    assert out["expected_shortfall_10_pnl"] is not None
    assert 0 <= out["prob_return_ge_50pct"] <= 1
    assert 0 <= out["prob_return_ge_100pct"] <= 1
    assert 0 <= out["prob_loss_ge_50pct"] <= 1
    assert out["trimmed_mean_return_on_debit"] < out["expected_return_on_debit"]


def test_guidance_accounts_for_earnings_implied_move_context():
    row={
        "contract_symbol":"EARN","state":"investigate","score":80.0,
        "entry_quote":2.0,"prob_profit":0.64,"prob_profit_ci95":[0.58,0.69],
        "prob_total_premium_loss":0.18,"expected_return_on_debit":0.28,
        "trimmed_mean_return_on_debit":0.24,"median_return_on_debit":0.10,
        "p10_pnl_per_contract":-70.0,"dte":14,"expiration":"2030-01-18",
        "effective_samples":120,"theta_cost_pct_per_day":0.015,
        "spread_pct":0.05,"risks":[],"reasons":[],"type":"call",
    }
    g=_guidance_payload(
        [row],
        {"mean_roc_auc":0.56,"historical_lift_ci95":[0.01,0.05]},
        current_regime="Uptrend / Normal Vol",
        context={"earnings":{"days_to_earnings":5,"avg_abs_1d_move":0.06}},
        relative_strength={"vs_spy_20d":0.02},
        model_probability=0.60,
        options_summary={"implied_moves_by_expiration":{"2030-01-18":{"move":0.09}}},
    )
    x=g["best_overall"]
    comp=x["all_data_components"]
    assert comp["expiration_implied_move"]==0.09
    assert comp["historical_avg_abs_earnings_move"]==0.06
    assert comp["earnings_implied_vs_historical_ratio"]==1.5
    assert comp["event_risk_points"] < 0
    assert any("above the historical average earnings move" in z for z in x["risks"])

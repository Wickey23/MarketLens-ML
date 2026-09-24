import pandas as pd
from src.opportunity_radar import build_opportunity_radar, _guidance_payload

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

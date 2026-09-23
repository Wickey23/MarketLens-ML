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

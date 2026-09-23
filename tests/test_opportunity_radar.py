import pandas as pd
from src.opportunity_radar import build_opportunity_radar

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

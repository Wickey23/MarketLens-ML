import math
import src.options_data as od
from src.options_data import _i, enrich_contract


def test_safe_integer_parsing_handles_nan_and_missing_values():
    assert _i(float("nan"))==0
    assert _i(None)==0
    assert _i("12")==12


def test_enriched_contract_keeps_core_risk_fields():
    row={
        "type":"call","dte":14,"strike":100.0,"breakeven":102.0,
        "ask":2.0,"mid":1.9,"iv":0.25,"open_interest":500,"volume":100,
        "spread_pct":0.05,
    }
    out=enrich_contract(row,spot=100.0,annual_rv=0.20,risk_free=0.04)
    assert out["entry_debit_per_contract"]==200.0
    assert out["max_loss_per_contract"]==200.0
    assert out["theta_cost_pct_per_day"] is not None
    assert 0 <= out["risk_neutral_prob_itm"] <= 1


def test_tradier_option_snapshot_uses_realtime_chain(monkeypatch):
    def fake_get(path,params):
        if path.endswith("expirations"):
            return {"expirations":{"date":["2099-01-16"]}}
        return {"options":{"option":[{
            "symbol":"ABC990116C00100000",
            "option_type":"call",
            "expiration_date":"2099-01-16",
            "strike":100.0,
            "bid":4.9,
            "ask":5.1,
            "last":5.0,
            "volume":100,
            "open_interest":500,
            "bid_date":4072381200000,
            "ask_date":4072381200000,
            "trade_date":4072381200000,
            "greeks":{
                "delta":0.55,"gamma":0.02,"theta":-0.05,"vega":0.08,
                "smv_vol":0.30,"updated_at":"2099-01-01 12:00:00"
            }
        }]}}
    monkeypatch.setattr(od,"_tradier_get",fake_get)
    monkeypatch.setattr(od,"_risk_free_rate",lambda:(0.04,"test"))
    out=od._tradier_option_snapshot("ABC",100.0,0.20,1,8)
    assert out["source"]=="Tradier Brokerage API"
    assert out["realtime"] is True
    assert out["greeks_frequency"]=="hourly"
    q=out["contracts"][0]
    assert q["contract_symbol"]=="ABC990116C00100000"
    assert q["bid"]==4.9
    assert q["ask"]==5.1
    assert q["delta"]==0.55
    assert q["theta_per_contract_per_day"]==-5.0
    assert q["greeks_source"]=="Tradier / ORATS (hourly)"

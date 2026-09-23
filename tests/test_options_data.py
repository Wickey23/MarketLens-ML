import math
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

import src.universe_scanner as us
from datetime import datetime, timezone
from src.fast_refresh import scan_is_fresh


def test_choose_research_universe_preserves_core_and_manual(monkeypatch):
    monkeypatch.setattr(us,"rank_market_universe",lambda limit=12:[
        {"ticker":"NVDA","scan_score":12.0},
        {"ticker":"AAPL","scan_score":11.0},
        {"ticker":"MSFT","scan_score":10.0},
    ][:limit])
    out=us.choose_research_universe(["SPY","QQQ"],["SOFI"],max_total=5)
    assert out["selected"]==["SPY","QQQ","SOFI","NVDA","AAPL"]
    assert out["manual"]==["SOFI"]
    assert out["broad_universe_size"]>=70
    assert out["deep_analysis_limit"]==5


def test_choose_research_universe_deduplicates(monkeypatch):
    monkeypatch.setattr(us,"rank_market_universe",lambda limit=12:[
        {"ticker":"SPY","scan_score":12.0},
        {"ticker":"NVDA","scan_score":11.0},
    ])
    out=us.choose_research_universe(["SPY"],["SPY","NVDA"],max_total=3)
    assert out["selected"]==["SPY","NVDA"]


def test_fast_refresh_reuses_recent_universe_scan():
    assert scan_is_fresh({"generated_at":datetime.now(timezone.utc).isoformat()},2.0) is True
    assert scan_is_fresh({"generated_at":"2000-01-01T00:00:00+00:00"},2.0) is False
    assert scan_is_fresh({},2.0) is False

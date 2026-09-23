import src.universe_scanner as us


def test_choose_research_universe_preserves_core_and_manual(monkeypatch):
    monkeypatch.setattr(us,"rank_market_universe",lambda limit=12:[
        {"ticker":"NVDA","scan_score":12.0},
        {"ticker":"AAPL","scan_score":11.0},
        {"ticker":"MSFT","scan_score":10.0},
    ][:limit])
    out=us.choose_research_universe(["SPY","QQQ"],["SOFI"],max_total=5)
    assert out["selected"]==["SPY","QQQ","SOFI","NVDA","AAPL"]
    assert out["manual"]==["SOFI"]


def test_choose_research_universe_deduplicates(monkeypatch):
    monkeypatch.setattr(us,"rank_market_universe",lambda limit=12:[
        {"ticker":"SPY","scan_score":12.0},
        {"ticker":"NVDA","scan_score":11.0},
    ])
    out=us.choose_research_universe(["SPY"],["SPY","NVDA"],max_total=3)
    assert out["selected"]==["SPY","NVDA"]

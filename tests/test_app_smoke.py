import app as market_app


def test_health_route():
    client=market_app.app.test_client()
    r=client.get('/api/health')
    assert r.status_code==200
    assert r.get_json()['status']=='ok'


def test_home_renders_template():
    client=market_app.app.test_client()
    r=client.get('/')
    assert r.status_code==200
    assert b'MarketLens' in r.data
    assert b'Options Lab' in r.data


def test_data_route_is_no_cache(monkeypatch):
    monkeypatch.setattr(market_app,'read_data',lambda:{'generated_at':'2026-01-01T00:00:00Z','horizon_days':5,'tickers':[],'errors':[]})
    client=market_app.app.test_client()
    r=client.get('/api/data')
    assert r.status_code==200
    assert r.headers['Cache-Control'].startswith('no-store')
    assert r.get_json()['tickers']==[]


def test_research_trigger_rejects_invalid_ticker_before_dispatch():
    client=market_app.app.test_client()
    r=client.post('/api/run-research',json={'ticker':'$BAD'})
    assert r.status_code==400


def test_research_trigger_requires_server_token(monkeypatch):
    monkeypatch.delenv('MARKETLENS_CONTROL_KEY',raising=False)
    monkeypatch.delenv('GITHUB_ACTIONS_TOKEN',raising=False)
    client=market_app.app.test_client()
    r=client.post('/api/run-research',json={'ticker':'SPY'})
    assert r.status_code==503


def test_live_quote_route(monkeypatch):
    monkeypatch.setattr(market_app,'live_quote',lambda ticker:{
        'ticker':ticker,'price':123.45,'change_pct':0.01,'provider':'test',
        'realtime':False,'delayed':True,'market_timestamp':'2026-01-01T15:30:00+00:00'
    })
    client=market_app.app.test_client()
    r=client.get('/api/live-quote?ticker=SPY')
    assert r.status_code==200
    j=r.get_json()
    assert j['ok'] is True
    assert j['ticker']=='SPY'
    assert j['price']==123.45
    assert r.headers['Cache-Control'].startswith('no-store')


def test_live_quote_rejects_invalid_ticker():
    client=market_app.app.test_client()
    r=client.get('/api/live-quote?ticker=$BAD')
    assert r.status_code==400


def test_live_quotes_batch_route(monkeypatch):
    monkeypatch.setattr(market_app,'live_quotes',lambda tickers:([
        {'ticker':t,'price':100.0+i,'change_pct':0.001*i,'provider':'test','delayed':True}
        for i,t in enumerate(tickers)
    ],[]))
    client=market_app.app.test_client()
    r=client.get('/api/live-quotes?tickers=SPY,QQQ')
    assert r.status_code==200
    j=r.get_json()
    assert j['ok'] is True
    assert [x['ticker'] for x in j['quotes']]==['SPY','QQQ']
    assert r.headers['Cache-Control'].startswith('no-store')


def test_live_quotes_rejects_invalid_list():
    client=market_app.app.test_client()
    r=client.get('/api/live-quotes?tickers=SPY,$BAD')
    assert r.status_code==400


def test_live_quote_prefers_provider_when_key_present(monkeypatch):
    market_app._live_quote_cache.clear()
    monkeypatch.delenv("TRADIER_ACCESS_TOKEN",raising=False)
    monkeypatch.setenv("FINNHUB_API_KEY","test-key")
    monkeypatch.setattr(market_app,"_finnhub_live_quote",lambda ticker,key:{
        "ticker":ticker,"price":200.0,"change_pct":0.02,"provider":"Finnhub quote",
        "realtime":True,"delayed":False
    })
    monkeypatch.setattr(market_app,"_yahoo_live_quote",lambda ticker:(_ for _ in ()).throw(AssertionError("Yahoo fallback should not run")))
    q=market_app.live_quote("SPY")
    assert q["price"]==200.0
    assert q["provider"]=="Finnhub quote"
    assert q["realtime"] is True


def test_market_stream_status_without_token(monkeypatch):
    monkeypatch.delenv("TRADIER_ACCESS_TOKEN",raising=False)
    client=market_app.app.test_client()
    r=client.get("/api/market-stream/status")
    assert r.status_code==200
    j=r.get_json()
    assert j["configured"] is False
    assert j["mode"]=="near-live-polling"


def test_market_stream_session_requires_token(monkeypatch):
    monkeypatch.delenv("MARKETLENS_CONTROL_KEY",raising=False)
    monkeypatch.delenv("TRADIER_ACCESS_TOKEN",raising=False)
    client=market_app.app.test_client()
    r=client.post("/api/market-stream/session")
    assert r.status_code==503


def test_market_stream_session_returns_browser_safe_session(monkeypatch):
    monkeypatch.delenv("MARKETLENS_CONTROL_KEY",raising=False)
    monkeypatch.setenv("TRADIER_ACCESS_TOKEN","secret")
    monkeypatch.setattr(market_app,"_create_tradier_market_session",lambda:"session-123")
    client=market_app.app.test_client()
    r=client.post("/api/market-stream/session")
    assert r.status_code==200
    j=r.get_json()
    assert j["ok"] is True
    assert j["sessionid"]=="session-123"
    assert "TRADIER_ACCESS_TOKEN" not in r.get_data(as_text=True)


def test_live_quote_prefers_tradier_when_configured(monkeypatch):
    market_app._live_quote_cache.clear()
    monkeypatch.setenv("TRADIER_ACCESS_TOKEN","tradier-test")
    monkeypatch.setenv("FINNHUB_API_KEY","finnhub-test")
    monkeypatch.setattr(market_app,"_tradier_live_quote",lambda ticker,token:{
        "ticker":ticker,"price":300.0,"change_pct":0.01,"provider":"Tradier Brokerage API",
        "realtime":True,"delayed":False
    })
    monkeypatch.setattr(market_app,"_finnhub_live_quote",lambda *args:(_ for _ in ()).throw(AssertionError("Finnhub should not run")))
    q=market_app.live_quote("SPY")
    assert q["price"]==300.0
    assert q["provider"]=="Tradier Brokerage API"
    assert q["realtime"] is True


def test_control_key_protects_stream_session(monkeypatch):
    monkeypatch.setenv("MARKETLENS_CONTROL_KEY","control-secret")
    monkeypatch.setenv("TRADIER_ACCESS_TOKEN","tradier-secret")
    monkeypatch.setattr(market_app,"_create_tradier_market_session",lambda:"session-123")
    client=market_app.app.test_client()
    denied=client.post("/api/market-stream/session")
    assert denied.status_code==401
    assert denied.get_json()["requires_control_key"] is True
    allowed=client.post("/api/market-stream/session",headers={"X-MarketLens-Key":"control-secret"})
    assert allowed.status_code==200
    assert allowed.get_json()["sessionid"]=="session-123"


def test_control_key_protects_research_trigger_before_github_token(monkeypatch):
    monkeypatch.setenv("MARKETLENS_CONTROL_KEY","control-secret")
    monkeypatch.delenv("GITHUB_ACTIONS_TOKEN",raising=False)
    client=market_app.app.test_client()
    denied=client.post("/api/run-research",json={"ticker":"SPY"})
    assert denied.status_code==401
    allowed=client.post(
        "/api/run-research",
        json={"ticker":"SPY"},
        headers={"X-MarketLens-Key":"control-secret"},
    )
    assert allowed.status_code==503

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

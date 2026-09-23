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

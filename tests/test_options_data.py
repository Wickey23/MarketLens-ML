import math
import src.options_data as od
from src.options_data import _i, enrich_contract, tradier_market_clock


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
        if path.endswith("quotes"):
            return {"quotes":{"quote":{
                "symbol":"ABC","bid":99.9,"ask":100.1,"last":100.0,
                "bid_date":4072381200000,"ask_date":4072381200000,"trade_date":4072381200000
            }}}
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
    assert out["underlying_price"]==100.0
    assert out["greeks_frequency"]=="hourly"
    q=out["contracts"][0]
    assert q["contract_symbol"]=="ABC990116C00100000"
    assert q["bid"]==4.9
    assert q["ask"]==5.1
    assert q["breakeven"]==105.1
    assert q["delta"]==0.55
    assert q["theta_per_contract_per_day"]==-5.0
    assert q["greeks_source"]=="Tradier / ORATS (hourly)"


def test_tradier_market_clock_open(monkeypatch):
    monkeypatch.setenv("TRADIER_ACCESS_TOKEN","test-token")
    monkeypatch.setattr(od,"_tradier_get",lambda path,params:{
        "clock":{
            "date":"2026-09-24","description":"Market is open",
            "state":"open","timestamp":1790250000,
            "next_change":"16:00","next_state":"postmarket"
        }
    })
    out=tradier_market_clock()
    assert out["state"]=="open"
    assert out["source"]=="Tradier"


def test_tradier_market_clock_unconfigured(monkeypatch):
    monkeypatch.delenv("TRADIER_ACCESS_TOKEN",raising=False)
    out=tradier_market_clock()
    assert out["state"]=="unconfigured"


def test_occ_parser_supports_alpaca_contract_symbols():
    meta=od._parse_occ_symbol("NVDA261002C00210000")
    assert meta["type"]=="call"
    assert meta["expiration"]=="2026-10-02"
    assert meta["strike"]==210.0


def test_alpaca_opra_snapshot_is_execution_grade(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID","key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY","secret")
    monkeypatch.setenv("ALPACA_OPTIONS_FEED","opra")
    monkeypatch.setenv("ALPACA_STOCK_FEED","sip")
    def fake_get(url,params=None):
        if "/stocks/" in url:
            return {
                "latestQuote":{"bp":99.9,"ap":100.1,"t":"2026-09-24T19:30:00Z"},
                "latestTrade":{"p":100.0,"t":"2026-09-24T19:30:00Z"},
                "dailyBar":{"o":99.0,"h":101.0,"l":98.5},
                "prevDailyBar":{"c":99.5},
            }
        return {"snapshots":{
            "ABC261002C00100000":{
                "latestQuote":{"bp":4.9,"ap":5.1,"t":"2026-09-24T19:30:00Z"},
                "latestTrade":{"p":5.0,"t":"2026-09-24T19:29:59Z"},
                "dailyBar":{"v":200},
                "impliedVolatility":0.30,
                "greeks":{"delta":0.55,"gamma":0.02,"theta":-0.05,"vega":0.08},
            }
        }}
    monkeypatch.setattr(od,"_alpaca_get",fake_get)
    monkeypatch.setattr(od,"_risk_free_rate",lambda:(0.04,"test"))
    out=od._alpaca_option_snapshot("ABC",100.0,0.20,2,8)
    assert out["provider_key"]=="alpaca_opra"
    assert out["realtime"] is True
    q=out["contracts"][0]
    assert q["provider_key"]=="alpaca_opra"
    assert q["bid"]==4.9
    assert q["ask"]==5.1
    assert q["delta"]==0.55


def test_multi_provider_merge_keeps_authoritative_quote(monkeypatch):
    monkeypatch.setattr(od,"_risk_free_rate",lambda:(0.04,"test"))
    base={
        "contract_symbol":"ABC261002C00100000","type":"call","expiration":"2026-10-02",
        "dte":8,"strike":100.0,"iv":0.30,"volume":100,"open_interest":500,
        "in_the_money":False,"breakeven":105.1,"breakeven_move":0.051,
        "spread_pct":0.02,
    }
    tradier={
        "source":"Tradier Brokerage API","provider_key":"tradier","feed":"consolidated",
        "underlying_price":100.0,
        "underlying_quote":{"price":100.0,"provider":"Tradier","provider_key":"tradier","feed":"consolidated","realtime":True,"consolidated":True,"market_timestamp":"2026-09-24T19:30:00Z"},
        "contracts":[{**base,"provider":"Tradier","provider_key":"tradier","feed":"consolidated","realtime":True,"consolidated":True,"bid":4.9,"ask":5.1,"mid":5.0,"quote_time":"2026-09-24T19:29:55Z"}],
        "contracts_scanned":1,
    }
    indicative={
        "source":"Alpaca indicative options","provider_key":"alpaca_indicative","feed":"indicative",
        "underlying_price":100.01,
        "underlying_quote":{"price":100.01,"provider":"Alpaca","provider_key":"alpaca_iex","feed":"iex","realtime":True,"consolidated":False,"market_timestamp":"2026-09-24T19:30:01Z"},
        "contracts":[{**base,"provider":"Alpaca indicative","provider_key":"alpaca_indicative","feed":"indicative","realtime":True,"consolidated":False,"bid":4.95,"ask":5.05,"mid":5.0,"quote_time":"2026-09-24T19:30:01Z"}],
        "contracts_scanned":1,
    }
    out=od._merge_option_snapshots("ABC",[tradier,indicative],0.20)
    q=out["contracts"][0]
    assert q["selected_quote_provider_key"]=="tradier"
    assert q["execution_realtime"] is True
    assert len(q["provider_candidates"])==2

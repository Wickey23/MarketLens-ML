from datetime import datetime, timezone
from src.ai_paper_trader import run_ai_paper_portfolio, performance_summary

def snap():
    q={"contract_symbol":"ABC1","type":"call","expiration":"2099-12-31","dte":14,"strike":100,
       "ask":1.0,"bid":.95,"mid":.975,"iv":.25,"spread_pct":.05,"theta_cost_pct_per_day":.01,"last_trade":datetime.now(timezone.utc).isoformat()}
    r={"contract_symbol":"ABC1","type":"call","expiration":"2099-12-31","dte":14,"strike":100,
       "score":80,"prob_profit":.62,"expected_pnl_per_contract":18,"historical_scope":"same regime",
       "reasons":["test"],"risks":[]}
    return {"generated_at":"2026-01-01T00:00:00Z","tickers":[{"ticker":"ABC",
        "options":{"chain":{"contracts":[q]},"opportunity_radar":{"opportunities":[r]}}}]}

def test_ai_paper_trader_enters_without_real_order():
    state={"starting_cash":10000.0,"cash":10000.0,"open":[],"closed":[],"equity_history":[],"decisions":[]}
    out=run_ai_paper_portfolio(snap(),state=state)
    assert len(out["open"])==1
    assert out["open"][0]["ticker"]=="ABC"
    assert out["cash"]<10000
    assert out["decisions"][0]["action"]=="paper_buy"
    summary=performance_summary(out)
    assert summary["open_positions"]==1


def test_ai_rejects_zero_dte_autonomous_entry():
    s=snap()
    q=s["tickers"][0]["options"]["chain"]["contracts"][0]
    r=s["tickers"][0]["options"]["opportunity_radar"]["opportunities"][0]
    q["dte"]=0
    r["dte"]=0
    state={"starting_cash":10000.0,"cash":10000.0,"open":[],"closed":[],"equity_history":[],"decisions":[]}
    out=run_ai_paper_portfolio(s,state=state)
    assert out["open"]==[]
    assert any(d.get("action")=="skip" and "DTE outside autonomous policy" in d.get("reason","") for d in out["decisions"])


def test_ai_rejects_stale_quote():
    s=snap()
    s["tickers"][0]["options"]["chain"]["contracts"][0]["last_trade"]="2020-01-01T00:00:00+00:00"
    state={"starting_cash":10000.0,"cash":10000.0,"open":[],"closed":[],"equity_history":[],"decisions":[]}
    out=run_ai_paper_portfolio(s,state=state)
    assert out["open"]==[]
    assert any("stale" in d.get("reason","") for d in out["decisions"])


def test_legacy_same_day_position_exits_on_bid():
    from datetime import date
    today=date.today().isoformat()
    s=snap()
    q=s["tickers"][0]["options"]["chain"]["contracts"][0]
    q["expiration"]=today
    q["dte"]=0
    q["bid"]=1.20
    q["ask"]=1.25
    state={
        "starting_cash":10000.0,"cash":9900.0,
        "open":[{
            "id":"legacy","ticker":"ABC","contract_key":"ABC1","type":"call",
            "strike":100,"expiration":today,"qty":1,"entry_price":1.0,
            "entry_cost":100.0,"opened_at":"2026-01-01T00:00:00+00:00"
        }],
        "closed":[],"equity_history":[],"decisions":[]
    }
    out=run_ai_paper_portfolio(s,state=state)
    assert out["open"]==[]
    assert out["closed"][0]["exit_reason"]=="time_risk"
    assert out["closed"][0]["exit_price"]==1.20


def test_ai_does_not_mark_with_mid_when_bid_unavailable():
    s=snap()
    q=s["tickers"][0]["options"]["chain"]["contracts"][0]
    q["bid"]=0
    q["mid"]=1.50
    state={
        "starting_cash":10000.0,"cash":9900.0,
        "open":[{
            "id":"legacy","ticker":"ABC","contract_key":"ABC1","type":"call",
            "strike":100,"expiration":"2099-12-31","qty":1,"entry_price":1.0,
            "entry_cost":100.0,"opened_at":"2026-01-01T00:00:00+00:00",
            "last_mark":0.9
        }],
        "closed":[],"equity_history":[],"decisions":[]
    }
    out=run_ai_paper_portfolio(s,state=state)
    assert out["open"][0]["last_mark"] is None

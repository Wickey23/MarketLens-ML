from datetime import datetime, timezone, timedelta
from src.ai_paper_trader import (
    CURRENT_STRATEGY_VERSION,
    run_ai_paper_portfolio,
    performance_summary,
    paper_to_real_readiness,
    strategy_performance_summary,
    forward_validation_summary,
    load_state,
    save_state,
)

TEST_NOW=datetime(2026,9,24,15,0,tzinfo=timezone.utc)

def snap():
    q={"contract_symbol":"ABC1","type":"call","expiration":"2099-12-31","dte":14,"strike":100,
       "ask":1.0,"bid":.95,"mid":.975,"iv":.25,"spread_pct":.05,"theta_cost_pct_per_day":.01,"last_trade":TEST_NOW.isoformat(),"quote_age_hours":0.0}
    r={"contract_symbol":"ABC1","type":"call","expiration":"2099-12-31","dte":14,"strike":100,
       "score":80,"prob_profit":.62,"expected_pnl_per_contract":18,"historical_scope":"same regime",
       "reasons":["test"],"risks":[]}
    return {"generated_at":"2026-09-24T14:59:00Z","tickers":[{"ticker":"ABC",
        "research_refreshed_at":"2026-09-24T14:00:00Z",
        "evidence":{"mean_roc_auc":0.55},
        "options":{"chain":{"contracts":[q],"realtime":True,"source":"test realtime"},"opportunity_radar":{"opportunities":[r]}}}]}

def test_ai_paper_trader_enters_without_real_order():
    state={"starting_cash":10000.0,"cash":10000.0,"open":[],"closed":[],"equity_history":[],"decisions":[]}
    out=run_ai_paper_portfolio(snap(),state=state,now_dt=TEST_NOW)
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
    out=run_ai_paper_portfolio(s,state=state,now_dt=TEST_NOW)
    assert out["open"]==[]
    assert any(d.get("action")=="skip" and "DTE outside autonomous policy" in d.get("reason","") for d in out["decisions"])


def test_ai_rejects_stale_quote():
    s=snap()
    s["tickers"][0]["options"]["chain"]["contracts"][0]["last_trade"]="2020-01-01T00:00:00+00:00"
    s["tickers"][0]["options"]["chain"]["contracts"][0]["quote_age_hours"]=100.0
    state={"starting_cash":10000.0,"cash":10000.0,"open":[],"closed":[],"equity_history":[],"decisions":[]}
    out=run_ai_paper_portfolio(s,state=state,now_dt=TEST_NOW)
    assert out["open"]==[]
    assert any("stale" in d.get("reason","") for d in out["decisions"])


def test_legacy_same_day_position_exits_on_bid():
    from datetime import date
    today=TEST_NOW.date().isoformat()
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
    out=run_ai_paper_portfolio(s,state=state,now_dt=TEST_NOW)
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
    out=run_ai_paper_portfolio(s,state=state,now_dt=TEST_NOW)
    assert out["open"][0]["last_mark"] is None


def test_ai_exit_is_logged_in_decision_history():
    from datetime import date
    today=TEST_NOW.date().isoformat()
    s=snap()
    q=s["tickers"][0]["options"]["chain"]["contracts"][0]
    q["expiration"]=today
    q["dte"]=0
    q["bid"]=1.20
    state={
        "starting_cash":10000.0,"cash":9900.0,
        "open":[{
            "id":"legacy2","ticker":"ABC","contract_key":"ABC1","type":"call",
            "strike":100,"expiration":today,"qty":1,"entry_price":1.0,
            "entry_cost":100.0,"opened_at":"2026-01-01T00:00:00+00:00"
        }],
        "closed":[],"equity_history":[],"decisions":[]
    }
    out=run_ai_paper_portfolio(s,state=state,now_dt=TEST_NOW)
    assert any(d.get("action")=="paper_exit" and d.get("reason")=="time_risk" for d in out["decisions"])


def test_open_equity_marks_zero_when_contract_has_no_bid():
    s=snap()
    q=s["tickers"][0]["options"]["chain"]["contracts"][0]
    q["bid"]=0
    state={
        "starting_cash":10000.0,"cash":9900.0,
        "open":[{
            "id":"open0","ticker":"ABC","contract_key":"ABC1","type":"call",
            "strike":100,"expiration":"2099-12-31","qty":1,"entry_price":1.0,
            "entry_cost":100.0,"opened_at":"2026-01-01T00:00:00+00:00"
        }],
        "closed":[],"equity_history":[],"decisions":[]
    }
    out=run_ai_paper_portfolio(s,state=state,now_dt=TEST_NOW)
    assert out["equity_history"][-1]["open_value"]==0
    assert out["equity_history"][-1]["equity"]==9900


def test_ai_uses_guided_best_overall_contract():
    s=snap()
    q2={
        "contract_symbol":"ABC2","type":"call","expiration":"2099-12-31","dte":21,"strike":105,
        "ask":0.8,"bid":0.75,"mid":0.775,"iv":0.25,"spread_pct":0.065,
        "theta_cost_pct_per_day":0.01,"last_trade":TEST_NOW.isoformat(),"quote_age_hours":0.0
    }
    r2={
        "contract_symbol":"ABC2","type":"call","expiration":"2099-12-31","dte":21,"strike":105,
        "score":76,"combined_evidence_score":84,"prob_profit":0.64,
        "expected_pnl_per_contract":22,"historical_scope":"same regime",
        "reasons":["guided"],"risks":[]
    }
    radar=s["tickers"][0]["options"]["opportunity_radar"]
    radar["opportunities"].append(r2)
    radar["guidance"]={"state":"strong_candidates","best_overall":r2}
    s["tickers"][0]["options"]["chain"]["contracts"].append(q2)
    state={"starting_cash":10000.0,"cash":10000.0,"open":[],"closed":[],"equity_history":[],"decisions":[]}
    out=run_ai_paper_portfolio(s,state=state,now_dt=TEST_NOW)
    assert len(out["open"])==1
    assert out["open"][0]["contract_key"]=="ABC2"
    assert out["open"][0]["entry_combined_evidence_score"]==84


def test_readiness_requires_forward_sample():
    state={"starting_cash":10000.0,"cash":10000.0,"open":[],"closed":[],"equity_history":[],"decisions":[]}
    out=paper_to_real_readiness(state)
    assert out["state"]=="collecting_forward_data"
    assert out["all_checks_pass"] is False
    sample=next(x for x in out["checks"] if x["id"]=="forward_sample")
    assert sample["target"]==30
    assert sample["pass"] is False


def test_readiness_does_not_approve_losing_30_trade_record():
    closed=[]
    for i in range(30):
        opened=datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(days=i)
        closed.append({
            "ticker":["AAA","BBB","CCC"][i%3],
            "pnl":-5.0,
            "entry_cost":100.0,
            "strategy_version":CURRENT_STRATEGY_VERSION,
            "entry_market_data_realtime":True,
            "opened_at":opened.isoformat(),
            "closed_at":(opened+timedelta(hours=1)).isoformat(),
        })
    state={
        "starting_cash":10000.0,
        "cash":9850.0,
        "open":[],
        "closed":closed,
        "equity_history":[
            {"at":"2026-01-01T00:00:00Z","equity":10000.0,"cash":10000.0,"open_value":0.0},
            {"at":"2026-02-01T00:00:00Z","equity":9850.0,"cash":9850.0,"open_value":0.0},
        ],
        "decisions":[],
    }
    out=paper_to_real_readiness(state)
    assert out["state"]=="paper_results_not_ready"
    assert out["all_checks_pass"] is False
    assert next(x for x in out["checks"] if x["id"]=="net_pnl")["pass"] is False
    assert next(x for x in out["checks"] if x["id"]=="avg_trade")["pass"] is False


def test_current_readiness_ignores_legacy_trade():
    legacy={
        "ticker":"QQQ","pnl":500.0,"entry_cost":100.0,
        "strategy_version":"legacy","opened_at":"2026-01-01T15:00:00+00:00",
        "closed_at":"2026-01-01T16:00:00+00:00",
    }
    state={"starting_cash":10000.0,"cash":10500.0,"open":[],"closed":[legacy],"equity_history":[],"decisions":[]}
    readiness=paper_to_real_readiness(state)
    assert readiness["closed_trades"]==0
    assert readiness["state"]=="collecting_forward_data"
    assert strategy_performance_summary(state)["total_pnl"]==0.0


def test_ai_rejects_delayed_chain_for_current_strategy():
    s=snap()
    s["tickers"][0]["options"]["chain"]["realtime"]=False
    state={"starting_cash":10000.0,"cash":10000.0,"open":[],"closed":[],"equity_history":[],"decisions":[]}
    out=run_ai_paper_portfolio(s,state=state,now_dt=TEST_NOW)
    assert out["open"]==[]
    assert any("real-time option data required" in d.get("reason","") for d in out["decisions"])


def test_ai_does_not_enter_outside_regular_market_window():
    s=snap()
    after_hours=datetime(2026,9,24,22,0,tzinfo=timezone.utc)
    state={"starting_cash":10000.0,"cash":10000.0,"open":[],"closed":[],"equity_history":[],"decisions":[]}
    out=run_ai_paper_portfolio(s,state=state,now_dt=after_hours)
    assert out["open"]==[]
    assert out["paper_market_session_open"] is False


def test_forward_validation_is_current_strategy_only():
    opened=datetime(2026,1,1,tzinfo=timezone.utc)
    current={
        "ticker":"AAA","qty":1,"pnl":20.0,"entry_cost":100.0,
        "entry_prob_profit":0.60,"entry_expected_pnl":10.0,
        "strategy_version":CURRENT_STRATEGY_VERSION,
        "entry_market_data_realtime":True,
        "opened_at":opened.isoformat(),"closed_at":(opened+timedelta(days=1)).isoformat(),
    }
    legacy={**current,"pnl":-999.0,"strategy_version":"legacy"}
    out=forward_validation_summary({"closed":[current,legacy]})
    assert out["closed_trades"]==1
    assert out["mean_actual_pnl_per_contract"]==20.0
    assert out["mean_expected_pnl_error_per_contract"]==10.0


def test_ai_respects_tradier_closed_market_clock():
    s=snap()
    s["market_clock"]={"source":"Tradier","state":"closed"}
    state={"starting_cash":10000.0,"cash":10000.0,"open":[],"closed":[],"equity_history":[],"decisions":[]}
    out=run_ai_paper_portfolio(s,state=state,now_dt=TEST_NOW)
    assert out["open"]==[]
    assert out["paper_market_session_open"] is False
    assert out["market_clock_state"]=="closed"


def test_state_persistence_is_atomic_and_round_trips(tmp_path):
    path=tmp_path/"state.json"
    state={"starting_cash":10000.0,"cash":9990.0,"open":[],"closed":[],"equity_history":[],"decisions":[]}
    save_state(state,path)
    assert load_state(path)==state
    assert not (tmp_path/"state.json.tmp").exists()


def test_corrupt_existing_state_refuses_silent_reset(tmp_path):
    path=tmp_path/"state.json"
    path.write_text("{not-json",encoding="utf-8")
    try:
        load_state(path)
    except RuntimeError as exc:
        assert "refusing to reset history" in str(exc)
    else:
        raise AssertionError("Corrupt existing state must not silently reset")


def test_ai_rejects_missing_deep_research_evidence():
    s=snap()
    s["tickers"][0].pop("research_refreshed_at",None)
    s["tickers"][0]["evidence"]={}
    state={"starting_cash":10000.0,"cash":10000.0,"open":[],"closed":[],"equity_history":[],"decisions":[]}
    out=run_ai_paper_portfolio(s,state=state,now_dt=TEST_NOW)
    assert out["open"]==[]
    assert any("recent deep research evidence required" in d.get("reason","") for d in out["decisions"])

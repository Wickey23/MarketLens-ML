from datetime import datetime, timezone

from src.provider_router import choose_option_quote, choose_underlying_quote


NOW=datetime(2026,9,24,19,30,tzinfo=timezone.utc)


def test_consolidated_realtime_beats_newer_indicative_quote():
    rows=[
        {
            "provider":"Tradier","provider_key":"tradier","feed":"consolidated",
            "realtime":True,"consolidated":True,"bid":10.0,"ask":10.1,
            "quote_time":"2026-09-24T19:29:55+00:00",
        },
        {
            "provider":"Alpaca indicative","provider_key":"alpaca_indicative","feed":"indicative",
            "realtime":True,"consolidated":False,"bid":10.02,"ask":10.12,
            "quote_time":"2026-09-24T19:29:59+00:00",
        },
    ]
    out=choose_option_quote(rows,now=NOW)
    assert out["provider_key"]=="tradier"
    assert out["execution_realtime"] is True


def test_two_consolidated_feeds_agree_for_high_confidence():
    rows=[
        {
            "provider":"Tradier","provider_key":"tradier","feed":"consolidated",
            "realtime":True,"consolidated":True,"bid":10.00,"ask":10.10,
            "quote_time":"2026-09-24T19:29:58+00:00",
        },
        {
            "provider":"Alpaca OPRA","provider_key":"alpaca_opra","feed":"opra",
            "realtime":True,"consolidated":True,"bid":10.01,"ask":10.11,
            "quote_time":"2026-09-24T19:29:59+00:00",
        },
    ]
    out=choose_option_quote(rows,now=NOW)
    assert out["data_confidence"]=="high"
    assert out["execution_realtime"] is True
    assert len(out["provider_candidates"])==2


def test_material_disagreement_marks_conflict():
    rows=[
        {
            "provider":"Tradier","provider_key":"tradier","feed":"consolidated",
            "realtime":True,"consolidated":True,"bid":10.0,"ask":10.1,
            "quote_time":"2026-09-24T19:29:59+00:00",
        },
        {
            "provider":"Alpaca OPRA","provider_key":"alpaca_opra","feed":"opra",
            "realtime":True,"consolidated":True,"bid":10.5,"ask":10.6,
            "quote_time":"2026-09-24T19:29:58+00:00",
        },
    ]
    out=choose_option_quote(rows,now=NOW)
    assert out["data_confidence"]=="conflict"
    assert out["execution_realtime"] is False


def test_underlying_prefers_authoritative_feed_over_delayed_yahoo():
    out=choose_underlying_quote([
        {
            "provider":"Yahoo","provider_key":"yahoo","price":101.0,
            "realtime":False,"consolidated":False,
            "market_timestamp":"2026-09-24T19:30:00+00:00",
        },
        {
            "provider":"Alpaca SIP","provider_key":"alpaca_sip","price":100.9,
            "realtime":True,"consolidated":True,
            "market_timestamp":"2026-09-24T19:29:58+00:00",
        },
    ],now=NOW)
    assert out["provider_key"]=="alpaca_sip"
    assert out["data_confidence"]=="medium"

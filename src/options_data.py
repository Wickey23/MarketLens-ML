from __future__ import annotations
from datetime import datetime, timezone, date, timedelta
import math
import os
import re
import json
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo
from statistics import NormalDist
import yfinance as yf

from src.provider_router import choose_option_quote, choose_underlying_quote, PROVIDER_QUALITY


def _f(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _i(x, default=0):
    try:
        v=float(x)
        return int(v) if math.isfinite(v) else default
    except Exception:
        return default




def _tradier_get(path, params):
    token=os.getenv("TRADIER_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Tradier production token is not configured")
    query=urllib.parse.urlencode(params)
    url=f"https://api.tradier.com/v1/{path}?{query}"
    req=urllib.request.Request(
        url,
        headers={
            "Authorization":f"Bearer {token}",
            "Accept":"application/json",
            "User-Agent":"MarketLens-ML/1.0",
        },
    )
    with urllib.request.urlopen(req,timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value,list) else [value]


def tradier_market_clock():
    """Best-effort authoritative market state from Tradier production."""
    if not os.getenv("TRADIER_ACCESS_TOKEN"):
        return {"source":"Tradier","state":"unconfigured"}
    try:
        body=_tradier_get("markets/clock",{})
        clock=body.get("clock") or {}
        return {
            "source":"Tradier",
            "state":str(clock.get("state") or "unknown").lower(),
            "date":clock.get("date"),
            "description":clock.get("description"),
            "timestamp":clock.get("timestamp"),
            "next_change":clock.get("next_change"),
            "next_state":clock.get("next_state"),
        }
    except Exception as exc:
        code=getattr(exc,"code",None)
        return {
            "source":"Tradier",
            "state":"unknown",
            "error":f"HTTP {code}" if code is not None else type(exc).__name__,
        }


def _tradier_expirations(ticker, max_expiries):
    body=_tradier_get("markets/options/expirations",{
        "symbol":ticker,
        "includeAllRoots":"false",
        "strikes":"false",
        "contractSize":"false",
        "expirationType":"false",
    })
    dates=((body.get("expirations") or {}).get("date"))
    return [str(x) for x in _as_list(dates) if x][:max_expiries]


def _tradier_underlying_quote(ticker, now):
    body=_tradier_get("markets/quotes",{"symbols":ticker,"greeks":"false"})
    raw=(body.get("quotes") or {}).get("quote")
    rows=_as_list(raw)
    if not rows:
        raise ValueError("Tradier underlying quote unavailable")
    row=rows[0] or {}
    bid=_f(row.get("bid")); ask=_f(row.get("ask")); last=_f(row.get("last"))
    mid=(bid+ask)/2 if bid is not None and ask is not None and ask>=bid and (bid>0 or ask>0) else None
    price=last if last is not None and last>0 else mid
    if price is None or price<=0:
        raise ValueError("Tradier underlying price unavailable")
    quote_ms=max(_i(row.get("bid_date"),0),_i(row.get("ask_date"),0),_i(row.get("trade_date"),0))
    quote_dt=datetime.fromtimestamp(quote_ms/1000,timezone.utc) if quote_ms else None
    age=max(0.0,(now-quote_dt).total_seconds()/3600.0) if quote_dt else None
    previous_close=_f(row.get("prevclose"))
    if previous_close is None:
        previous_close=_f(row.get("close"))
    return {
        "price":price,"bid":bid,"ask":ask,"last":last,
        "previous_close":previous_close,
        "change":_f(row.get("change")),
        "change_pct":_f(row.get("change_percentage")),
        "quote_age_hours":_f(age),
        "quote_time":quote_dt.isoformat() if quote_dt else None,
        "market_timestamp":quote_dt.isoformat() if quote_dt else None,
        "provider":"Tradier Brokerage API",
        "provider_key":"tradier",
        "feed":"consolidated",
        "realtime":True,
        "consolidated":True,
    }


def _tradier_rows_for_expiration(ticker, expiration, spot, market_today, now, strikes_each_side):
    body=_tradier_get("markets/options/chains",{
        "symbol":ticker,
        "expiration":expiration,
        "greeks":"true",
    })
    raw=_as_list((body.get("options") or {}).get("option"))
    parsed=[]
    dte=max((date.fromisoformat(expiration)-market_today).days,0)
    for row in raw:
        strike=_f(row.get("strike"))
        side=str(row.get("option_type") or "").lower()
        if strike is None or side not in ("call","put"):
            continue
        bid=_f(row.get("bid")); ask=_f(row.get("ask")); last=_f(row.get("last"))
        mid=(bid+ask)/2 if bid is not None and ask is not None and ask>=bid and (bid>0 or ask>0) else last
        entry_quote=ask if ask is not None and ask>0 else mid
        breakeven=(strike+entry_quote if side=="call" else strike-entry_quote) if entry_quote is not None else None
        spread=(ask-bid) if bid is not None and ask is not None else None
        spread_pct=(spread/mid) if spread is not None and mid and mid>0 else None
        quote_ms=max(_i(row.get("bid_date"),0),_i(row.get("ask_date"),0))
        quote_dt=datetime.fromtimestamp(quote_ms/1000,timezone.utc) if quote_ms else None
        quote_age=max(0.0,(now-quote_dt).total_seconds()/3600.0) if quote_dt else None
        trade_ms=_i(row.get("trade_date"),0)
        trade_dt=datetime.fromtimestamp(trade_ms/1000,timezone.utc) if trade_ms else None
        g=row.get("greeks") or {}
        iv=_f(g.get("smv_vol"))
        if iv is None: iv=_f(g.get("mid_iv"))
        parsed.append({
            "contract_symbol":str(row.get("symbol") or ""),
            "type":side,
            "expiration":str(row.get("expiration_date") or expiration),
            "dte":dte,
            "strike":strike,
            "bid":bid,"ask":ask,"mid":_f(mid),"last":last,
            "iv":iv,
            "volume":_i(row.get("volume")),
            "open_interest":_i(row.get("open_interest")),
            "in_the_money":bool((side=="call" and spot>strike) or (side=="put" and spot<strike)),
            "last_trade":trade_dt.isoformat() if trade_dt else None,
            "quote_age_hours":_f(quote_age),
            "quote_time":quote_dt.isoformat() if quote_dt else None,
            "provider":"Tradier Brokerage API",
            "provider_key":"tradier",
            "feed":"consolidated",
            "realtime":True,
            "consolidated":True,
            "breakeven":_f(breakeven),
            "breakeven_move":_f((breakeven/spot)-1) if breakeven and spot else None,
            "spread_pct":_f(spread_pct),
            "provider_greeks":{
                "delta":_f(g.get("delta")),
                "gamma":_f(g.get("gamma")),
                "theta":_f(g.get("theta")),
                "vega":_f(g.get("vega")),
                "updated_at":g.get("updated_at"),
            },
        })
    keep=[]
    for side in ("call","put"):
        rows=[x for x in parsed if x["type"]==side]
        rows.sort(key=lambda x:abs(x["strike"]-spot))
        keep.extend(rows[:strikes_each_side*2+1])
    return keep


def _tradier_option_snapshot(ticker, spot, annual_rv, max_expiries, strikes_each_side):
    now=datetime.now(timezone.utc)
    market_today=datetime.now(ZoneInfo("America/New_York")).date()
    risk_free,risk_free_source=_risk_free_rate()
    underlying=_tradier_underlying_quote(ticker,now)
    market_spot=float(underlying["price"])
    expiries=_tradier_expirations(ticker,max_expiries)
    rows=[]
    for exp in expiries:
        rows.extend(_tradier_rows_for_expiration(ticker,exp,market_spot,market_today,now,strikes_each_side))
    liquid=[]
    for row in rows:
        if not row["mid"] or row["mid"]<=0 or not (row["open_interest"]>=25 or row["volume"]>=5):
            continue
        g=row.pop("provider_greeks",{}) or {}
        enriched=enrich_contract(row,market_spot,annual_rv,risk_free)
        if g.get("delta") is not None: enriched["delta"]=g["delta"]
        if g.get("gamma") is not None: enriched["gamma"]=g["gamma"]
        if g.get("theta") is not None:
            enriched["theta_per_share_per_day"]=g["theta"]
            enriched["theta_per_contract_per_day"]=_f(g["theta"]*100)
            debit=enriched.get("entry_debit_per_contract") or 0
            enriched["theta_cost_pct_per_day"]=_f(abs(enriched["theta_per_contract_per_day"])/debit) if debit>0 else None
        if g.get("vega") is not None:
            enriched["vega_per_share_per_vol_point"]=g["vega"]
            enriched["vega_per_contract_per_vol_point"]=_f(g["vega"]*100)
        enriched["greeks_source"]="Tradier / ORATS (hourly)"
        enriched["greeks_updated_at"]=g.get("updated_at")
        liquid.append(enriched)
    liquid.sort(key=lambda r:(r["dte"],abs((r["strike"] or market_spot)-market_spot),r["spread_pct"] if r["spread_pct"] is not None else 99))
    return {
        "source":"Tradier Brokerage API",
        "provider_key":"tradier",
        "feed":"consolidated",
        "underlying_price":market_spot,
        "underlying_quote":underlying,
        "quote_note":"Production Tradier brokerage market data is real-time for U.S. stocks/options. Greeks and volatility are ORATS data updated hourly.",
        "retrieved_at":now.isoformat(),
        "risk_free_rate":risk_free,
        "risk_free_source":risk_free_source,
        "expirations":expiries,
        "contracts":liquid[:240],
        "contracts_scanned":len(rows),
        "liquid_contracts":len(liquid),
        "realtime":True,
        "greeks_frequency":"hourly",
    }


def _alpaca_credentials():
    key=os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
    secret=os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    return key,secret


def _alpaca_get(url, params=None):
    key,secret=_alpaca_credentials()
    if not key or not secret:
        raise RuntimeError("Alpaca market-data credentials are not configured")
    if params:
        url=url+"?"+urllib.parse.urlencode(params)
    req=urllib.request.Request(
        url,
        headers={
            "APCA-API-KEY-ID":key,
            "APCA-API-SECRET-KEY":secret,
            "Accept":"application/json",
            "User-Agent":"MarketLens-ML/1.0",
        },
    )
    with urllib.request.urlopen(req,timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def _alpaca_stock_quote(ticker, now):
    feed=(os.getenv("ALPACA_STOCK_FEED") or "iex").strip().lower()
    body=_alpaca_get(
        f"https://data.alpaca.markets/v2/stocks/{urllib.parse.quote(ticker,safe='')}/snapshot",
        {"feed":feed},
    )
    q=body.get("latestQuote") or body.get("latest_quote") or {}
    trade=body.get("latestTrade") or body.get("latest_trade") or {}
    daily=body.get("dailyBar") or body.get("daily_bar") or {}
    prevbar=body.get("prevDailyBar") or body.get("prev_daily_bar") or {}
    bid=_f(q.get("bp") if "bp" in q else q.get("bid_price"))
    ask=_f(q.get("ap") if "ap" in q else q.get("ask_price"))
    last=_f(trade.get("p") if "p" in trade else trade.get("price"))
    mid=(bid+ask)/2 if bid is not None and ask is not None and ask>=bid and (bid>0 or ask>0) else None
    price=last if last is not None and last>0 else mid
    if price is None or price<=0:
        raise ValueError("Alpaca underlying quote unavailable")
    quote_time=q.get("t") or q.get("timestamp") or trade.get("t") or trade.get("timestamp")
    prev=_f(prevbar.get("c") if "c" in prevbar else prevbar.get("close"))
    consolidated=(feed=="sip")
    return {
        "price":price,"bid":bid,"ask":ask,"last":last,
        "previous_close":prev,
        "open":_f(daily.get("o") if "o" in daily else daily.get("open")),
        "high":_f(daily.get("h") if "h" in daily else daily.get("high")),
        "low":_f(daily.get("l") if "l" in daily else daily.get("low")),
        "quote_time":quote_time,
        "market_timestamp":quote_time,
        "provider":"Alpaca Market Data",
        "provider_key":"alpaca_sip" if consolidated else "alpaca_iex",
        "feed":feed,
        "realtime":True,
        "consolidated":consolidated,
    }


_OCC_RE=re.compile(r"^(.+?)(\d{6})([CP])(\d{8})$")


def _parse_occ_symbol(symbol):
    m=_OCC_RE.match(str(symbol or "").upper())
    if not m:
        return None
    root,ymd,cp,strike_raw=m.groups()
    try:
        expiration=date(2000+int(ymd[:2]),int(ymd[2:4]),int(ymd[4:6])).isoformat()
        strike=int(strike_raw)/1000.0
    except Exception:
        return None
    return {"root":root,"expiration":expiration,"type":"call" if cp=="C" else "put","strike":strike}


def _alpaca_option_snapshot(ticker, spot, annual_rv, max_expiries, strikes_each_side):
    now=datetime.now(timezone.utc)
    market_today=datetime.now(ZoneInfo("America/New_York")).date()
    feed=(os.getenv("ALPACA_OPTIONS_FEED") or "indicative").strip().lower()
    if feed not in ("opra","indicative"):
        feed="indicative"
    body=_alpaca_get(
        f"https://data.alpaca.markets/v1beta1/options/snapshots/{urllib.parse.quote(ticker,safe='')}",
        {
            "feed":feed,
            "limit":1000,
            "expiration_date_gte":market_today.isoformat(),
            "expiration_date_lte":(market_today+timedelta(days=60)).isoformat(),
            "strike_price_gte":max(0.01,float(spot)*0.70),
            "strike_price_lte":float(spot)*1.30,
        },
    )
    snapshots=body.get("snapshots") or {}
    risk_free,risk_free_source=_risk_free_rate()
    underlying=_alpaca_stock_quote(ticker,now)
    market_spot=float(underlying["price"])
    consolidated=(feed=="opra")
    provider_key="alpaca_opra" if consolidated else "alpaca_indicative"
    parsed=[]
    expirations=set()
    for symbol,snap in snapshots.items():
        meta=_parse_occ_symbol(symbol)
        if not meta:
            continue
        exp=meta["expiration"]
        dte=max((date.fromisoformat(exp)-market_today).days,0)
        q=snap.get("latestQuote") or snap.get("latest_quote") or {}
        tr=snap.get("latestTrade") or snap.get("latest_trade") or {}
        daily=snap.get("dailyBar") or snap.get("daily_bar") or {}
        g=snap.get("greeks") or {}
        bid=_f(q.get("bp") if "bp" in q else q.get("bid_price"))
        ask=_f(q.get("ap") if "ap" in q else q.get("ask_price"))
        last=_f(tr.get("p") if "p" in tr else tr.get("price"))
        mid=(bid+ask)/2 if bid is not None and ask is not None and ask>=bid and (bid>0 or ask>0) else last
        entry=ask if ask is not None and ask>0 else mid
        strike=meta["strike"]
        breakeven=(strike+entry if meta["type"]=="call" else strike-entry) if entry is not None else None
        spread=(ask-bid) if bid is not None and ask is not None else None
        quote_time=q.get("t") or q.get("timestamp")
        iv=_f(snap.get("impliedVolatility") if "impliedVolatility" in snap else snap.get("implied_volatility"))
        row={
            "contract_symbol":str(symbol),
            "type":meta["type"],"expiration":exp,"dte":dte,"strike":strike,
            "bid":bid,"ask":ask,"mid":_f(mid),"last":last,
            "iv":iv,
            "volume":_i(daily.get("v") if "v" in daily else daily.get("volume")),
            "open_interest":0,
            "in_the_money":bool((meta["type"]=="call" and market_spot>strike) or (meta["type"]=="put" and market_spot<strike)),
            "last_trade":tr.get("t") or tr.get("timestamp"),
            "quote_time":quote_time,
            "quote_age_hours":None,
            "provider":"Alpaca OPRA" if consolidated else "Alpaca indicative options",
            "provider_key":provider_key,
            "feed":feed,
            "realtime":True,
            "consolidated":consolidated,
            "breakeven":_f(breakeven),
            "breakeven_move":_f((breakeven/market_spot)-1) if breakeven and market_spot else None,
            "spread_pct":_f((spread/mid) if spread is not None and mid and mid>0 else None),
            "provider_greeks":{
                "delta":_f(g.get("delta")),
                "gamma":_f(g.get("gamma")),
                "theta":_f(g.get("theta")),
                "vega":_f(g.get("vega")),
                "updated_at":quote_time,
            },
        }
        parsed.append(row)
        expirations.add(exp)
    selected_exp=sorted(expirations)[:max_expiries]
    keep=[]
    for exp in selected_exp:
        for side in ("call","put"):
            rows=[x for x in parsed if x["expiration"]==exp and x["type"]==side]
            rows.sort(key=lambda x:abs(x["strike"]-market_spot))
            keep.extend(rows[:strikes_each_side*2+1])
    liquid=[]
    for row in keep:
        if not row.get("mid") or row["mid"]<=0:
            continue
        g=row.pop("provider_greeks",{}) or {}
        enriched=enrich_contract(row,market_spot,annual_rv,risk_free)
        for name in ("delta","gamma"):
            if g.get(name) is not None:
                enriched[name]=g[name]
        if g.get("theta") is not None:
            enriched["theta_per_share_per_day"]=g["theta"]
            enriched["theta_per_contract_per_day"]=_f(g["theta"]*100)
            debit=enriched.get("entry_debit_per_contract") or 0
            enriched["theta_cost_pct_per_day"]=_f(abs(enriched["theta_per_contract_per_day"])/debit) if debit>0 else None
        if g.get("vega") is not None:
            enriched["vega_per_share_per_vol_point"]=g["vega"]
            enriched["vega_per_contract_per_vol_point"]=_f(g["vega"]*100)
        enriched["greeks_source"]="Alpaca snapshot model"
        enriched["greeks_updated_at"]=g.get("updated_at")
        liquid.append(enriched)
    liquid.sort(key=lambda r:(r["dte"],abs((r["strike"] or market_spot)-market_spot),r["spread_pct"] if r["spread_pct"] is not None else 99))
    return {
        "source":"Alpaca OPRA" if consolidated else "Alpaca indicative options",
        "provider_key":provider_key,
        "feed":feed,
        "underlying_price":market_spot,
        "underlying_quote":underlying,
        "quote_note":(
            "Alpaca OPRA is the consolidated options feed."
            if consolidated else
            "Alpaca indicative quotes are modified derivatives of OPRA and are not execution-grade."
        ),
        "retrieved_at":now.isoformat(),
        "risk_free_rate":risk_free,
        "risk_free_source":risk_free_source,
        "expirations":selected_exp,
        "contracts":liquid[:240],
        "contracts_scanned":len(parsed),
        "liquid_contracts":len(liquid),
        "realtime":consolidated,
        "quote_realtime":True,
        "consolidated":consolidated,
        "greeks_frequency":"snapshot/model",
    }


def _cdf(z):
    return NormalDist().cdf(z)


def _pdf(z):
    return math.exp(-0.5*z*z)/math.sqrt(2*math.pi)


def _risk_free_rate():
    """Best-effort annualized risk-free proxy from the 13-week Treasury index."""
    try:
        h=yf.Ticker("^IRX").history(period="5d",auto_adjust=False)
        if h is not None and not h.empty:
            v=float(h["Close"].dropna().iloc[-1])/100.0
            if 0 <= v <= .25:
                return v,"^IRX 13-week Treasury yield"
    except Exception:
        pass
    return .04,"4% fallback"


def _bs_metrics(side,spot,strike,dte,iv,r):
    if not spot or not strike or not iv or iv<=0 or dte<=0:
        return {}
    T=max(dte/365.0,1/3650)
    sigma=max(iv,1e-6)
    root=math.sqrt(T)
    d1=(math.log(spot/strike)+(r+0.5*sigma*sigma)*T)/(sigma*root)
    d2=d1-sigma*root
    nd1=_cdf(d1); nd2=_cdf(d2)
    disc=math.exp(-r*T)
    if side=="call":
        price=spot*nd1-strike*disc*nd2
        delta=nd1
        theta=(-(spot*_pdf(d1)*sigma)/(2*root)-r*strike*disc*nd2)/365
        prob_itm=_cdf(d2)
    else:
        price=strike*disc*_cdf(-d2)-spot*_cdf(-d1)
        delta=nd1-1
        theta=(-(spot*_pdf(d1)*sigma)/(2*root)+r*strike*disc*_cdf(-d2))/365
        prob_itm=_cdf(-d2)
    gamma=_pdf(d1)/(spot*sigma*root)
    vega=(spot*_pdf(d1)*root)/100
    return {
        "model_price":_f(price),
        "delta":_f(delta),
        "gamma":_f(gamma),
        "theta_per_share_per_day":_f(theta),
        "theta_per_contract_per_day":_f(theta*100),
        "vega_per_share_per_vol_point":_f(vega),
        "vega_per_contract_per_vol_point":_f(vega*100),
        "risk_neutral_prob_itm":_f(prob_itm),
    }


def enrich_contract(r, spot, annual_rv, risk_free):
    dte=max(r["dte"],1)
    sigma=max(annual_rv or 0,1e-6)
    move=sigma*math.sqrt(dte/252)
    target=r["breakeven"]
    if target and spot>0 and move>0:
        z=math.log(target/spot)/move
        p_above=1-_cdf(z)
        r["historical_vol_prob_breakeven"]=_f(p_above if r["type"]=="call" else 1-p_above)
    else:
        r["historical_vol_prob_breakeven"]=None
    r["realized_move_to_expiry"]=_f(move)
    r["iv_rv_ratio"]=_f(r["iv"]/sigma) if r["iv"] is not None and sigma>0 else None
    entry=(r.get("ask") if r.get("ask") and r.get("ask")>0 else r.get("mid")) or 0
    r["entry_debit_per_contract"]=_f(entry*100)
    r["max_loss_per_contract"]=_f(entry*100)
    r["max_profit_per_contract"]=None if r["type"]=="call" else _f(max(r["strike"]-entry,0)*100)
    r["return_if_intrinsic_at_spot"]=_f(((max(spot-r["strike"],0) if r["type"]=="call" else max(r["strike"]-spot,0))-entry)/entry) if entry>0 else None
    intrinsic=max(spot-r["strike"],0) if r["type"]=="call" else max(r["strike"]-spot,0)
    r["intrinsic_value"]=_f(intrinsic)
    r["extrinsic_value"]=_f(max((r["mid"] or 0)-intrinsic,0))
    r.update(_bs_metrics(r["type"],spot,r["strike"],dte,r["iv"],risk_free))

    flags=[]
    if r["open_interest"]>=500: flags.append("high OI")
    if r["volume"]>=100: flags.append("active")
    if r["spread_pct"] is not None and r["spread_pct"]<=.10: flags.append("tight spread")
    if r["iv_rv_ratio"] is not None and r["iv_rv_ratio"]>=1.25: flags.append("IV > realized")
    if r["iv_rv_ratio"] is not None and r["iv_rv_ratio"]<=.90: flags.append("IV < realized")
    if abs(r.get("theta_per_contract_per_day") or 0) >= 25: flags.append("high theta")
    r["research_flags"]=flags

    # Neutral contract-quality diagnostics. These describe execution/risk characteristics,
    # not whether the user should buy or sell the contract.
    checks=[]
    spread=r.get("spread_pct")
    if spread is None: checks.append({"name":"Spread","state":"unknown","detail":"No usable bid/ask spread"})
    elif spread<=.10: checks.append({"name":"Spread","state":"strong","detail":"Relatively tight quoted spread"})
    elif spread<=.20: checks.append({"name":"Spread","state":"mixed","detail":"Moderate quoted spread"})
    else: checks.append({"name":"Spread","state":"weak","detail":"Wide quoted spread may materially affect execution"})
    oi=r.get("open_interest",0); vol=r.get("volume",0)
    if oi>=500 or vol>=100: checks.append({"name":"Liquidity","state":"strong","detail":"Higher open interest or trading activity"})
    elif oi>=100 or vol>=20: checks.append({"name":"Liquidity","state":"mixed","detail":"Moderate open interest/activity"})
    else: checks.append({"name":"Liquidity","state":"weak","detail":"Lower open interest/activity"})
    theta=abs(r.get("theta_per_contract_per_day") or 0)
    debit=r.get("entry_debit_per_contract") or 0
    theta_ratio=(theta/debit) if debit>0 else None
    r["theta_cost_pct_per_day"]=_f(theta_ratio)
    if theta_ratio is None: checks.append({"name":"Time decay","state":"unknown","detail":"Theta unavailable"})
    elif theta_ratio<=.01: checks.append({"name":"Time decay","state":"strong","detail":"Lower modeled daily decay relative to debit"})
    elif theta_ratio<=.025: checks.append({"name":"Time decay","state":"mixed","detail":"Meaningful modeled daily decay"})
    else: checks.append({"name":"Time decay","state":"weak","detail":"High modeled daily decay relative to debit"})
    ratio=r.get("iv_rv_ratio")
    if ratio is None: checks.append({"name":"Volatility pricing","state":"unknown","detail":"IV/realized comparison unavailable"})
    elif ratio>=1.25: checks.append({"name":"Volatility pricing","state":"caution","detail":"IV materially exceeds recent realized volatility"})
    elif ratio<=.90: checks.append({"name":"Volatility pricing","state":"notable","detail":"IV is below recent realized volatility"})
    else: checks.append({"name":"Volatility pricing","state":"neutral","detail":"IV is near recent realized volatility"})
    r["quality_checks"]=checks
    return r



def _yahoo_option_snapshot(ticker, spot, annual_rv, max_expiries, strikes_each_side):
    t=yf.Ticker(ticker)
    expiries=list(t.options)[:max_expiries]
    rows=[]
    now=datetime.now(timezone.utc)
    market_today=datetime.now(ZoneInfo("America/New_York")).date()
    risk_free,risk_free_source=_risk_free_rate()
    for exp in expiries:
        try:
            chain=t.option_chain(exp)
            dte=max((date.fromisoformat(exp)-market_today).days,0)
            for side,df in (("call",chain.calls),("put",chain.puts)):
                if df is None or df.empty:
                    continue
                x=df.copy()
                x["distance"]=(x["strike"]-spot).abs()
                x=x.sort_values("distance").head(strikes_each_side*2+1)
                for _,row in x.iterrows():
                    bid=_f(row.get("bid")); ask=_f(row.get("ask")); last=_f(row.get("lastPrice"))
                    mid=(bid+ask)/2 if bid is not None and ask is not None and ask>=bid and (bid>0 or ask>0) else last
                    strike=_f(row.get("strike"))
                    entry=ask if ask is not None and ask>0 else mid
                    breakeven=(strike+entry if side=="call" else strike-entry) if strike is not None and entry is not None else None
                    spread=(ask-bid) if bid is not None and ask is not None else None
                    last_trade=row.get("lastTradeDate")
                    if hasattr(last_trade,"to_pydatetime"):
                        last_trade=last_trade.to_pydatetime()
                    if isinstance(last_trade,datetime):
                        if last_trade.tzinfo is None:
                            last_trade=last_trade.replace(tzinfo=timezone.utc)
                        else:
                            last_trade=last_trade.astimezone(timezone.utc)
                        last_trade=last_trade.isoformat()
                    elif last_trade is not None:
                        last_trade=str(last_trade)
                    rows.append({
                        "contract_symbol":str(row.get("contractSymbol") or ""),
                        "type":side,"expiration":exp,"dte":dte,"strike":strike,
                        "bid":bid,"ask":ask,"mid":_f(mid),"last":last,
                        "iv":_f(row.get("impliedVolatility")),
                        "volume":_i(row.get("volume")),
                        "open_interest":_i(row.get("openInterest")),
                        "in_the_money":bool(row.get("inTheMoney",False)),
                        "last_trade":last_trade,
                        "quote_time":last_trade,
                        "quote_age_hours":None,
                        "provider":"Yahoo Finance via yfinance",
                        "provider_key":"yahoo",
                        "feed":"delayed/best-effort",
                        "realtime":False,
                        "consolidated":False,
                        "breakeven":_f(breakeven),
                        "breakeven_move":_f((breakeven/spot)-1) if breakeven else None,
                        "spread_pct":_f((spread/mid) if spread is not None and mid and mid>0 else None),
                    })
        except Exception:
            continue
    liquid=[enrich_contract(r,spot,annual_rv,risk_free) for r in rows
            if r["mid"] and r["mid"]>0 and (r["open_interest"]>=25 or r["volume"]>=5)]
    liquid.sort(key=lambda r:(r["dte"],abs((r["strike"] or spot)-spot),r["spread_pct"] if r["spread_pct"] is not None else 99))
    return {
        "source":"Yahoo Finance via yfinance",
        "provider_key":"yahoo",
        "feed":"delayed/best-effort",
        "underlying_price":_f(spot),
        "underlying_quote":{
            "price":_f(spot),"provider":"Yahoo/research close","provider_key":"yahoo",
            "feed":"delayed/best-effort","realtime":False,"consolidated":False,
        },
        "realtime":False,
        "consolidated":False,
        "greeks_frequency":"snapshot/model",
        "quote_note":"Yahoo options are a delayed/best-effort fallback and are not execution-grade.",
        "retrieved_at":now.isoformat(),
        "risk_free_rate":risk_free,
        "risk_free_source":risk_free_source,
        "expirations":expiries,
        "contracts":liquid[:240],
        "contracts_scanned":len(rows),
        "liquid_contracts":len(liquid),
    }


def _snapshot_error(name, exc):
    code=getattr(exc,"code",None)
    detail=f"HTTP {code}" if code is not None else type(exc).__name__
    return {"provider":name,"ok":False,"error":detail}


def _merge_option_snapshots(ticker, snapshots, annual_rv):
    now=datetime.now(timezone.utc)
    if not snapshots:
        raise RuntimeError("No option data provider returned a usable snapshot")
    primary=max(snapshots,key=lambda s:PROVIDER_QUALITY.get(s.get("provider_key"),0))
    risk_free,risk_free_source=_risk_free_rate()
    underlying_candidates=[s.get("underlying_quote") for s in snapshots if s.get("underlying_quote")]
    underlying=choose_underlying_quote(underlying_candidates,now=now)
    market_spot=_f((underlying or {}).get("price")) or _f(primary.get("underlying_price"))
    if not market_spot:
        raise RuntimeError("No usable underlying quote from configured providers")

    maps={
        s.get("provider_key"):{x.get("contract_symbol"):x for x in (s.get("contracts") or []) if x.get("contract_symbol")}
        for s in snapshots
    }
    merged=[]
    for base in primary.get("contracts") or []:
        symbol=base.get("contract_symbol")
        candidates=[]
        matching=[]
        for s in snapshots:
            row=(maps.get(s.get("provider_key")) or {}).get(symbol)
            if not row:
                continue
            matching.append(row)
            candidates.append({
                "provider":row.get("provider") or s.get("source"),
                "provider_key":row.get("provider_key") or s.get("provider_key"),
                "feed":row.get("feed") or s.get("feed"),
                "realtime":row.get("realtime") if row.get("realtime") is not None else s.get("realtime"),
                "consolidated":row.get("consolidated") if row.get("consolidated") is not None else s.get("consolidated"),
                "bid":row.get("bid"),"ask":row.get("ask"),"last":row.get("last"),
                "quote_time":row.get("quote_time") or row.get("last_trade"),
            })
        chosen=choose_option_quote(candidates,now=now)
        row=dict(base)
        if chosen:
            row.update({
                "bid":chosen.get("bid"),"ask":chosen.get("ask"),"mid":chosen.get("mid"),"last":chosen.get("last"),
                "quote_time":chosen.get("quote_time"),"quote_age_hours":chosen.get("quote_age_hours"),
                "selected_quote_provider":chosen.get("provider"),
                "selected_quote_provider_key":chosen.get("provider_key"),
                "selected_quote_feed":chosen.get("feed"),
                "data_confidence":chosen.get("data_confidence"),
                "provider_agreement_pct":chosen.get("provider_agreement_pct"),
                "provider_candidates":chosen.get("provider_candidates"),
                "execution_realtime":chosen.get("execution_realtime",False),
                "realtime":chosen.get("realtime",False),
                "consolidated":chosen.get("consolidated",False),
            })
            entry=chosen.get("ask") if chosen.get("ask") and chosen.get("ask")>0 else chosen.get("mid")
            if entry is not None:
                row["breakeven"]=row["strike"]+entry if row["type"]=="call" else row["strike"]-entry
                row["breakeven_move"]=_f(row["breakeven"]/market_spot-1)
                mid=chosen.get("mid")
                spread=(chosen.get("ask")-chosen.get("bid")) if chosen.get("ask") is not None and chosen.get("bid") is not None else None
                row["spread_pct"]=_f(spread/mid) if spread is not None and mid and mid>0 else None
        best_meta=max(
            matching or [base],
            key=lambda x:PROVIDER_QUALITY.get(x.get("provider_key"),0)
        )
        if best_meta.get("iv") is not None:
            row["iv"]=best_meta.get("iv")
        row["open_interest"]=max([_i(x.get("open_interest")) for x in matching] or [_i(row.get("open_interest"))])
        row["volume"]=max([_i(x.get("volume")) for x in matching] or [_i(row.get("volume"))])
        row=enrich_contract(row,market_spot,annual_rv,risk_free)
        # Prefer provider Greeks from the best metadata row when they are present.
        for key in ("delta","gamma","theta_per_share_per_day","theta_per_contract_per_day",
                    "vega_per_share_per_vol_point","vega_per_contract_per_vol_point",
                    "greeks_source","greeks_updated_at"):
            if best_meta.get(key) is not None:
                row[key]=best_meta.get(key)
        debit=row.get("entry_debit_per_contract") or 0
        theta=abs(row.get("theta_per_contract_per_day") or 0)
        if debit>0:
            row["theta_cost_pct_per_day"]=_f(theta/debit)
        merged.append(row)

    merged.sort(key=lambda r:(r["dte"],abs((r["strike"] or market_spot)-market_spot),r["spread_pct"] if r["spread_pct"] is not None else 99))
    providers=[{
        "source":s.get("source"),"provider_key":s.get("provider_key"),"feed":s.get("feed"),
        "realtime":bool(s.get("realtime")),"consolidated":bool(s.get("consolidated")),
        "contracts":len(s.get("contracts") or []),
    } for s in snapshots]
    execution_count=sum(1 for x in merged if x.get("execution_realtime") is True)
    confidences={}
    for x in merged:
        k=x.get("data_confidence") or "unknown"
        confidences[k]=confidences.get(k,0)+1
    return {
        "source":"MarketLens multi-provider router",
        "provider_key":"multi",
        "feed":"best trustworthy per contract",
        "underlying_price":market_spot,
        "underlying_quote":underlying,
        "quote_note":"Quotes are selected per contract by provider quality, feed authority, freshness, and cross-provider agreement. A newer indicative/delayed quote cannot override a trustworthy consolidated real-time quote.",
        "retrieved_at":now.isoformat(),
        "risk_free_rate":risk_free,
        "risk_free_source":risk_free_source,
        "expirations":sorted({x.get("expiration") for x in merged if x.get("expiration")}),
        "contracts":merged[:240],
        "contracts_scanned":sum(int(s.get("contracts_scanned") or 0) for s in snapshots),
        "liquid_contracts":len(merged),
        "realtime":bool(merged) and execution_count==len(merged),
        "execution_grade_contracts":execution_count,
        "data_confidence_counts":confidences,
        "providers":providers,
        "provider_count":len(providers),
        "greeks_frequency":"provider-dependent",
    }


def option_snapshot(ticker:str, spot:float, annual_rv:float|None=None, max_expiries:int=6, strikes_each_side:int=8):
    """Aggregate configured option providers and select the freshest trustworthy quote.

    Provider authority beats raw timestamp freshness: consolidated real-time OPRA
    data can never be displaced by a newer indicative or delayed quote.
    """
    snapshots=[]
    errors=[]
    if os.getenv("TRADIER_ACCESS_TOKEN"):
        try:
            snapshots.append(_tradier_option_snapshot(ticker,spot,annual_rv,max_expiries,strikes_each_side))
        except Exception as exc:
            errors.append(_snapshot_error("Tradier",exc))

    key,secret=_alpaca_credentials()
    if key and secret:
        try:
            snapshots.append(_alpaca_option_snapshot(ticker,spot,annual_rv,max_expiries,strikes_each_side))
        except Exception as exc:
            errors.append(_snapshot_error("Alpaca",exc))

    # Yahoo remains a low-authority reference/fallback. It cannot override
    # consolidated real-time data but keeps research functioning without paid feeds.
    try:
        snapshots.append(_yahoo_option_snapshot(ticker,spot,annual_rv,max_expiries,strikes_each_side))
    except Exception as exc:
        errors.append(_snapshot_error("Yahoo",exc))

    out=_merge_option_snapshots(ticker,snapshots,annual_rv)
    out["provider_errors"]=errors
    out["provider_warning"]="; ".join(f'{x["provider"]}: {x["error"]}' for x in errors) if errors else None
    return out

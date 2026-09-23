from __future__ import annotations
from datetime import datetime, timezone, date
import math
from zoneinfo import ZoneInfo
from statistics import NormalDist
import yfinance as yf


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


def option_snapshot(ticker:str, spot:float, annual_rv:float|None=None, max_expiries:int=6, strikes_each_side:int=8):
    """Best-effort delayed option-chain snapshot from yfinance.

    Greeks are Black-Scholes estimates using the chain IV and a Treasury-yield proxy.
    They are model estimates, not exchange-provided values.
    """
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
                    breakeven=None
                    if strike is not None and mid is not None:
                        breakeven=strike+mid if side=="call" else strike-mid
                    spread=(ask-bid) if bid is not None and ask is not None else None
                    spread_pct=(spread/mid) if spread is not None and mid and mid>0 else None
                    last_trade=row.get("lastTradeDate")
                    quote_age_hours=None
                    if hasattr(last_trade,"to_pydatetime"):
                        last_trade=last_trade.to_pydatetime()
                    if isinstance(last_trade,datetime):
                        if last_trade.tzinfo is None:
                            last_trade=last_trade.replace(tzinfo=timezone.utc)
                        else:
                            last_trade=last_trade.astimezone(timezone.utc)
                        quote_age_hours=max(0.0,(now-last_trade).total_seconds()/3600.0)
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
                        "quote_age_hours":_f(quote_age_hours),
                        "breakeven":_f(breakeven),
                        "breakeven_move":_f((breakeven/spot)-1) if breakeven else None,
                        "spread_pct":_f(spread_pct),
                    })
        except Exception:
            continue

    liquid=[enrich_contract(r,spot,annual_rv,risk_free) for r in rows
            if r["mid"] and r["mid"]>0 and (r["open_interest"]>=25 or r["volume"]>=5)]
    liquid.sort(key=lambda r:(r["dte"],abs((r["strike"] or spot)-spot),r["spread_pct"] if r["spread_pct"] is not None else 99))
    return {
        "source":"Yahoo Finance via yfinance",
        "quote_note":"Quotes may be delayed or stale; verify with a brokerage before acting. Greeks are Black-Scholes estimates without dividend-yield or early-exercise adjustments, not exchange-provided values.",
        "retrieved_at":now.isoformat(),
        "risk_free_rate":risk_free,
        "risk_free_source":risk_free_source,
        "expirations":expiries,
        "contracts":liquid[:240],
        "contracts_scanned":len(rows),
        "liquid_contracts":len(liquid),
    }

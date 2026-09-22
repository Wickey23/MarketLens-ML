from __future__ import annotations
from datetime import datetime, timezone
import math
from statistics import NormalDist
import yfinance as yf

def _f(x):
    try:
        v=float(x); return v if math.isfinite(v) else None
    except Exception:return None

def _cdf(z): return NormalDist().cdf(z)
def enrich_contract(r, spot, annual_rv):
    dte=max(r["dte"],1); sigma=max(annual_rv or 0,1e-6); move=sigma*math.sqrt(dte/252)
    # Lognormal probability proxy from realized volatility; research benchmark, not a forecast.
    target=r["breakeven"]
    if target and spot>0 and move>0:
        z=math.log(target/spot)/move
        p_above=1-_cdf(z)
        r["historical_vol_prob_breakeven"]= _f(p_above if r["type"]=="call" else 1-p_above)
    else:r["historical_vol_prob_breakeven"]=None
    r["realized_move_to_expiry"]=_f(move)
    r["iv_rv_ratio"]=_f(r["iv"]/sigma) if r["iv"] is not None and sigma>0 else None
    r["max_loss_per_contract"]=_f((r["mid"] or 0)*100)
    # Quality flags are descriptive filters, not trade recommendations.
    flags=[]
    if r["open_interest"]>=500: flags.append("high OI")
    if r["volume"]>=100: flags.append("active")
    if r["spread_pct"] is not None and r["spread_pct"]<=.10: flags.append("tight spread")
    if r["iv_rv_ratio"] is not None and r["iv_rv_ratio"]>=1.25: flags.append("IV > realized")
    if r["iv_rv_ratio"] is not None and r["iv_rv_ratio"]<=.90: flags.append("IV < realized")
    r["research_flags"]=flags
    return r

def option_snapshot(ticker:str, spot:float, annual_rv:float|None=None, max_expiries:int=4, strikes_each_side:int=6):
    """Best-effort delayed option-chain snapshot from yfinance.
    Intended for research; timestamps and quotes may be delayed/stale.
    """
    t=yf.Ticker(ticker); expiries=list(t.options)[:max_expiries]; rows=[]
    now=datetime.now(timezone.utc)
    for exp in expiries:
        try:
            chain=t.option_chain(exp)
            dte=max((datetime.fromisoformat(exp).date()-now.date()).days,0)
            for side,df in (("call",chain.calls),("put",chain.puts)):
                if df is None or df.empty: continue
                x=df.copy(); x["distance"]=(x["strike"]-spot).abs()
                x=x.sort_values("distance").head(strikes_each_side*2+1)
                for _,r in x.iterrows():
                    bid=_f(r.get("bid")); ask=_f(r.get("ask")); last=_f(r.get("lastPrice"))
                    mid=(bid+ask)/2 if bid is not None and ask is not None and ask>=bid and (bid>0 or ask>0) else last
                    strike=_f(r.get("strike")); breakeven=None
                    if strike is not None and mid is not None: breakeven=strike+mid if side=="call" else strike-mid
                    spread=(ask-bid) if bid is not None and ask is not None else None
                    spread_pct=(spread/mid) if spread is not None and mid and mid>0 else None
                    rows.append({"type":side,"expiration":exp,"dte":dte,"strike":strike,"bid":bid,"ask":ask,"mid":_f(mid),
                      "last":last,"iv":_f(r.get("impliedVolatility")),"volume":int(r.get("volume") or 0),"open_interest":int(r.get("openInterest") or 0),
                      "in_the_money":bool(r.get("inTheMoney",False)),"breakeven":_f(breakeven),"breakeven_move":_f((breakeven/spot)-1) if breakeven else None,
                      "spread_pct":_f(spread_pct)})
        except Exception:
            continue
    liquid=[enrich_contract(r,spot,annual_rv) for r in rows if r["mid"] and r["mid"]>0 and (r["open_interest"]>=50 or r["volume"]>=10)]
    liquid.sort(key=lambda r:(r["dte"],abs((r["strike"] or spot)-spot),r["spread_pct"] if r["spread_pct"] is not None else 99))
    return {"source":"Yahoo Finance via yfinance","quote_note":"Quotes may be delayed or stale; verify with a brokerage before acting.","retrieved_at":now.isoformat(),
      "expirations":expiries,"contracts":liquid[:120],"contracts_scanned":len(rows),"liquid_contracts":len(liquid)}

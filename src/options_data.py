from __future__ import annotations
from datetime import datetime, timezone
import math
import yfinance as yf

def _f(x):
    try:
        v=float(x); return v if math.isfinite(v) else None
    except Exception:return None

def option_snapshot(ticker:str, spot:float, max_expiries:int=4, strikes_each_side:int=6):
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
    liquid=[r for r in rows if r["mid"] and r["mid"]>0 and (r["open_interest"]>=50 or r["volume"]>=10)]
    liquid.sort(key=lambda r:(r["dte"],abs((r["strike"] or spot)-spot),r["spread_pct"] if r["spread_pct"] is not None else 99))
    return {"source":"Yahoo Finance via yfinance","quote_note":"Quotes may be delayed or stale; verify with a brokerage before acting.","retrieved_at":now.isoformat(),
      "expirations":expiries,"contracts":liquid[:120],"contracts_scanned":len(rows),"liquid_contracts":len(liquid)}

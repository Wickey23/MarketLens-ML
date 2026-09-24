from __future__ import annotations

from datetime import datetime, timezone
import math


PROVIDER_QUALITY = {
    "tradier": 100,
    "alpaca_opra": 100,
    "alpaca_sip": 95,
    "alpaca_iex": 80,
    "finnhub": 70,
    "alpaca_indicative": 55,
    "yahoo": 25,
    "snapshot": 10,
}


def _f(value):
    try:
        x=float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def parse_timestamp(value):
    if not value:
        return None
    if isinstance(value,(int,float)):
        try:
            return datetime.fromtimestamp(float(value),timezone.utc)
        except Exception:
            return None
    try:
        dt=datetime.fromisoformat(str(value).replace("Z","+00:00"))
        if dt.tzinfo is None:
            dt=dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def quote_age_seconds(candidate, now=None):
    now=now or datetime.now(timezone.utc)
    dt=parse_timestamp(candidate.get("quote_time") or candidate.get("market_timestamp"))
    if dt is None:
        return None
    return max(0.0,(now-dt).total_seconds())


def _quality(candidate):
    key=str(candidate.get("provider_key") or "").lower()
    return PROVIDER_QUALITY.get(key,0)


def _price(candidate):
    p=_f(candidate.get("price"))
    if p is not None and p>0:
        return p
    bid=_f(candidate.get("bid")); ask=_f(candidate.get("ask"))
    if bid is not None and ask is not None and ask>=bid and (bid>0 or ask>0):
        return (bid+ask)/2
    return _f(candidate.get("last"))


def _mid(candidate):
    bid=_f(candidate.get("bid")); ask=_f(candidate.get("ask"))
    if bid is not None and ask is not None and bid>=0 and ask>0 and ask>=bid:
        return (bid+ask)/2
    return _price(candidate)


def _valid_underlying(candidate):
    p=_price(candidate)
    return p is not None and p>0


def _valid_option(candidate):
    bid=_f(candidate.get("bid")); ask=_f(candidate.get("ask"))
    return bid is not None and ask is not None and bid>=0 and ask>0 and ask>=bid


def _rank(candidate, now):
    quality=_quality(candidate)
    age=quote_age_seconds(candidate,now)
    freshness=-(age if age is not None else 10**9)
    realtime=1 if candidate.get("realtime") is True else 0
    consolidated=1 if candidate.get("consolidated") is True else 0
    return (quality,realtime,consolidated,freshness)


def _agreement(selected, candidates):
    selected_mid=_mid(selected)
    if selected_mid is None or selected_mid<=0:
        return None,None
    comparable=[]
    for row in candidates:
        if row is selected:
            continue
        m=_mid(row)
        if m is None or m<=0:
            continue
        # Delayed Yahoo is still useful as a sanity check, but do not let it
        # establish "high" agreement by itself.
        comparable.append((row,abs(m-selected_mid)/selected_mid))
    if not comparable:
        return None,None
    best=min(x[1] for x in comparable)
    return best,comparable


def _confidence(selected,candidates):
    diff,comparable=_agreement(selected,candidates)
    q=_quality(selected)
    if diff is not None and diff>0.02 and comparable:
        trusted_peer=any(_quality(r)>=80 for r,_ in comparable)
        if trusted_peer:
            return "conflict",diff
    trusted=[
        r for r in candidates
        if _quality(r)>=90 and r.get("realtime") is True
    ]
    if q>=90 and selected.get("realtime") is True:
        if len(trusted)>=2 and diff is not None and diff<=0.005:
            return "high",diff
        return "medium",diff
    if q>=70 and selected.get("realtime") is True:
        return "medium",diff
    return "low",diff


def _compact_candidates(candidates,now):
    out=[]
    for r in sorted(candidates,key=lambda x:_rank(x,now),reverse=True):
        out.append({
            "provider":r.get("provider"),
            "provider_key":r.get("provider_key"),
            "feed":r.get("feed"),
            "realtime":bool(r.get("realtime")),
            "consolidated":bool(r.get("consolidated")),
            "bid":_f(r.get("bid")),
            "ask":_f(r.get("ask")),
            "last":_f(r.get("last")),
            "price":_price(r),
            "quote_time":r.get("quote_time") or r.get("market_timestamp"),
            "quote_age_seconds":quote_age_seconds(r,now),
        })
    return out


def choose_underlying_quote(candidates,now=None):
    now=now or datetime.now(timezone.utc)
    valid=[dict(x) for x in candidates if _valid_underlying(x)]
    if not valid:
        return None
    selected=max(valid,key=lambda x:_rank(x,now))
    confidence,diff=_confidence(selected,valid)
    out=dict(selected)
    out["price"]=_price(selected)
    out["selected_provider_quality"]=_quality(selected)
    out["data_confidence"]=confidence
    out["provider_agreement_pct"]=(diff*100 if diff is not None else None)
    out["quote_age_seconds"]=quote_age_seconds(selected,now)
    out["provider_candidates"]=_compact_candidates(valid,now)
    return out


def choose_option_quote(candidates,now=None):
    now=now or datetime.now(timezone.utc)
    valid=[dict(x) for x in candidates if _valid_option(x)]
    if not valid:
        return None
    selected=max(valid,key=lambda x:_rank(x,now))
    confidence,diff=_confidence(selected,valid)
    out=dict(selected)
    bid=_f(out.get("bid")); ask=_f(out.get("ask"))
    out["mid"]=(bid+ask)/2
    out["selected_provider_quality"]=_quality(selected)
    out["data_confidence"]=confidence
    out["provider_agreement_pct"]=(diff*100 if diff is not None else None)
    out["quote_age_seconds"]=quote_age_seconds(selected,now)
    out["quote_age_hours"]=(out["quote_age_seconds"]/3600.0 if out["quote_age_seconds"] is not None else None)
    out["execution_realtime"]=bool(
        out.get("realtime") is True
        and out.get("consolidated") is True
        and _quality(out)>=90
        and confidence!="conflict"
    )
    out["provider_candidates"]=_compact_candidates(valid,now)
    return out

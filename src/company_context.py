from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from difflib import SequenceMatcher
import yfinance as yf


def _iso(value):
    if value is None:
        return None
    try:
        if hasattr(value, "to_pydatetime"):
            value = value.to_pydatetime()
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.isoformat()
        return str(value)
    except Exception:
        return None


def _earnings_dates(ticker_obj, limit=4):
    rows = []
    try:
        df = ticker_obj.get_earnings_dates(limit=limit)
        if df is not None and not df.empty:
            for idx, row in df.iterrows():
                rows.append({
                    "date": _iso(idx),
                    "eps_estimate": _num(row.get("EPS Estimate")),
                    "reported_eps": _num(row.get("Reported EPS")),
                    "surprise_pct": _num(row.get("Surprise(%)")),
                })
    except Exception:
        pass
    return rows


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _extract_news_item(item):
    content = item.get("content") or {}
    provider = content.get("provider") or {}
    canonical = content.get("canonicalUrl") or {}
    click = content.get("clickThroughUrl") or {}
    title = item.get("title") or content.get("title")
    publisher = item.get("publisher") or provider.get("displayName")
    link = item.get("link") or canonical.get("url") or click.get("url")
    published = item.get("providerPublishTime") or content.get("pubDate") or content.get("displayTime")
    if isinstance(published, (int, float)):
        published = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()
    elif published is not None:
        published = str(published)
    if not title:
        return None
    return {
        "title": str(title),
        "publisher": str(publisher) if publisher else None,
        "url": str(link) if link else None,
        "published_at": published,
    }


def _catalyst_flags(headlines):
    text = " ".join((h.get("title") or "").lower() for h in headlines)
    groups = {
        "earnings/guidance": ["earnings", "guidance", "revenue", "profit", "eps", "forecast"],
        "analyst activity": ["upgrade", "downgrade", "price target", "analyst"],
        "regulatory/legal": ["sec ", "regulator", "lawsuit", "investigation", "antitrust", "recall"],
        "capital activity": ["offering", "buyback", "dividend", "debt", "convertible"],
        "product/operations": ["launch", "delivery", "production", "factory", "partnership", "contract"],
    }
    return [name for name, words in groups.items() if any(w in text for w in words)]



_POSITIVE_TERMS = {
    "beat": 1.0, "beats": 1.0, "surge": 0.8, "surges": 0.8, "growth": 0.5,
    "raises guidance": 1.2, "raised guidance": 1.2, "upgrade": 0.8, "upgraded": 0.8,
    "record": 0.5, "approval": 0.9, "approved": 0.9, "partnership": 0.5,
    "contract win": 0.8, "buyback": 0.5, "strong demand": 0.8, "outperform": 0.6,
    "profit rises": 0.8, "revenue rises": 0.7, "launch": 0.3,
}
_NEGATIVE_TERMS = {
    "miss": -1.0, "misses": -1.0, "cuts guidance": -1.2, "cut guidance": -1.2,
    "downgrade": -0.8, "downgraded": -0.8, "lawsuit": -0.6, "investigation": -0.8,
    "recall": -0.7, "warning": -0.5, "weak demand": -0.8, "layoffs": -0.5,
    "offering": -0.5, "dilution": -0.8, "fraud": -1.0, "probe": -0.7,
    "profit falls": -0.8, "revenue falls": -0.7, "guidance below": -1.0,
}
_SOURCE_QUALITY = {
    "reuters": 1.0, "associated press": 1.0, "ap": 1.0, "bloomberg": 1.0,
    "wall street journal": .95, "wsj": .95, "cnbc": .9, "financial times": .95,
    "marketwatch": .8, "barrons": .85, "barron's": .85, "yahoo finance": .75,
    "benzinga": .65, "seeking alpha": .55, "motley fool": .45,
}

def _published_dt(value):
    if not value:
        return None
    try:
        dt=datetime.fromisoformat(str(value).replace("Z","+00:00"))
        if dt.tzinfo is None:
            dt=dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def _source_weight(name):
    s=str(name or "").lower()
    for key,weight in _SOURCE_QUALITY.items():
        if key in s:
            return weight
    return .6

def _headline_direction(title):
    text=" "+re.sub(r"\s+"," ",str(title or "").lower())+" "
    score=0.0
    hits=[]
    for term,val in {**_POSITIVE_TERMS,**_NEGATIVE_TERMS}.items():
        if term in text:
            score+=val
            hits.append(term)
    # Negation guard for common constructions.
    if any(p in text for p in (" not expected to "," fails to "," failed to ")):
        score*=-.5
    return max(-2.0,min(2.0,score)),hits

def _news_signal(headlines, now=None):
    now=now or datetime.now(timezone.utc)
    scored=[]
    seen_titles=[]
    for h in headlines or []:
        title=str(h.get("title") or "").strip()
        if not title:
            continue
        norm=re.sub(r"[^a-z0-9 ]+"," ",title.lower())
        norm=re.sub(r"\s+"," ",norm).strip()
        duplicate=max((SequenceMatcher(None,norm,prev).ratio() for prev in seen_titles),default=0.0)>=.82
        if not duplicate:
            seen_titles.append(norm)
        dt=_published_dt(h.get("published_at"))
        age_hours=max(0.0,(now-dt).total_seconds()/3600.0) if dt else None
        freshness=(1.0 if age_hours is not None and age_hours<=6 else
                   .8 if age_hours is not None and age_hours<=24 else
                   .5 if age_hours is not None and age_hours<=72 else
                   .25 if age_hours is not None and age_hours<=168 else .15)
        direction,hits=_headline_direction(title)
        source_quality=_source_weight(h.get("publisher"))
        novelty=.2 if duplicate else 1.0
        relevance=1.0 if hits else .35
        weight=freshness*source_quality*novelty*relevance
        scored.append({
            **h,
            "direction":_num(direction),
            "freshness_weight":_num(freshness),
            "source_quality":_num(source_quality),
            "novelty_weight":_num(novelty),
            "relevance_weight":_num(relevance),
            "weight":_num(weight),
            "age_hours":_num(age_hours),
            "matched_terms":hits[:6],
            "duplicate_like":duplicate,
        })
    usable=[x for x in scored if abs(float(x.get("direction") or 0))>0 and float(x.get("weight") or 0)>.08]
    denom=sum(float(x.get("weight") or 0) for x in usable)
    raw=(sum(float(x.get("direction") or 0)*float(x.get("weight") or 0) for x in usable)/denom) if denom else 0.0
    signal=max(-1.0,min(1.0,raw/1.5))
    quality=(sum(float(x.get("source_quality") or 0)*float(x.get("weight") or 0) for x in usable)/denom) if denom else 0.0
    freshness=(sum(float(x.get("freshness_weight") or 0)*float(x.get("weight") or 0) for x in usable)/denom) if denom else 0.0
    confidence=min(1.0,(len(usable)/4.0)*.45+quality*.30+freshness*.25) if usable else 0.0
    positive=sum(1 for x in usable if float(x.get("direction") or 0)>0)
    negative=sum(1 for x in usable if float(x.get("direction") or 0)<0)
    conflict=(positive>0 and negative>0 and min(positive,negative)/max(positive,negative)>=.5)
    if conflict:
        confidence*=.7
    return {
        "score":_num(signal),
        "confidence":_num(confidence),
        "positive_headlines":positive,
        "negative_headlines":negative,
        "usable_headlines":len(usable),
        "total_headlines":len(scored),
        "conflicting":bool(conflict),
        "method":"Freshness × source quality × relevance × novelty, with bounded headline-direction lexicon. News is a context modifier, not a standalone trade signal.",
        "headlines":scored,
    }

def company_context(ticker: str, news_limit: int = 8):
    """Best-effort company/event context from yfinance.

    Headlines are surfaced as context, not converted into a directional recommendation.
    """
    t = yf.Ticker(ticker)
    news = []
    try:
        for item in (t.news or [])[: max(news_limit * 2, news_limit)]:
            parsed = _extract_news_item(item)
            if parsed:
                news.append(parsed)
            if len(news) >= news_limit:
                break
    except Exception:
        pass

    earnings = _earnings_dates(t, limit=4)
    now = datetime.now(timezone.utc)
    next_earnings = None
    days_to_earnings = None
    for row in earnings:
        try:
            dt = datetime.fromisoformat(row["date"].replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt >= now:
                next_earnings = row["date"]
                days_to_earnings = max((dt.date() - now.date()).days, 0)
                break
        except Exception:
            continue

    info = {}
    try:
        fast = t.fast_info
        info = {
            "market_cap": _num(getattr(fast, "market_cap", None)),
            "currency": getattr(fast, "currency", None),
            "exchange": getattr(fast, "exchange", None),
        }
    except Exception:
        pass

    news_signal=_news_signal(news,now=now)

    return {
        "news": news_signal.get("headlines") or news,
        "news_signal": news_signal,
        "news_note": "Recent headlines are scored as a bounded catalyst/context modifier using freshness, source quality, relevance and novelty. News alone cannot create a BUY candidate.",
        "catalyst_flags": _catalyst_flags(news),
        "earnings": {
            "next_earnings": next_earnings,
            "days_to_earnings": days_to_earnings,
            "recent_and_upcoming": earnings,
        },
        "company": info,
    }

from __future__ import annotations

from datetime import datetime, timezone
import math
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

    return {
        "news": news,
        "news_note": "Recent headlines are context only; MarketLens does not treat headline wording as a validated directional signal.",
        "catalyst_flags": _catalyst_flags(news),
        "earnings": {
            "next_earnings": next_earnings,
            "days_to_earnings": days_to_earnings,
            "recent_and_upcoming": earnings,
        },
        "company": info,
    }

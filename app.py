from flask import Flask, jsonify, render_template, request
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hmac
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from src.provider_router import choose_underlying_quote

app = Flask(__name__)
DATA = Path(__file__).with_name("data") / "dashboard.json"

_TRIGGER_COOLDOWN_SECONDS = 30
_TRIGGER_REMOTE_COOLDOWN_SECONDS = 300
_LIVE_QUOTE_TTL_SECONDS = 5
_RESEARCH_DATA_TTL_SECONDS = 15
_trigger_last_seen = {}
_live_quote_cache = {}
_research_data_cache = {"at":None,"payload":None}


def _control_authorized():
    """Protect expensive/control-plane endpoints when an app key is configured."""
    expected=os.getenv("MARKETLENS_CONTROL_KEY")
    if not expected:
        return True
    provided=request.headers.get("X-MarketLens-Key") or ""
    return hmac.compare_digest(str(provided),str(expected))


def _latest_fast_refresh_age_seconds(token):
    """Best-effort cross-instance cooldown using GitHub's own run history."""
    url="https://api.github.com/repos/Wickey23/MarketLens-ML/actions/workflows/fast-refresh.yml/runs?branch=main&per_page=1"
    req=urllib.request.Request(
        url,
        headers={
            "Authorization":f"Bearer {token}",
            "Accept":"application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28",
            "User-Agent":"MarketLens-ML",
        },
    )
    with urllib.request.urlopen(req,timeout=6) as response:
        body=json.loads(response.read().decode("utf-8"))
    rows=body.get("workflow_runs") or []
    if not rows:
        return None
    created=rows[0].get("created_at")
    if not created:
        return None
    dt=datetime.fromisoformat(str(created).replace("Z","+00:00"))
    if dt.tzinfo is None:
        dt=dt.replace(tzinfo=timezone.utc)
    return max(0.0,(datetime.now(timezone.utc)-dt.astimezone(timezone.utc)).total_seconds())


def read_data():
    """Read the latest generated research state from the dedicated data branch."""
    now=time.monotonic()
    cached=_research_data_cache.get("payload")
    cached_at=_research_data_cache.get("at")
    if cached is not None and cached_at is not None and now-cached_at<_RESEARCH_DATA_TTL_SECONDS:
        return dict(cached)
    url = "https://raw.githubusercontent.com/Wickey23/MarketLens-ML/market-data/data/dashboard.json"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "MarketLens-ML/1.0", "Cache-Control": "no-cache"},
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
            payload["_data_source"] = "github-market-data-live-v3"
            _research_data_cache["at"]=now
            _research_data_cache["payload"]=payload
            return dict(payload)
    except Exception as exc:
        if DATA.exists():
            payload = json.loads(DATA.read_text(encoding="utf-8"))
            payload["_data_source"] = "bundled-fallback"
            payload["_live_data_error"] = str(exc)
            return payload
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "horizon_days": 5,
            "tickers": [],
            "errors": ["Live research snapshot unavailable"],
            "_live_data_error": str(exc),
        }


def _snapshot_quote(ticker):
    payload = read_data()
    row = next(
        (x for x in payload.get("tickers", []) if str(x.get("ticker", "")).upper() == ticker),
        None,
    )
    if not row:
        return None
    return {
        "ticker": ticker,
        "price": row.get("price"),
        "previous_close": None,
        "open": None,
        "high": None,
        "low": None,
        "change": None,
        "change_pct": row.get("change_1d"),
        "market_timestamp": row.get("fast_refreshed_at") or row.get("as_of"),
        "exchange": None,
        "currency": None,
        "provider": "MarketLens research snapshot",
        "provider_key":"snapshot",
        "feed":"snapshot",
        "consolidated":False,
        "realtime": False,
        "delayed": True,
        "note": "Near-live quote provider unavailable; showing the latest research snapshot.",
    }


def _tradier_live_quote(ticker, token):
    """Server-side Tradier quote fallback; the permanent token never reaches the browser."""
    query=urllib.parse.urlencode({"symbols":ticker,"greeks":"false"})
    url=f"https://api.tradier.com/v1/markets/quotes?{query}"
    req=urllib.request.Request(
        url,
        headers={
            "Authorization":f"Bearer {token}",
            "Accept":"application/json",
            "User-Agent":"MarketLens-ML/1.0",
        },
    )
    with urllib.request.urlopen(req,timeout=5) as response:
        body=json.loads(response.read().decode("utf-8"))
    row=(body.get("quotes") or {}).get("quote")
    if isinstance(row,list):
        row=row[0] if row else None
    if not isinstance(row,dict):
        raise ValueError("Tradier quote unavailable")
    def num(value):
        try:
            return float(value) if value is not None else None
        except Exception:
            return None
    bid=num(row.get("bid")); ask=num(row.get("ask")); last=num(row.get("last"))
    price=last if last is not None and last>0 else ((bid+ask)/2 if bid is not None and ask is not None and ask>=bid else None)
    prev=num(row.get("prevclose"))
    if prev is None:
        prev=num(row.get("close"))
    if price is None or price<=0:
        raise ValueError("Tradier price unavailable")
    change=(price-prev) if prev not in (None,0) else num(row.get("change"))
    change_pct=(change/prev) if change is not None and prev not in (None,0) else None
    ts=max(
        int(num(row.get("bid_date")) or 0),
        int(num(row.get("ask_date")) or 0),
        int(num(row.get("trade_date")) or 0),
    )
    return {
        "ticker":ticker,
        "price":price,
        "previous_close":prev,
        "open":num(row.get("open")),
        "high":num(row.get("high")),
        "low":num(row.get("low")),
        "bid":bid,
        "ask":ask,
        "change":change,
        "change_pct":change_pct,
        "market_timestamp":datetime.fromtimestamp(ts/1000,timezone.utc).isoformat() if ts else None,
        "exchange":row.get("exch"),
        "currency":"USD",
        "provider":"Tradier Brokerage API",
        "provider_key":"tradier",
        "feed":"consolidated",
        "consolidated":True,
        "realtime":True,
        "delayed":False,
        "note":"Production Tradier brokerage quote. Real-time availability depends on the configured account entitlement.",
    }


def _finnhub_live_quote(ticker, api_key):
    """Optional provider-backed quote path for lower-latency production data."""
    query=urllib.parse.urlencode({"symbol":ticker,"token":api_key})
    url=f"https://finnhub.io/api/v1/quote?{query}"
    req=urllib.request.Request(
        url,
        headers={"User-Agent":"MarketLens-ML/1.0","Accept":"application/json"},
    )
    with urllib.request.urlopen(req,timeout=5) as response:
        body=json.loads(response.read().decode("utf-8"))
    price=body.get("c")
    prev=body.get("pc")
    if price in (None,0):
        raise ValueError("Provider quote unavailable")
    ts=body.get("t")
    change=body.get("d")
    change_pct=(float(body["dp"])/100.0) if body.get("dp") is not None else (
        (float(price)/float(prev)-1.0) if prev not in (None,0) else None
    )
    return {
        "ticker":ticker,
        "price":float(price),
        "previous_close":float(prev) if prev is not None else None,
        "open":float(body["o"]) if body.get("o") is not None else None,
        "high":float(body["h"]) if body.get("h") is not None else None,
        "low":float(body["l"]) if body.get("l") is not None else None,
        "change":float(change) if change is not None else None,
        "change_pct":change_pct,
        "market_timestamp":datetime.fromtimestamp(int(ts),timezone.utc).isoformat() if ts else None,
        "exchange":None,
        "currency":"USD",
        "provider":"Finnhub quote",
        "provider_key":"finnhub",
        "feed":"provider",
        "consolidated":False,
        "realtime":True,
        "delayed":False,
        "note":"Provider-backed quote path. Actual exchange entitlements depend on the configured provider account.",
    }


def _alpaca_live_quote(ticker,key,secret):
    feed=(os.getenv("ALPACA_STOCK_FEED") or "iex").strip().lower()
    if feed not in ("sip","iex","delayed_sip"):
        feed="iex"
    symbol=urllib.parse.quote(ticker,safe="")
    url=f"https://data.alpaca.markets/v2/stocks/{symbol}/snapshot?"+urllib.parse.urlencode({"feed":feed})
    req=urllib.request.Request(
        url,
        headers={
            "APCA-API-KEY-ID":key,
            "APCA-API-SECRET-KEY":secret,
            "Accept":"application/json",
            "User-Agent":"MarketLens-ML/1.0",
        },
    )
    with urllib.request.urlopen(req,timeout=5) as response:
        body=json.loads(response.read().decode("utf-8"))
    q=body.get("latestQuote") or body.get("latest_quote") or {}
    tr=body.get("latestTrade") or body.get("latest_trade") or {}
    daily=body.get("dailyBar") or body.get("daily_bar") or {}
    prevbar=body.get("prevDailyBar") or body.get("prev_daily_bar") or {}
    def num(v):
        try:return float(v) if v is not None else None
        except Exception:return None
    bid=num(q.get("bp") if "bp" in q else q.get("bid_price"))
    ask=num(q.get("ap") if "ap" in q else q.get("ask_price"))
    last=num(tr.get("p") if "p" in tr else tr.get("price"))
    price=last if last is not None and last>0 else ((bid+ask)/2 if bid is not None and ask is not None and ask>=bid else None)
    if price is None or price<=0:
        raise ValueError("Alpaca quote unavailable")
    prev=num(prevbar.get("c") if "c" in prevbar else prevbar.get("close"))
    ts=q.get("t") or q.get("timestamp") or tr.get("t") or tr.get("timestamp")
    consolidated=feed=="sip"
    delayed=feed=="delayed_sip"
    return {
        "ticker":ticker,"price":price,"previous_close":prev,
        "open":num(daily.get("o") if "o" in daily else daily.get("open")),
        "high":num(daily.get("h") if "h" in daily else daily.get("high")),
        "low":num(daily.get("l") if "l" in daily else daily.get("low")),
        "bid":bid,"ask":ask,"last":last,
        "change":(price-prev) if prev not in (None,0) else None,
        "change_pct":((price/prev)-1) if prev not in (None,0) else None,
        "market_timestamp":ts,
        "exchange":None,"currency":"USD",
        "provider":"Alpaca Market Data",
        "provider_key":"alpaca_sip" if consolidated else ("yahoo" if delayed else "alpaca_iex"),
        "feed":feed,
        "consolidated":consolidated,
        "realtime":not delayed,
        "delayed":delayed,
        "note":"Alpaca stock snapshot. SIP is consolidated; IEX is real-time exchange-subset data.",
    }


def _yahoo_live_quote(ticker):
    """Best-effort near-live underlying quote without an extra server dependency."""
    symbol = urllib.parse.quote(ticker, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        "?interval=1m&range=1d&includePrePost=true"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 MarketLens/1.0",
            "Accept": "application/json",
            "Cache-Control": "no-cache",
        },
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        body = json.loads(response.read().decode("utf-8"))

    result = ((body.get("chart") or {}).get("result") or [None])[0]
    if not result:
        raise ValueError("No Yahoo quote returned")

    meta = result.get("meta") or {}
    timestamps = result.get("timestamp") or []
    price = meta.get("regularMarketPrice")
    if price is None:
        closes = (
            (((result.get("indicators") or {}).get("quote")) or [{}])[0].get("close")
            or []
        )
        usable = [x for x in closes if x is not None]
        price = usable[-1] if usable else None

    prev = meta.get("previousClose")
    if prev is None:
        prev = meta.get("chartPreviousClose")

    change = (
        float(price) - float(prev)
        if price is not None and prev not in (None, 0)
        else None
    )
    change_pct = change / float(prev) if change is not None and prev not in (None, 0) else None
    ts = meta.get("regularMarketTime") or (timestamps[-1] if timestamps else None)

    return {
        "ticker": ticker,
        "price": float(price) if price is not None else None,
        "previous_close": float(prev) if prev is not None else None,
        "open": meta.get("regularMarketOpen"),
        "high": meta.get("regularMarketDayHigh"),
        "low": meta.get("regularMarketDayLow"),
        "change": change,
        "change_pct": change_pct,
        "market_timestamp": datetime.fromtimestamp(int(ts), timezone.utc).isoformat() if ts else None,
        "exchange": meta.get("exchangeName"),
        "currency": meta.get("currency"),
        "market_state": meta.get("marketState"),
        "provider": "Yahoo Finance chart",
        "provider_key":"yahoo",
        "feed":"best-effort",
        "consolidated":False,
        "realtime": False,
        "delayed": True,
        "note": "Best-effort near-live underlying quote; may be delayed. Options and model evidence remain research snapshots.",
    }


def live_quote(ticker):
    now = time.monotonic()
    cached = _live_quote_cache.get(ticker)
    if cached and now - cached["at"] < _LIVE_QUOTE_TTL_SECONDS:
        return cached["payload"]

    candidates=[]
    errors=[]
    tradier_token=os.getenv("TRADIER_ACCESS_TOKEN")
    if tradier_token:
        try:candidates.append(_tradier_live_quote(ticker,tradier_token))
        except Exception as exc:errors.append({"provider":"Tradier","error":type(exc).__name__})

    alpaca_key=os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
    alpaca_secret=os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if alpaca_key and alpaca_secret:
        try:candidates.append(_alpaca_live_quote(ticker,alpaca_key,alpaca_secret))
        except Exception as exc:errors.append({"provider":"Alpaca","error":type(exc).__name__})

    provider_key=os.getenv("FINNHUB_API_KEY")
    if provider_key:
        try:candidates.append(_finnhub_live_quote(ticker,provider_key))
        except Exception as exc:errors.append({"provider":"Finnhub","error":type(exc).__name__})

    try:candidates.append(_yahoo_live_quote(ticker))
    except Exception as exc:errors.append({"provider":"Yahoo","error":type(exc).__name__})

    out=choose_underlying_quote(candidates,now=datetime.now(timezone.utc))
    if out is None:
        out=_snapshot_quote(ticker)
    if out is not None:
        out["ticker"]=ticker
        out["delayed"]=not bool(out.get("realtime"))
        out["provider_errors"]=errors
        out["served_at"] = datetime.now(timezone.utc).isoformat()
        out["note"]="MarketLens selected the freshest trustworthy configured quote; authoritative real-time feeds outrank indicative/delayed sources."
        _live_quote_cache[ticker] = {"at": now, "payload": out}
    return out


def live_quotes(tickers):
    quotes = []
    errors = []
    workers = max(1, min(6, len(tickers)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(live_quote, ticker): ticker for ticker in tickers}
        for job in as_completed(jobs):
            ticker = jobs[job]
            try:
                quote = job.result()
                if quote:
                    quotes.append(quote)
                else:
                    errors.append({"ticker": ticker, "error": "Quote unavailable"})
            except Exception as exc:
                errors.append({"ticker": ticker, "error": str(exc)})
    quotes.sort(key=lambda x: tickers.index(x["ticker"]))
    return quotes, errors


@app.after_request
def security_headers(response):
    response.headers.setdefault("X-Content-Type-Options","nosniff")
    response.headers.setdefault("X-Frame-Options","DENY")
    response.headers.setdefault("Referrer-Policy","strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy","camera=(), microphone=(), geolocation=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; connect-src 'self' wss://ws.tradier.com; "
        "img-src 'self' data: https:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )
    return response


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/data")
def data():
    response = jsonify(read_data())
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/api/live-quote")
def api_live_quote():
    ticker = str(request.args.get("ticker") or "").upper().strip()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,7}", ticker):
        return jsonify({"ok": False, "error": "Invalid ticker"}), 400
    quote = live_quote(ticker)
    if quote is None:
        return jsonify({"ok": False, "error": "Quote unavailable", "ticker": ticker}), 502
    response = jsonify({"ok": True, **quote})
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.get("/api/live-quotes")
def api_live_quotes():
    raw = str(request.args.get("tickers") or "").upper().strip()
    tickers = [x.strip() for x in raw.split(",") if x.strip()]
    if not tickers:
        tickers = ["SPY", "VOO", "QQQ", "VXUS"]
    tickers = list(dict.fromkeys(tickers))
    if len(tickers) > 12 or any(
        not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,7}", ticker) for ticker in tickers
    ):
        return jsonify({"ok": False, "error": "Invalid ticker list"}), 400

    quotes, errors = live_quotes(tickers)
    response = jsonify(
        {
            "ok": bool(quotes),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "quotes": quotes,
            "errors": errors,
            "quote_note":"Multi-provider quote router: authoritative real-time feeds outrank indicative/delayed sources; provider agreement is reported with each quote.",
        }
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response




def _create_tradier_market_session():
    """Create a short-lived browser-safe Tradier streaming session.

    The long-lived access token never leaves the server. The returned session
    identifier can be used by the browser to open Tradier's market WebSocket.
    """
    token=os.getenv("TRADIER_ACCESS_TOKEN")
    if not token:
        return None
    req=urllib.request.Request(
        "https://api.tradier.com/v1/markets/events/session",
        data=b"",
        method="POST",
        headers={
            "Authorization":f"Bearer {token}",
            "Accept":"application/json",
            "User-Agent":"MarketLens-ML/1.0",
        },
    )
    with urllib.request.urlopen(req,timeout=8) as response:
        body=json.loads(response.read().decode("utf-8"))
    stream=body.get("stream") or {}
    sessionid=stream.get("sessionid")
    if not sessionid:
        raise ValueError("Tradier did not return a market streaming session")
    return sessionid


@app.get("/api/market-stream/status")
def market_stream_status():
    configured=bool(os.getenv("TRADIER_ACCESS_TOKEN"))
    return jsonify({
        "configured":configured,
        "provider":"tradier" if configured else "polling-fallback",
        "mode":"websocket" if configured else "near-live-polling",
        "stocks":"real-time with eligible Tradier brokerage data access" if configured else "near-live Yahoo/Finnhub overlay",
        "options":"real-time with eligible Tradier brokerage data access" if configured else "scheduled snapshot",
    })


@app.post("/api/market-stream/session")
def market_stream_session():
    if not os.getenv("TRADIER_ACCESS_TOKEN"):
        return jsonify({
            "ok":False,
            "error":"Streaming provider is not configured",
            "required_env":"TRADIER_ACCESS_TOKEN",
            "fallback":"MarketLens will continue using the near-live polling overlay and scheduled option snapshots.",
        }),503
    if not os.getenv("MARKETLENS_CONTROL_KEY"):
        return jsonify({
            "ok":False,
            "error":"MarketLens control protection is not configured",
            "required_env":"MARKETLENS_CONTROL_KEY",
        }),503
    if not _control_authorized():
        return jsonify({"ok":False,"error":"Control authorization required","requires_control_key":True}),401
    try:
        sessionid=_create_tradier_market_session()
        return jsonify({
            "ok":True,
            "provider":"Tradier",
            "sessionid":sessionid,
            "websocket_url":"wss://ws.tradier.com/v1/markets/events",
            "filters":["quote","trade","summary"],
            "note":"Real-time availability depends on the data entitlements of the configured Tradier brokerage account.",
        })
    except urllib.error.HTTPError as exc:
        return jsonify({"ok":False,"error":f"Tradier streaming session failed ({exc.code})"}),502
    except Exception:
        return jsonify({"ok":False,"error":"Tradier streaming session unavailable"}),502

@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "time": datetime.now(timezone.utc).isoformat(),
            "commit": os.getenv("VERCEL_GIT_COMMIT_SHA"),
            "data_branch": "market-data",
            "live_quote_ttl_seconds": _LIVE_QUOTE_TTL_SECONDS,
            "research_data_ttl_seconds": _RESEARCH_DATA_TTL_SECONDS,
            "live_provider": ("tradier-rest+stream" if os.getenv("TRADIER_ACCESS_TOKEN") else ("finnhub" if os.getenv("FINNHUB_API_KEY") else "yahoo-fallback")),
            "market_stream_configured": bool(os.getenv("TRADIER_ACCESS_TOKEN")),
            "control_key_configured": bool(os.getenv("MARKETLENS_CONTROL_KEY")),
        }
    )


@app.post("/api/run-research")
def run_research():
    body = request.get_json(silent=True) or {}
    ticker = str(body.get("ticker") or "").upper().strip()
    if ticker and not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,7}", ticker):
        return jsonify({"ok": False, "error": "Invalid ticker"}), 400

    token = os.getenv("GITHUB_ACTIONS_TOKEN")
    if not token:
        return jsonify({"ok": False, "error": "Server trigger is not configured"}), 503
    if not os.getenv("MARKETLENS_CONTROL_KEY"):
        return jsonify({
            "ok":False,
            "error":"MarketLens control protection is not configured",
            "required_env":"MARKETLENS_CONTROL_KEY",
        }),503
    if not _control_authorized():
        return jsonify({"ok":False,"error":"Control authorization required","requires_control_key":True}),401

    forwarded = request.headers.get("X-Forwarded-For", "")
    client = (forwarded.split(",")[0].strip() if forwarded else request.remote_addr) or "unknown"
    now = time.monotonic()
    last = _trigger_last_seen.get(client)
    if last is not None and now - last < _TRIGGER_COOLDOWN_SECONDS:
        wait = max(1, int(_TRIGGER_COOLDOWN_SECONDS - (now - last)))
        return jsonify({"ok": False, "error": f"Please wait {wait}s before starting another refresh"}), 429
    try:
        remote_age=_latest_fast_refresh_age_seconds(token)
    except Exception:
        remote_age=None
    if remote_age is not None and remote_age < _TRIGGER_REMOTE_COOLDOWN_SECONDS:
        wait=max(1,int(_TRIGGER_REMOTE_COOLDOWN_SECONDS-remote_age))
        return jsonify({"ok":False,"error":f"A refresh already ran recently. Try again in about {wait}s"}),429
    _trigger_last_seen[client] = now

    payload = {"ref": "main"}
    if ticker:
        payload["inputs"] = {"ticker": ticker}
    workflow = "fast-refresh.yml"
    url = f"https://api.github.com/repos/Wickey23/MarketLens-ML/actions/workflows/{workflow}/dispatches"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "MarketLens-ML",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            return jsonify(
                {"ok": True, "status": "queued", "ticker": ticker or None, "workflow": workflow}
            )
    except urllib.error.HTTPError as exc:
        return jsonify({"ok": False, "error": f"GitHub trigger failed ({exc.code})"}), 502
    except Exception:
        return jsonify({"ok": False, "error": "GitHub trigger unavailable"}), 502

from flask import Flask, jsonify, render_template, request
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

app = Flask(__name__)
DATA = Path(__file__).with_name("data") / "dashboard.json"

_TRIGGER_COOLDOWN_SECONDS = 30
_LIVE_QUOTE_TTL_SECONDS = 10
_trigger_last_seen = {}
_live_quote_cache = {}


def read_data():
    """Read the latest generated research state from the dedicated data branch."""
    url = "https://raw.githubusercontent.com/Wickey23/MarketLens-ML/market-data/data/dashboard.json"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "MarketLens-ML/1.0", "Cache-Control": "no-cache"},
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
            payload["_data_source"] = "github-market-data-live-v3"
            return payload
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
        "realtime": False,
        "delayed": True,
        "note": "Near-live quote provider unavailable; showing the latest research snapshot.",
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
        "realtime":True,
        "delayed":False,
        "note":"Provider-backed quote path. Actual exchange entitlements depend on the configured provider account.",
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
        "realtime": False,
        "delayed": True,
        "note": "Best-effort near-live underlying quote; may be delayed. Options and model evidence remain research snapshots.",
    }


def live_quote(ticker):
    now = time.monotonic()
    cached = _live_quote_cache.get(ticker)
    if cached and now - cached["at"] < _LIVE_QUOTE_TTL_SECONDS:
        return cached["payload"]

    provider_key=os.getenv("FINNHUB_API_KEY")
    if provider_key:
        try:
            out=_finnhub_live_quote(ticker,provider_key)
        except Exception:
            out=None
    else:
        out=None
    if out is None:
        try:
            out=_yahoo_live_quote(ticker)
        except Exception:
            out=_snapshot_quote(ticker)

    if out is not None:
        out["served_at"] = datetime.now(timezone.utc).isoformat()
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
            "quote_note": ("Provider-backed underlying quotes." if os.getenv("FINNHUB_API_KEY") else "Near-live Yahoo Finance underlying quotes. Exchange/broker data may be delayed or differ."),
        }
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "time": datetime.now(timezone.utc).isoformat(),
            "commit": os.getenv("VERCEL_GIT_COMMIT_SHA"),
            "data_branch": "market-data",
            "live_quote_ttl_seconds": _LIVE_QUOTE_TTL_SECONDS,
            "live_provider": "finnhub" if os.getenv("FINNHUB_API_KEY") else "yahoo-fallback",
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

    forwarded = request.headers.get("X-Forwarded-For", "")
    client = (forwarded.split(",")[0].strip() if forwarded else request.remote_addr) or "unknown"
    now = time.monotonic()
    last = _trigger_last_seen.get(client)
    if last is not None and now - last < _TRIGGER_COOLDOWN_SECONDS:
        wait = max(1, int(_TRIGGER_COOLDOWN_SECONDS - (now - last)))
        return jsonify({"ok": False, "error": f"Please wait {wait}s before starting another refresh"}), 429
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

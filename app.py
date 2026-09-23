from flask import Flask, jsonify, render_template, request
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import time
import urllib.error
import urllib.request

app = Flask(__name__)
# deployment marker removed
DATA = Path(__file__).with_name("data") / "dashboard.json"
_TRIGGER_COOLDOWN_SECONDS = 30
_trigger_last_seen = {}
_live_quote_cache = {}
_LIVE_QUOTE_TTL_SECONDS = 8



def read_data():
    # dashboard.json changes much more often than the application deployment.
    # Read the current main-branch snapshot directly so Vercel never serves
    # the copy that happened to be bundled at build time.
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
        # Keep the bundled snapshot only as an outage fallback.
        if DATA.exists():
            payload = json.loads(DATA.read_text(encoding="utf-8"))
            payload["_data_source"] = "bundled-fallback"
            payload["_live_data_error"] = str(exc)
            return payload
        return {"generated_at": datetime.now(timezone.utc).isoformat(), "horizon_days": 5, "tickers": [], "errors": ["Live research snapshot unavailable"], "_live_data_error": str(exc)}


def _snapshot_quote(ticker):
    payload=read_data()
    row=next((x for x in payload.get("tickers",[]) if str(x.get("ticker","")).upper()==ticker),None)
    if not row:
        return None
    return {
        "ticker":ticker,
        "price":row.get("price"),
        "previous_close":None,
        "open":None,
        "high":None,
        "low":None,
        "change":None,
        "change_pct":row.get("change_1d"),
        "market_timestamp":row.get("fast_refreshed_at") or row.get("as_of"),
        "provider":"market-data snapshot",
        "realtime":False,
        "delayed":True,
    }

def _yahoo_live_quote(ticker):
    # Best-effort near-live fallback requiring no additional dependency.
    # Yahoo timestamps/availability can be delayed and rate-limited.
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1m&range=1d&includePrePost=true"
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 MarketLens/1.0","Cache-Control":"no-cache"})
    with urllib.request.urlopen(req,timeout=5) as response:
        body=json.loads(response.read().decode("utf-8"))
    result=((body.get("chart") or {}).get("result") or [None])[0]
    if not result:
        return None
    meta=result.get("meta") or {}
    price=meta.get("regularMarketPrice")
    prev=meta.get("chartPreviousClose") or meta.get("previousClose")
    change=(float(price)-float(prev)) if price is not None and prev not in (None,0) else None
    change_pct=(change/float(prev)) if change is not None and prev not in (None,0) else None
    ts=meta.get("regularMarketTime")
    market_timestamp=datetime.fromtimestamp(ts,timezone.utc).isoformat() if ts else None
    return {
        "ticker":ticker,
        "price":price,
        "previous_close":prev,
        "open":meta.get("regularMarketOpen"),
        "high":meta.get("regularMarketDayHigh"),
        "low":meta.get("regularMarketDayLow"),
        "change":change,
        "change_pct":change_pct,
        "market_timestamp":market_timestamp,
        "exchange":meta.get("exchangeName"),
        "market_state":meta.get("marketState"),
        "provider":"Yahoo Finance chart",
        "realtime":False,
        "delayed":True,
        "note":"Best-effort near-live underlying quote; may be delayed. Options remain on the scheduled research snapshot."
    }

def live_quote(ticker):
    now=time.monotonic()
    cached=_live_quote_cache.get(ticker)
    if cached and now-cached["at"]<_LIVE_QUOTE_TTL_SECONDS:
        return cached["payload"]
    out=None
    try:
        out=_yahoo_live_quote(ticker)
    except Exception:
        out=None
    if out is None:
        out=_snapshot_quote(ticker)
    if out is not None:
        out["served_at"]=datetime.now(timezone.utc).isoformat()
        _live_quote_cache[ticker]={"at":now,"payload":out}
    return out

@app.get("/api/live-quote")
def api_live_quote():
    ticker=str(request.args.get("ticker") or "").upper().strip()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,7}",ticker):
        return jsonify({"ok":False,"error":"Invalid ticker"}),400
    q=live_quote(ticker)
    if q is None:
        return jsonify({"ok":False,"error":"Quote unavailable","ticker":ticker}),502
    response=jsonify({"ok":True,**q})
    response.headers["Cache-Control"]="no-store, no-cache, must-revalidate, max-age=0"
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

@app.get("/api/health")
def health():
    return jsonify({"status":"ok","time":datetime.now(timezone.utc).isoformat(),"commit":os.getenv("VERCEL_GIT_COMMIT_SHA"),"data_branch":"market-data"})

@app.post("/api/run-research")
def run_research():
    body=request.get_json(silent=True) or {}
    ticker=str(body.get("ticker") or "").upper().strip()
    if ticker and not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,7}",ticker):
        return jsonify({"ok":False,"error":"Invalid ticker"}),400

    token=os.getenv("GITHUB_ACTIONS_TOKEN")
    if not token:
        return jsonify({"ok":False,"error":"Server trigger is not configured"}),503

    # This public UI can launch only the lightweight refresh workflow.
    # Heavy research remains scheduled/manual so a public request cannot
    # repeatedly consume the expensive model-training workflow.
    forwarded=request.headers.get("X-Forwarded-For","")
    client=(forwarded.split(",")[0].strip() if forwarded else request.remote_addr) or "unknown"
    now=time.monotonic()
    last=_trigger_last_seen.get(client)
    if last is not None and now-last<_TRIGGER_COOLDOWN_SECONDS:
        wait=max(1,int(_TRIGGER_COOLDOWN_SECONDS-(now-last)))
        return jsonify({"ok":False,"error":f"Please wait {wait}s before starting another refresh"}),429
    _trigger_last_seen[client]=now

    payload={"ref":"main"}
    if ticker:
        payload["inputs"]={"ticker":ticker}
    workflow="fast-refresh.yml"
    url=f"https://api.github.com/repos/Wickey23/MarketLens-ML/actions/workflows/{workflow}/dispatches"
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),method="POST",headers={
        "Authorization":f"Bearer {token}",
        "Accept":"application/vnd.github+json",
        "X-GitHub-Api-Version":"2022-11-28",
        "User-Agent":"MarketLens-ML",
    })
    try:
        with urllib.request.urlopen(req,timeout=10):
            return jsonify({"ok":True,"status":"queued","ticker":ticker or None,"workflow":workflow})
    except urllib.error.HTTPError as e:
        return jsonify({"ok":False,"error":f"GitHub trigger failed ({e.code})"}),502
    except Exception:
        return jsonify({"ok":False,"error":"GitHub trigger unavailable"}),502

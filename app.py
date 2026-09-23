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



def read_data():
    # dashboard.json changes much more often than the application deployment.
    # Read the current main-branch snapshot directly so Vercel never serves
    # the copy that happened to be bundled at build time.
    url = "https://raw.githubusercontent.com/Wickey23/MarketLens-ML/main/data/dashboard.json"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "MarketLens-ML/1.0", "Cache-Control": "no-cache"},
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
            payload["_data_source"] = "github-main-live-v2"
            return payload
    except Exception as exc:
        # Keep the bundled snapshot only as an outage fallback.
        if DATA.exists():
            payload = json.loads(DATA.read_text(encoding="utf-8"))
            payload["_data_source"] = "bundled-fallback"
            payload["_live_data_error"] = str(exc)
            return payload
        return {"generated_at": datetime.now(timezone.utc).isoformat(), "horizon_days": 5, "tickers": [], "errors": ["Live research snapshot unavailable"], "_live_data_error": str(exc)}

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
    return jsonify({"status":"ok","time":datetime.now(timezone.utc).isoformat()})

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

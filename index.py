from flask import Flask, jsonify, render_template_string
from datetime import datetime, timezone

app = Flask(__name__)

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MarketLens ML</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;font-family:Inter,ui-sans-serif,system-ui;background:#080b10;color:#f4f7fb}.wrap{max-width:1180px;margin:auto;padding:28px}.nav{display:flex;justify-content:space-between;align-items:center;margin-bottom:54px}.brand{font-weight:800;font-size:21px;letter-spacing:-.5px}.pill{border:1px solid #263142;border-radius:999px;padding:8px 12px;color:#9fb0c7;font-size:12px}.hero h1{font-size:54px;line-height:1.02;letter-spacing:-2.5px;max-width:800px;margin:0 0 18px}.hero p{color:#91a0b5;font-size:18px;max-width:680px;line-height:1.6}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:42px}.card{background:#0e131b;border:1px solid #1c2635;border-radius:18px;padding:20px;min-height:150px}.ticker{color:#8fa0b7;font-size:13px}.value{font-size:29px;font-weight:750;margin-top:18px}.muted{color:#728198;font-size:12px;margin-top:8px}.status{margin-top:30px;border-top:1px solid #1a2230;padding-top:22px;color:#728198;font-size:13px}@media(max-width:800px){.grid{grid-template-columns:1fr 1fr}.hero h1{font-size:40px}}@media(max-width:480px){.grid{grid-template-columns:1fr}}
</style></head>
<body><main class="wrap"><nav class="nav"><div class="brand">MarketLens <span style="color:#718096">ML</span></div><div class="pill">Research Engine · V1</div></nav>
<section class="hero"><h1>Market intelligence built around probabilities, not predictions.</h1><p>Machine-learning research for trend detection, regime classification and out-of-sample testing. MarketLens separates model signals from demonstrated historical performance.</p></section>
<section class="grid">
<div class="card"><div class="ticker">SPY · 5 DAY</div><div class="value">Pipeline Active</div><div class="muted">Walk-forward model</div></div>
<div class="card"><div class="ticker">MARKET REGIME</div><div class="value">Analyzing</div><div class="muted">Trend + volatility</div></div>
<div class="card"><div class="ticker">VALIDATION</div><div class="value">Walk Forward</div><div class="muted">No random train/test split</div></div>
<div class="card"><div class="ticker">MODELS</div><div class="value">2 Active</div><div class="muted">Logistic · Random Forest</div></div>
</section><div class="status">Live web shell deployed from GitHub main · ML data integration in progress · Research only</div></main></body></html>"""

@app.get("/")
def home():
    return render_template_string(HTML)

@app.get("/api/health")
def health():
    return jsonify({"status":"ok","service":"MarketLens-ML","time":datetime.now(timezone.utc).isoformat()})

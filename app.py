from flask import Flask, jsonify, render_template_string
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import urllib.error
import urllib.request

app = Flask(__name__)
DATA = Path(__file__).with_name("data") / "dashboard.json"

HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MarketLens ML</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root{color-scheme:dark;--bg:#070a0f;--p:#0d121a;--l:#1c2635;--m:#8190a5;--t:#f3f6fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--t);font-family:Inter,system-ui}
.w{max-width:1280px;margin:auto;padding:26px}.nav{display:flex;justify-content:space-between;gap:12px;align-items:center}
.brand{font-size:21px;font-weight:800}.badge,.tab{border:1px solid var(--l);border-radius:999px;padding:8px 12px;color:#a9b5c5;font-size:12px;background:transparent}
.tab{cursor:pointer}.tab.active{background:#e9eef5;color:#090c11}.hero{margin:44px 0 24px}.hero h1{font-size:44px;letter-spacing:-2px;margin:0 0 10px}
.hero p,.sub,.foot{color:var(--m)}.tabs{display:flex;gap:8px;margin:20px 0;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:13px}.card,.box{background:var(--p);border:1px solid var(--l);border-radius:16px;padding:18px}
.label{color:var(--m);font-size:11px;letter-spacing:.08em}.value{font-size:26px;font-weight:760;margin-top:12px}.sub{font-size:12px;margin-top:7px}
.main{display:grid;grid-template-columns:2fr 1fr;gap:13px;margin-top:13px}.chart{height:340px}.table{margin-top:13px;overflow:auto}
table{width:100%;border-collapse:collapse;font-size:13px;min-width:760px}th,td{text-align:left;padding:11px;border-bottom:1px solid var(--l);white-space:nowrap}
th{color:var(--m);font-weight:500}.foot{font-size:11px;line-height:1.6;margin:24px 0}
@media(max-width:850px){.grid{grid-template-columns:1fr 1fr}.main{grid-template-columns:1fr}.nav{align-items:flex-start;flex-direction:column}}
@media(max-width:480px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body><main class="w">
<nav class="nav"><div class="brand">MarketLens <span style="color:#718096">ML</span></div>
<div style="display:flex;gap:8px;flex-wrap:wrap"><button class="tab" id="runBtn" onclick="runResearch()">Run research</button><div class="badge" id="runStatus">Ready</div><div class="badge" id="updated">Loading…</div></div></nav>
<section class="hero"><h1>Probabilistic market and options research.</h1><p>Walk-forward validation, regime detection, historical calibration and contract-level options research.</p></section>
<div class="tabs" id="tabs"></div>
<section class="grid">
<div class="card"><div class="label">LATEST PRICE</div><div class="value" id="price">—</div><div class="sub" id="returns">—</div></div>
<div class="card"><div class="label">5-DAY UP MODEL OUTPUT</div><div class="value" id="prob">—</div><div class="sub">Ensemble output, not a guarantee</div></div>
<div class="card"><div class="label">MARKET REGIME</div><div class="value" id="regime" style="font-size:18px">—</div><div class="sub">Trend + realized volatility</div></div>
<div class="card"><div class="label">20-DAY VOLATILITY</div><div class="value" id="vol">—</div><div class="sub" id="rsi">—</div></div>
</section>
<section class="main"><div class="box chart"><canvas id="chart"></canvas></div><div class="card"><div class="label">MODEL PROBABILITIES</div><div id="models"></div><div class="label" style="margin-top:25px">252-DAY DRAWDOWN</div><div class="value" id="dd">—</div></div></section>
<section class="box" style="margin-top:13px"><div class="label">DECISION EVIDENCE</div><div class="value" id="estate" style="font-size:22px">—</div><div class="sub" id="evidenceNote">—</div></section>
<section class="grid" style="margin-top:13px">
<div class="card"><div class="label">HISTORICAL BASE RATE</div><div class="value" id="base">—</div></div>
<div class="card"><div class="label">SIMILAR SIGNALS</div><div class="value" id="simrate">—</div><div class="sub" id="simn">—</div></div>
<div class="card"><div class="label">SIMILAR SETUP AVG RETURN</div><div class="value" id="simret">—</div></div>
<div class="card"><div class="label">SIGNAL LIFT VS BASE</div><div class="value" id="lift">—</div></div>
</section>
<section class="box table"><div class="label">OPTIONS LAB · UNDERLYING DISTRIBUTION</div><div class="sub" id="optstatus">—</div>
<table><thead><tr><th>Horizon</th><th>Positive rate</th><th>Median return</th><th>10th pct</th><th>90th pct</th><th>1σ move</th></tr></thead><tbody id="optionsRows"></tbody></table></section>
<section class="box table"><div class="label">OPTION CHAIN RESEARCH</div><div class="sub" id="chainNote">—</div>
<table><thead><tr><th>Type</th><th>Expiration</th><th>DTE</th><th>Strike</th><th>Mid</th><th>IV</th><th>Breakeven</th><th>BE move</th><th>OI</th><th>Volume</th><th>Spread</th><th>IV/RV</th><th>BE prob*</th><th>Max debit risk</th><th>Flags</th></tr></thead><tbody id="chainRows"></tbody></table></section>
<section class="box table"><div class="label">OUT-OF-SAMPLE WALK-FORWARD METRICS</div>
<table><thead><tr><th>Model</th><th>Accuracy</th><th>F1</th><th>ROC AUC</th><th>Brier ↓</th><th>N</th></tr></thead><tbody id="metrics"></tbody></table></section>
<div class="foot">Research and education only. Historical results and model outputs do not guarantee future performance. *Breakeven probability is a realized-volatility benchmark, not a calibrated forecast.</div>
</main>
<script>
let P,C,ch;
const pc=x=>x==null?'—':(Number(x)*100).toFixed(1)+'%';
const money=x=>x==null?'—':'$'+Number(x).toLocaleString(undefined,{maximumFractionDigits:2});
async function runResearch(){
  const b=document.getElementById('runBtn'),s=document.getElementById('runStatus');
  b.disabled=true;s.textContent='Starting…';
  try{
    const r=await fetch('/api/run-research',{method:'POST'}),j=await r.json();
    s.textContent=r.ok?'Queued ✓':(j.error||'Failed');
    if(r.ok)setTimeout(()=>s.textContent='Running in GitHub Actions…',1800);
  }catch(e){s.textContent='Failed'}
  finally{setTimeout(()=>b.disabled=false,5000)}
}
async function load(){
  const r=await fetch('/api/data',{cache:'no-store'});P=await r.json();
  updated.textContent='Updated '+new Date(P.generated_at).toLocaleString();
  tabs.innerHTML='';
  (P.tickers||[]).forEach((x,i)=>{const b=document.createElement('button');b.className='tab'+(i?'':' active');b.textContent=x.ticker;b.onclick=()=>sel(x.ticker,b);tabs.appendChild(b)});
  if(P.tickers&&P.tickers.length)sel(P.tickers[0].ticker,tabs.children[0]); else updated.textContent='Waiting for research data';
}
function sel(t,b){
  C=P.tickers.find(x=>x.ticker===t);document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));b.classList.add('active');
  price.textContent=money(C.price);returns.textContent='1D '+pc(C.change_1d)+' · 5D '+pc(C.change_5d);prob.textContent=pc(C.probability_5d_up);
  regime.textContent=C.regime||'—';vol.textContent=pc(C.volatility_20d);rsi.textContent='RSI 14: '+(C.rsi_14==null?'—':Number(C.rsi_14).toFixed(1));dd.textContent=pc(C.drawdown_252);
  const e=C.evidence||{};estate.textContent=e.state||'Awaiting refreshed research';evidenceNote.textContent=(e.validation_quality||'Validation pending')+' · mean AUC '+(e.mean_roc_auc==null?'—':Number(e.mean_roc_auc).toFixed(3));
  base.textContent=pc(C.base_up_rate);const s=C.similar_setups||{};simrate.textContent=pc(s.actual_up_rate);simn.textContent=(s.observations||0)+' comparable observations';simret.textContent=pc(s.mean_forward_return);lift.textContent=(s.actual_up_rate==null||C.base_up_rate==null)?'—':((s.actual_up_rate-C.base_up_rate)*100).toFixed(1)+' pp';
  const o=C.options||{};optstatus.textContent=o.status||'Awaiting options data';
  optionsRows.innerHTML=Object.entries(o.horizons||{}).map(([h,x])=>'<tr><td>'+h+' days</td><td>'+pc(x.positive_rate)+'</td><td>'+pc(x.median_return)+'</td><td>'+pc(x.p10)+'</td><td>'+pc(x.p90)+'</td><td>'+pc(x.realized_move_1sd)+'</td></tr>').join('');
  const oc=o.chain||{};chainNote.textContent=oc.quote_note||oc.error||'Awaiting option-chain refresh';
  chainRows.innerHTML=(oc.contracts||[]).slice(0,40).map(x=>'<tr><td>'+x.type+'</td><td>'+x.expiration+'</td><td>'+x.dte+'</td><td>'+money(x.strike)+'</td><td>'+money(x.mid)+'</td><td>'+pc(x.iv)+'</td><td>'+money(x.breakeven)+'</td><td>'+pc(x.breakeven_move)+'</td><td>'+x.open_interest+'</td><td>'+x.volume+'</td><td>'+pc(x.spread_pct)+'</td><td>'+(x.iv_rv_ratio==null?'—':Number(x.iv_rv_ratio).toFixed(2)+'×')+'</td><td>'+pc(x.historical_vol_prob_breakeven)+'</td><td>'+money(x.max_loss_per_contract)+'</td><td>'+(x.research_flags||[]).join(', ')+'</td></tr>').join('');
  models.innerHTML=Object.entries(C.model_probabilities||{}).map(([k,v])=>'<div style="display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #1c2635"><span>'+k.replace('_',' ')+'</span><b>'+pc(v)+'</b></div>').join('');
  metrics.innerHTML=(C.metrics||[]).map(m=>'<tr><td>'+m.model.replace('_',' ')+'</td><td>'+pc(m.accuracy)+'</td><td>'+pc(m.f1)+'</td><td>'+(m.roc_auc==null?'—':Number(m.roc_auc).toFixed(3))+'</td><td>'+(m.brier==null?'—':Number(m.brier).toFixed(3))+'</td><td>'+m.observations+'</td></tr>').join('');
  draw();
}
function draw(){
  if(ch)ch.destroy();
  ch=new Chart(chart,{type:'line',data:{labels:(C.history||[]).map(x=>x.date),datasets:[{label:C.ticker+' close',data:(C.history||[]).map(x=>x.close),borderWidth:2,pointRadius:0}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8190a5'}}},scales:{x:{ticks:{color:'#657489',maxTicksLimit:8},grid:{display:false}},y:{ticks:{color:'#657489'},grid:{color:'#151d29'}}}}});
}
load();
</script></body></html>"""

def read_data():
    if DATA.exists():
        return json.loads(DATA.read_text(encoding="utf-8"))
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "horizon_days": 5, "tickers": [], "errors": ["Awaiting first research run"]}

@app.get("/")
def home():
    return render_template_string(HTML)

@app.get("/api/data")
def data():
    return jsonify(read_data())

@app.get("/api/health")
def health():
    return jsonify({"status":"ok","time":datetime.now(timezone.utc).isoformat()})

@app.post("/api/run-research")
def run_research():
    token=os.getenv("GITHUB_ACTIONS_TOKEN")
    if not token:
        return jsonify({"ok":False,"error":"Server trigger is not configured"}),503
    url="https://api.github.com/repos/Wickey23/MarketLens-ML/actions/workflows/daily-research.yml/dispatches"
    req=urllib.request.Request(url,data=json.dumps({"ref":"main"}).encode(),method="POST",headers={
        "Authorization":f"Bearer {token}","Accept":"application/vnd.github+json",
        "X-GitHub-Api-Version":"2022-11-28","User-Agent":"MarketLens-ML"})
    try:
        with urllib.request.urlopen(req,timeout=10):
            return jsonify({"ok":True,"status":"queued"})
    except urllib.error.HTTPError as e:
        return jsonify({"ok":False,"error":f"GitHub trigger failed ({e.code})"}),502
    except Exception:
        return jsonify({"ok":False,"error":"GitHub trigger unavailable"}),502

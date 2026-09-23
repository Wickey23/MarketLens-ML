from flask import Flask, jsonify, render_template_string, request
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import urllib.error
import urllib.request

app = Flask(__name__)
# deployment marker removed
DATA = Path(__file__).with_name("data") / "dashboard.json"

HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MarketLens ML</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root{color-scheme:dark;--bg:#070a0f;--p:#0d121a;--p2:#111823;--l:#1c2635;--m:#8190a5;--t:#f3f6fa;--good:#6ee7a8;--warn:#f5c76b;--bad:#ff8b8b;--blue:#8bb7ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--t);font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif}.w{max-width:1380px;margin:auto;padding:24px}
.nav{display:flex;justify-content:space-between;gap:12px;align-items:center;position:sticky;top:0;background:rgba(7,10,15,.94);backdrop-filter:blur(12px);z-index:10;padding:10px 0}
.brand{font-size:21px;font-weight:800}.badge,.btn,.tab,.input,.select{border:1px solid var(--l);border-radius:10px;padding:9px 12px;color:#c0cad8;font-size:12px;background:#0a0f16}.btn,.tab{cursor:pointer}.btn:hover,.tab:hover{border-color:#38506f}.btn.primary,.tab.active{background:#e9eef5;color:#090c11;border-color:#e9eef5}
.search{display:flex;gap:8px;flex-wrap:wrap}.input{min-width:150px;text-transform:uppercase}.hero{margin:34px 0 18px}.hero h1{font-size:42px;letter-spacing:-1.7px;margin:0 0 8px}.hero p,.sub,.foot,.muted{color:var(--m)}
.navtabs,.tickerTabs{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}.view{display:none}.view.active{display:block}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.card,.box{background:var(--p);border:1px solid var(--l);border-radius:15px;padding:17px}.label{color:var(--m);font-size:10px;letter-spacing:.09em}.value{font-size:25px;font-weight:760;margin-top:10px}.sub{font-size:12px;margin-top:6px;line-height:1.5}
.main{display:grid;grid-template-columns:2fr 1fr;gap:12px;margin-top:12px}.chart{height:320px}.table{margin-top:12px;overflow:auto}
table{width:100%;border-collapse:collapse;font-size:12px;min-width:900px}th,td{text-align:left;padding:10px;border-bottom:1px solid var(--l);white-space:nowrap}th{color:var(--m);font-weight:500;position:sticky;top:0;background:var(--p)}
.good{color:var(--good)}.warn{color:var(--warn)}.bad{color:var(--bad)}.blue{color:var(--blue)}.bestrow{background:rgba(110,231,168,.08)}.besttag{display:inline-block;border:1px solid var(--good);color:var(--good);border-radius:999px;padding:4px 7px;font-size:10px;font-weight:800}
.kv{display:flex;justify-content:space-between;gap:12px;padding:9px 0;border-bottom:1px solid var(--l)}.kv:last-child{border-bottom:0}
.explain{line-height:1.55;font-size:13px}.explain li{margin:8px 0}.newsitem{padding:12px 0;border-bottom:1px solid var(--l)}.newsitem a{color:#dbe8ff;text-decoration:none}.newsitem a:hover{text-decoration:underline}
.panel{background:var(--p2);border:1px solid var(--l);border-radius:13px;padding:14px}.contractPanel{display:none;margin-top:12px}.contractPanel.active{display:block}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.range{width:180px}.small{font-size:11px}.pill{display:inline-block;border:1px solid var(--l);border-radius:999px;padding:5px 8px;margin:3px;font-size:11px;color:#aebbd0}
.statline{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;margin-top:12px}.stat{background:#0b1119;border:1px solid var(--l);padding:11px;border-radius:11px}
.right{text-align:right}.foot{font-size:11px;line-height:1.6;margin:22px 0}.empty{padding:26px;color:var(--m);text-align:center}
@media(max-width:950px){.grid{grid-template-columns:1fr 1fr}.grid3,.grid2,.main{grid-template-columns:1fr}.statline{grid-template-columns:1fr 1fr}.nav{align-items:flex-start;flex-direction:column;position:static}.hero h1{font-size:34px}}
@media(max-width:520px){.grid{grid-template-columns:1fr}.w{padding:14px}.statline{grid-template-columns:1fr}}
</style>
</head>
<body><main class="w">
<nav class="nav">
  <div class="brand">MarketLens <span style="color:#718096">ML</span></div>
  <div class="search">
    <input class="input" id="tickerInput" maxlength="8" placeholder="TSLA, NVDA, AAPL">
    <button class="btn primary" onclick="analyzeTicker()">Analyze ticker</button>
    <button class="btn" id="runBtn" onclick="runResearch()">Refresh current</button>
    <span class="badge" id="runStatus">Ready</span>
    <span class="badge" id="updated">Loading…</span>
  </div>
</nav>

<section class="hero"><h1>Options research that explains itself.</h1><p>Evidence, events, Greeks, contract economics and paper trading — separated from the decision you make.</p></section>

<div class="navtabs">
  <button class="tab active" onclick="showView('research',this)">Research</button>
  <button class="tab" onclick="showView('options',this)">Options Lab</button>
  <button class="tab" onclick="showView('sim',this)">Simulator</button>
  <button class="tab" onclick="showView('ai',this)">AI Portfolio</button>
</div>
<div class="tickerTabs" id="tabs"></div>

<section id="researchView" class="view active">
<section class="grid">
<div class="card"><div class="label">LATEST PRICE</div><div class="value" id="price">—</div><div class="sub" id="returns">—</div></div>
<div class="card"><div class="label">5-DAY UP MODEL OUTPUT</div><div class="value" id="prob">—</div><div class="sub">Model output, not a guaranteed probability</div></div>
<div class="card"><div class="label">MARKET REGIME</div><div class="value" id="regime" style="font-size:18px">—</div><div class="sub" id="relative">—</div></div>
<div class="card"><div class="label">20-DAY VOLATILITY</div><div class="value" id="vol">—</div><div class="sub" id="rsi">—</div></div>
</section>
<section class="main"><div class="box chart"><canvas id="chart"></canvas></div><div class="card"><div class="label">MODEL PROBABILITIES</div><div id="models"></div><div class="label" style="margin-top:22px">252-DAY DRAWDOWN</div><div class="value" id="dd">—</div></div></section>
<section class="grid2" style="margin-top:12px">
<div class="box"><div class="label">MARKETLENS EXPLAINS</div><ul class="explain" id="plain"></ul></div>
<div class="box"><div class="label">DECISION EVIDENCE</div><div class="value" id="estate" style="font-size:21px">—</div><div class="sub" id="evidenceNote">—</div><div id="evidenceMore" style="margin-top:12px"></div></div>
</section>
<section class="grid" style="margin-top:12px">
<div class="card"><div class="label">HISTORICAL BASE RATE</div><div class="value" id="base">—</div></div>
<div class="card"><div class="label">SIMILAR SIGNALS</div><div class="value" id="simrate">—</div><div class="sub" id="simn">—</div></div>
<div class="card"><div class="label">SIMILAR SETUP AVG RETURN</div><div class="value" id="simret">—</div></div>
<div class="card"><div class="label">SIGNAL LIFT VS BASE</div><div class="value" id="lift">—</div></div>
</section>
<section class="grid2" style="margin-top:12px">
<div class="box"><div class="label">EARNINGS & CATALYST RISK</div><div class="value" id="earnings" style="font-size:20px">—</div><div class="sub" id="catalysts">—</div></div>
<div class="box"><div class="label">RECENT NEWS CONTEXT</div><div id="news"></div></div>
</section>
<section class="box table"><div class="label">OUT-OF-SAMPLE WALK-FORWARD METRICS</div>
<table><thead><tr><th>Model</th><th>Accuracy</th><th>F1</th><th>ROC AUC</th><th>Brier ↓</th><th>N</th></tr></thead><tbody id="metrics"></tbody></table></section>
</section>

<section id="optionsView" class="view">
<section class="grid3">
<div class="card"><div class="label">REALIZED VOL · 20D</div><div class="value" id="optRV">—</div></div>
<div class="card"><div class="label">NEAREST ATM STRADDLE MOVE</div><div class="value" id="atmMove">—</div><div class="sub" id="atmExp">—</div></div>
<div class="card"><div class="label">EARNINGS</div><div class="value" id="optEarnings" style="font-size:20px">—</div></div>
</section>
<section class="box table"><div class="label">UNDERLYING HISTORICAL DISTRIBUTION</div><div class="sub" id="optstatus">—</div>
<table><thead><tr><th>Horizon</th><th>Positive rate</th><th>Median</th><th>10th pct</th><th>25th pct</th><th>75th pct</th><th>90th pct</th><th>1σ move</th></tr></thead><tbody id="optionsRows"></tbody></table></section>
<section class="box" id="bestOptionBox"><div class="label">FACT-BASED CONTRACT HIGHLIGHT</div><div class="sub">Loading contract screen…</div></section>
<section class="box table">
<div class="controls"><div><div class="label">OPTION CHAIN</div><div class="sub" id="chainNote">—</div></div>
<select id="typeFilter" class="select" onchange="renderChain()"><option value="all">Calls + puts</option><option value="call">Calls</option><option value="put">Puts</option></select>
<select id="expFilter" class="select" onchange="renderChain()"><option value="all">All expirations</option></select></div>
<table><thead><tr><th></th><th>Type</th><th>Exp</th><th>DTE</th><th>Strike</th><th>Bid</th><th>Ask</th><th>Mid</th><th>IV</th><th>Delta</th><th>Gamma</th><th>Theta/day*</th><th>Vega/1pt*</th><th>Breakeven</th><th>BE move</th><th>OI</th><th>Volume</th><th>Spread</th></tr></thead><tbody id="chainRows"></tbody></table>
</section>
<section id="contractPanel" class="box contractPanel">
<div class="controls"><div><div class="label">CONTRACT ANALYZER</div><div class="value" id="contractTitle" style="font-size:21px">—</div></div><select id="paperQty" class="select"><option>1</option><option>2</option><option>3</option><option>5</option><option>10</option></select><select id="paperOrderType" class="select" onchange="toggleLimit()"><option value="market">Market</option><option value="limit">Limit</option></select><input id="paperLimit" class="input" style="display:none;min-width:100px;width:110px" type="number" min="0.01" step="0.01" placeholder="Limit $"><button class="btn primary" onclick="paperTrade()">Review paper order</button></div>
<div id="orderTicket" class="panel" style="display:none;margin-top:12px"></div>
<div class="statline" id="contractStats"></div>
<div class="grid2" style="margin-top:12px"><div class="panel"><div class="label">CONTRACT CHECKLIST</div><div id="contractChecks"></div></div><div class="panel"><div class="label">TRADE MATH AT EXPIRATION</div><div id="tradeMath"></div></div></div>
<div class="grid2" style="margin-top:12px">
<div class="panel"><div class="label">IN PLAIN ENGLISH</div><ul class="explain" id="contractExplain"></ul></div>
<div class="panel"><div class="label">SCENARIO SIMULATOR</div>
<div class="kv"><span>Stock move</span><span id="scMoveLabel">0%</span></div><input class="range" id="scMove" type="range" min="-20" max="20" value="0" step="1" oninput="scenario()">
<div class="kv"><span>Days forward</span><span id="scDaysLabel">0</span></div><input class="range" id="scDays" type="range" min="0" max="30" value="0" step="1" oninput="scenario()">
<div class="kv"><span>IV change</span><span id="scIvLabel">0 pts</span></div><input class="range" id="scIv" type="range" min="-30" max="30" value="0" step="1" oninput="scenario()">
<div class="value" id="scenarioValue" style="font-size:22px">—</div><div class="sub" id="scenarioText">Black-Scholes estimate using current chain IV. Not a guaranteed future quote.</div>
</div></div>
</section>
</section>

<section id="simView" class="view">
<section class="grid">
<div class="card"><div class="label">STARTING PAPER CASH</div><div class="value">$10,000</div></div>
<div class="card"><div class="label">PAPER EQUITY</div><div class="value" id="paperEquity">—</div></div>
<div class="card"><div class="label">UNREALIZED P/L</div><div class="value" id="paperUnreal">—</div></div>
<div class="card"><div class="label">REALIZED P/L</div><div class="value" id="paperReal">—</div></div>
</section><section class="grid" style="margin-top:12px">
<div class="card"><div class="label">BUYING POWER</div><div class="value" id="paperBuying">—</div></div>
<div class="card"><div class="label">OPEN OPTION VALUE</div><div class="value" id="paperOptionValue">—</div></div>
<div class="card"><div class="label">OPEN POSITIONS</div><div class="value" id="paperCount">—</div></div>
<div class="card"><div class="label">PRACTICE MODE</div><div class="value" style="font-size:18px">Long options</div><div class="sub">Simulated fills use quoted ask to buy and bid to sell.</div></div>
</section>
<section class="box table"><div class="controls"><div><div class="label">OPEN PAPER POSITIONS</div><div class="sub">Entry uses ask when available; exit/mark uses bid when available. This intentionally includes the quoted spread.</div></div><button class="btn" onclick="resetPaper()">Reset simulator</button></div>
<table><thead><tr><th>Ticker</th><th>Contract</th><th>Opened</th><th>Entry</th><th>Current exit mark</th><th>P/L</th><th>Underlying</th><th>Underlying move</th><th>Entry context</th><th></th></tr></thead><tbody id="paperOpen"></tbody></table></section>
<section class="box table"><div class="label">PENDING PAPER ORDERS</div><div class="sub">Limit orders fill when the simulated executable quote reaches your limit.</div><table><thead><tr><th>Ticker</th><th>Contract</th><th>Qty</th><th>Limit</th><th>Current ask</th><th>Status</th><th></th></tr></thead><tbody id="paperPending"></tbody></table></section>
<section class="box table"><div class="label">CLOSED PAPER POSITIONS</div>
<table><thead><tr><th>Ticker</th><th>Contract</th><th>Entry</th><th>Exit</th><th>P/L</th><th>Opened</th><th>Closed</th></tr></thead><tbody id="paperClosed"></tbody></table></section>
<section class="box" style="margin-top:12px"><div class="label">HISTORICAL REPLAY · UNDERLYING ONLY</div><div class="sub">Use this to inspect what happened after an earlier date without pretending we have historical option quotes. Exact option replay requires historical chain data from a dedicated provider.</div>
<div class="controls" style="margin-top:10px"><select class="select" id="replayDate" onchange="replay()"></select><select class="select" id="replayH" onchange="replay()"><option value="5">5 days</option><option value="10">10 days</option><option value="20">20 days</option><option value="30">30 days</option></select></div>
<div class="value" id="replayResult" style="font-size:20px">—</div><div class="sub" id="replayText">—</div></section>
</section>

<section id="aiView" class="view">
<section class="grid"><div class="card"><div class="label">AI PAPER EQUITY</div><div class="value" id="aiEquity">—</div><div class="sub">Started with $10,000 simulated capital</div></div><div class="card"><div class="label">TOTAL P/L</div><div class="value" id="aiPnl">—</div></div><div class="card"><div class="label">WIN RATE</div><div class="value" id="aiWin">—</div><div class="sub" id="aiTrades">—</div></div><div class="card"><div class="label">MAX DRAWDOWN</div><div class="value" id="aiDd">—</div></div></section>
<section class="grid2" style="margin-top:12px"><div class="box chart"><div class="label">AI PORTFOLIO EQUITY</div><canvas id="aiChart"></canvas></div><div class="box"><div class="label">OPPORTUNITY INBOX</div><div class="sub">Only setups clearing the research thresholds appear here.</div><div id="aiInbox"></div></div></section>
<section class="box table"><div class="label">OPEN AI PAPER POSITIONS</div><table><thead><tr><th>Ticker</th><th>Contract</th><th>Qty</th><th>Entry</th><th>Mark</th><th>P/L</th><th>Score</th><th>Why</th></tr></thead><tbody id="aiOpen"></tbody></table></section>
<section class="box table"><div class="label">WHAT HAS ACTUALLY WORKED?</div><div class="sub">Forward paper results grouped by facts captured at entry. Small samples are not treated as an edge.</div><div id="aiAttrib"></div></section>\n<section class="box table"><div class="label">AI DECISION LOG</div><table><thead><tr><th>Time</th><th>Ticker</th><th>Contract</th><th>Decision</th><th>Qty</th><th>Price</th><th>Score</th><th>Reason</th></tr></thead><tbody id="aiDecisions"></tbody></table></section>
<section class="box table"><div class="label">CLOSED AI PAPER TRADES</div><table><thead><tr><th>Ticker</th><th>Contract</th><th>Entry</th><th>Exit</th><th>P/L</th><th>Exit reason</th><th>Opened</th><th>Closed</th></tr></thead><tbody id="aiClosed"></tbody></table></section></section>

<div class="foot">Research and education only. MarketLens separates model evidence from the decision you make. Options can lose 100% of premium. Quotes may be delayed. Greeks and scenario values are model estimates; verify live quotes and contract details with your brokerage before acting.</div>
</main>

<script>
let P,C,ch,selectedContract,aiCh;
const pc=x=>x==null?'—':(Number(x)*100).toFixed(1)+'%';
const money=x=>x==null?'—':'$'+Number(x).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
const num=(x,d=2)=>x==null?'—':Number(x).toFixed(d);
const esc=s=>String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]));

function showView(v,b){
  document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.navtabs .tab').forEach(x=>x.classList.remove('active'));
  document.getElementById(v+'View').classList.add('active'); b.classList.add('active');
  if(v==='sim')renderPaper();
  if(v==='ai')renderAI();
}
async function runResearch(ticker=null){
  const b=document.getElementById('runBtn'),s=document.getElementById('runStatus');
  b.disabled=true;s.textContent='Starting…';
  const t=ticker || (C&&C.ticker) || '';
  try{
    const r=await fetch('/api/run-research',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ticker:t})});
    const j=await r.json(); s.textContent=r.ok?('Fast refresh queued '+(t||'core')+' ✓'):(j.error||'Failed');
    if(r.ok){
      setTimeout(()=>s.textContent='Market/options refresh running…',1700);
      const start=P&&P.generated_at;
      let tries=0;
      const poll=setInterval(async()=>{tries++;try{const rr=await fetch('/api/data',{cache:'no-store'}),nn=await rr.json();if(nn.generated_at!==start){clearInterval(poll);P=nn;await reloadData(t||null);s.textContent='Fresh data loaded ✓'}}catch(e){}if(tries>=30){clearInterval(poll);s.textContent='Refresh queued — check shortly'}},10000);
    }
  }catch(e){s.textContent='Failed'}
  finally{setTimeout(()=>b.disabled=false,3500)}
}
function analyzeTicker(){
  const t=tickerInput.value.trim().toUpperCase();
  if(!/^[A-Z][A-Z0-9.\-]{0,7}$/.test(t)){runStatus.textContent='Enter a valid ticker';return}
  const existing=P&&P.tickers&&P.tickers.find(x=>x.ticker===t);
  if(existing){const btn=[...tabs.children].find(x=>x.textContent===t);sel(t,btn);runStatus.textContent=t+' loaded · refreshing…'}
  runResearch(t);
}
async function reloadData(selectTicker=null){
  const r=await fetch('/api/data',{cache:'no-store'});const n=await r.json();
  const old=P&&P.generated_at;P=n;
  updated.textContent='Updated '+new Date(P.generated_at).toLocaleString();tabs.innerHTML='';
  (P.tickers||[]).forEach((x,i)=>{const b=document.createElement('button');b.className='tab'+((selectTicker?x.ticker===selectTicker:i===0)?' active':'');b.textContent=x.ticker;b.onclick=()=>sel(x.ticker,b);tabs.appendChild(b)});
  const target=(selectTicker&&P.tickers.find(x=>x.ticker===selectTicker))?selectTicker:(C&&P.tickers.find(x=>x.ticker===C.ticker)?C.ticker:(P.tickers[0]&&P.tickers[0].ticker));
  if(target){const b=[...tabs.children].find(x=>x.textContent===target);sel(target,b)}
  return old!==P.generated_at;
}
async function load(){
  try{
    const r=await fetch('/api/data',{cache:'no-store'});
    if(!r.ok)throw new Error('Data request '+r.status);
    P=await r.json();
    updated.textContent='Updated '+new Date(P.generated_at).toLocaleString();tabs.innerHTML='';
    (P.tickers||[]).forEach((x,i)=>{const b=document.createElement('button');b.className='tab'+(i?'':' active');b.textContent=x.ticker;b.onclick=()=>sel(x.ticker,b);tabs.appendChild(b)});
    if(P.tickers&&P.tickers.length)sel(P.tickers[0].ticker,tabs.children[0]); else updated.textContent='Waiting for research data';
  }catch(e){
    updated.textContent='Data unavailable';
    runStatus.textContent='Retrying data…';
    setTimeout(load,5000);
  }
}
function sel(t,b){
  C=P.tickers.find(x=>x.ticker===t);selectedContract=null;contractPanel.classList.remove('active');
  document.querySelectorAll('.tickerTabs .tab').forEach(x=>x.classList.remove('active'));if(b)b.classList.add('active');
  tickerInput.value=t;renderResearch();renderOptions();renderPaper();renderReplay();renderAI();
}
function renderResearch(){
  price.textContent=money(C.price);returns.textContent='1D '+pc(C.change_1d)+' · 5D '+pc(C.change_5d);prob.textContent=pc(C.probability_5d_up);
  regime.textContent=C.regime||'—';vol.textContent=pc(C.volatility_20d);rsi.textContent='RSI 14: '+(C.rsi_14==null?'—':num(C.rsi_14,1));dd.textContent=pc(C.drawdown_252);
  const rs=C.relative_strength||{}; relative.textContent=C.ticker==='SPY'?'Benchmark ticker':'20D vs SPY '+(rs.vs_spy_20d==null?'—':((rs.vs_spy_20d>=0?'+':'')+(rs.vs_spy_20d*100).toFixed(1)+' pp'));
  models.innerHTML=Object.entries(C.model_probabilities||{}).map(([k,v])=>'<div class="kv"><span>'+esc(k.replaceAll('_',' '))+'</span><b>'+pc(v)+'</b></div>').join('');
  const e=C.evidence||{},s=C.similar_setups||{};estate.textContent=e.state||'Awaiting evidence';evidenceNote.textContent=(e.validation_quality||'Validation pending')+' · mean AUC '+(e.mean_roc_auc==null?'—':num(e.mean_roc_auc,3));
  evidenceMore.innerHTML='<div class="kv"><span>Model agreement</span><b>'+pc(e.model_agreement)+'</b></div><div class="kv"><span>Comparable sample</span><b>'+(e.similar_sample_size||0)+'</b></div>';
  base.textContent=pc(C.base_up_rate);simrate.textContent=pc(s.actual_up_rate);simn.textContent=(s.observations||0)+' comparable observations';simret.textContent=pc(s.mean_forward_return);lift.textContent=(s.actual_up_rate==null||C.base_up_rate==null)?'—':((s.actual_up_rate-C.base_up_rate)*100).toFixed(1)+' pp';
  plain.innerHTML=(C.plain_language||['Awaiting refreshed research.']).map(x=>'<li>'+esc(x)+'</li>').join('');
  const ctx=C.company_context||{},er=ctx.earnings||{};earnings.textContent=er.days_to_earnings==null?'No upcoming date available':(er.days_to_earnings+' days to earnings');catalysts.textContent=(ctx.catalyst_flags||[]).length?'Headline themes: '+ctx.catalyst_flags.join(', '):(ctx.news_note||'No catalyst themes detected in the current headline set.');
  news.innerHTML=(ctx.news||[]).slice(0,5).map(n=>'<div class="newsitem">'+(n.url?'<a target="_blank" rel="noopener" href="'+esc(n.url)+'">'+esc(n.title)+'</a>':'<span>'+esc(n.title)+'</span>')+'<div class="sub">'+esc(n.publisher||'')+(n.published_at?' · '+esc(String(n.published_at).slice(0,10)):'')+'</div></div>').join('')||'<div class="empty">No recent headlines in this research snapshot.</div>';
  metrics.innerHTML=(C.metrics||[]).map(m=>'<tr><td>'+esc(m.model.replaceAll('_',' '))+'</td><td>'+pc(m.accuracy)+'</td><td>'+pc(m.f1)+'</td><td>'+num(m.roc_auc,3)+'</td><td>'+num(m.brier,3)+'</td><td>'+m.observations+'</td></tr>').join('');
  draw();
}
function renderOptions(){
  const o=C.options||{},sum=o.summary||{},ctx=C.company_context||{},er=ctx.earnings||{};
  optRV.textContent=pc(o.realized_vol_20d);atmMove.textContent=pc(sum.atm_straddle_implied_move);atmExp.textContent=sum.atm_expiration?('Nearest reference expiration '+sum.atm_expiration):'Awaiting chain';optEarnings.textContent=er.days_to_earnings==null?'—':er.days_to_earnings+' days';
  optstatus.textContent=o.status||'Awaiting options data';
  optionsRows.innerHTML=Object.entries(o.horizons||{}).map(([h,x])=>'<tr><td>'+h+' days</td><td>'+pc(x.positive_rate)+'</td><td>'+pc(x.median_return)+'</td><td>'+pc(x.p10)+'</td><td>'+pc(x.p25)+'</td><td>'+pc(x.p75)+'</td><td>'+pc(x.p90)+'</td><td>'+pc(x.realized_move_1sd)+'</td></tr>').join('');
  const oc=o.chain||{};chainNote.textContent=oc.quote_note||oc.error||'Awaiting option-chain refresh';
  expFilter.innerHTML='<option value="all">All expirations</option>'+(oc.expirations||[]).map(x=>'<option value="'+esc(x)+'">'+esc(x)+'</option>').join('');
  renderChain();
}
function contractScore(x){
 if(!x||!x.mid||x.mid<=0||x.dte==null)return null;
 const spread=x.spread_pct==null?1:Number(x.spread_pct),oi=Number(x.open_interest||0),v=Number(x.volume||0),theta=Math.abs(Number(x.theta_cost_pct_per_day||0)),be=Math.abs(Number(x.breakeven_move||0)),ivr=Number(x.iv_rv_ratio||1),dte=Number(x.dte||0);
 const liquidity=Math.min(25,(oi>=1000?14:oi>=250?9:4)+(v>=1000?11:v>=100?7:2)),execution=spread<=.02?20:spread<=.05?15:spread<=.10?9:spread<=.15?4:0,decay=dte===0?0:theta<=.03?20:theta<=.06?14:theta<=.10?8:theta<=.15?3:0,volatility=ivr<=.9?15:ivr<=1.05?11:ivr<=1.2?7:3,breakeven=be<=.005?12:be<=.01?9:be<=.02?5:2,time=dte>=5&&dte<=45?8:dte>=2?5:0;
 return {score:liquidity+execution+decay+volatility+breakeven+time,parts:['liquidity '+liquidity+'/25','execution '+execution+'/20','decay '+decay+'/20','IV/RV '+volatility+'/15','breakeven '+breakeven+'/12','time '+time+'/8']};
}
function renderChain(){
 const oc=(C.options||{}).chain||{},typ=typeFilter.value,exp=expFilter.value;let rows=(oc.contracts||[]).filter(x=>(typ==='all'||x.type===typ)&&(exp==='all'||x.expiration===exp));
 const ranked=rows.map(x=>({x,r:contractScore(x)})).filter(z=>z.r).sort((a,b)=>b.r.score-a.r.score),best=ranked[0];
 if(best){const x=best.x,r=best.r;bestOptionBox.innerHTML='<div class="label">FACT-BASED CONTRACT HIGHLIGHT</div><div class="value" style="font-size:20px">'+esc(C.ticker)+' '+esc(x.expiration)+' '+money(x.strike)+' '+esc(x.type.toUpperCase())+' <span class="besttag">TOP SCREEN '+r.score+'/100</span></div><div class="sub">Ranks contract mechanics only: liquidity, spread, modeled decay, IV versus realized volatility, breakeven distance and time to expiration. It does not decide market direction or tell you to buy. '+esc(r.parts.join(' · '))+'</div><button class="btn" style="margin-top:9px" onclick="selectContractFromButton(this)">Analyze highlighted contract</button>';}else bestOptionBox.innerHTML='<div class="label">FACT-BASED CONTRACT HIGHLIGHT</div><div class="sub">No contract currently passes the screen.</div>';
 chainRows.innerHTML=rows.slice(0,120).map(x=>{const isBest=best&&x===best.x;return '<tr class="'+(isBest?'bestrow':'')+'"><td>'+(isBest?'<span class="besttag">TOP SCREEN</span> ':'')+'<button class="btn" onclick="selectContractFromButton(this)">Analyze</button></td><td>'+x.type+'</td><td>'+x.expiration+'</td><td>'+x.dte+'</td><td>'+money(x.strike)+'</td><td>'+money(x.bid)+'</td><td>'+money(x.ask)+'</td><td>'+money(x.mid)+'</td><td>'+pc(x.iv)+'</td><td>'+num(x.delta,3)+'</td><td>'+num(x.gamma,4)+'</td><td>'+money(x.theta_per_contract_per_day)+'</td><td>'+money(x.vega_per_contract_per_vol_point)+'</td><td>'+money(x.breakeven)+'</td><td>'+pc(x.breakeven_move)+'</td><td>'+x.open_interest+'</td><td>'+x.volume+'</td><td>'+pc(x.spread_pct)+'</td></tr>'}).join('');
}
function selectContractFromButton(btn){const tr=btn.closest("tr");if(!tr)return;const cells=tr.children;selectContract("",Number(String(cells[4].textContent).replace(/[$,]/g,"")),cells[1].textContent.trim(),cells[2].textContent.trim())}\nfunction selectContract(sym,strike,type,exp){
  const rows=(((C.options||{}).chain||{}).contracts||[]);
  selectedContract=rows.find(x=>(sym&&x.contract_symbol===sym)||(!sym&&x.strike===strike&&x.type===type&&x.expiration===exp));
  if(!selectedContract)return;contractPanel.classList.add('active');
  const x=selectedContract;contractTitle.textContent=C.ticker+' '+x.expiration+' '+money(x.strike)+' '+x.type.toUpperCase();
  contractStats.innerHTML=[
    ['Premium mid',money(x.mid)],['Breakeven',money(x.breakeven)],['IV',pc(x.iv)],['Delta',num(x.delta,3)],['Theta/day',money(x.theta_per_contract_per_day)],
    ['Vega/1pt',money(x.vega_per_contract_per_vol_point)],['Spread',pc(x.spread_pct)],['Open interest',x.open_interest],['Max debit',money(x.max_loss_per_contract)],['BE benchmark',pc(x.historical_vol_prob_breakeven)]
  ].map(([a,b])=>'<div class="stat"><div class="label">'+a+'</div><div style="margin-top:7px;font-weight:700">'+b+'</div></div>').join('');
  contractExplain.innerHTML=contractExplanation(x).map(v=>'<li>'+esc(v)+'</li>').join('');
  const checks=x.quality_checks||[];
  contractChecks.innerHTML=checks.map(q=>'<div class="kv"><span>'+esc(q.name)+'</span><span><b>'+esc(q.state)+'</b><div class="sub">'+esc(q.detail)+'</div></span></div>').join('')||'<div class="sub">Awaiting refreshed contract diagnostics.</div>';
  const debit=(x.ask&&x.ask>0?x.ask:x.mid)||0, be=x.breakeven;
  tradeMath.innerHTML='<div class="kv"><span>1-contract debit</span><b>'+money(debit*100)+'</b></div>'+
    '<div class="kv"><span>Maximum loss (long option)</span><b>'+money(debit*100)+'</b></div>'+
    '<div class="kv"><span>Expiration breakeven</span><b>'+money(be)+'</b></div>'+
    '<div class="kv"><span>Move needed to breakeven</span><b>'+pc(Math.abs(x.breakeven_move||0))+'</b></div>'+
    '<div class="kv"><span>Modeled theta / day</span><b>'+money(x.theta_per_contract_per_day)+'</b></div>'+
    '<div class="kv"><span>Theta as % of debit / day</span><b>'+pc(x.theta_cost_pct_per_day)+'</b></div>';
  scMove.value=0;scDays.value=0;scIv.value=0;scenario();
  contractPanel.scrollIntoView({behavior:'smooth',block:'start'});
}
function contractExplanation(x){
  const out=[],delta=(x.delta||0)*100,theta=x.theta_per_contract_per_day,vega=x.vega_per_contract_per_vol_point;
  out.push('At the current model estimate, a $1 move in '+C.ticker+' changes this contract by roughly $'+Math.abs(delta).toFixed(0)+' initially from delta, all else equal. Direction depends on whether it is a call or put.');
  if(theta!=null)out.push('Time decay is currently about $'+Math.abs(theta).toFixed(2)+' per contract per day, holding price and volatility constant. Theta generally changes as expiration approaches.');
  if(vega!=null)out.push('A 1 percentage-point change in implied volatility changes the modeled contract value by about $'+Math.abs(vega).toFixed(2)+' per contract initially.');
  if(x.breakeven_move!=null)out.push('At expiration, the underlying must move to about '+money(x.breakeven)+' for this long option to break even, roughly '+pc(Math.abs(x.breakeven_move))+' from the current stock price.');
  if(x.iv_rv_ratio!=null)out.push('Current implied volatility is about '+Number(x.iv_rv_ratio).toFixed(2)+'× the recent realized-volatility estimate. This is a pricing comparison, not proof that the option is cheap or expensive.');
  const er=((C.company_context||{}).earnings||{}).days_to_earnings;if(er!=null&&er<=x.dte)out.push('This contract spans the next scheduled earnings date, so event risk and a post-event IV change can materially affect the option.');
  if(x.spread_pct!=null&&x.spread_pct>.15)out.push('The quoted bid/ask spread is wide relative to premium, which can make entry and exit materially more expensive.');
  return out;
}
function normcdf(x){const a1=.254829592,a2=-.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=.3275911;const s=x<0?-1:1;const z=Math.abs(x)/Math.sqrt(2);const t=1/(1+p*z);const erf=1-(((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*Math.exp(-z*z);return .5*(1+s*erf)}
function bs(side,S,K,T,sigma,r){
 if(T<=0)return Math.max(side==='call'?S-K:K-S,0);sigma=Math.max(sigma,.0001);const d1=(Math.log(S/K)+(r+.5*sigma*sigma)*T)/(sigma*Math.sqrt(T)),d2=d1-sigma*Math.sqrt(T);return side==='call'?S*normcdf(d1)-K*Math.exp(-r*T)*normcdf(d2):K*Math.exp(-r*T)*normcdf(-d2)-S*normcdf(-d1)
}
function scenario(){
 if(!selectedContract)return;const x=selectedContract,m=Number(scMove.value),d=Number(scDays.value),ivc=Number(scIv.value);scMoveLabel.textContent=(m>=0?'+':'')+m+'%';scDaysLabel.textContent=d;scIvLabel.textContent=(ivc>=0?'+':'')+ivc+' pts';
 const S=C.price*(1+m/100),days=Math.max(x.dte-d,0),iv=Math.max((x.iv||.01)+ivc/100,.001),r=((((C.options||{}).chain||{}).risk_free_rate)||.04),v=bs(x.type,S,x.strike,days/365,iv,r),entry=x.ask>0?x.ask:x.mid,pnl=(v-entry)*100;
 scenarioValue.textContent='Estimated value '+money(v)+' · P/L '+(pnl>=0?'+':'')+money(pnl);
 scenarioValue.className='value '+(pnl>=0?'good':'bad');scenarioText.textContent='Underlying '+money(S)+' · '+days+' DTE · IV '+pc(iv)+'. Black-Scholes scenario estimate; not a guaranteed quote.';
}
function paperState(){try{const s=JSON.parse(localStorage.getItem('marketlens_paper_v1'))||{};return{open:s.open||[],closed:s.closed||[],pending:s.pending||[],equityHistory:s.equityHistory||[]}}catch(e){return{open:[],closed:[],pending:[],equityHistory:[]}}}
function toggleLimit(){paperLimit.style.display=paperOrderType.value==='limit'?'block':'none';if(paperOrderType.value==='limit'&&selectedContract)paperLimit.value=num(selectedContract.mid,2)}
function savePaper(s){localStorage.setItem('marketlens_paper_v1',JSON.stringify(s))}
function paperTrade(){
 if(!selectedContract)return;const x=selectedContract,s=paperState(),entry=(x.ask&&x.ask>0)?x.ask:x.mid;if(!entry){runStatus.textContent='No usable entry quote';return}
 s.open.push({id:Date.now(),ticker:C.ticker,type:x.type,strike:x.strike,expiration:x.expiration,contract_symbol:x.contract_symbol||'',qty:1,entry_price:entry,entry_spot:C.price,entry_iv:x.iv,entry_delta:x.delta,entry_theta:x.theta_per_contract_per_day,entry_vega:x.vega_per_contract_per_vol_point,entry_spread_pct:x.spread_pct,entry_model_output:C.probability_5d_up,entry_evidence:(C.evidence||{}).state||null,opened_at:new Date().toISOString()});savePaper(s);runStatus.textContent='Paper trade added';renderPaper();
}
function processPending(s){
 s.pending.forEach(o=>{const q=lookupContract(o);if(q&&q.ask!=null&&q.ask<=o.limit_price)o._fill=Math.min(q.ask,o.limit_price)});
 const fills=s.pending.filter(o=>o._fill!=null);s.pending=s.pending.filter(o=>o._fill==null);
 fills.forEach(o=>{const {limit_price,created_at,status,_fill,...p}=o;s.open.push({...p,entry_price:_fill,opened_at:new Date().toISOString()})});return fills.length;
}
function lookupContract(p){const t=(P.tickers||[]).find(x=>x.ticker===p.ticker);if(!t)return null;return (((t.options||{}).chain||{}).contracts||[]).find(z=>(p.contract_symbol&&z.contract_symbol===p.contract_symbol)||(!p.contract_symbol&&z.type===p.type&&z.strike===p.strike&&z.expiration===p.expiration))||null}
function cancelPending(id){const s=paperState();s.pending=s.pending.filter(x=>x.id!==id);savePaper(s);renderPaper()}
function lookupMark(p){
 const t=(P.tickers||[]).find(x=>x.ticker===p.ticker);if(!t)return{mark:null,spot:null};
 const rows=((((t.options||{}).chain||{}).contracts)||[]);const x=rows.find(z=>(p.contract_symbol&&z.contract_symbol===p.contract_symbol)||(!p.contract_symbol&&z.type===p.type&&z.strike===p.strike&&z.expiration===p.expiration));
 if(x)return{mark:(x.bid&&x.bid>0)?x.bid:x.mid,spot:t.price};
 const expired=new Date(p.expiration+'T23:59:59Z')<new Date();if(expired&&t.price!=null){const intrinsic=Math.max(p.type==='call'?t.price-p.strike:p.strike-t.price,0);return{mark:intrinsic,spot:t.price}}
 return{mark:null,spot:t.price};
}
function renderPaper(){
 const s=paperState();const filled=processPending(s);if(filled)savePaper(s);let unreal=0,real=0,cash=10000;paperOpen.innerHTML='';paperClosed.innerHTML='';paperPending.innerHTML='';
 s.open.forEach(p=>{const q=lookupMark(p),cost=p.entry_price*100*p.qty;cash-=cost;const val=q.mark==null?null:q.mark*100*p.qty,pl=val==null?null:val-cost;if(pl!=null)unreal+=pl;paperOpen.innerHTML+='<tr><td>'+p.ticker+'</td><td>'+p.expiration+' '+money(p.strike)+' '+p.type+'</td><td>'+new Date(p.opened_at).toLocaleDateString()+'</td><td>'+money(p.entry_price)+'</td><td>'+money(q.mark)+'</td><td class="'+(pl==null?'':pl>=0?'good':'bad')+'">'+(pl==null?'—':((pl>=0?'+':'')+money(pl)))+'</td><td>'+money(q.spot)+'</td><td>'+((q.spot==null||p.entry_spot==null)?'—':pc(q.spot/p.entry_spot-1))+'</td><td>'+(p.entry_evidence?esc(p.entry_evidence):'—')+'</td><td><button class="btn" onclick="closePaper('+p.id+')">Close</button></td></tr>'});
 s.pending.forEach(p=>{const q=lookupContract(p);paperPending.innerHTML+='<tr><td>'+p.ticker+'</td><td>'+p.expiration+' '+money(p.strike)+' '+p.type+'</td><td>'+p.qty+'</td><td>'+money(p.limit_price)+'</td><td>'+money(q&&q.ask)+'</td><td class="warn">Pending</td><td><button class="btn" onclick="cancelPending('+p.id+')">Cancel</button></td></tr>'});if(!s.pending.length)paperPending.innerHTML='<tr><td colspan="7" class="empty">No pending orders.</td></tr>';
 s.closed.forEach(p=>{real+=p.pnl;cash+=p.pnl;paperClosed.innerHTML+='<tr><td>'+p.ticker+'</td><td>'+p.expiration+' '+money(p.strike)+' '+p.type+'</td><td>'+money(p.entry_price)+'</td><td>'+money(p.exit_price)+'</td><td class="'+(p.pnl>=0?'good':'bad')+'">'+(p.pnl>=0?'+':'')+money(p.pnl)+'</td><td>'+new Date(p.opened_at).toLocaleDateString()+'</td><td>'+new Date(p.closed_at).toLocaleDateString()+'</td></tr>'});
 const openVal=s.open.reduce((a,p)=>{const q=lookupMark(p);return a+(q.mark==null?0:q.mark*100*p.qty)},0),equity=cash+openVal;
 paperEquity.textContent=money(equity);paperBuying.textContent=money(cash);paperOptionValue.textContent=money(openVal);paperCount.textContent=s.open.length;paperUnreal.textContent=(unreal>=0?'+':'')+money(unreal);paperReal.textContent=(real>=0?'+':'')+money(real);paperUnreal.className='value '+(unreal>=0?'good':'bad');paperReal.className='value '+(real>=0?'good':'bad');
 if(!s.open.length)paperOpen.innerHTML='<tr><td colspan="10" class="empty">No open paper positions. Analyze a contract and choose Paper trade.</td></tr>';
 if(!s.closed.length)paperClosed.innerHTML='<tr><td colspan="7" class="empty">No closed paper positions yet.</td></tr>';
 s.equityHistory.push({t:new Date().toISOString(),equity});if(s.equityHistory.length>300)s.equityHistory=s.equityHistory.slice(-300);savePaper(s);
}
function closePaper(id){
 const s=paperState(),i=s.open.findIndex(x=>x.id===id);if(i<0)return;const p=s.open[i],q=lookupMark(p);if(q.mark==null){runStatus.textContent='No current exit quote';return}
 p.exit_price=q.mark;p.closed_at=new Date().toISOString();p.pnl=(p.exit_price-p.entry_price)*100*p.qty;s.open.splice(i,1);s.closed.push(p);savePaper(s);renderPaper();
}
function resetPaper(){if(confirm('Reset all paper-trading history in this browser?')){localStorage.removeItem('marketlens_paper_v1');renderPaper()}}
function renderReplay(){
 const h=C.history||[];replayDate.innerHTML=h.slice(0,-30).reverse().map(x=>'<option value="'+x.date+'">'+x.date+'</option>').join('');replay();
}
function replay(){
 if(!C||!replayDate.value)return;const h=C.history||[],i=h.findIndex(x=>x.date===replayDate.value),n=Number(replayH.value);if(i<0||i+n>=h.length){replayResult.textContent='Not enough subsequent data';replayText.textContent='Choose an earlier date.';return}
 const a=h[i].close,b=h[i+n].close,r=b/a-1;replayResult.textContent=C.ticker+' '+(r>=0?'+':'')+pc(r)+' over '+n+' trading days';replayResult.className='value '+(r>=0?'good':'bad');replayText.textContent='Underlying moved from '+money(a)+' to '+money(b)+'. This does not reconstruct historical option prices.';
}
function renderAI(){
 const a=(P&&P.ai_portfolio)||{},s=a.summary||{},hist=a.equity_history||[];
 aiEquity.textContent=money(s.equity);aiPnl.textContent=(Number(s.total_pnl||0)>=0?'+':'')+money(s.total_pnl||0);aiPnl.className='value '+(Number(s.total_pnl||0)>=0?'good':'bad');
 aiWin.textContent=s.win_rate==null?'—':pc(s.win_rate);aiTrades.textContent=(s.closed_trades||0)+' closed · '+(s.open_positions||0)+' open';aiDd.textContent=pc(s.max_drawdown);
 aiOpen.innerHTML=(a.open||[]).map(p=>'<tr><td>'+esc(p.ticker)+'</td><td>'+esc(p.expiration)+' '+money(p.strike)+' '+esc(p.type)+'</td><td>'+p.qty+'</td><td>'+money(p.entry_price)+'</td><td>'+money(p.last_mark)+'</td><td class="'+((p.unrealized_pnl||0)>=0?'good':'bad')+'">'+money(p.unrealized_pnl)+'</td><td>'+num(p.entry_score,1)+'</td><td>'+esc((p.entry_reasons||[]).join(' · '))+'</td></tr>').join('')||'<tr><td colspan="8" class="empty">No open AI paper positions.</td></tr>';
 aiClosed.innerHTML=(a.closed||[]).slice().reverse().map(p=>'<tr><td>'+esc(p.ticker)+'</td><td>'+esc(p.expiration)+' '+money(p.strike)+' '+esc(p.type)+'</td><td>'+money(p.entry_price)+'</td><td>'+money(p.exit_price)+'</td><td class="'+((p.pnl||0)>=0?'good':'bad')+'">'+money(p.pnl)+'</td><td>'+esc(p.exit_reason||'—')+'</td><td>'+esc(String(p.opened_at||'').slice(0,10))+'</td><td>'+esc(String(p.closed_at||'').slice(0,10))+'</td></tr>').join('')||'<tr><td colspan="8" class="empty">No closed AI paper trades yet.</td></tr>';
 const at=a.attribution||{}, sections=[['TYPE',at.by_type],['REGIME',at.by_regime],['DTE',at.by_dte],['SCORE',at.by_score],['HISTORICAL PROBABILITY',at.by_historical_probability],['IV ENVIRONMENT',at.by_iv_environment]];
 aiAttrib.innerHTML=sections.map(z=>'<div class="panel" style="margin-top:10px"><div class="label">'+z[0]+'</div><table><thead><tr><th>Group</th><th>Trades</th><th>Win rate</th><th>Total P/L</th><th>Avg P/L</th><th>Return on premium</th></tr></thead><tbody>'+((z[1]||[]).map(g=>'<tr><td>'+esc(g.group)+'</td><td>'+g.trades+'</td><td>'+pc(g.win_rate)+'</td><td class="'+((g.total_pnl||0)>=0?'good':'bad')+'">'+money(g.total_pnl)+'</td><td>'+money(g.avg_pnl)+'</td><td>'+pc(g.return_on_premium)+'</td></tr>').join('')||'<tr><td colspan="6" class="empty">Not enough closed trades yet.</td></tr>')+'</tbody></table></div>').join('')+'<div class="sub">'+esc(at.note||'Attribution begins after trades close.')+'</div>';
 aiDecisions.innerHTML=(a.decisions||[]).slice().reverse().map(d=>'<tr><td>'+esc(String(d.at||'').replace('T',' ').slice(0,16))+'</td><td>'+esc(d.ticker||'—')+'</td><td>'+esc(d.contract||'—')+'</td><td>'+esc(d.action||'—')+'</td><td>'+esc(d.qty||'—')+'</td><td>'+money(d.price)+'</td><td>'+num(d.score,1)+'</td><td>'+esc(d.reason||'—')+'</td></tr>').join('')||'<tr><td colspan="8" class="empty">No AI decisions recorded yet.</td></tr>';
 let opp=[];(P.tickers||[]).forEach(t=>{const r=((t.options||{}).opportunity_radar)||{};(r.opportunities||[]).forEach(x=>opp.push(Object.assign({ticker:t.ticker},x)))});opp.sort((x,y)=>(y.score||0)-(x.score||0));
 aiInbox.innerHTML=opp.slice(0,8).map(x=>'<div class="newsitem"><b>'+esc(x.ticker)+' '+esc(x.expiration)+' '+money(x.strike)+' '+esc(String(x.type||'').toUpperCase())+'</b> <span class="besttag">'+num(x.score,1)+'/100</span><div class="sub">Historical profitable replay '+pc(x.prob_profit)+' · mean P/L '+money(x.expected_pnl_per_contract)+' · '+esc(x.historical_scope||'')+'</div><div class="sub">'+esc((x.reasons||[]).join(' · '))+'</div></div>').join('')||'<div class="empty">No strong setup currently clears the Opportunity Radar thresholds.</div>';
 if(aiCh)aiCh.destroy();if(hist.length)aiCh=new Chart(document.getElementById('aiChart'),{type:'line',data:{labels:hist.map(x=>String(x.at||'').slice(5,16)),datasets:[{label:'AI paper equity',data:hist.map(x=>x.equity),borderWidth:2,pointRadius:0}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8190a5'}}},scales:{x:{ticks:{color:'#657489',maxTicksLimit:7},grid:{display:false}},y:{ticks:{color:'#657489'},grid:{color:'#151d29'}}}}});
}

function draw(){if(ch)ch.destroy();ch=new Chart(chart,{type:'line',data:{labels:(C.history||[]).map(x=>x.date),datasets:[{label:C.ticker+' close',data:(C.history||[]).map(x=>x.close),borderWidth:2,pointRadius:0}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8190a5'}}},scales:{x:{ticks:{color:'#657489',maxTicksLimit:8},grid:{display:false}},y:{ticks:{color:'#657489'},grid:{color:'#151d29'}}}}})}
load();
</script>
</body></html>"""

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
    return render_template_string(HTML)

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
    token=os.getenv("GITHUB_ACTIONS_TOKEN")
    if not token:
        return jsonify({"ok":False,"error":"Server trigger is not configured"}),503
    body=request.get_json(silent=True) or {}
    ticker=str(body.get("ticker") or "").upper().strip()
    if ticker and not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,7}",ticker):
        return jsonify({"ok":False,"error":"Invalid ticker"}),400
    payload={"ref":"main"}
    if ticker:
        payload["inputs"]={"ticker":ticker}
    workflow="fast-refresh.yml" if str(body.get("mode") or "fast")!="full" else "daily-research.yml"
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

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score,brier_score_loss,f1_score,roc_auc_score
from src.data_loader import download_prices
from src.features import add_target,build_features
from src.regime import classify_regime
from src.train import FEATURES
from src.walk_forward import expanding_predictions,model_library
TICKERS=["SPY","VOO","QQQ","VXUS"]; HORIZON=5
def sf(x):
    try:
        v=float(x); return v if np.isfinite(v) else None
    except: return None
def bucket_stats(p,y,r):
    out=[]
    for lo,hi in [(0,.45),(.45,.5),(.5,.55),(.55,.6),(.6,.65),(.65,.7),(.7,1.01)]:
        m=(p>=lo)&(p<hi); n=int(m.sum())
        if n: out.append({"bucket":f"{int(lo*100)}-{int(min(hi,1)*100)}%","n":n,"actual_up_rate":sf(y[m].mean()),"mean_forward_return":sf(r[m].mean())})
    return out
def analyze(ticker):
    raw=download_prices(ticker,"2010-01-01"); feat=build_features(raw); lab=add_target(feat,HORIZON).dropna(subset=FEATURES+["target"])
    latest=feat.dropna(subset=FEATURES).iloc[-1]; probs=[]; mp={}; metrics=[]
    for name,model in model_library().items():
        p=expanding_predictions(lab,FEATURES,model,horizon=HORIZON); probs.append(p)
        v=p.notna(); y=lab.loc[v,"target"].astype(int); pv=p[v]; pred=(pv>=.5).astype(int)
        metrics.append({"model":name,"accuracy":sf(accuracy_score(y,pred)),"f1":sf(f1_score(y,pred,zero_division=0)),"roc_auc":sf(roc_auc_score(y,pv)) if y.nunique()>1 else None,"brier":sf(brier_score_loss(y,pv)),"observations":int(len(y))})
        final=clone(model); final.fit(lab[FEATURES],lab["target"].astype(int)); mp[name]=sf(final.predict_proba(feat.dropna(subset=FEATURES).iloc[[-1]][FEATURES])[:,1][0])
    ens=np.nanmean(np.vstack([x.values for x in probs]),axis=0); ens=np.asarray(ens); valid=np.isfinite(ens)
    ep=ens[valid]; ey=lab["target"].to_numpy()[valid].astype(int); er=lab["forward_return"].to_numpy()[valid]
    base=sf(ey.mean()); current=sf(np.mean(list(mp.values())))
    # Similar historical setups: +/- 2.5 percentage points around today's ensemble output.
    sm=np.abs(ep-current)<=.025; sn=int(sm.sum())
    similar={"observations":sn,"actual_up_rate":sf(ey[sm].mean()) if sn else None,"mean_forward_return":sf(er[sm].mean()) if sn else None,"median_forward_return":sf(np.median(er[sm])) if sn else None,"base_up_rate":base}
    # Evidence card: describes the data without issuing a buy/sell instruction.
    aucs=[m["roc_auc"] for m in metrics if m["roc_auc"] is not None]; mean_auc=sf(np.mean(aucs)) if aucs else None
    agreement=sf(1-abs(mp["logistic"]-mp["random_forest"])) if len(mp)==2 else None
    lift=sf(similar["actual_up_rate"]-base) if similar["actual_up_rate"] is not None and base is not None else None
    if mean_auc is None: quality="Not available"
    elif mean_auc>=.58: quality="Stronger historical discrimination"
    elif mean_auc>=.53: quality="Modest historical discrimination"
    else: quality="Weak historical discrimination"
    evidence={"validation_quality":quality,"mean_roc_auc":mean_auc,"model_agreement":agreement,"historical_lift":lift,"similar_sample_size":sn,
      "notes":["Model output is not a calibrated real-world probability.","Compare signal lift with the unconditional base rate.","Give more weight to signals only when validation and sample size support them."]}
    close=raw["Close"]; daily=close.pct_change()
    return {"ticker":ticker,"as_of":str(raw.index[-1].date()),"price":sf(close.iloc[-1]),"change_1d":sf(daily.iloc[-1]),"change_5d":sf(close.pct_change(5).iloc[-1]),"probability_5d_up":current,"model_probabilities":mp,"regime":str(classify_regime(raw).iloc[-1]),"volatility_20d":sf(daily.rolling(20).std().iloc[-1]*np.sqrt(252)),"rsi_14":sf(latest["rsi_14"]),"drawdown_252":sf(latest["drawdown_252"]),"base_up_rate":base,"similar_setups":similar,"evidence":evidence,"calibration_buckets":bucket_stats(ep,ey,er),"metrics":metrics,"history":[{"date":str(i.date()),"close":sf(v)} for i,v in close.tail(180).items()]}
def main():
    p={"generated_at":datetime.now(timezone.utc).isoformat(),"horizon_days":HORIZON,"tickers":[],"errors":[]}
    for t in TICKERS:
        try:p["tickers"].append(analyze(t))
        except Exception as e:p["errors"].append({"ticker":t,"error":str(e)})
    Path("data").mkdir(exist_ok=True); Path("data/dashboard.json").write_text(json.dumps(p,indent=2),encoding="utf-8"); print(json.dumps({"tickers":[x["ticker"] for x in p["tickers"]],"errors":p["errors"]},indent=2))
if __name__=="__main__":main()

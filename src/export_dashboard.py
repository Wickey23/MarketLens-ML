from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from src.data_loader import download_prices
from src.features import add_target, build_features
from src.regime import classify_regime
from src.train import FEATURES
from src.walk_forward import expanding_predictions, model_library

TICKERS = ["SPY", "VOO", "QQQ", "VXUS"]
HORIZON = 5

def sf(x):
    try:
        v=float(x)
        return v if np.isfinite(v) else None
    except Exception: return None

def analyze(ticker):
    raw=download_prices(ticker,"2010-01-01")
    feat=build_features(raw)
    labeled=add_target(feat,HORIZON).dropna(subset=FEATURES+["target"])
    latest=feat.dropna(subset=FEATURES).iloc[-1]
    model_probs={}; metric_rows=[]
    for name,model in model_library().items():
        probs=expanding_predictions(labeled,FEATURES,model,horizon=HORIZON)
        valid=probs.notna(); y=labeled.loc[valid,"target"].astype(int); p=probs.loc[valid]; pred=(p>=.5).astype(int)
        metric_rows.append({"model":name,"accuracy":sf(accuracy_score(y,pred)),"f1":sf(f1_score(y,pred,zero_division=0)),"roc_auc":sf(roc_auc_score(y,p)) if y.nunique()>1 else None,"brier":sf(brier_score_loss(y,p)),"observations":int(len(y))})
        final=clone(model); final.fit(labeled[FEATURES],labeled["target"].astype(int))
        model_probs[name]=sf(final.predict_proba(feat.dropna(subset=FEATURES).iloc[[-1]][FEATURES])[:,1][0])
    close=raw["Close"]; daily=close.pct_change()
    return {"ticker":ticker,"as_of":str(raw.index[-1].date()),"price":sf(close.iloc[-1]),"change_1d":sf(daily.iloc[-1]),"change_5d":sf(close.pct_change(5).iloc[-1]),"probability_5d_up":sf(np.mean(list(model_probs.values()))),"model_probabilities":model_probs,"regime":str(classify_regime(raw).iloc[-1]),"volatility_20d":sf(daily.rolling(20).std().iloc[-1]*np.sqrt(252)),"rsi_14":sf(latest["rsi_14"]),"drawdown_252":sf(latest["drawdown_252"]),"metrics":metric_rows,"history":[{"date":str(i.date()),"close":sf(v)} for i,v in close.tail(180).items()]}

def main():
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"horizon_days":HORIZON,"tickers":[],"errors":[]}
    for t in TICKERS:
        try: payload["tickers"].append(analyze(t))
        except Exception as e: payload["errors"].append({"ticker":t,"error":str(e)})
    Path("data").mkdir(exist_ok=True)
    Path("data/dashboard.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({"tickers":[x["ticker"] for x in payload["tickers"]],"errors":payload["errors"]},indent=2))
if __name__=="__main__": main()

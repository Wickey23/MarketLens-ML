from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def model_library(random_state: int = 42):
    return {
        "logistic": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=random_state, n_jobs=-1),
    }

def expanding_predictions(data: pd.DataFrame, features: list[str], model, horizon: int = 5, min_train: int = 750, retrain_every: int = 20) -> pd.Series:
    """Leakage-safe expanding walk-forward predictions with a label embargo."""
    if len(data) <= min_train + horizon:
        raise ValueError("Not enough rows for requested walk-forward window.")
    probs = pd.Series(np.nan, index=data.index, name="probability")
    fitted = None
    last_train_end = None
    for i in range(min_train + horizon, len(data)):
        train_end = i - horizon
        if fitted is None or last_train_end is None or train_end - last_train_end >= retrain_every:
            fitted = clone(model)
            train = data.iloc[:train_end]
            fitted.fit(train[features], train["target"].astype(int))
            last_train_end = train_end
        probs.iloc[i] = fitted.predict_proba(data.iloc[[i]][features])[:, 1][0]
    return probs

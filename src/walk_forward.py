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
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=10, random_state=random_state, n_jobs=-1),
    }


def expanding_predictions(data: pd.DataFrame, features: list[str], model, min_train: int = 750, retrain_every: int = 20) -> pd.Series:
    """Generate predictions using only observations available before prediction time."""
    if len(data) <= min_train:
        raise ValueError("Not enough rows for requested walk-forward window.")
    probs = pd.Series(np.nan, index=data.index, name="probability")
    fitted = None
    for i in range(min_train, len(data)):
        if fitted is None or (i - min_train) % retrain_every == 0:
            fitted = clone(model)
            fitted.fit(data.iloc[:i][features], data.iloc[:i]["target"].astype(int))
        probs.iloc[i] = fitted.predict_proba(data.iloc[[i]][features])[:, 1][0]
    return probs

from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from src.data_loader import download_prices
from src.features import add_target, build_features

FEATURES = ["ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d", "ma_dist_10", "ma_dist_20", "ma_dist_50", "ma_dist_200", "rsi_14", "vol_20", "vol_60", "volume_change_5", "drawdown_252"]


def dataset(ticker: str = "SPY", start: str = "2010-01-01", horizon: int = 5) -> pd.DataFrame:
    return add_target(build_features(download_prices(ticker, start)), horizon).dropna(subset=FEATURES + ["target"])


def evaluate_walk_forward(data: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    X, y = data[FEATURES], data["target"].astype(int)
    # Gap the folds by the forecast horizon so training labels cannot use
    # future prices that overlap the beginning of the validation fold.
    splitter = TimeSeriesSplit(n_splits=5, gap=horizon)
    models = {
        "logistic": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=10, random_state=42, n_jobs=-1),
    }
    rows = []
    for name, model in models.items():
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X), 1):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            prob = model.predict_proba(X.iloc[test_idx])[:, 1]
            pred = (prob >= 0.5).astype(int)
            rows.append({"model": name, "fold": fold, "accuracy": accuracy_score(y.iloc[test_idx], pred), "f1": f1_score(y.iloc[test_idx], pred), "roc_auc": roc_auc_score(y.iloc[test_idx], prob)})
    return pd.DataFrame(rows)


def main() -> None:
    data = dataset()
    scores = evaluate_walk_forward(data)
    print(scores.groupby("model")[["accuracy", "f1", "roc_auc"]].mean().round(4))
    model = RandomForestClassifier(n_estimators=400, min_samples_leaf=10, random_state=42, n_jobs=-1)
    model.fit(data[FEATURES], data["target"].astype(int))
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "ticker": "SPY", "horizon": 5}, "models/spy_rf.joblib")
    print("Saved models/spy_rf.joblib")


if __name__ == "__main__":
    main()

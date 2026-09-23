from __future__ import annotations
import joblib
from src.data_loader import download_prices
from src.features import build_features


def latest_probability(ticker: str = "SPY", model_path: str = "models/spy_rf.joblib") -> float:
    bundle = joblib.load(model_path)
    trained_ticker=str(bundle.get("ticker") or "SPY").upper()
    requested=str(ticker).upper()
    if requested!=trained_ticker:
        raise ValueError(f"Model at {model_path} was trained for {trained_ticker}, not {requested}.")
    frame = build_features(download_prices(requested)).dropna(subset=bundle["features"])
    row = frame[bundle["features"]].iloc[[-1]]
    return float(bundle["model"].predict_proba(row)[0, 1])


if __name__ == "__main__":
    p = latest_probability()
    print(f"SPY probability of positive 5-day return: {p:.1%}")

import numpy as np
import pandas as pd
from src.features import add_target, build_features


def test_features_and_target_exist():
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    close = pd.Series(np.linspace(100, 150, len(idx)), index=idx)
    df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * .99, "Close": close, "Volume": 1_000_000}, index=idx)
    out = add_target(build_features(df), 5)
    assert "rsi_14" in out.columns
    assert "forward_return" in out.columns
    assert "target" in out.columns
    assert out["target"].tail(5).isna().all()

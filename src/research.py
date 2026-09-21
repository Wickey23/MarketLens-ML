from __future__ import annotations
import pandas as pd
from src.backtest import backtest_probabilities, summary
from src.train import FEATURES, dataset
from src.walk_forward import expanding_predictions, model_library


def run_research(ticker: str = "SPY", threshold: float = 0.55) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = dataset(ticker=ticker)
    models = model_library()
    rows = []
    outputs = []
    for name, model in models.items():
        prob = expanding_predictions(data, FEATURES, model)
        valid = prob.notna()
        frame = pd.DataFrame(index=data.index[valid])
        frame["probability"] = prob[valid]
        frame["asset_return"] = data.loc[valid, "Close"].pct_change().fillna(0)
        tested = backtest_probabilities(frame, threshold=threshold)
        stats = summary(tested["strategy_return"])
        bh = summary(tested["asset_return"])
        rows.append({"model": name, **stats, "buy_hold_return": bh["total_return"], "observations": len(tested)})
        tested["model"] = name
        outputs.append(tested)
    return pd.DataFrame(rows), pd.concat(outputs)


if __name__ == "__main__":
    results, _ = run_research()
    print(results.to_string(index=False))

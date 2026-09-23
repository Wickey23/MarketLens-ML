# MarketLens ML

MarketLens is an options-research and paper-trading platform that combines market context, leakage-safe walk-forward ML validation, option-chain economics, historical scenario replay, an Opportunity Radar, and two separate paper accounts.

It is a research and decision-support system. It does not place brokerage orders and does not guarantee profitable outcomes.

## Product workflow

1. **Research** — inspect price, regime, realized volatility, model output, historical base rates, comparable signals, catalysts, news context, and out-of-sample metrics.
2. **Options Lab** — compare delayed option-chain quotes, liquidity, spreads, implied volatility, estimated Greeks, breakeven, time decay, and expiration scenarios.
3. **Simulator** — manually practice long-option market and limit orders with a browser-local $10,000 paper account.
4. **AI Portfolio** — forward-test autonomous paper decisions from Opportunity Radar under conservative sizing and entry guardrails.

## Architecture

- **Flask/Vercel frontend:** `app.py` + `templates/index.html`
- **Heavy research:** GitHub Actions + Python ML modules
- **Fast market refresh:** `src/fast_refresh.py`
- **Daily walk-forward research:** `src/export_dashboard.py`
- **Opportunity Radar:** `src/opportunity_radar.py`
- **Autonomous paper account:** `src/ai_paper_trader.py`
- **Live generated state:** `market-data` Git branch, `data/dashboard.json` and `data/ai_paper_portfolio.json`

The production application reads generated JSON from the dedicated `market-data` branch. This keeps frequent market snapshots separate from application-code commits and avoids unnecessary Vercel deployments.

## Validation

Directional models currently include Logistic Regression and Random Forest. Walk-forward predictions use an expanding training window with a label embargo equal to the forecast horizon to avoid target leakage.

CI checks:

- Python compilation for `src`, `app.py`, and `index.py`
- Inline frontend JavaScript syntax
- Flask route smoke tests
- Feature/target construction
- Opportunity Radar behavior
- Autonomous paper-trader behavior and entry guardrails

## Autonomous paper guardrails

The AI portfolio is paper-only. New autonomous positions require Opportunity Radar qualification and additional constraints including minimum score, bounded DTE, acceptable spread, controlled theta burden, plausible implied-volatility inputs, positive historical replay economics, and fixed-fraction premium-at-risk sizing.

Existing trades retain the facts captured at entry so later strategy changes do not rewrite their history.

## Live quote overlay

The web app now has a separate underlying-quote path from the heavier research snapshots:

- If `FINNHUB_API_KEY` is configured on Vercel, MarketLens uses the provider-backed quote endpoint first.
- Without that secret, the app falls back to Yahoo Finance's 1-minute chart feed.
- Underlying quotes are cached server-side for roughly 10 seconds and the open browser polls the active ticker every 15 seconds.
- Option chains, Greeks, Opportunity Radar, and model evidence remain snapshot-based and refresh through GitHub Actions every 30 minutes during the configured weekday market window.
- The browser checks for a new research/options snapshot every two minutes and reloads it without requiring a Vercel redeploy.

The active quote provider is exposed by `/api/health`. Exact exchange entitlements depend on the configured provider account.


Option-chain and company context are retrieved through Yahoo Finance/yfinance and may be delayed, stale, incomplete, or inconsistent. Greeks are Black-Scholes estimates. Historical contract replay applies today's strike and premium economics to historical underlying moves; it is not historical option-chain reconstruction.

Model probabilities are research outputs, not guaranteed real-world probabilities. Current model validation should always be checked before giving directional output substantial weight.

## Local setup

For the web application:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

For the full research stack:

```bash
pip install -r requirements-ml.txt
python -m src.fast_refresh
python -m src.export_dashboard
python -m pytest -q
```

## Deployment

Vercel serves the lightweight Flask application from `main`. GitHub Actions perform the heavier market/ML work and write generated state to `market-data`.

The public refresh endpoint can launch only the lightweight fast-refresh workflow and is rate-limited. Full research remains scheduled or manually dispatched through GitHub Actions.

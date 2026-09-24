# MarketLens ML

MarketLens is an options-research and forward paper-testing platform. It combines leakage-aware walk-forward ML validation, market regime/context, live or fallback option-chain data, historical payoff replay, an Opportunity Radar, a manual paper simulator, and an isolated autonomous paper portfolio.

MarketLens is research software. It does not submit brokerage orders and does not guarantee profitable outcomes.

## Current workflow

1. **Research** — inspect price, regime, realized volatility, model output, historical base rates, comparable signals, catalysts, news context, and out-of-sample metrics.
2. **Options Lab** — compare bid/ask execution, liquidity, IV, Greeks, breakeven, theta, expirations, and scenario risk.
3. **Simulator** — manually practice long-option entries/exits with a browser-local $10,000 paper account.
4. **AI Portfolio** — prospectively test qualifying MarketLens setups in a separate persistent paper account.
5. **Forward Validation** — compare realized paper outcomes against the historical replay evidence captured before each entry.
6. **Readiness Review** — current-strategy evidence remains in collection until the minimum forward sample, observation span, breadth, P/L, concentration, and drawdown gates are met.

## Architecture

- Flask/Vercel web app: `app.py`, `index.py`, `templates/index.html`
- Heavy research: GitHub Actions + Python ML modules
- Fast market/options refresh: `src/fast_refresh.py`
- Daily walk-forward research: `src/export_dashboard.py`
- Opportunity Radar: `src/opportunity_radar.py`
- Options providers/math: `src/options_data.py`
- Autonomous paper account: `src/ai_paper_trader.py`
- Persistent generated state: `market-data` branch
  - `data/dashboard.json`
  - `data/ai_paper_portfolio.json`

The web application reads generated research state from `market-data`. Vercel deployments are disabled for that branch so frequent JSON refreshes do not create web builds.

## Market data

### Tradier production path

When `TRADIER_ACCESS_TOKEN` is configured:

- REST underlying quotes prefer Tradier.
- Options chains use Tradier production data.
- Option-chain economics use the same Tradier underlying quote as the option snapshot.
- Tradier/ORATS Greeks are used when available.
- The browser can request a short-lived Tradier streaming session without receiving the permanent token.
- The autonomous paper strategy only opens new positions from option snapshots explicitly marked real-time.
- Tradier's market clock is attached to research snapshots and helps gate simulated execution.

The permanent Tradier token stays server-side / in GitHub Actions secrets. The project contains no brokerage-order submission code.

### Fallback path

Without a production Tradier token, MarketLens can continue research with Yahoo/yfinance and optional Finnhub underlying quotes. Fallback option data may be delayed or stale and **does not qualify for the current autonomous forward experiment**.

## ML validation

Directional models currently include Logistic Regression and Random Forest.

- Features use information available at or before each row.
- The target is a future 5-trading-day return.
- Walk-forward predictions use expanding training windows.
- Training labels are embargoed by the forecast horizon to avoid overlap into each validation observation.
- Similar-signal lift uncertainty uses paired moving-block bootstrap resampling, rather than subtracting a fixed base rate from a subset-only interval.
- Model outputs are not presented as guaranteed or perfectly calibrated real-world probabilities.

Historical option payoff replay applies today's strike and executable long-entry premium to historical underlying moves. Because multi-day forward returns overlap, MarketLens reports an overlap-adjusted effective sample size and uses it for replay uncertainty/qualification instead of treating every daily row as independent. The replay does **not** reconstruct historical option IV/Greeks paths and does not prove a durable edge. The forward paper record is the primary evidence for the actual strategy.

## Autonomous paper strategy

Current strategy generation: `v3_realtime_session`.

New paper entries require, among other controls:

- current strategy score threshold
- 3–45 DTE
- valid two-sided bid/ask quote
- quote freshness within the configured limit
- real-time option-chain source
- maximum spread
- maximum theta burden relative to debit
- plausible IV input
- positive historical replay economics
- minimum historical replay profit frequency
- fixed-fraction premium-at-risk sizing
- conservative U.S. regular-session window
- Tradier market-clock state when available

Long entries are simulated at the ask and exits at the bid. Expired options settle from intrinsic value rather than stale post-expiry quotes.

Legacy experiments remain preserved in history but cannot count toward current-strategy readiness, attribution, or adaptive weighting.

## Paper-to-real evidence gates

The current server-side readiness review uses only current-strategy trades entered from real-time data. Defaults include:

- at least 30 closed forward paper trades
- at least 21 days of observation
- positive net paper P/L
- positive average trade
- maximum drawdown no worse than 15%
- largest winner no more than 50% of gross winning P/L
- at least 3 tickers represented

Passing the gates means the paper record is ready for human review. It is not a guarantee and does not automatically enable real-money trading.

## Security and control plane

Secrets must never be committed.

Production/environment variables:

- `TRADIER_ACCESS_TOKEN` — Tradier production token for market data
- `GITHUB_ACTIONS_TOKEN` — server-side token used only to dispatch the lightweight refresh workflow
- `FINNHUB_API_KEY` — optional underlying-quote fallback
- `MARKETLENS_CONTROL_KEY` — app control key protecting refresh and streaming-session endpoints

Control endpoints fail closed once their server-side capability token is present: Tradier streaming will not expose a session endpoint unless `MARKETLENS_CONTROL_KEY` is also configured, and the GitHub refresh trigger follows the same rule. The browser requests the key only for the current session and stores it in `sessionStorage`; it is sent only to MarketLens control endpoints.

The refresh endpoint also checks GitHub's workflow history for a cross-instance cooldown so multiple Vercel instances cannot bypass the in-memory limiter.

## CI

Every push to `main` runs:

- Python compilation
- inline browser JavaScript syntax check
- Flask route/API smoke tests
- feature/target tests
- options/Tradier integration tests
- Opportunity Radar tests
- market-universe scanner tests
- autonomous paper-trader and readiness tests
- block-bootstrap uncertainty tests

## Local setup

Web app:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Full research stack:

```bash
pip install -r requirements-ml.txt
python -m src.fast_refresh
python -m src.export_dashboard
python -m pytest -q
```

## Deployment

Vercel serves the Flask application from `main`. GitHub Actions perform the heavier market/ML work and commit generated state to `market-data`.

There should be only one production Vercel project connected to this repository. Duplicate Vercel projects cause every `main` commit to build twice and should be disconnected or removed from Vercel.

## Important limitations

- Options can lose 100% of premium.
- Real-time market-data availability depends on provider/account entitlements.
- ORATS/Tradier Greeks have their own update frequency and are not tick-by-tick risk estimates.
- Black-Scholes fallback metrics omit some real-world effects such as dividends and American early exercise.
- Historical replay is not historical option-price reconstruction.
- A profitable paper sample can fail out of sample later.

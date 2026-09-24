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

### Multi-provider router

MarketLens does not blindly take the newest timestamp. It normalizes every configured provider, ranks the feed by authority, checks freshness, and compares providers for material disagreement.

Current provider roles:

- **Tradier production** — consolidated real-time U.S. stock/options data when the brokerage account is entitled; also supplies the market clock and browser-safe streaming session.
- **Alpaca OPRA / SIP** — consolidated real-time options/stock feeds when the configured Alpaca plan is entitled.
- **Alpaca IEX** — real-time stock exchange-subset data, useful for cross-checking but lower authority than SIP/consolidated feeds.
- **Alpaca indicative options** — near-real-time indicative options quotes; useful as a secondary reference but deliberately not execution-grade.
- **Finnhub** — optional underlying-quote cross-check.
- **Yahoo/yfinance** — low-authority delayed/best-effort research fallback.

For each selected quote MarketLens can expose the winning provider, feed, quote age, cross-provider agreement, confidence state, and the candidate provider quotes that were compared. A newer indicative or delayed quote cannot override a trustworthy consolidated feed merely because its timestamp is newer.

Option quotes are considered execution-grade for the autonomous paper strategy only when the selected source is consolidated real-time data, its timestamp is present and no more than 60 seconds old, and trusted providers are not in material conflict. The browser polls the selected option contract every five seconds when Tradier streaming is not supplying it; underlying REST quotes use a five-second cache.

When both Tradier and Alpaca OPRA/SIP are available, close agreement can raise the data confidence to **high**. Material disagreement between trusted providers produces **conflict** and blocks autonomous paper entry.

The permanent provider credentials stay server-side / in Vercel and GitHub Actions secrets. The project contains no brokerage-order submission code.

### Provider configuration

Supported environment variables:

- `TRADIER_ACCESS_TOKEN`
- `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY`
- `ALPACA_OPTIONS_FEED` — `indicative` or `opra`
- `ALPACA_STOCK_FEED` — normally `iex` or `sip`; when omitted, MarketLens defaults to SIP when OPRA is selected and IEX otherwise
- `FINNHUB_API_KEY` — optional underlying cross-check

If no execution-grade provider is configured, research continues using lower-authority fallbacks, but those quotes **do not qualify for the current autonomous forward experiment**.

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
- execution-grade consolidated real-time option quote with a fresh timestamp
- no material trusted-provider quote conflict
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
- `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` — optional Alpaca market-data credentials
- `ALPACA_OPTIONS_FEED` / `ALPACA_STOCK_FEED` — optional Alpaca entitlement/feed selection
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
- Tradier/Alpaca option integration and multi-provider routing tests
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

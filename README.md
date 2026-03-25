# Volume Profile Predictive Model — v2

> **Previous version:** [v1.0-tick-data](../../tree/v1.0-tick-data) — the original pipeline built on 6 months of raw tick data.
> **See also:** [CHANGELOG.md](CHANGELOG.md) for the full version history.

---

After finishing v1, I had a working ML pipeline and a model with real predictive edge — 84% accuracy, AUC of 0.95. But there was one thing I couldn't ignore: 183 days is not enough data to trust a backtest. You can get lucky over 6 months. You cannot get lucky over 3 years across multiple market regimes.

So v2 is about two things. First, replacing the raw tick data pipeline with a cleaner API-based approach that makes it easy to pull as much historical data as needed. Second, building a proper backtester that models real trading costs — because a strategy that looks great before fees can look terrible after them. And I wanted to find out which one this was.

---

## What Changed from v1

### Data Source
v1 used raw Binance tick data downloaded manually as monthly zip files and loaded into DuckDB to avoid RAM crashes. It worked, but it was slow to set up and limited to however many months I had downloaded.

v2 replaces all of that with a single API call to Binance's public futures endpoint. No authentication required, no manual downloads, no DuckDB. The API returns 1-minute OHLCV candles which are aggregated directly into the volume profile. Three years of data downloads in about 10 minutes.

The volume profile built from 1-minute candles is marginally less precise than one built from individual ticks, but the difference at $10 bucket resolution is negligible — and the v1 validation against ATAS already confirmed the engine works correctly.

### Data Range
- v1: August 2025 → January 2026 (183 days, 1 market regime)
- v2: January 2023 → January 2026 (1,126 days, multiple regimes)

This matters because the 2023-2026 period includes a bear market recovery, the 2024 bull run and halving, and the 2025-2026 continuation. A strategy that only works in one regime is not a strategy — it's luck.

### Backtester
v1 had no backtester. v2 adds `backtest.py`, which simulates all three trading strategies against the full 3-year dataset with realistic costs.

**Entry logic:**
- Wait for price to revisit the previous session's POC on a 15-minute candle — first touch only
- Entry at POC price (limit order)
- TP at previous session's VAH (longs) or VAL (shorts)
- SL at the nearest LVN (Low Volume Node) on the correct side of the POC, detected from the previous day's volume profile
- Skip the trade if R:R < 1.5

**Risk management:**
- 3% of capital risked per trade
- Position size capped at 3x leverage to prevent runaway compounding

**Costs modelled:**
- Maker fee: 0.02% (entry and TP exits — limit orders)
- Taker fee: 0.05% (SL exits — market orders)
- Slippage: $10 flat per side

---

## Project Structure

The pipeline is now 7 modules. `data_loader.py` and the DuckDB dependency are gone. Everything else carries over from v1 with minor updates.

| Module | File | Description |
|---|---|---|
| API Loader | `api_loader.py` | Fetches 1-min klines from Binance API |
| Volume Profile Engine | `volume_profile.py` | Builds daily VP and computes POC, VAH, VAL |
| Visualizer | `visualize.py` | Plots VP for sanity checking |
| Feature Engineering | `features.py` | Extracts ML features from VP levels |
| Label Generator | `labels.py` | Creates binary labels for supervised learning |
| Model | `model.py` | Trains and evaluates the ML model |
| Evaluation | `evaluation.py` | Measures predictive edge |
| Backtester | `backtest.py` | Simulates trading strategies with realistic costs |

Run the pipeline:
```bash
python api_loader.py        # fetch klines — only needed once
python main.py --skip-fetch # build VP → features → labels → model → evaluation
python backtest.py          # run backtest separately
```

---

## api_loader.py

This replaces `data_loader.py` entirely. It hits Binance's public futures API endpoint (`/fapi/v1/klines`) and handles pagination automatically — Binance returns a maximum of 1,000 candles per request, so the loader loops and stitches them together until the full date range is covered.

It outputs two files:
- `data/df_klines_1m.csv` — 1.6 million 1-minute candles
- `data/df_klines_15m.csv` — the same data aggregated to 15-minute candles, used by the backtester

No API key required. No DuckDB. No manual downloads.

---

## volume_profile.py

Same algorithm as v1 — $10 price buckets, 70% value area, POC/VAH/VAL computed per day. The only difference is the input: instead of querying DuckDB for raw ticks, it reads from `df_klines_1m.csv` and aggregates by close price per candle.

1,127 days of volume profile computed in under 5 seconds.

---

## Model Results (v2 — 1,126 days)

With 6x more data the model numbers are more trustworthy.

| Metric | v1 (183 days) | v2 (1,126 days) |
|---|---|---|
| Accuracy | 84% | 84% |
| Baseline | 63.9% | 70.5% |
| Improvement | +20.1% | +15.5% |
| AUC | 0.95 | 0.937 |

The accuracy held. The AUC held. The improvement over baseline dropped slightly because the 3-year dataset has a higher baseline — BTC accepted above the previous VAH 70.5% of the time over this period, reflecting the strong bull market of 2023-2026.

The strongest predictor remained `price_vs_prev_poc` with a coefficient of +3.06 — if today's price is above yesterday's POC, the model already has its most powerful bullish signal before the session even starts.

---

## Backtest Results (v2 — 1,126 days)

This is where things got honest.

| Strategy | Trades | Win Rate | Gross P&L | Fees | Net P&L | Return | Sharpe |
|---|---|---|---|---|---|---|---|
| VAH Long | 211 | 19.0% | +$3,453 | -$7,495 | -$4,042 | -40.4% | -2.22 |
| POC Bullish | 219 | 27.9% | +$11,173 | -$9,143 | +$2,030 | +20.3% | 0.62 |
| POC Bearish | 158 | 25.9% | +$8,796 | -$6,458 | +$2,338 | +23.4% | 0.70 |
| Buy & Hold | — | — | — | — | +$40,686 | +406.9% | — |

The strategies that trade with the trend (POC Bullish and POC Bearish) show positive returns after realistic fees. VAH Long underperforms because the VAH is often close to the entry price, making the R:R unfavourable even with the LVN stop loss.

None of the strategies beat buy and hold — but that was expected. BTC went from $16k to $96k over this period. No active day trading strategy paying 0.02-0.05% fees per round trip is going to beat that in a pure 6x bull run. The more relevant question is whether the edge is real and whether it survives in a sideways or bearish regime. That requires further testing.

**The honest takeaway:** the backtest revealed that transaction costs are the biggest enemy at this trade frequency. The strategies have gross edge (positive P&L before fees on POC Bullish and POC Bearish) but the net edge after costs is thin. The next step is reducing trade frequency and improving signal quality rather than taking every setup.

---

## What This Version Doesn't Do (Yet)

- **Walk-forward validation** — the backtest uses a single fixed train/test split. A proper walk-forward test would retrain the model on a rolling window to simulate live deployment more accurately.
- **Trend day entries** — the backtester only enters when price revisits the POC. Days where price opens and never looks back (trend days) are skipped entirely. These are a separate setup that needs its own logic.
- **Multiple instruments** — the pipeline is instrument-agnostic. The same code could run on ES futures, NQ, crude oil or any liquid instrument with accessible 1-min data. Testing across instruments would confirm whether the VP edge generalises.
- **Power BI dashboard** — the three output CSVs (`bt_trades.csv`, `bt_equity.csv`, `bt_summary.csv`) are designed to feed directly into a Power BI dashboard for visual strategy comparison. This is the next step.

---

## Future Improvements

These are ordered by priority — the same logic as v1 but updated to reflect what the backtest revealed.

1. **Reduce trade frequency** — apply stricter signal filters to take only the highest-quality setups. Fewer trades with better R:R will dramatically reduce the fee drag.
2. **Walk-forward validation** — replace the static 80/20 split with a rolling window to measure how the model performs on truly unseen data over time.
3. **Power BI dashboard** — visualise the equity curves, trade log, and strategy comparison in an interactive dashboard.
4. **Trend day detection** — build a separate label and entry logic for days where price opens and moves directionally without revisiting the POC.
5. **Naked POC tracking** — identify unvisited POCs from previous sessions and add them as features. These act as magnets in VP theory.
6. **More complex models** — test Random Forest and XGBoost against the logistic regression baseline.
7. **Live signal generator** — connect to Binance WebSocket for real-time end-of-day signal generation.
# Changelog

## v2.0 — api-klines branch *(current)*

**Data**
- Replaced manual tick data downloads + DuckDB with Binance public API (`api_loader.py`)
- Expanded dataset from 183 days to 1,126 days (Jan 2023 → Jan 2026)
- Added 15-minute candle generation for intraday backtest entry detection

**Backtester**
- Built `backtest.py` — simulates 3 trading strategies with realistic costs
- Entry: first 15-min candle touching prev_POC (limit order)
- SL: nearest LVN from previous day's VP
- TP: prev_VAH (longs) / prev_VAL (shorts)
- Costs: 0.02% maker fee, 0.05% taker fee, $10 slippage per side
- Leverage cap: 3x to prevent runaway position sizing
- R:R filter: minimum 1.5 required to take the trade

**Model**
- Retrained on 1,126 days — accuracy held at 84%, AUC at 0.937
- Baseline increased from 63.9% to 70.5% (stronger bull market trend in 3yr dataset)

**Backtest results (1,126 days, 3x leverage, 0.02% maker / 0.05% taker / $10 slippage)**

| Strategy | Trades | Win Rate | Gross P&L | Fees | Net P&L | Return | Sharpe |
|---|---|---|---|---|---|---|---|
| VAH Long | 211 | 19.0% | +$3,453 | -$7,495 | -$4,042 | -40.4% | -2.22 |
| POC Bullish | 219 | 27.9% | +$11,173 | -$9,143 | +$2,030 | +20.3% | 0.62 |
| POC Bearish | 158 | 25.9% | +$8,796 | -$6,458 | +$2,338 | +23.4% | 0.70 |
| Buy & Hold | — | — | — | — | +$40,686 | +406.9% | — |

All three strategies show positive gross edge. Transaction costs are the primary challenge at this trade frequency.

---

## v1.0 — main branch *(tag: v1.0-tick-data)*

**Data**
- Raw BTCUSDT perpetual futures tick data from Binance (data.binance.vision)
- 6 months: August 2025 → January 2026 (183 trading days)
- Loaded via DuckDB to avoid RAM exhaustion from pandas

**Pipeline**
- `data_loader.py` — loads tick zips into DuckDB
- `volume_profile.py` — builds daily VP, computes POC/VAH/VAL ($10 buckets, 70% rule)
- `visualize.py` — validates VP against ATAS (max $100 difference)
- `features.py` — 15 VP-derived features using shift(1)
- `labels.py` — 3 binary labels (VAH acceptance, POC bullish, POC bearish)
- `model.py` — Logistic Regression + StandardScaler, 80/20 split
- `evaluation.py` — accuracy, ROC curve, AUC

**Model results**
- Accuracy: 84% on unseen test set (+20.1% over 63.9% baseline)
- AUC: 0.95
- Strongest predictor: `price_vs_prev_poc` (coefficient +2.44)
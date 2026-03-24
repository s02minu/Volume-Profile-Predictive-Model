"""
backtest.py  (v4)
-----------------
Backtests 3 VP-based strategies over all 183 days.

Entry logic:
  - Wait for price to revisit prev_POC on a 15-min candle (first touch)
  - SL: nearest LVN on the correct side of prev_POC (from previous day's VP)
  - TP: prev_VAH (longs) or prev_VAL (shorts)
  - Skip if: no LVN found | R:R < MIN_RR | bad geometry | no POC touch

LVN detection parameters (tuned on real data):
  - THRESHOLD    = 0.65  (bucket volume < 65% of neighbours average)
  - NEIGHBOURS   = 3     (compare against 3 buckets each side)
  - SEARCH_RANGE = 60    (search up to 60 buckets = $600 from POC)

Risk: 3% of capital per trade

Outputs:
  data/bt_trades.csv    every individual trade
  data/bt_equity.csv    daily equity curve per strategy + buy & hold
  data/bt_summary.csv   win rate, sharpe, max drawdown, P&L per strategy

Requires:
  data/df_labels.csv    signals + prev levels
  data/df_15min.csv     15-min candles
  data/df_vp.csv        volume profile buckets for LVN detection
"""

import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 10_000
RISK_PCT         = 0.03
MIN_RR           = 1.5

# Realistic Binance USDT-M Futures costs
# Entry at POC = limit order = maker fee
# TP exit      = limit order = maker fee
# SL exit      = market order = taker fee
# Slippage     = conservative flat estimate per side
MAKER_FEE   = 0.0002   # 0.02%
TAKER_FEE   = 0.0005   # 0.05%
SLIPPAGE    = 10.0     # $10 per side (entry + exit)
MAX_LEVERAGE = 3.0     # max position value = 3x capital

# LVN detection — tuned on real BTCUSDT VP data
LVN_THRESHOLD    = 0.65   # volume < 65% of neighbour average = LVN
LVN_NEIGHBOURS   = 3      # buckets each side for comparison
LVN_SEARCH_RANGE = 60     # max buckets ($600) to search from POC

LABELS_PATH   = "data/df_labels.csv"
INTRADAY_PATH = "data/df_15min.csv"
VP_PATH       = "data/df_vp.csv"

STRATEGIES = {
    "VAH Long":    {"signal": "label_vah_acceptance", "direction": "long"},
    "POC Bullish": {"signal": "label_poc_bullish",    "direction": "long"},
    "POC Bearish": {"signal": "label_poc_bearish",    "direction": "short"},
}


# ── LOAD DATA ─────────────────────────────────────────────────────────────
def load_data():
    print("Loading data...")

    df_signals = pd.read_csv(LABELS_PATH, parse_dates=["date"])
    df_signals = df_signals.sort_values("date").reset_index(drop=True)
    df_signals = df_signals.dropna(subset=["prev_poc", "prev_vah", "prev_val"])

    df_15min = pd.read_csv(INTRADAY_PATH, parse_dates=["datetime"])
    df_15min["date"] = pd.to_datetime(df_15min["date"]).dt.date
    df_15min = df_15min.sort_values("datetime").reset_index(drop=True)

    df_vp = pd.read_csv(VP_PATH, parse_dates=["date"])
    df_vp["date"] = pd.to_datetime(df_vp["date"]).dt.date

    print(f"  Signals:        {len(df_signals)} days")
    print(f"  15-min candles: {len(df_15min):,}")
    print(f"  VP buckets:     {len(df_vp):,}")
    return df_signals, df_15min, df_vp


# ── LVN DETECTION ─────────────────────────────────────────────────────────
def find_lvn(df_vp, date, poc_price, direction):
    """
    Find the nearest LVN to prev_POC on the correct side.

    long  → search below poc_price (SL candidate)
    short → search above poc_price (SL candidate)

    Returns LVN price (float) or None.
    """
    day_vp = df_vp[df_vp["date"] == date].copy()
    if day_vp.empty:
        return None

    day_vp  = day_vp.sort_values("price_bucket").reset_index(drop=True)
    buckets = day_vp["price_bucket"].values
    volumes = day_vp["volume"].values
    n       = len(buckets)

    # Compute LVN flags
    lvn_flags = []
    for i in range(n):
        lo = max(0, i - LVN_NEIGHBOURS)
        hi = min(n, i + LVN_NEIGHBOURS + 1)
        nb = list(volumes[lo:i]) + list(volumes[i+1:hi])
        if not nb:
            lvn_flags.append(False)
            continue
        lvn_flags.append(volumes[i] < LVN_THRESHOLD * np.mean(nb))

    # Find POC index
    poc_idx = int(np.argmin(np.abs(buckets - poc_price)))

    if direction == "long":
        start = max(0, poc_idx - LVN_SEARCH_RANGE)
        for i in range(poc_idx - 1, start - 1, -1):
            if lvn_flags[i]:
                return float(buckets[i])
    else:
        end = min(n, poc_idx + LVN_SEARCH_RANGE)
        for i in range(poc_idx + 1, end):
            if lvn_flags[i]:
                return float(buckets[i])

    return None


# ── FIND POC TOUCH ON 15-MIN ──────────────────────────────────────────────
def find_poc_touch(df_15min, trade_date, poc_price, direction):
    """
    Find the first 15-min candle where price touches prev_POC.
    long  → candle low  <= poc_price
    short → candle high >= poc_price

    Returns (touch_time, remaining_candles) or (None, None).
    """
    if isinstance(trade_date, pd.Timestamp):
        trade_date = trade_date.date()

    day = df_15min[df_15min["date"] == trade_date].sort_values("datetime")
    if day.empty:
        return None, None

    for _, candle in day.iterrows():
        touched = (
            (direction == "long"  and candle["low"]  <= poc_price) or
            (direction == "short" and candle["high"] >= poc_price)
        )
        if touched:
            remaining = day[day["datetime"] > candle["datetime"]]
            return candle["datetime"], remaining

    return None, None


# ── CHECK TP / SL ─────────────────────────────────────────────────────────
def check_tp_sl(remaining, entry, tp, sl, direction, eod_close):
    """Walk remaining 15-min candles to find TP or SL hit."""
    if remaining is None or remaining.empty:
        return "EOD", eod_close

    for _, candle in remaining.iterrows():
        if direction == "long":
            tp_hit = candle["high"] >= tp
            sl_hit = candle["low"]  <= sl
        else:
            tp_hit = candle["low"]  <= tp
            sl_hit = candle["high"] >= sl

        if tp_hit and sl_hit:
            if direction == "long":
                return ("TP", tp) if (tp - entry) <= (entry - sl) else ("SL", sl)
            else:
                return ("TP", tp) if (entry - tp) <= (sl - entry) else ("SL", sl)
        elif tp_hit:
            return "TP", tp
        elif sl_hit:
            return "SL", sl

    return "EOD", eod_close


# ── POSITION SIZE ─────────────────────────────────────────────────────────
def position_size(capital, entry, sl_distance):
    """
    Risk exactly RISK_PCT of capital.
    qty = risk_amount / sl_distance

    Capped at MAX_LEVERAGE × capital so position value
    never exceeds a realistic margin requirement.
    """
    if sl_distance <= 0:
        return 0
    qty_risk = (capital * RISK_PCT) / sl_distance
    qty_max  = (capital * MAX_LEVERAGE) / entry
    return min(qty_risk, qty_max)


# ── TRANSACTION COSTS ────────────────────────────────────────────────────
def transaction_costs(entry, tp, sl, qty, outcome):
    """
    Calculate realistic trading costs for a single trade.

    Entry:  always limit order (maker fee + slippage)
    TP hit: limit order (maker fee + slippage)
    SL hit: market order (taker fee + slippage) — worse fill
    EOD:    market order (taker fee + slippage)

    Returns total cost in USD (always positive — it's a deduction).
    """
    position_value_entry = qty * entry

    # Entry cost — always maker + slippage
    entry_cost = (position_value_entry * MAKER_FEE) + SLIPPAGE

    # Exit cost — depends on outcome
    if outcome == "TP":
        exit_value = qty * tp
        exit_cost  = (exit_value * MAKER_FEE) + SLIPPAGE
    elif outcome == "SL":
        exit_value = qty * sl
        exit_cost  = (exit_value * TAKER_FEE) + SLIPPAGE
    else:  # EOD — market close
        exit_value = qty * tp   # approximate — actual EOD price used in pnl
        exit_cost  = (exit_value * TAKER_FEE) + SLIPPAGE

    return round(entry_cost + exit_cost, 4)


# ── RUN STRATEGY ──────────────────────────────────────────────────────────
def run_strategy(df_signals, df_15min, df_vp, name, signal_col, direction):
    trades       = []
    capital      = INITIAL_CAPITAL
    equity_curve = []

    skip = {"no_lvn": 0, "rr": 0, "geometry": 0, "no_touch": 0}

    # Pre-sort VP dates for fast lookup
    vp_dates = sorted(df_vp["date"].unique())

    for _, row in df_signals.iterrows():
        date      = row["date"]
        signal    = row[signal_col]
        prev_poc  = row["prev_poc"]
        prev_vah  = row["prev_vah"]
        prev_val  = row["prev_val"]
        eod_close = row["close"]
        trade_date = pd.to_datetime(date).date()

        if signal == 1:
            entry = prev_poc
            tp    = prev_vah if direction == "long" else prev_val

            # Geometry check
            if direction == "long"  and tp <= entry:
                skip["geometry"] += 1; continue
            if direction == "short" and tp >= entry:
                skip["geometry"] += 1; continue

            # Get previous VP date
            prior = [d for d in vp_dates if d < trade_date]
            if not prior:
                skip["no_lvn"] += 1; continue
            prev_vp_date = prior[-1]

            # Find LVN for SL
            lvn = find_lvn(df_vp, prev_vp_date, prev_poc, direction)
            if lvn is None:
                skip["no_lvn"] += 1; continue
            sl = lvn

            # Final geometry check with SL
            if direction == "long"  and not (tp > entry > sl):
                skip["geometry"] += 1; continue
            if direction == "short" and not (tp < entry < sl):
                skip["geometry"] += 1; continue

            # R:R check
            tp_dist = abs(tp - entry)
            sl_dist = abs(entry - sl)
            rr = tp_dist / sl_dist if sl_dist > 0 else 0
            if rr < MIN_RR:
                skip["rr"] += 1; continue

            # Find POC touch on 15-min
            touch_time, remaining = find_poc_touch(
                df_15min, trade_date, prev_poc, direction
            )
            if touch_time is None:
                skip["no_touch"] += 1; continue

            # Check TP / SL
            outcome, exit_price = check_tp_sl(
                remaining, entry, tp, sl, direction, eod_close
            )

            qty = position_size(capital, entry, sl_dist)

            if outcome == "TP":
                pnl = qty * tp_dist
            elif outcome == "SL":
                pnl = -(qty * sl_dist)   # actual loss based on real qty
            else:
                pnl = qty * (exit_price - entry) if direction == "long" \
                      else qty * (entry - exit_price)

            # Deduct realistic fees + slippage
            costs = transaction_costs(entry, tp, sl, qty, outcome)
            pnl   = pnl - costs

            capital += pnl

            trades.append({
                "date":          date,
                "strategy_name": name,
                "direction":     direction,
                "entry":         round(entry, 2),
                "tp":            round(tp, 2),
                "sl":            round(sl, 2),
                "lvn_sl":        round(lvn, 2),
                "tp_distance":   round(tp_dist, 2),
                "sl_distance":   round(sl_dist, 2),
                "rr_ratio":      round(rr, 2),
                "qty":           round(qty, 6),
                "outcome":       outcome,
                "exit_price":    round(exit_price, 2) if exit_price else None,
                "costs":         costs,
                "pnl":           round(pnl, 2),
                "pnl_gross":     round(pnl + costs, 2),
                "capital_after": round(capital, 2),
                "touch_time":    touch_time,
            })

        equity_curve.append({
            "date":          date,
            "capital":       round(capital, 2),
            "strategy_name": name,
        })

    print(f"  Skip — no LVN:     {skip['no_lvn']}")
    print(f"  Skip — bad R:R:    {skip['rr']}")
    print(f"  Skip — geometry:   {skip['geometry']}")
    print(f"  Skip — no touch:   {skip['no_touch']}")

    return pd.DataFrame(trades), pd.DataFrame(equity_curve)


# ── METRICS ───────────────────────────────────────────────────────────────
def compute_metrics(name, trades_df, equity_df):
    if trades_df.empty:
        return {
            "strategy": name, "total_trades": 0, "win_rate": 0,
            "total_pnl": 0, "avg_win": 0, "avg_loss": 0,
            "profit_factor": 0, "max_drawdown_pct": 0,
            "sharpe_ratio": 0, "final_capital": INITIAL_CAPITAL,
            "return_pct": 0,
        }

    wins   = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]

    gross_profit = wins["pnl"].sum()
    gross_loss   = abs(losses["pnl"].sum())

    equity    = equity_df["capital"].values
    peak      = np.maximum.accumulate(equity)
    dd        = (equity - peak) / peak * 100

    daily_ret = equity_df["capital"].pct_change().dropna()
    sharpe    = round(
        (daily_ret.mean() / daily_ret.std()) * np.sqrt(365), 2
    ) if daily_ret.std() > 0 else 0

    final = round(equity_df["capital"].iloc[-1], 2)

    return {
        "strategy":         name,
        "total_trades":     len(trades_df),
        "win_rate":         round(len(wins) / len(trades_df) * 100, 1),
        "total_pnl":        round(trades_df["pnl"].sum(), 2),
        "total_costs":      round(trades_df["costs"].sum(), 2),
        "total_pnl_gross":  round(trades_df["pnl_gross"].sum(), 2),
        "avg_win":          round(wins["pnl"].mean(),   2) if not wins.empty   else 0,
        "avg_loss":         round(losses["pnl"].mean(), 2) if not losses.empty else 0,
        "profit_factor":    round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
        "max_drawdown_pct": round(dd.min(), 2),
        "sharpe_ratio":     sharpe,
        "final_capital":    final,
        "return_pct":       round((final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2),
    }


# ── BUY AND HOLD ──────────────────────────────────────────────────────────
def buy_and_hold(df_signals):
    btc = INITIAL_CAPITAL / df_signals.iloc[0]["open"]
    curve = []
    for _, row in df_signals.iterrows():
        curve.append({
            "date":          row["date"],
            "capital":       round(btc * row["close"], 2),
            "strategy_name": "Buy & Hold",
        })
    return pd.DataFrame(curve)


# ── MAIN ──────────────────────────────────────────────────────────────────
def run_backtest():
    df_signals, df_15min, df_vp = load_data()

    all_trades, all_equity, all_metrics = [], [], []

    for name, cfg in STRATEGIES.items():
        print(f"\nRunning: {name}...")
        t_df, e_df = run_strategy(
            df_signals, df_15min, df_vp,
            name, cfg["signal"], cfg["direction"]
        )
        m = compute_metrics(name, t_df, e_df)
        all_trades.append(t_df)
        all_equity.append(e_df)
        all_metrics.append(m)

        print(f"  Trades taken:   {m['total_trades']}")
        print(f"  Win Rate:       {m['win_rate']}%")
        print(f"  Avg Win:        ${m['avg_win']:,.2f}")
        print(f"  Avg Loss:       ${m['avg_loss']:,.2f}")
        print(f"  Profit Factor:  {m['profit_factor']}")
        print(f"  Total P&L:      ${m['total_pnl']:,.2f}  (gross: ${m['total_pnl_gross']:,.2f})")
        print(f"  Total Costs:    ${m['total_costs']:,.2f}  (fees + slippage)")
        print(f"  Max Drawdown:   {m['max_drawdown_pct']}%")
        print(f"  Sharpe:         {m['sharpe_ratio']}")
        print(f"  Return:         {m['return_pct']}%")

    # Buy & Hold
    print("\nRunning: Buy & Hold...")
    bh      = buy_and_hold(df_signals)
    bh_eq   = bh["capital"].values
    bh_peak = np.maximum.accumulate(bh_eq)
    bh_dd   = round(((bh_eq - bh_peak) / bh_peak * 100).min(), 2)
    bh_ret  = round((bh["capital"].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2)
    all_equity.append(bh)
    all_metrics.append({
        "strategy": "Buy & Hold", "total_trades": 1, "win_rate": "-",
        "total_pnl": round(bh["capital"].iloc[-1] - INITIAL_CAPITAL, 2),
        "avg_win": "-", "avg_loss": "-", "profit_factor": "-",
        "max_drawdown_pct": bh_dd, "sharpe_ratio": "-",
        "final_capital": bh["capital"].iloc[-1], "return_pct": bh_ret,
    })
    print(f"  Return:       {bh_ret}%")
    print(f"  Max Drawdown: {bh_dd}%")

    # Save
    trades_out = pd.concat(
        [t for t in all_trades if not t.empty], ignore_index=True
    ) if any(not t.empty for t in all_trades) else pd.DataFrame()

    equity_out  = pd.concat(all_equity,  ignore_index=True)
    metrics_out = pd.DataFrame(all_metrics)

    trades_out.to_csv("data/bt_trades.csv",   index=False)
    equity_out.to_csv("data/bt_equity.csv",   index=False)
    metrics_out.to_csv("data/bt_summary.csv", index=False)

    print("\n✓ Saved:")
    print("  data/bt_trades.csv")
    print("  data/bt_equity.csv")
    print("  data/bt_summary.csv")
    print("\nBacktest complete!")

    return trades_out, equity_out, metrics_out


if __name__ == "__main__":
    run_backtest()
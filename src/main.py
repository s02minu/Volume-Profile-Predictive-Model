"""
main.py
-------
Runs the full Volume Profile ML pipeline (v2).

Usage:
  python -m src.main                 # full pipeline (fetch + build + train)
  python -m src.main --skip-fetch    # skip API fetch (uses existing kline CSVs)
"""

import argparse
import pandas as pd

from src.config import DATA_DIR
from src.data.api_loader import run_api_loader
from src.data.volume_profile import (
    build_daily_volume_profile,
    compute_vp_levels,
    extract_daily_ohlc,
    save_to_csv,
    load_from_csv,
)
from src.features.features import build_features
from src.features.labels import build_labels
from src.model.model import prepare_data, train_model, save_model
from src.model.evaluation import load_model, evaluate_model


def run_pipeline(skip_fetch=False):

    # step 1 — fetch klines from Binance API
    if not skip_fetch:
        print("=== STEP 1: Fetching Klines ===")
        df_1m, df_15m = run_api_loader()
    else:
        print("=== STEP 1: Loading Klines from Disk ===")
        df_1m = pd.read_csv(DATA_DIR / "df_klines_1m.csv")
        df_15m = pd.read_csv(DATA_DIR / "df_klines_15m.csv")
        print(f"  Loaded {len(df_1m):,} 1-min candles")
        print(f"  Loaded {len(df_15m):,} 15-min candles")

    # step 2 — build volume profile
    print("\n=== STEP 2: Building Volume Profile ===")
    df_vp = build_daily_volume_profile(df_1m, bucket_size=10)
    df_levels = compute_vp_levels(df_vp)
    df_ohlc = extract_daily_ohlc(df_1m)
    save_to_csv(df_vp, df_levels, df_15m)
    df_ohlc.to_csv(DATA_DIR / "df_ohlc.csv", index=False)

    # step 3 — build features
    print("\n=== STEP 3: Engineering Features ===")
    df_vp, df_levels = load_from_csv()
    df_features = build_features(df_levels, df_vp)
    df_features.to_csv(DATA_DIR / "df_features.csv", index=False)

    # step 4 — build labels
    print("\n=== STEP 4: Building Labels ===")
    df_labels = build_labels(df_features)
    df_labels.to_csv(DATA_DIR / "df_labels.csv", index=False)

    # step 5 — train model and save to disk
    print("\n=== STEP 5: Training Model ===")
    X, y, feature_cols = prepare_data(df_labels)
    model, scaler, X_test_scaled, y_test, y_pred = train_model(X, y)
    save_model(model, scaler)

    # step 6 — evaluate model performance
    print("\n=== STEP 6: Evaluating Model ===")
    model, scaler = load_model()
    df_labels = pd.read_csv(DATA_DIR / "df_labels.csv")
    X, y, feature_cols = prepare_data(df_labels)
    y_pred, y_prob, auc = evaluate_model(model, scaler, X, y, feature_cols)

    print("\n=== PIPELINE COMPLETE ===")
    print(f"Final AUC Score: {auc:.3f}")
    print("All outputs saved to data/")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    parser = argparse.ArgumentParser(description="Volume Profile ML Pipeline")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip API fetch and use existing kline CSVs")
    args = parser.parse_args()

    run_pipeline(skip_fetch=args.skip_fetch)

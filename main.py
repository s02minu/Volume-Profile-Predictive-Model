import pandas as pd
from data_loader import load_raw_ticks_to_duckdb
from volume_profile import build_daily_volume_profile, compute_vp_levels, extract_daily_ohlc, extract_intraday_ohlc, save_to_csv, load_from_csv
from features import build_features
from labels import build_labels
from model import prepare_data, train_model, save_model
from evaluation import load_model, evaluate_model

data_path = r"MarketData"

def run_pipeline():

    # step 1 — load raw ticks into DuckDB
    print("=== STEP 1: Loading Data ===")
    con = load_raw_ticks_to_duckdb(data_path)

    # step 2 — build volume profile and save to disk
    print("\n=== STEP 2: Building Volume Profile ===")
    df_vp = build_daily_volume_profile(con, bucket_size=10)
    df_levels = compute_vp_levels(df_vp)
    df_ohlc = extract_daily_ohlc(con)
    df_15min = extract_intraday_ohlc(con, timeframe_minutes=15)
    save_to_csv(df_vp, df_levels, df_15min)
    df_ohlc.to_csv('data/df_ohlc.csv', index=False)
            
    # step 3 — build features
    print("\n=== STEP 3: Engineering Features ===")
    df_vp, df_levels = load_from_csv()
    df_features = build_features(df_levels, df_vp)
    df_features.to_csv('data/df_features.csv', index=False)

    # step 4 — build labels
    print("\n=== STEP 4: Building Labels ===")
    df_labels = build_labels(df_features)
    df_labels.to_csv('data/df_labels.csv', index=False)

    # step 5 — train model and save to disk
    print("\n=== STEP 5: Training Model ===")
    X, y, feature_cols = prepare_data(df_labels)
    model, scaler, X_test_scaled, y_test, y_pred = train_model(X, y)
    save_model(model, scaler)

    # step 6 — evaluate model performance
    print("\n=== STEP 6: Evaluating Model ===")
    model, scaler = load_model()
    df_labels = pd.read_csv('data/df_labels.csv')
    X, y, feature_cols = prepare_data(df_labels)
    y_pred, y_prob, auc = evaluate_model(model, scaler, X, y, feature_cols)

    print("\n=== PIPELINE COMPLETE ===")
    print(f"Final AUC Score: {auc:.3f}")
    print("All outputs saved to data/")


if __name__ == "__main__":
    run_pipeline()
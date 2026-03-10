import pandas as pd
import numpy as np
from volume_profile import load_from_csv


def build_features(df_levels, df_vp):

    # step 1 — load true daily OHLC from disk (extracted from raw ticks in volume_profile.py)
    daily_ohlc = pd.read_csv('data/df_ohlc.csv')

    # step 2 — get buy and sell volume totals per day from df_vp
    daily_volume = df_vp.groupby('date').agg(
        total_buy_volume=('buy_volume', 'sum'),
        total_sell_volume=('sell_volume', 'sum')
    ).reset_index()

    # step 3 — merge VP levels + OHLC + volume into one dataframe
    df = pd.merge(df_levels, daily_ohlc, on='date', how='inner')
    df = pd.merge(df, daily_volume, on='date', how='inner')

    # step 4 — delta and buy ratio
    df['delta'] = df['total_buy_volume'] - df['total_sell_volume']
    df['buy_ratio'] = df['total_buy_volume'] / df['total_volume'].replace(0, np.nan)

    # step 5 — value area coverage
    # how much of the day's total price range was covered by the value area
    df['va_width'] = df['vah'] - df['val']
    df['day_range'] = df['high'] - df['low']
    df['va_coverage'] = df['va_width'] / df['day_range'].replace(0, np.nan)

    # step 6 — classify day type using va_coverage and delta
    # accumulation = tight range + net buying
    # distribution = tight range + net selling
    # trending = wide range relative to value area
    # neutral = inconclusive
    def classify_day(row):
        if row['va_coverage'] >= 0.6 and row['delta'] > 0:
            return 'accumulation'
        elif row['va_coverage'] >= 0.6 and row['delta'] < 0:
            return 'distribution'
        elif row['va_coverage'] <= 0.4:
            return 'trending'
        else:
            return 'neutral'

    df['day_type'] = df.apply(classify_day, axis=1)

    # step 7 — POC position within value area
    # 0 = POC at bottom of VA, 1 = POC at top of VA
    df['poc_position'] = (df['poc'] - df['val']) / (df['vah'] - df['val']).replace(0, np.nan)

    # step 8 — shift all VP features by 1 day to get previous day's profile
    # yesterday's VP is the input, today's price action is the outcome
    shift_cols = [
        'poc', 'vah', 'val', 'va_width', 'total_volume',
        'delta', 'buy_ratio', 'va_coverage', 'poc_position', 'day_type'
    ]

    # cleaner way to shift multiple columns at once
    df[[f'prev_{c}' for c in shift_cols]] = df[shift_cols].shift(1)

    # step 9 — POC direction: did today's POC move up or down vs yesterday?
    # 1 = moved up, -1 = moved down, 0 = unchanged
    df['poc_direction'] = (df['poc'] - df['prev_poc']).apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )

    # step 10 — distance from today's open to yesterday's key levels
    # tells us where price started the day relative to the previous profile
    # positive = open is above the level = buy imbalance, negative = open is below = sell imbalance
    df['dist_prev_poc'] = df['open'] - df['prev_poc']
    df['dist_prev_vah'] = df['open'] - df['prev_vah']
    df['dist_prev_val'] = df['open'] - df['prev_val']
    
    
    # step 11 — is today's close above or below yesterday's POC?
    # 1 = above, 0 = below
    df['price_vs_prev_poc'] = (df['close'] > df['prev_poc']).astype(int)


    # step 12 — drop first row since it has no previous day data
    df = df.dropna(subset=['prev_poc']).reset_index(drop=True)

    print(f"Features built for {len(df)} days")
    print("\nFeature columns:")
    print([col for col in df.columns if col.startswith('prev_') or col in [
        'poc_direction', 'price_vs_prev_poc', 'day_type',
        'dist_prev_poc', 'dist_prev_vah', 'dist_prev_val'
    ]])
    print("\nSample:")
    print(df.head(5))

    return df


if __name__ == "__main__":
    df_vp, df_levels = load_from_csv()
    df_features = build_features(df_levels, df_vp)
    df_features.to_csv('data/df_features.csv', index=False)
    print("\nSaved df_features to data/df_features.csv")
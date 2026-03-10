import pandas as pd


def build_labels(df_features):

    df = df_features.copy()

    # ── LABEL 1 ─────────────────────────────────────────────────────────────
    # Did price accept above yesterday's VAH today?
    # defined as today's close being above yesterday's VAH
    # 1 = yes (bullish acceptance above value area)
    # 0 = no (price stayed inside or below value area)
    df['label_vah_acceptance'] = (df['close'] > df['prev_vah']).astype(int)

    # ── LABEL 2 ─────────────────────────────────────────────────────────────
    # Did price open above yesterday's POC and close above it too?
    # this captures bullish continuation — price started above fair value
    # and stayed there, confirming the move
    # 1 = yes (bullish continuation)
    # 0 = no
    df['label_poc_bullish'] = (
        (df['dist_prev_poc'] > 0) & (df['close'] > df['prev_poc'])
    ).astype(int)

    # ── LABEL 3 ─────────────────────────────────────────────────────────────
    # Did price open below yesterday's POC and close below it too?
    # this captures bearish continuation
    # 1 = yes (bearish continuation)
    # 0 = no
    df['label_poc_bearish'] = (
        (df['dist_prev_poc'] < 0) & (df['close'] < df['prev_poc'])
    ).astype(int)

    # ── CLASS BALANCE SUMMARY ────────────────────────────────────────────────
    total = len(df)

    for label in ['label_vah_acceptance', 'label_poc_bullish', 'label_poc_bearish']:
        positive = df[label].sum()
        negative = total - positive
        print(f"\n{label}:")
        print(f"  Yes (1): {positive} ({positive/total*100:.1f}%)")
        print(f"  No  (0): {negative} ({negative/total*100:.1f}%)")

    # sanity check — print monthly win rate for label_vah_acceptance
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month')['label_vah_acceptance'].mean() * 100
    print("\nMonthly VAH acceptance rate:")
    print(monthly.round(1))
    
    return df



if __name__ == "__main__":
    df_features = pd.read_csv('data/df_features.csv')
    df = build_labels(df_features)
    df.to_csv('data/df_labels.csv', index=False)
    print("\nSaved df_labels to data/df_labels.csv")
    
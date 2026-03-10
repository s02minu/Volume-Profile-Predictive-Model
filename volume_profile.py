import pandas as pd
from data_loader import load_raw_ticks_to_duckdb

data_path = r"MarketData"

def build_daily_volume_profile(con, bucket_size=10):
    df_vp = con.execute(f'''
        select 
            cast(epoch_ms(time) as date) as date,
            floor(price/{bucket_size}) * {bucket_size} as price_bucket,
            sum(qty) as volume,
            sum(case when is_buyer_maker = false then qty else 0 end) as buy_volume,
            sum(case when is_buyer_maker = true then qty else 0 end) as sell_volume
        from trades
        group by date, price_bucket
        order by date, price_bucket    
    ''').df()

    print(f"Total rows: {len(df_vp):,}")
    print(f"Unique days: {df_vp['date'].nunique()}")
    print("\nSample:")
    print(df_vp.head(10))
    
    return df_vp


def compute_vp_levels(df_vp):
    results = []

    for date, day_profile in df_vp.groupby('date'):
        day_profile = day_profile.sort_values('volume', ascending=False)
        poc = day_profile.iloc[0]['price_bucket']

        total_volume = day_profile['volume'].sum()
        value_area_target = total_volume * 0.70

        day_profile = day_profile.sort_values('price_bucket')

        poc_idx = day_profile[day_profile['price_bucket'] == poc].index[0]
        poc_pos = day_profile.index.get_loc(poc_idx)

        accumulated_volume = day_profile.iloc[poc_pos]['volume']
        upper = poc_pos
        lower = poc_pos

        while accumulated_volume < value_area_target and (upper < len(day_profile) - 1 or lower > 0):
            upper_vol = day_profile.iloc[upper + 1]['volume'] if upper + 1 < len(day_profile) else 0
            lower_vol = day_profile.iloc[lower - 1]['volume'] if lower - 1 >= 0 else 0

            if upper_vol >= lower_vol:
                upper += 1
                accumulated_volume += upper_vol
            else:
                lower -= 1
                accumulated_volume += lower_vol

        vah = day_profile.iloc[upper]['price_bucket']
        val = day_profile.iloc[lower]['price_bucket']

        results.append({
            'date': date,
            'poc': poc,
            'vah': vah,
            'val': val,
            'total_volume': total_volume
        })

    df_levels = pd.DataFrame(results)
    df_levels = df_levels.sort_values('date').reset_index(drop=True)

    print(f"Computed VP levels for {len(df_levels)} days")
    print("\nSample")
    print(df_levels.head(10))

    return df_levels


def save_to_csv(df_vp, df_levels):
    # saving processed data to disk so other modules dont need to reprocess
    df_vp.to_csv('data/df_vp.csv', index=False)
    df_levels.to_csv('data/df_levels.csv', index=False)
    print("Saved df_vp and df_levels to data/")


def load_from_csv():
    # loading the pre-processed data from disk — fast, no DuckDB needed
    df_vp = pd.read_csv('data/df_vp.csv')
    df_levels = pd.read_csv('data/df_levels.csv')
    print("Loaded df_vp and df_levels from data/")
    return df_vp, df_levels


def extract_daily_ohlc(con):
    # extracting daily open, high, low and close directly from raw ticks
    # this gives us the true closing price for each day
    df_ohlc = con.execute('''
        select
            cast(epoch_ms(time) as date) as date,
            first(price order by time) as open,
            max(price) as high,
            min(price) as low,
            last(price order by time) as close
        from trades
        group by date
        order by date
    ''').df()

    print(f"Extracted OHLC for {len(df_ohlc)} days")
    print(df_ohlc.head())

    return df_ohlc

if __name__ == "__main__":
    con = load_raw_ticks_to_duckdb(data_path)
    df_vp = build_daily_volume_profile(con, bucket_size=10)
    df_levels = compute_vp_levels(df_vp)
    df_ohlc = extract_daily_ohlc(con)
    save_to_csv(df_vp, df_levels)
    df_ohlc.to_csv('data/df_ohlc.csv', index=False)
    print("Saved df_ohlc to data/df_ohlc.csv")
import pandas as pd
import os

# Check signals for 300808 or other stocks in the screenshots
code = '300808'
cache_file = 'data/signal_cache/weak_signals.parquet'

if os.path.exists(cache_file):
    df = pd.read_parquet(cache_file)
    df_stock = df[df['code'] == code].copy()
    df_stock['date'] = pd.to_datetime(df_stock['date'])
    df_stock = df_stock.sort_values('date')
    
    # Check Jan 1 to Feb 13, 2026
    start_date = '2026-01-01'
    end_date = '2026-02-13'
    
    mask = (df_stock['date'] >= start_date) & (df_stock['date'] <= end_date)
    df_subset = df_stock[mask]
    
    print(f"--- Signals for {code} ({start_date} to {end_date}) ---")
    cols = [c for c in df_subset.columns if c.startswith('Signal_')]
    for _, row in df_subset.iterrows():
        triggered = [c.replace('Signal_', '') for c in cols if row[c]]
        if triggered:
            print(f"{row['date'].strftime('%Y-%m-%d')}: {', '.join(triggered)}")
else:
    print("Cache file not found.")

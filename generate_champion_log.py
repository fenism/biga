import pandas as pd
import json
import os
from data_loader import DataLoader
from signal_cache import SignalCacheReader
from backtest import BacktestEngine

def generate_champion_trades():
    # 1. Config
    bt_start = "2023-01-01"
    bt_end = "2026-02-20"
    initial_capital = 100000.0
    stop_loss = 0.05
    take_profit = 0.10
    max_hold = 20
    position_sizing = 0.05
    max_positions = 20
    
    # Champion combination
    champion_weak = ['RSI_Rev', 'Spring', 'Money_Flow', 'Limit_Open']
    champion_strong = []
    
    print(f"Loading data for combo: {champion_weak}...")
    
    # 2. Load cached signals directly from parquet to ensure consistency
    cache_reader = SignalCacheReader()
    weak_path = os.path.join(cache_reader.cache_dir, "weak_signals.parquet")
    if not os.path.exists(weak_path):
        print(f"❌ Error: {weak_path} not found. Please build cache first.")
        return
        
    df_w_full = pd.read_parquet(weak_path)
    df_w_full['date'] = pd.to_datetime(df_w_full['date'])
    df_w_full = df_w_full[(df_w_full['date'] >= pd.to_datetime(bt_start)) & (df_w_full['date'] <= pd.to_datetime(bt_end))]
    
    # Filter for the champion combo (OR logic for candidates)
    w_cols = [f'Signal_{s}' for s in champion_weak]
    avail = [c for c in w_cols if c in df_w_full.columns]
    if not avail:
        print(f"❌ Error: Targeted signals {champion_weak} not found in cache.")
        return
    mask = df_w_full[avail].any(axis=1)
    df_weak = df_w_full[mask].copy()
    
    # Prepare signal_df for backtest
    signal_df = df_weak
    for c in w_cols:
        if c not in signal_df.columns:
            signal_df[c] = False
    
    # 3. Run Backtest
    loader = DataLoader() # Assuming it finds the data/ dir
    engine = BacktestEngine(loader, initial_capital)
    
    equity_df, trade_df = engine.run(
        signal_df=signal_df,
        start_date=bt_start,
        end_date=bt_end,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
        max_hold_days=max_hold,
        position_sizing=position_sizing,
        max_positions=max_positions
    )
    
    # 4. Save Trades
    output_path = "champion_trades_log.csv"
    trade_df.to_csv(output_path, index=False)
    print(f"✅ Champion trade log saved to {output_path}")
    print(f"Summary: Return {((equity_df['total_assets'].iloc[-1]/100000)-1)*100:.2f}%")

if __name__ == "__main__":
    generate_champion_trades()

import os
import pandas as pd
from backtest import BacktestEngine
from data_loader import DataLoader
from signal_cache import SignalCacheReader

def verify():
    loader = DataLoader()
    cache_reader = SignalCacheReader()
    
    # Combined champion targets
    # Note: 'Limit_Open' maps to 'Signal_Limit' in the parquet columns
    # 'RSI_Rev', 'Spring', 'Money_Flow' are exact matches.
    targets = ['RSI_Rev', 'Spring', 'Money_Flow', 'Limit']
    
    start_date = '2023-01-01'
    end_date = '2026-02-23'
    
    print("Loading signal cache...")
    weak_path = os.path.join(cache_reader.cache_dir, "weak_signals.parquet")
    df_w_full = pd.read_parquet(weak_path)
    
    # Filter by date
    df_w_full['date'] = pd.to_datetime(df_w_full['date'])
    
    # Extract signals
    all_cols = [f'Signal_{s}' for s in targets]
    available_cols = [c for c in all_cols if c in df_w_full.columns]
    print(f"Found columns: {available_cols}")
    
    mask = df_w_full[available_cols].any(axis=1)
    signal_df = df_w_full[mask].copy()
    
    signal_df['signal_count'] = signal_df[available_cols].fillna(False).astype(int).sum(axis=1)
    
    # Final date filter
    signal_df = signal_df[(signal_df['date'] >= pd.to_datetime(start_date)) & (signal_df['date'] <= pd.to_datetime(end_date))]
    signal_df = signal_df.sort_values(by=['date', 'signal_count'], ascending=[True, False])
    
    print(f"Total entries filtered: {len(signal_df)}")
    if len(signal_df) == 0:
        print("No signals found in the given range!")
        return

    engine = BacktestEngine(loader, initial_capital=100000.0)
    
    print("Running A-Share Compliant Backtest...")
    equity, trades = engine.run(
        signal_df, 
        start_date, 
        end_date,
        stop_loss_pct=0.1,
        take_profit_pct=0.2,
        max_hold_days=10,
        position_sizing=0.2,
        max_positions=5
    )
    
    if not equity.empty:
        final_equity = equity.iloc[-1]['total_assets']
        ret_pct = (final_equity - 100000.0) / 100000.0
        
        equity['hwm'] = equity['total_assets'].cummax()
        drawdown = ((equity['total_assets'] - equity['hwm']) / equity['hwm']).min()
            
        print(f"\n=== RESULTS ===")
        print(f"Final Equity: {final_equity:.2f}")
        print(f"Total Return: {ret_pct*100:.2f}%")
        print(f"Max Drawdown: {drawdown*100:.2f}%")
        print(f"Total Trades: {len(trades)}")
        
        trades.to_csv('champion_trades_log_COMPLIANT.csv', index=False)
        print("Detailed log saved to champion_trades_log_COMPLIANT.csv")

if __name__ == "__main__":
    verify()

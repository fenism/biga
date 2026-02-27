import os
import sys
import pandas as pd
from itertools import combinations
import time
import concurrent.futures

# Ensure imports work from within the directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data_loader import DataLoader
from backtest import BacktestEngine
from signal_cache import SignalCacheReader

MARKET_DATA_DIR = "data/market_data"

_GLOBAL_LOADER = None
_GLOBAL_DF_S = None
_GLOBAL_DF_W = None

def worker(args):
    global _GLOBAL_LOADER, _GLOBAL_DF_S, _GLOBAL_DF_W
    """
    Worker function to test a single strategy combination.
    """
    s_combo, w_combo, start_date, end_date, initial_capital, stop_loss, take_profit, max_hold, position_sizing, max_positions, use_sector_filter, use_ambush_filter = args
    
    # Needs a local loader for thread/process safety (though reading CSV is mostly safe)
    if _GLOBAL_LOADER is None:
        _GLOBAL_LOADER = DataLoader(MARKET_DATA_DIR)
        
        # Load datasets ONCE per worker to avoid 510 IPC serialization tasks
        cache_reader = SignalCacheReader()
        strong_path = os.path.join(cache_reader.cache_dir, "strong_signals.parquet")
        if os.path.exists(strong_path):
            df = pd.read_parquet(strong_path)
            df['date'] = pd.to_datetime(df['date'])
            _GLOBAL_DF_S = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
        else:
            _GLOBAL_DF_S = pd.DataFrame()
            
        weak_path = os.path.join(cache_reader.cache_dir, "weak_signals.parquet")
        if os.path.exists(weak_path):
            df = pd.read_parquet(weak_path)
            df['date'] = pd.to_datetime(df['date'])
            _GLOBAL_DF_W = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
        else:
            _GLOBAL_DF_W = pd.DataFrame()
            
    loader = _GLOBAL_LOADER
    df_s_full = _GLOBAL_DF_S
    df_w_full = _GLOBAL_DF_W
    
    local_engine = BacktestEngine(loader, initial_capital)
    
    signal_df = pd.DataFrame()
    df_strong = pd.DataFrame()
    df_weak = pd.DataFrame()
    
    all_selected = s_combo + w_combo
    
    if s_combo and not df_s_full.empty:
        s_cols = [f'Signal_{s}' for s in s_combo]
        avail = [c for c in s_cols if c in df_s_full.columns]
        if len(avail) > 0: 
            mask = df_s_full[avail].any(axis=1)
            df_strong = df_s_full[mask].copy()
            
    if w_combo and not df_w_full.empty:
        w_cols = [f'Signal_{s}' for s in w_combo]
        avail = [c for c in w_cols if c in df_w_full.columns]
        if len(avail) > 0:
            mask = df_w_full[avail].any(axis=1)
            df_weak = df_w_full[mask].copy()
            
    if not df_strong.empty and not df_weak.empty:
        signal_df = pd.merge(df_strong, df_weak, on=['code', 'name', 'date'], how='outer', suffixes=('', '_weak'))
    elif not df_strong.empty:
        signal_df = df_strong
    else:
        signal_df = df_weak
        
    if signal_df.empty:
        return None
        
    # Final AND filter
    s_cols = [f'Signal_{s}' for s in all_selected]
    for c in s_cols:
        if c not in signal_df.columns:
            signal_df[c] = False
            
    signal_df[s_cols] = signal_df[s_cols].fillna(False)
    mask = signal_df[s_cols].any(axis=1)
    signal_df = signal_df[mask].copy()
    
    signal_df = signal_df.drop_duplicates(subset=['code', 'date'])
    
    if len(signal_df) == 0:
        return None
        
    # Add triggered_strategies column and calculate total_signal_count
    all_possible = s_cols if s_combo else w_cols  # The required ones, but wait, we want to count ALL available signals from the full set.
    
    # Add triggered_strategies column and calculate total_signal_count
    # Vectorized computation of signal count instead of slow row applying
    full_cols = [c for c in (df_s_full.columns.tolist() + df_w_full.columns.tolist()) if c.startswith('Signal_')]
    
    # Fast row sum of boolean columns
    available_cols = [c for c in full_cols if c in signal_df.columns]
    signal_df['signal_count'] = signal_df[available_cols].fillna(False).astype(int).sum(axis=1)
        
    # Sort signals by date and then by signal_count DESCENDING so BacktestEngine prioritizes them
    signal_df = signal_df.sort_values(by=['date', 'signal_count'], ascending=[True, False])
        
    # Run Backtest
    equity, trades = local_engine.run(signal_df, start_date, end_date, stop_loss, take_profit, max_hold, position_sizing, max_positions, use_sector_filter=use_sector_filter, use_ambush_filter=use_ambush_filter)
    
    if len(trades) > 0:
        final_equity = equity.iloc[-1]['total_assets']
        ret_pct = (final_equity - initial_capital) / initial_capital
        winning_trades = trades[trades['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades)
        
        equity['high_water_mark'] = equity['total_assets'].cummax()
        drawdown = ((equity['total_assets'] - equity['high_water_mark']) / equity['high_water_mark']).min()
        
        combo_name = " + ".join(all_selected)
        return {
            "Strategy": combo_name,
            "Return (%)": round(ret_pct * 100, 2),
            "Win Rate (%)": round(win_rate * 100, 2),
            "Trades": len(trades),
            "Max Drawdown (%)": round(drawdown * 100, 2),
            "Final Equity": round(final_equity, 2)
        }
    return None


def run_optimization():
    print(">>> 开始自动化策略组合寻优 (Automated Multiprocessing Strategy Optimizer) <<<")
    
    # Configuration matches Web UI Defaults for theoretical maximums
    bt_start = "2023-01-01" 
    bt_end = "2026-02-25"
    initial_capital = 100000.0  # 10万元
    stop_loss = 0.10
    take_profit = None          # No hard TP — use trailing exits only
    max_hold = None             # No hard max hold — let winners run
    position_sizing = 0.02     # Risk 2% per trade
    max_positions = 5          # User constraint: Max 5 positions
    use_sector_filter = True   # Phase 16: Industry Resonance
    use_ambush_filter = True   # Phase 19: Early Bird Ambush Filter (T-1)
    
    cache_reader = SignalCacheReader()
    
    strong_strats = ['Z_Score', 'RS', 'TKOS', 'DTR_Plus', 'Fighting', 'UA', 'HMC', 'OBO', 'LCS']
    weak_strats = ['HLP3', 'Limit', 'RSI_Rev', 'Spring', 'Pinbar', 'Money_Flow', 'Double_Vol', 'UA_Weak', 'Limit_Open']
    
    combos_to_test = []
    # Test singles
    for s in strong_strats: combos_to_test.append(([s], []))
    for w in weak_strats: combos_to_test.append(([], [w]))
    
    # Test combinations of 2, 3, and 4 (Strong internally, Weak internally)
    for i in range(2, 5):
        for combo in combinations(strong_strats, i): combos_to_test.append((list(combo), []))
        for combo in combinations(weak_strats, i): combos_to_test.append(([], list(combo)))
        
    print(f"Total Combinations to Backtest: {len(combos_to_test)}")
    
    df_s_full = pd.DataFrame()
    df_w_full = pd.DataFrame()

    # Build task arguments
    tasks = []
    for s_combo, w_combo in combos_to_test:
        tasks.append((
            s_combo, w_combo, bt_start, bt_end, initial_capital, 
            stop_loss, take_profit, max_hold, position_sizing, max_positions, use_sector_filter, use_ambush_filter
        ))

    results = []
    t0 = time.time()
    
    # Execute using ProcessPoolExecutor for maximum execution speed
    import gc
    import concurrent.futures
    import os
    
    max_workers = min(5, os.cpu_count() or 1)
    print(f"Spawning {max_workers} processes to evaluate combinations...")
    
    count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            count += 1
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception as e:
                print(f"Combination generated an exception: {e}")
                
            # Keep console output clean and force GC
            if count % 10 == 0:
                print(f"Processed {count}/{len(tasks)} combinations... split-time: {time.time()-t0:.1f}s")
                # ---> SAVE PARTIAL RESULTS TO JSON <---
                if len(results) > 0:
                    pd.DataFrame(results).to_json("optimization_results.json", orient="records", indent=4)
                gc.collect()

    print(f"Total execution time: {time.time()-t0:.1f}s")
    
    # Final save
    if len(results) > 0:
        pd.DataFrame(results).to_json("optimization_results.json", orient="records", indent=4)
    
    # Sort and Display
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        top_returns = res_df.sort_values(by="Return (%)", ascending=False).head(15)
        print("\n🏆 Top 15 Strategy Combinations by Return:")
        print(top_returns.to_markdown(index=False))
        
        print("\n🛡️ Top 15 Strategy Combinations by Win Rate (Min 10 trades):")
        top_wr = res_df[res_df['Trades'] >= 10].sort_values(by="Win Rate (%)", ascending=False).head(15)
        print(top_wr.to_markdown(index=False))
    else:
        print("No trades generated across any combinations.")
        
if __name__ == "__main__":
    run_optimization()

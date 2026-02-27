import pandas as pd
import os
from indicators import Indicators
from exit_signals import ExitSignals
from data_loader import DataLoader
import datetime

market_dir = "data/market_data"
loader = DataLoader(market_dir)

end_date = datetime.datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.datetime.now() - datetime.timedelta(days=100)).strftime("%Y-%m-%d")

stock_files = [f for f in os.listdir(market_dir) if f.endswith(".csv") and len(f) == 10]
found = 0
for f in stock_files[:300]:  # scan first 300 stocks
    code = f.split(".")[0]
    df = loader.get_k_data(code, start_date, end_date)
    if len(df) < 50: continue
    
    df = Indicators.add_all_indicators(df)
    exits = ExitSignals.detect(df, source_type='strong')
    
    # Let's find a stock that had a UA Reversal recently
    recent_exits = exits.tail(10)
    
    ua_idx = recent_exits[recent_exits['Exit_UA_Reversal']].index
    if not ua_idx.empty:
        trigger_date = df.loc[ua_idx[0], 'date']
        print(f"Found UA Reversal! Code: {code} on Date: {trigger_date.strftime('%Y-%m-%d')}")
        found += 1
    
    ma_idx = recent_exits[recent_exits['Exit_Break_MA20']].index
    if not ma_idx.empty and found < 2:
        trigger_date = df.loc[ma_idx[0], 'date']
        print(f"Found Break MA20! Code: {code} on Date: {trigger_date.strftime('%Y-%m-%d')}")
        found += 1
        
    if found >= 2:
        break

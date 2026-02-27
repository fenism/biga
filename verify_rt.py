
from amarket_framework.tencent_loader import TencentLoader
import pandas as pd

def test_rt():
    loader = TencentLoader()
    # Test Index
    sh_index = "sh000001"
    # Test Stock
    stock_code = "600000"
    full_stock = loader.get_full_code(stock_code)
    
    codes = [sh_index, full_stock]
    print(f"Fetching quotes for: {codes}")
    rt_df = loader.fetch_realtime_quotes(codes)
    
    if rt_df.empty:
        print("Error: Received empty dataframe")
        return
        
    print("\n--- Real-time Quotes ---")
    print(rt_df[['code', 'name', 'open', 'high', 'low', 'close', 'volume', 'pctChg']])
    
    # Check for missing values
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if (rt_df[col] == 0).any():
            print(f"Warning: Zero value found in column {col}")

if __name__ == "__main__":
    test_rt()

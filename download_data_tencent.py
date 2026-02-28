import requests
import pandas as pd
import os
import concurrent.futures
import time
import json
import random
from datetime import datetime

DATA_DIR = "stock_app/data/market_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Headers to mimic browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "http://gu.qq.com/"
}

def get_stock_list_local():
    list_path = os.path.join(DATA_DIR, "stock_list.csv")
    if os.path.exists(list_path):
        print(f"Loading stock list from: {list_path}")
        return pd.read_csv(list_path, dtype={'code': str})
    else:
        print("Error: stock_list.csv not found! Please run the previous script to generate the list or fetch it via akshare/baostock first.")
        # Fallback: maybe use akshare to get list if not exists?
        # For now assume it exists as per previous steps.
        return pd.DataFrame()

def download_stock_tencent(stock_info):
    code = stock_info['code']
    name = stock_info['name']
    
    # Determine prefix (sh/sz/bj)
    # 6xx -> sh, 0xx/3xx -> sz, 8xx/4xx -> bj (Tencent might not support BJ well, or uses different prefix)
    # Tencent mostly strictly sh/sz.
    # 60, 68 -> sh
    # 00, 30 -> sz
    
    if code.startswith('6') or code.startswith('51') or code.startswith('56') or code.startswith('58'):
        symbol = f"sh{code}"
    elif code.startswith('0') or code.startswith('3') or code.startswith('15') or code.startswith('16'):
        symbol = f"sz{code}"
    elif code.startswith('8') or code.startswith('4'):
        # Beijing exchange might accept 'bj' or might not be fully supported by this endpoint.
        # Let's try 'bj' or skip. Detailed check needed. 
        # Usually web.ifzq.gtimg.cn supports sh/sz mainly.
        # Let's skip BJ for now to ensure stability or try to map.
        return f"Skipped {code} (BSE/Other)"
    else:
        symbol = f"sz{code}" # Default fallback
    
    # Target file
    file_path = os.path.join(DATA_DIR, f"{code}.csv")
    
    existing_df = None
    last_date = ""
    
    # Incremental check
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path)
            if not existing_df.empty and 'date' in existing_df.columns:
                last_date = str(existing_df['date'].iloc[-1])
                today_str = datetime.now().strftime("%Y-%m-%d")
                
                # If already up to date, skip
                if last_date == today_str:
                    return f"Skipped {code} (Up to date)"
        except Exception as e:
            print(f"Error reading existing file {code}: {e}")
            existing_df = None # Fallback to full download

    # Tencent API
    # qfq = forward adjusted
    # If incrementing, 30 points is enough for a week's gap. 
    # If last_date exists, fetch 30, else fetch 640.
    points = 640 if not last_date else 40
    
    api_url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var=kline_dayqfq&param={symbol},day,,,{points},qfq"
    
    try:
        resp = requests.get(api_url, headers=HEADERS, timeout=5)
        if resp.status_code != 200:
            return f"Error {code}: HTTP {resp.status_code}"
            
        content = resp.text
        if "=" in content:
            json_str = content.split("=", 1)[1]
        else:
            json_str = content
            
        data = json.loads(json_str)
        stock_data = data.get('data', {}).get(symbol, {})
        
        kline_list = stock_data.get('qfqday', [])
        if not kline_list:
             kline_list = stock_data.get('day', [])
             
        if not kline_list:
            return f"Warning {code}: No Data found"
            
        records = []
        for item in kline_list:
            if len(item) < 6: continue
            record = {
                'date': item[0],
                'open': float(item[1]),
                'close': float(item[2]),
                'high': float(item[3]),
                'low': float(item[4]),
                'volume': float(item[5])
            }
            records.append(record)
            
        if not records:
            return f"Warning {code}: Parsed Empty"
            
        new_df = pd.DataFrame(records)
        
        # Merge with existing
        if existing_df is not None:
            df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['date']).sort_values('date')
        else:
            df = new_df
            
        # Recalculate basic metrics if missing or just for safety
        df['pctChg'] = df['close'].pct_change() * 100
        df['pctChg'] = df['pctChg'].fillna(0)
        df['amount'] = df['close'] * df['volume'] * 100 
        df['turn'] = 0.0 # placeholder
        
        df.to_csv(file_path, index=False)
        return f"Success {code} (Incremental)" if last_date else f"Success {code} (Full)"
        
    except Exception as e:
        return f"Error {code}: {str(e)}"

def main():
    print(">>> 启动腾讯财经数据下载 (Tencent API) <<<")
    
    # 1. Load Local List
    stocks = get_stock_list_local()
    if stocks.empty:
        print("Stock list is empty. Please ensure stock_list.csv exists.")
        return
        
    stock_infos = stocks.to_dict('records')
    print(f"加载股票列表: {len(stocks)} 只。")
    
    # 2. Parallel Download
    # Tencent API is robust, can handle higher concurrency.
    max_workers = 10
    
    start_time = time.time()
    done = 0
    total = len(stocks)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_stock_tencent, info): info for info in stock_infos}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            # Optional: print errors
            if "Error" in res or "Warning" in res:
                 # Print only genuine errors, ignore warnings to reduce noise if many
                 if "No Data" not in res and "Skipped" not in res:
                     print(res)
            
            done += 1
            if done % 100 == 0:
                elapsed = time.time() - start_time
                speed = done / elapsed if elapsed > 0 else 0
                print(f"Progress: {done}/{total} | Speed: {speed:.1f}/s")
                
    print(f"\n全部下载完成! 数据存储于: {DATA_DIR}")

if __name__ == "__main__":
    main()

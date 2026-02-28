
import akshare as ak
import baostock as bs
import pandas as pd
import os
import concurrent.futures
from datetime import datetime
import time

DATA_DIR = "stock_app/data/market_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def get_stock_list_baostock():
    print("尝试使用 Baostock 获取股票列表 (Fallback)...")
    lg = bs.login()
    if lg.error_code != '0':
        raise Exception(f"Baostock login failed: {lg.error_msg}")
        
    date = datetime.now().strftime("%Y-%m-%d")
    # Query all A shares
    # Baostock doesn't have a single "all stocks" API easily, usually query by index or just iterating?
    # Actually query_stock_basic() works.
    
    rs = bs.query_stock_basic()
    if rs.error_code != '0':
        bs.logout()
        raise Exception(f"Baostock query failed: {rs.error_msg}")
        
    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())
    
    bs.logout()
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    # columns: code, code_name, ipoDate, outDate, type, status
    
    # ETF code prefixes (SH: 51x/58x/56x, SZ: 15x/16x)
    ETF_PREFIXES = ('510', '511', '512', '513', '515', '516', '518',
                    '560', '561', '562', '563', '588',
                    '159')
    
    filtered_list = []
    
    for _, row in df.iterrows():
        # Baostock code format: sh.600000
        full_code = row['code']
        name = row['code_name']
        status = row['status'] # 1=listed
        stock_type = row['type'] # 1=stock, 2=index, 3=other(ETF/bond)
        
        # Clean code (600000)
        code = full_code.split('.')[-1]
        
        if status != '1':
            continue
        
        # Include: stocks (type=1) + ETFs (type=3 with ETF prefix)
        is_etf = code.startswith(ETF_PREFIXES)
        
        if stock_type == '1':
            # Regular stock filters
            if code.startswith('8') or code.startswith('4') or code.startswith('92'): continue # BSE
            if code.startswith('688'): continue # STAR
            if code.startswith('200') or code.startswith('900'): continue # B-shares
            if 'ST' in name: continue
            filtered_list.append({'code': code, 'name': name, 'is_etf': False})
        elif is_etf:
            # ETF — include without ST filter
            filtered_list.append({'code': code, 'name': name, 'is_etf': True})
        # else: skip (indices, bonds, etc.)
        
    etf_count = sum(1 for x in filtered_list if x['is_etf'])
    stock_count = len(filtered_list) - etf_count
    print(f"Baostock 获取到 {stock_count} 只股票 + {etf_count} 只ETF = {len(filtered_list)} 总计。")
    return pd.DataFrame(filtered_list)

def download_stock(stock_info):
    code = stock_info['code']
    name = stock_info['name']
    is_etf = stock_info.get('is_etf', False)
    file_path = os.path.join(DATA_DIR, f"{code}.csv")
    
    if os.path.exists(file_path):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path)).date()
            if mtime == datetime.now().date():
                return f"Skipped {code} - Already updated today"
            
            # Additional check: read the last line of CSV to see if we have enough recent data
            existing_df = pd.read_csv(file_path)
            if not existing_df.empty:
                last_date_str = str(existing_df['date'].iloc[-1])
                # Normalize date string (some are YYYY-MM-DD, some are YYYYMMDD)
                last_date = pd.to_datetime(last_date_str).date()
                if last_date >= datetime.now().date():
                     return f"Skipped {code} - Data current (last: {last_date})"
        except: pass
            
    try:
        start_date = "20240101" 
        end_date = datetime.now().strftime("%Y%m%d")
        
        # Retry loop
        for i in range(3):
            try:
                if is_etf:
                    # ETF uses fund_etf_hist_em
                    df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
                else:
                    # Regular stock
                    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
                break
            except:
                time.sleep(1)
        else:
            return f"Error {code}: Akshare fetch failed"
            
        if df is None or df.empty:
            return f"Warning {code}: Empty"
            
        rename_map = {
            "日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", 
            "成交量": "volume", "成交额": "amount", "换手率": "turn"
        }
        df = df.rename(columns=rename_map)
        cols = ["date", "open", "high", "low", "close", "volume", "amount", "turn"]
        df = df[[c for c in cols if c in df.columns]]
        
        df.to_csv(file_path, index=False)
        return f"Success {code}{' (ETF)' if is_etf else ''}"
        
    except Exception as e:
        return f"Error {code}: {str(e)}"

def main():
    print(">>> 开始构建本地离线数据仓库 (v2.2 Hybrid Mode) <<<")
    
    # 1. Get List via Baostock (More stable)
    try:
        stocks = get_stock_list_baostock()
        stocks.to_csv(os.path.join(DATA_DIR, "stock_list.csv"), index=False)
    except Exception as e:
        print(f"Critical Error: Failed to get stock list. {e}")
        return

    stock_infos = stocks.to_dict('records')
    
    # 2. Parallel Download
    max_workers = 5
    print(f"启动 {max_workers} 线程下载 {len(stocks)} 只股票数据 (AkShare Source)...")
    
    done = 0
    total = len(stocks)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_stock, info): info for info in stock_infos}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            done += 1
            if done % 100 == 0:
                print(f"Progress: {done}/{total}")

    print(f"\n下载完成。")

if __name__ == "__main__":
    main()

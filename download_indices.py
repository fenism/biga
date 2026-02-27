import requests, json, pandas as pd, os

MARKET_DATA_DIR = "/Users/swag/Library/CloudStorage/GoogleDrive-fenism@gmail.com/其他计算机/我的 Mac Pro/文档/analyse/stock_app/data/market_data"

def download_one(c, sym):
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var=kline_dayqfq&param={sym},day,,,600,qfq"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200: return False
        content = r.text
        if "=" in content: json_str = content.split("=", 1)[1]
        else: json_str = content
        data = json.loads(json_str)
        k_data = data.get('data', {}).get(sym, {})
        klines = k_data.get('qfqday', []) or k_data.get('day', [])
        
        cols = ['date', 'open', 'close', 'high', 'low', 'volume']
        recs = []
        for k in klines:
            if len(k) < 6: continue
            recs.append({
                'date': k[0], 'open': k[1], 'close': k[2], 
                'high': k[3], 'low': k[4], 'volume': k[5]
            })
        if recs:
            df = pd.DataFrame(recs)
            # ensure sorting by date just in case
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(by='date').reset_index(drop=True)
            df.to_csv(os.path.join(MARKET_DATA_DIR, f"{c}.csv"), index=False)
            print(f"✅ Downloaded {c} - {len(df)} rows")
            return True
    except Exception as e:
        print(f"Error {c}: {e}")
        return False
        
download_one('399001.SZ', 'sz399001')
download_one('399006.SZ', 'sz399006')

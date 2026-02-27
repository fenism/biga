import requests
import pandas as pd
from datetime import datetime
from amarket_framework.timezone_utils import get_beijing_now

class TencentLoader:
    def __init__(self):
        self.session = requests.Session()
        # Headers to mimic a browser, though Tencent API is generally open
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

    def get_full_code(self, code):
        """
        Convert 6-digit code to Tencent format (shXXXXXX or szXXXXXX)
        """
        code = str(code).zfill(6)
        if code.startswith('6') or code.startswith('9'):
            return f"sh{code}"
        else:
            return f"sz{code}"

    def fetch_realtime_quotes(self, codes):
        """
        Fetch real-time data for a list of stock codes.
        codes: list of strings, e.g., ["sh000001", "sz399001"]
        Returns: DataFrame with columns [code, name, price, pct_chg, volume(hands), amount, time]
        """
        if not codes:
            return pd.DataFrame()
            
        code_str = ",".join(codes)
        url = f"http://qt.gtimg.cn/q={code_str}"
        
        try:
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            text = response.text
            
            data_list = []
            # Response format: v_sh000001="1~上证指数~000001~3268.00~...";
            lines = text.split(";")
            
            for line in lines:
                if "=" not in line:
                    continue
                    
                key, val = line.split("=", 1)
                val = val.strip('"')
                parts = val.split("~")
                
                if len(parts) < 30:
                    continue
                    
                # Parse key fields
                # 1: Name, 3: Current Price, 4: Prev Close, 5: Open, 33: High, 34: Low, 31: Change, 32: Change%, 6: Volume (hands), 37: Amount (wan), 30: Time
                code = key.split("_")[-1]
                name = parts[1]
                price = float(parts[3])
                open_p = float(parts[5])
                high = float(parts[33])
                low = float(parts[34])
                pct_chg = float(parts[32])
                volume = float(parts[6]) # Keep in hands (1 hand = 100 shares) to match K-line data
                amount = float(parts[37]) * 10000 # Convert wan to raw
                
                data_list.append({
                    "code": code,
                    "name": name,
                    "close": price,
                    "open": open_p,
                    "high": high,
                    "low": low,
                    "pctChg": pct_chg,
                    "volume": volume,
                    "amount": amount,
                    "timestamp": get_beijing_now() # Beijing timezone (UTC+8)
                })
                
            return pd.DataFrame(data_list)
            
        except Exception as e:
            print(f"Error fetching realtime quotes: {e}")
            return pd.DataFrame()

    def fetch_k_line(self, code, day_count=300):
        """
        Fetch daily K-line data for EMA calculation.
        code: e.g. "sh000001"
        """
        # Mapping standard prefix to Tencent format if needed, but usually sh000001 works
        url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{day_count},qfq"
        
        try:
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Navigate JSON structure: data -> code -> day
            if "data" not in data or code not in data["data"]:
                return pd.DataFrame()
                
            k_data = data["data"][code].get("day", [])
            
            if not k_data:
                return pd.DataFrame()
                
            # Tencent K-line format: [date, open, close, high, low, volume, ...others]
            # Data might have 9 or 10 columns. We only need the first 6.
            
            clean_data = []
            for item in k_data:
                # Ensure we have at least 6 columns
                if len(item) >= 6:
                    clean_data.append(item[:6])
            
            df = pd.DataFrame(clean_data, columns=["date", "open", "close", "high", "low", "volume"])
            df["date"] = pd.to_datetime(df["date"])
            
            for col in ["open", "close", "high", "low", "volume"]:
                df[col] = pd.to_numeric(df[col])
                
            # Add pctChg for volatility calc
            df['pctChg'] = df['close'].pct_change() * 100
            
            return df.set_index("date")
            
        except Exception as e:
            print(f"Error fetching k-line for {code}: {e}")
            return pd.DataFrame()

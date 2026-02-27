import baostock as bs
import pandas as pd
from datetime import datetime, timedelta
from amarket_framework.timezone_utils import get_beijing_now

class BaostockLoader:
    def __init__(self):
        self.logged_in = False
    
    def login(self):
        if not self.logged_in:
            try:
                lg = bs.login()
                if lg.error_code == '0':
                    self.logged_in = True
                    print(f"Baostock login success: {lg.error_msg}")
                else:
                    print(f"Baostock login failed: {lg.error_msg}")
            except Exception as e:
                print(f"Baostock login exception: {e}")
    
    def logout(self):
        if self.logged_in:
            try:
                bs.logout()
                self.logged_in = False
                print("Baostock logout success")
            except Exception as e:
                print(f"Baostock logout exception: {e}")

    def _query_data(self, code, fields, start_date, end_date, frequency="d", adjustflag="3"):
        """Generic internal method to query and process pagination"""
        try:
            self.login()
            rs = bs.query_history_k_data_plus(code, fields,
                start_date=start_date, end_date=end_date,
                frequency=frequency, adjustflag=adjustflag)
            
            if rs.error_code != '0':
                print(f"Query failed for {code}: {rs.error_msg}")
                return pd.DataFrame()

            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if not data_list:
                return pd.DataFrame()
                
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'pctChg', 'turn', 'peTTM', 'pbMRQ']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            return df
        except Exception as e:
            print(f"Exception querying data for {code}: {e}")
            return pd.DataFrame()

    def fetch_daily_kline(self, code, start_date=None, end_date=None, limit_days=365):
        """
        Fetch daily K-line data
        Standard indices: sh.000001 (ShangZheng), sz.399001 (ShenZheng)
        Standard stocks: sh.600000, sz.000001
        """
        if not end_date:
            end_date = get_beijing_now().strftime('%Y-%m-%d')
        if not start_date:
            start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=limit_days)
            start_date = start_dt.strftime('%Y-%m-%d')
            
        # Fields for indices differ slightly from stocks (indices don't have turn/pe/pb usually)
        if code.startswith("sh.000") or code.startswith("sz.399"):
            fields = "date,code,open,high,low,close,volume,amount,pctChg"
        else:
            fields = "date,code,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ"
            
        return self._query_data(code, fields, start_date, end_date)

    def fetch_index_data(self, code="sh.000001", days=365):
        """Wrapper for fetching index data"""
        return self.fetch_daily_kline(code, limit_days=days)

    def search_stock(self, code_name):
        """Search for a stock code by name (partial support via baostock usually requires full code)"""
        # Baostock doesn't have a strong 'search by name' API, assumes known codes.
        pass

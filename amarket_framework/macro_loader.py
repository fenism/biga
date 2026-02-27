import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from amarket_framework.timezone_utils import get_beijing_now

class MacroLoader:
    def __init__(self):
        pass

    def fetch_market_margin(self):
        """
        Fetch margin financing balance (financing + securities lending).
        Returns: {
            "date": str (YYYY-MM-DD),
            "margin_balance": float (in Billion CNY),
            "details": str
        }
        """
        try:
            # 1. Fetch History for Trend (Last 365 Days)
            end_date = get_beijing_now()
            start_date = end_date - timedelta(days=365)
            
            # SSE History
            try:
                df_sh = ak.stock_margin_sse(start_date=start_date.strftime("%Y%m%d"), end_date=end_date.strftime("%Y%m%d"))
                # Columns: 信用交易日期, 融资余额, ...
                df_sh['date'] = pd.to_datetime(df_sh['信用交易日期'], format='%Y%m%d')
                df_sh['sh_balance'] = df_sh['融资余额'].astype(float)
                df_sh = df_sh[['date', 'sh_balance']].set_index('date')
            except Exception as e:
                print(f"Error fetching SSE history: {e}")
                df_sh = pd.DataFrame()

            # SZSE History (macro_china_market_margin_sz returns all history)
            try:
                df_sz_all = ak.macro_china_market_margin_sz()
                # Columns: 日期, 融资余额, ...
                df_sz_all['date'] = pd.to_datetime(df_sz_all['日期'])
                df_sz_all['sz_balance'] = df_sz_all['融资余额'].astype(float)
                
                # Filter last 365 days (convert start_date to naive datetime for comparison)
                df_sz = df_sz_all[df_sz_all['date'] >= start_date.replace(tzinfo=None)].copy()
                df_sz = df_sz[['date', 'sz_balance']].set_index('date').sort_index()
            except Exception as e:
                print(f"Error fetching SZSE history: {e}")
                df_sz = pd.DataFrame()

            # Merge and Sum
            if not df_sh.empty and not df_sz.empty:
                # Ensure SH is sorted too
                df_sh = df_sh.sort_index()
                
                df_total = df_sh.join(df_sz, how='inner')
                
                # ADAPTIVE UNIT CORRECTION
                # Standard SH balance is ~8e11 (800 Billion) in Yuan
                # Standard SZ balance is ~7e11 (700 Billion) in Yuan
                # Logic: Check magnitude of the latest SZ value
                
                latest_sh_raw = df_total['sh_balance'].iloc[-1]
                latest_sz_raw = df_total['sz_balance'].iloc[-1]
                
                # Method 1: Absolute magnitude check
                # If SZ is too small (e.g. 7e7 -> 70 Million), it's likely "Wan Yuan", needs * 10000
                if latest_sz_raw < 1e9 and latest_sh_raw > 1e11: 
                    df_total['sz_balance'] = df_total['sz_balance'] * 10000
                    latest_sz_raw = df_total['sz_balance'].iloc[-1]  # Update after correction
                
                # Method 2: Ratio check (SZ should be roughly 0.7-1.2x of SH)
                # If ratio is way off, one of them has wrong unit
                if latest_sh_raw > 0:
                    ratio = latest_sz_raw / latest_sh_raw
                    if ratio > 3:  # SZ way too big, likely unit error (e.g. should be Wan but treated as Yuan*10000)
                        df_total['sz_balance'] = df_total['sz_balance'] / 10000
                    elif ratio < 0.01:  # SZ way too small, likely Wan treated as Yuan
                        df_total['sz_balance'] = df_total['sz_balance'] * 10000
                    
                df_total['total_balance'] = df_total['sh_balance'] + df_total['sz_balance']
                
                # Latest Value
                latest = df_total.iloc[-1]
                latest_date = latest.name.strftime("%Y-%m-%d")
                latest_val = latest['total_balance']
                
                # Convert history to simple list or dict for plotting
                history_data = df_total['total_balance'].reset_index()
                history_data['date'] = history_data['date'].dt.strftime('%Y-%m-%d')
                
            else:
                # Fallback to single point if history fails
                return self._fetch_margin_snapshot_fallback()

            return {
                "date": latest_date,
                "margin_balance": latest_val / 1e8, # Convert to Billion
                "details": f"SH: {latest['sh_balance']/1e8:.2f} + SZ: {latest['sz_balance']/1e8:.2f}",
                "history": history_data  # DataFrame with date, total_balance
            }

        except Exception as e:
            print(f"Error fetching margin data: {e}")
            return self._fetch_margin_snapshot_fallback()

    def _fetch_margin_snapshot_fallback(self):
        """Fallback to original snapshot method if history fails"""
        try:
            # Original logic...
            # SSE gives a range, so we get the latest valid trading day
            df_sh = ak.stock_margin_sse(start_date="20240101", end_date=get_beijing_now().strftime("%Y%m%d"))
            latest_sh = df_sh.iloc[-1]
            val_sh = float(latest_sh['融资余额'])
            date_val = str(latest_sh['信用交易日期'])
            date_sh = f"{date_val[:4]}-{date_val[4:6]}-{date_val[6:]}"
            
            df_sz = ak.stock_margin_szse(date=date_val) 
            val_sz = float(df_sz['融资余额（元）'].sum())
            
            total_margin = val_sh + val_sz
            return {
                "date": date_sh,
                "margin_balance": total_margin / 1e8,
                "details": f"SH: {val_sh/1e8:.2f} + SZ: {val_sz/1e8:.2f}",
                "history": None
            }
        except Exception as e:
             return {"date": "N/A", "margin_balance": 0, "error": str(e), "history": None}


    def fetch_money_supply(self):
        """
        Fetch M1/M2 historical data for trend analysis.
        Returns:
            dict with keys: date (latest), m1_yoy, m2_yoy, scissors, history (DataFrame)
        """
        try:
            df = ak.macro_china_money_supply()
            # Columns: 月份, 货币和准货币(M2)-同比增长, 货币(M1)-同比增长
            
            # Parse dates
            df['date'] = pd.to_datetime(df['月份'], format='%Y.%m', errors='coerce')
            
            # Check if parsing was successful
            if df['date'].notna().any():
                # At least some dates parsed successfully
                # Drop rows with invalid dates and sort
                df = df[df['date'].notna()].copy()
                df = df.sort_values('date', ascending=False).reset_index(drop=True)
            else:
                # All dates failed to parse, use raw string dates
                print(f"Warning: Failed to parse money supply dates, using string fallback")
                df = df.reset_index(drop=True)
                df['date'] = df['月份'].astype(str)
            
            if df.empty:
                return {"date": "N/A", "m1_yoy": 0, "m2_yoy": 0, "scissors": 0, "history": None}
            
            # Extract M1, M2 YoY growth rates
            df['m1_yoy'] = df['货币(M1)-同比增长'].astype(float)
            df['m2_yoy'] = df['货币和准货币(M2)-同比增长'].astype(float)
            df['scissors'] = df['m1_yoy'] - df['m2_yoy']
            
            # Latest value (first row after DESC sort)
            latest = df.iloc[0]
            if isinstance(latest['date'], pd.Timestamp):
                date_str = latest['date'].strftime('%Y-%m')
            else:
                date_str = str(latest['date'])
            
            m1_yoy = float(latest['m1_yoy'])
            m2_yoy = float(latest['m2_yoy'])
            scissors = float(latest['scissors'])
            
            # Prepare history for charting (reverse to ascending for plotly)
            history = df[['date', 'm1_yoy', 'm2_yoy', 'scissors']].sort_values('date', ascending=True).copy()
            if isinstance(history['date'].iloc[0], pd.Timestamp):
                history['date'] = history['date'].dt.strftime('%Y-%m')
            else:
                history['date'] = history['date'].astype(str)
            
            return {
                "date": date_str,
                "m1_yoy": m1_yoy,
                "m2_yoy": m2_yoy,
                "scissors": scissors,
                "history": history
            }
        except Exception as e:
            print(f"Error fetching money supply: {e}")
            return {"date": "N/A", "m1_yoy": 0, "m2_yoy": 0, "scissors": 0, "history": None}


import pandas as pd
import numpy as np
import os
import datetime

class DataLoader:
    _raw_csv_cache = {} # Class-level shared memory to prevent IO bottlenecks

    def __init__(self, data_dir=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if data_dir is None:
            self.data_dir = os.path.join(base_dir, "data", "market_data")
        else:
            self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory {self.data_dir} does not exist. Please run download_data.py first.")
            
        self._df_cache = {}

    def get_stock_list(self, date=None):
        """Fetch stock list from local warehouse."""
        list_path = os.path.join(self.data_dir, "stock_list.csv")
        if os.path.exists(list_path):
            return pd.read_csv(list_path, dtype={'code': str})
        else:
            return pd.DataFrame(columns=['code', 'name'])

    def get_k_data(self, code, start_date, end_date, compute_indicators=True):
        """
        Fetch K-line data from local CSV.
        """
        # Ensure code is 6 digits string
        code = str(code).zfill(6)
        file_path = os.path.join(self.data_dir, f"{code}.csv")
        
        if not os.path.exists(file_path):
            # Suppress normal missing warnings in loop
            return pd.DataFrame()

        try:
            if code not in self._df_cache:
                # O(1) Shared Memory fetch instead of 3ms Hard Drive IO penalty
                if code in DataLoader._raw_csv_cache:
                    df = DataLoader._raw_csv_cache[code].copy()
                else:
                    df = pd.read_csv(file_path)
                    DataLoader._raw_csv_cache[code] = df.copy()
                    
                # Standardize dates
                if 'date' not in df.columns:
                    self._df_cache[code] = pd.DataFrame()
                else:
                    df['date'] = pd.to_datetime(df['date'])
                    
                    if compute_indicators:
                        # Inject Indicators computation so cache holds them
                        from indicators import Indicators
                        from exit_signals import ExitSignals
                        try:
                            df = Indicators.add_all_indicators(df)
                            
                            # Pre-compute Exits for the whole timeline
                            exits_df = ExitSignals.detect(df, 'both')
                            
                            # Vectorize the exit string extraction to prevent python loop overhead
                            exit_cols = [c for c in exits_df.columns if c.startswith('Exit_')]
                            if exit_cols:
                                conditions = [exits_df[c] == True for c in exit_cols]
                                choices = exit_cols
                                df['Exits'] = np.select(conditions, choices, default="")
                            else:
                                df['Exits'] = ""
                            
                        except Exception as e:
                            import traceback
                            print(f"[DataLoader] Failed to compute indicators for {code}: {e}")
                            traceback.print_exc()
                            self._df_cache[code] = pd.DataFrame()
                        else:
                            # ========= CRITICAL MEMORY FIX ==========
                            # Drop the 60+ technical indicator columns to prevent OS Kernel Panic (OOM).
                            # BacktestEngine only needs basic OHLC + the calculated Exits + Life Lines.
                            keep_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'turn', 'Exits', 'MA20', 'EMA200']
                            actual_keep = [c for c in keep_cols if c in df.columns]
                            
                            # Hard cap cache size to 400 stocks
                            if len(self._df_cache) > 400:
                                self._df_cache.clear()
                                
                            self._df_cache[code] = df[actual_keep].copy()
                    else:
                        # If compute_indicators is False, we just need basic OHLC and Exits
                        # This happens during Backtest sweeps to avoid loading 60+ heavy indicators
                        from exit_signals import ExitSignals
                        try:
                            exits_df = ExitSignals.detect(df, 'both')
                            exit_cols = [c for c in exits_df.columns if c.startswith('Exit_')]
                            if exit_cols:
                                conditions = [exits_df[c] == True for c in exit_cols]
                                choices = exit_cols
                                df['Exits'] = np.select(conditions, choices, default="")
                            else:
                                df['Exits'] = ""
                        except Exception as e:
                            print(f"[DataLoader] Failed to compute exits for {code}: {e}")
                            df['Exits'] = ""
                            
                        keep_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'turn', 'Exits']
                        actual_keep = [c for c in keep_cols if c in df.columns]
                        
                        if len(self._df_cache) > 400:
                            self._df_cache.clear()
                        self._df_cache[code] = df[actual_keep].copy()
            
            df = self._df_cache[code]
            if df.empty:
                return df
            
            # Filter using already-parsed pd.Timestamp objects
            # start_dt = pd.to_datetime(start_date)
            # end_dt = pd.to_datetime(end_date)
            # (Passing already parsed datetime objects bypasses Regex parsing inside the 510-loop)
            start_dt = start_date
            end_dt = end_date
            
            # Filter
            mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
            res = df.loc[mask].copy()
            
            if res.empty:
                print(f"[DataLoader] Data empty after filtering. Range: {start_dt} - {end_dt}. File Range: {df['date'].min()} - {df['date'].max()}")
                
            return res
            
        except Exception as e:
            print(f"[DataLoader] Error reading {code}: {e}")
            return pd.DataFrame()

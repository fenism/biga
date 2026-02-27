"""
信号缓存管理模块
用于预计算和缓存所有股票的策略信号，提升筛选性能
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from strong_strategies import StrongStrategies
from weak_strategies import WeakStrategies




import concurrent.futures

_GLOBAL_INDEX_DF = None

def get_index_df(index_path):
    global _GLOBAL_INDEX_DF
    if _GLOBAL_INDEX_DF is None and index_path and os.path.exists(index_path):
        try:
            _GLOBAL_INDEX_DF = pd.read_csv(index_path)
            _GLOBAL_INDEX_DF['date'] = pd.to_datetime(_GLOBAL_INDEX_DF['date'])
        except Exception:
            pass
    return _GLOBAL_INDEX_DF

def _process_stock(args):
    """
    Worker function for processing a single stock.
    Must be top-level for pickling in multiprocessing.
    """
    code, name, start_date, end_date, data_dir, index_path = args
    
    index_df = get_index_df(index_path)
    
    # Re-instantiate loader (lightweight) or just use pd.read_csv
    # Ideally use DataLoader but need to ensure it's imported. It is.
    loader = DataLoader(data_dir)
    
    strong_results = []
    weak_results = []
    
    try:
        # Load data
        df = loader.get_k_data(code, start_date, end_date)
        if df.empty or len(df) < 100:
            return None, None
            
        # 1. Strong Strategies
        try:
            strong_signals = StrongStrategies.check_all_strong_strategies(df, index_df=index_df)
            if not strong_signals.empty:
                strong_signals['code'] = code
                strong_signals['name'] = name
                strong_signals['date'] = df['date'].values
                strong_results = strong_signals
        except Exception as e:
            # print(f"⚠️ {code} Strong Error: {e}")
            pass
            
        # 2. Weak Strategies
        try:
            weak_signals = WeakStrategies.check_all_weak_strategies(df)
            if not weak_signals.empty:
                weak_signals['code'] = code
                weak_signals['name'] = name
                weak_signals['date'] = df['date'].values
                weak_results = weak_signals
        except Exception as e:
            # print(f"⚠️ {code} Weak Error: {e}")
            pass
            
        return strong_results, weak_results
        
    except Exception as e:
        return None, None


class SignalCacheBuilder:
    """信号缓存构建器"""
    
    def __init__(self, data_dir=None, cache_dir=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if data_dir is None:
            self.data_dir = os.path.join(base_dir, "data", "market_data")
        else:
            self.data_dir = data_dir
            
        if cache_dir is None:
            self.cache_dir = os.path.join(base_dir, "data", "signal_cache")
        else:
            self.cache_dir = cache_dir
            
        self.loader = DataLoader(self.data_dir)
        
        # 确保缓存目录存在
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def build_all_signals(self, start_date: str = None, end_date: str = None, 
                         progress_callback=None) -> bool:
        """
        构建所有股票的信号缓存 (Multi-processing Optimized)
        """
        try:
            # 1. 获取股票列表
            stock_list = self.loader.get_stock_list()
            if stock_list.empty:
                print("❌ 股票列表为空，请先下载数据")
                return False
            
            total_stocks = len(stock_list)
            # CPU cores - 1 to leave room for UI
            max_workers = max(1, os.cpu_count() - 1)
            print(f"📊 开始加速构建缓存，共 {total_stocks} 只股票 (并行核心: {max_workers})...")
            
            # 2. 确定日期范围
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                # 默认从一年半前开始
                start_date = (datetime.now() - pd.Timedelta(days=550)).strftime("%Y-%m-%d")
            
            # 3. 准备任务参数
            index_path = os.path.join(self.data_dir, "000001.SH.csv")
            tasks = []
            for _, row in stock_list.iterrows():
                tasks.append((row['code'], row.get('name', ''), start_date, end_date, self.data_dir, index_path))
            
            strong_records = []
            weak_records = []
            
            # 5. 并行执行
            completed = 0
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Use map for order preservation if needed, or submit for control
                # Submit is better for progress bar
                futures = [executor.submit(_process_stock, t) for t in tasks]
                
                for future in concurrent.futures.as_completed(futures):
                    s_res, w_res = future.result()
                    
                    if s_res is not None and not s_res.empty:
                        strong_records.append(s_res)
                    if w_res is not None and not w_res.empty:
                        weak_records.append(w_res)
                        
                    completed += 1
                    if progress_callback:
                        if completed == 1 or completed == total_stocks or completed % 20 == 0:
                            progress_callback(completed, total_stocks, f"Processing... {completed}/{total_stocks}")
                    elif completed % 100 == 0:
                        print(f"进度: {completed}/{total_stocks}")

            # 6. 合并保存
            print("正在合并数据...")
            if strong_records:
                strong_df = pd.concat(strong_records, ignore_index=True)
                strong_path = os.path.join(self.cache_dir, "strong_signals.parquet")
                strong_df.to_parquet(strong_path, index=False, compression='snappy')
                print(f"✅ 强势信号缓存已保存: {len(strong_df)} 条记录")
            
            if weak_records:
                weak_df = pd.concat(weak_records, ignore_index=True)
                weak_path = os.path.join(self.cache_dir, "weak_signals.parquet")
                weak_df.to_parquet(weak_path, index=False, compression='snappy')
                print(f"✅ 弱势信号缓存已保存: {len(weak_df)} 条记录")
            
            # 7. 元数据
            metadata = {
                "cache_version": "1.1_multiprocess",
                "last_build_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_date_range": [start_date, end_date],
                "total_stocks": total_stocks,
                "strong_strategies": ["Z_Score", "RS", "TKOS", "DTR_Plus", "Fighting", "UA", "HMC"],
                "weak_strategies": ["HLP3", "Limit", "RSI_Rev", "Spring", "Pinbar", "Money_Flow", "UA_Weak", "DBL_VOL"]
            }
            
            metadata_path = os.path.join(self.cache_dir, "cache_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print("🎉 信号缓存构建完成！")
            return True
            
        except Exception as e:
            print(f"❌ 缓存构建失败: {e}")
            import traceback
            traceback.print_exc()
            return False



class SignalCacheReader:
    """信号缓存读取器"""
    
    def __init__(self, cache_dir=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if cache_dir is None:
            self.cache_dir = os.path.join(base_dir, "data", "signal_cache")
        else:
            self.cache_dir = cache_dir
    
    def is_cache_valid(self) -> Tuple[bool, str]:
        """
        检查缓存是否有效
        
        :return: (是否有效, 提示信息)
        """
        metadata_path = os.path.join(self.cache_dir, "cache_metadata.json")
        strong_path = os.path.join(self.cache_dir, "strong_signals.parquet")
        weak_path = os.path.join(self.cache_dir, "weak_signals.parquet")
        
        # 检查文件是否存在
        if not os.path.exists(metadata_path):
            return False, "缓存元数据不存在"
        if not os.path.exists(strong_path):
            return False, "强势信号缓存不存在"
        if not os.path.exists(weak_path):
            return False, "弱势信号缓存不存在"
        
        # 读取元数据
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            build_time = metadata.get('last_build_time', 'Unknown')
            date_range = metadata.get('data_date_range', [])
            
            return True, f"缓存有效 | 构建时间: {build_time} | 数据范围: {date_range[0]} ~ {date_range[1]}"
        except:
            return False, "缓存元数据损坏"
    
    def get_metadata(self) -> Optional[dict]:
        """获取缓存元数据"""
        metadata_path = os.path.join(self.cache_dir, "cache_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def filter_strong_stocks(self, selected_strategies: List[str], 
                            start_date: str, end_date: str) -> pd.DataFrame:
        """
        筛选符合强势策略的股票
        
        :param selected_strategies: 选中的策略列表，如 ['Z_Score', 'DTR_Plus']
        :param start_date: 筛选开始日期
        :param end_date: 筛选结束日期
        :return: 符合条件的股票DataFrame
        """
        strong_path = os.path.join(self.cache_dir, "strong_signals.parquet")
        
        if not os.path.exists(strong_path):
            raise FileNotFoundError("强势信号缓存不存在，请先构建缓存")
        
        # 读取缓存
        df = pd.read_parquet(strong_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # 日期过滤
        mask_date = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
        df = df[mask_date]
        
        # 策略过滤
        signal_cols = [f'Signal_{s}' for s in selected_strategies]
        
        # 确保所有信号列都存在
        available_cols = [col for col in signal_cols if col in df.columns]
        if not available_cols:
            return pd.DataFrame(columns=['code', 'name', 'date', 'triggered_strategies'])
        
        # 筛选：至少触发一个策略
        mask_signal = df[available_cols].any(axis=1)
        result = df[mask_signal].copy()
        
        # 添加触发的策略列表
        def get_triggered(row):
            triggered = []
            for col in available_cols:
                if row[col]:
                    triggered.append(col.replace('Signal_', ''))
            return ', '.join(triggered)
        
        result['triggered_strategies'] = result.apply(get_triggered, axis=1)
        
        # 返回关键列
        return result[['code', 'name', 'date', 'triggered_strategies'] + available_cols]
    
    def filter_weak_stocks(self, selected_strategies: List[str], 
                          start_date: str, end_date: str) -> pd.DataFrame:
        """
        筛选符合弱势策略的股票
        
        :param selected_strategies: 选中的策略列表，如 ['HLP3', 'Limit']
        :param start_date: 筛选开始日期
        :param end_date: 筛选结束日期
        :return: 符合条件的股票DataFrame
        """
        weak_path = os.path.join(self.cache_dir, "weak_signals.parquet")
        
        if not os.path.exists(weak_path):
            raise FileNotFoundError("弱势信号缓存不存在，请先构建缓存")
        
        # 读取缓存
        df = pd.read_parquet(weak_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # 日期过滤
        mask_date = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
        df = df[mask_date]
        
        # 策略过滤
        signal_cols = [f'Signal_{s}' for s in selected_strategies]
        
        # 确保所有信号列都存在
        available_cols = [col for col in signal_cols if col in df.columns]
        if not available_cols:
            return pd.DataFrame(columns=['code', 'name', 'date', 'triggered_strategies'])
        
        # 筛选：至少触发一个策略
        mask_signal = df[available_cols].any(axis=1)
        result = df[mask_signal].copy()
        
        # 添加触发的策略列表
        def get_triggered(row):
            triggered = []
            for col in available_cols:
                if row[col]:
                    triggered.append(col.replace('Signal_', ''))
            return ', '.join(triggered)
        
        result['triggered_strategies'] = result.apply(get_triggered, axis=1)
        
        # 返回关键列
        return result[['code', 'name', 'date', 'triggered_strategies'] + available_cols]
    def get_stock_signals(self, code: str) -> pd.DataFrame:
        """
        获取单个股票的所有缓存信号
        
        :param code: 股票代码
        :return: 包含该股票所有信号的DataFrame (Strong + Weak), 索引为日期
        """
        strong_path = os.path.join(self.cache_dir, "strong_signals.parquet")
        weak_path = os.path.join(self.cache_dir, "weak_signals.parquet")
        
        strong_df = pd.DataFrame()
        weak_df = pd.DataFrame()
        
        # Helper to read filtered
        # Note: 'code' column must exist in parquet for filtering
        def read_filtered(path):
            if not os.path.exists(path):
                return pd.DataFrame()
            try:
                # Try reading with filters (requires pyarrow/fastparquet and partition/or dict check)
                # If code is a column, this works.
                # Cast code to str to be safe if parquet stored as str
                return pd.read_parquet(path, filters=[('code', '==', str(code))])
            except Exception:
                # Fallback: Read all and filter (Slower but safe if filters fail or engine missing)
                try:
                    df_all = pd.read_parquet(path)
                    return df_all[df_all['code'] == str(code)]
                except:
                    return pd.DataFrame()

        strong_df = read_filtered(strong_path)
        weak_df = read_filtered(weak_path)
        
        # Merge
        # Both should have 'date' and 'code'
        if strong_df.empty and weak_df.empty:
            return pd.DataFrame()
            
        # Process Strong
        if not strong_df.empty:
            strong_df['date'] = pd.to_datetime(strong_df['date'])
            strong_df.set_index('date', inplace=True)
            # Keep only Signal columns
            s_cols = [c for c in strong_df.columns if c.startswith('Signal_')]
            strong_df = strong_df[s_cols]
        
        # Process Weak
        if not weak_df.empty:
            weak_df['date'] = pd.to_datetime(weak_df['date'])
            weak_df.set_index('date', inplace=True)
            w_cols = [c for c in weak_df.columns if c.startswith('Signal_')]
            weak_df = weak_df[w_cols]
            
        # Join
        # Outer join to keep all dates
        full_df = pd.concat([strong_df, weak_df], axis=1)
        
        # Remove duplicate columns if any (unlikely unless strategy name collision)
        full_df = full_df.loc[:, ~full_df.columns.duplicated()]
        
        # Fill NaN with False (if a date exists in one but not other)
        full_df = full_df.fillna(False)
        
        return full_df

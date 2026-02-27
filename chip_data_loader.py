"""
筹码分布数据加载器
使用 akshare 获取股票的获利盘比例数据，支持本地缓存
"""

import pandas as pd
import numpy as np
import akshare as ak
import os
import datetime
from pathlib import Path


class ChipDataLoader:
    """筹码分布数据加载器"""
    
    # 缓存目录
    CACHE_DIR = Path(__file__).parent / "data" / "chip_cache"
    
    @staticmethod
    def _ensure_cache_dir():
        """确保缓存目录存在"""
        ChipDataLoader.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _get_cache_path(code):
        """获取股票的缓存文件路径"""
        ChipDataLoader._ensure_cache_dir()
        return ChipDataLoader.CACHE_DIR / f"{code}_chip.csv"
    
    @staticmethod
    def _is_cache_valid(cache_path, max_age_hours=24):
        """
        检查缓存是否有效
        
        :param cache_path: 缓存文件路径
        :param max_age_hours: 最大缓存年龄（小时）
        :return: 缓存是否有效
        """
        if not cache_path.exists():
            return False
        
        # 检查文件修改时间
        mtime = datetime.datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.datetime.now() - mtime
        
        return age.total_seconds() < max_age_hours * 3600
    
    @staticmethod
    def get_chip_data(code, use_cache=True, force_refresh=False):
        """
        获取单只股票的筹码分布数据
        
        :param code: 股票代码（6位数字，如 '000001'）
        :param use_cache: 是否使用缓存
        :param force_refresh: 是否强制刷新（忽略缓存）
        :return: DataFrame with columns ['date', 'winner_pct']
                 如果获取失败，返回空DataFrame
        """
        cache_path = ChipDataLoader._get_cache_path(code)
        
        # 尝试使用缓存
        if use_cache and not force_refresh and ChipDataLoader._is_cache_valid(cache_path):
            try:
                df = pd.read_csv(cache_path)
                df['date'] = pd.to_datetime(df['date'])
                return df
            except Exception:
                pass  # 缓存读取失败，继续尝试网络获取
        
        # 从 akshare 获取数据
        try:
            # akshare的stock_cyq_em接口
            df = ak.stock_cyq_em(symbol=code, adjust='')
            
            if df.empty:
                return pd.DataFrame(columns=['date', 'winner_pct'])
            
            # 数据清洗和转换
            result = pd.DataFrame()
            result['date'] = pd.to_datetime(df['日期'])
            # 获利比例从0-1转换为0-100
            result['winner_pct'] = df['获利比例'] * 100
            
            # 保存到缓存
            if use_cache:
                try:
                    result.to_csv(cache_path, index=False)
                except Exception:
                    pass  # 缓存保存失败不影响主流程
            
            return result
            
        except Exception as e:
            # 网络请求失败或其他异常
            # 返回空DataFrame
            return pd.DataFrame(columns=['date', 'winner_pct'])
    
    @staticmethod
    def merge_with_kline(kline_df, chip_df):
        """
        将筹码数据合并到K线数据中
        
        :param kline_df: K线数据，必须包含 'date' 列
        :param chip_df: 筹码数据，包含 ['date', 'winner_pct']
        :return: 合并后的DataFrame
        """
        if chip_df.empty:
            # 如果筹码数据为空，添加空列
            kline_df = kline_df.copy()
            kline_df['winner_pct'] = np.nan
            return kline_df
        
        # 按日期合并，左连接保留所有K线数据
        merged = pd.merge(kline_df, chip_df, on='date', how='left')
        
        # 对于缺失的获利盘数据，可以选择：
        # 1. 保留NaN
        # 2. 前向填充（假设获利盘变化缓慢）
        # 3. 后向填充
        # 这里选择前向填充
        merged['winner_pct'] = merged['winner_pct'].fillna(method='ffill')
        
        return merged
    
    @staticmethod
    def batch_get_chip_data(codes, progress_callback=None, use_cache=True):
        """
        批量获取多只股票的筹码数据
        
        :param codes: 股票代码列表
        :param progress_callback: 进度回调函数 callback(current, total, code)
        :param use_cache: 是否使用缓存
        :return: dict {code: chip_df}
        """
        results = {}
        total = len(codes)
        
        for idx, code in enumerate(codes):
            if progress_callback:
                progress_callback(idx + 1, total, code)
            
            chip_df = ChipDataLoader.get_chip_data(code, use_cache=use_cache)
            results[code] = chip_df
            
            # 添加小延迟，避免请求过快
            if not use_cache or chip_df.empty:
                import time
                time.sleep(0.5)  # 500ms延迟
        
        return results
    
    @staticmethod
    def clear_cache(code=None):
        """
        清除缓存
        
        :param code: 股票代码，如果为None则清除所有缓存
        """
        if code:
            cache_path = ChipDataLoader._get_cache_path(code)
            if cache_path.exists():
                cache_path.unlink()
        else:
            # 清除所有缓存
            if ChipDataLoader.CACHE_DIR.exists():
                for cache_file in ChipDataLoader.CACHE_DIR.glob("*.csv"):
                    cache_file.unlink()
    
    @staticmethod
    def get_cache_info():
        """
        获取缓存信息
        
        :return: dict with cache statistics
        """
        if not ChipDataLoader.CACHE_DIR.exists():
            return {
                'total_files': 0,
                'total_size_mb': 0,
                'oldest_cache': None,
                'newest_cache': None
            }
        
        cache_files = list(ChipDataLoader.CACHE_DIR.glob("*.csv"))
        
        if not cache_files:
            return {
                'total_files': 0,
                'total_size_mb': 0,
                'oldest_cache': None,
                'newest_cache': None
            }
        
        total_size = sum(f.stat().st_size for f in cache_files)
        mtimes = [datetime.datetime.fromtimestamp(f.stat().st_mtime) for f in cache_files]
        
        return {
            'total_files': len(cache_files),
            'total_size_mb': total_size / 1024 / 1024,
            'oldest_cache': min(mtimes),
            'newest_cache': max(mtimes)
        }


# 使用示例
if __name__ == "__main__":
    # 测试单只股票
    print("测试获取 000001 的筹码数据...")
    df = ChipDataLoader.get_chip_data('000001')
    print(f"获取到 {len(df)} 条记录")
    if not df.empty:
        print(df.head())
        print(f"\n获利盘范围: {df['winner_pct'].min():.2f}% - {df['winner_pct'].max():.2f}%")
    
    # 测试缓存
    print("\n测试缓存...")
    cache_info = ChipDataLoader.get_cache_info()
    print(f"缓存文件数: {cache_info['total_files']}")
    print(f"缓存大小: {cache_info['total_size_mb']:.2f} MB")

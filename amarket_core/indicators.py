import pandas as pd
import numpy as np

def calculate_ema(series, span=200):
    """Calculates Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()

def calculate_volatility(df, window=20):
    """
    Calculates volatility using standard deviation of log returns 
    or just ATR-like range. Here we use std dev of percent change.
    """
    if 'pctChg' not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    
    # Rolling standard deviation of daily percentage change
    return df['pctChg'].rolling(window=window).std()

def detect_volatility_contraction(vol_series, lookback=60, threshold_quantile=0.2):
    """
    Detects if current volatility is in the lowest quantile of the lookback period.
    Returns: (is_contracting, current_vol_rank_percentile)
    """
    if len(vol_series) < lookback:
        return False, 1.0
        
    recent_window = vol_series.iloc[-lookback:]
    current_vol = vol_series.iloc[-1]
    
    # Calculate percentile of current vol within the recent window
    rank = (recent_window < current_vol).mean()
    
    is_contracting = rank <= threshold_quantile
    return is_contracting, rank

def calculate_relative_strength(series_a, series_b, window=20):
    """
    Calculates Relative Strength of A divided by B.
    Returns the ratio series and its trend (SMA of ratio).
    """
    # Ensure aligned dates
    common_idx = series_a.index.intersection(series_b.index)
    a = series_a.loc[common_idx]
    b = series_b.loc[common_idx]
    
    # Avoid division by zero
    ratio = a / b.replace(0, np.nan)
    
    # Normalize start to 100 for better visualization
    if not ratio.empty:
        ratio = ratio / ratio.iloc[0] * 100
        
    ratio_trend = ratio.rolling(window=window).mean()
    
    return ratio, ratio_trend

def calculate_nhr(df_list, current_date, lookback=250):
    """
    Calculate New High / New Low Ratio.
    df_list: list of dataframes for individual stocks.
    This is computationally expensive if fetching real-time on client side.
    Usually requires a pre-computed database.
    
    Simplified proxy: Use Index Advance/Decline or simple price position check.
    Here we define a simple placeholder that returns 0.5 (neutral) 
    unless we implement full market scan.
    """
    # TODO: Implement full market scan or use cached aggregate data/proxy
    return 0.5

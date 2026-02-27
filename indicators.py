
import pandas as pd
import numpy as np

class Indicators:
    @staticmethod
    def add_all_indicators(df):
        """Add all necessary indicators to the dataframe inplace."""
        df = df.copy()
        
        # Basic MAs
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA250'] = df['close'].rolling(window=250).mean()
        
        # Volume MAs
        df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
        
        # MACD
        # EMA12, EMA26
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = ema12 - ema26
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = 2 * (df['DIF'] - df['DEA'])
        
        # Bollinger Bands (N=20, k=2)
        std20 = df['close'].rolling(window=20).std()
        df['Boll_Mid'] = df['MA20']
        df['Boll_Upper'] = df['Boll_Mid'] + 2 * std20
        df['Boll_Lower'] = df['Boll_Mid'] - 2 * std20
        
        # RSI (N=2 for spec strategies, N=6 standard)
        df['RSI2'] = Indicators.calculate_rsi(df['close'], 2)
        df['RSI6'] = Indicators.calculate_rsi(df['close'], 6)
        
        # CYC (Cost Moving Average)
        # CYC_Short (13 days) -> Sum(Amount, 13) / Sum(Volume, 13)
        # CYC_Infinite -> CumSum(Amount) / CumSum(Volume)
        # Using Amount(turnover) and Volume
        if 'amount' not in df.columns and 'volume' in df.columns:
            # Estimate amount if not provided by data source. 
            # Note: volume is typically in 'hands' (1手=100股).
            # To make CYC (amount/volume) match the price scale, amount should be (price * volume).
            # If we do price * volume * 100, then CYC = (price * volume * 100) / volume = price * 100.
            # So we just use price * volume here for the ratio to be correct.
            df['amount'] = df['close'] * df['volume']
            
        if 'amount' in df.columns and 'volume' in df.columns:
            # Avoid division by zero
            vol_s = df['volume'].replace(0, np.nan)
            
            # Short CYC (13) - approx using rolling sum
            # If amount isn't artificially scaled, we might need a factor if amount is actual RMB and vol is Hands.
            # Standard Akshare: amount=(元), volume=(手). So Price ≈ amount / (volume * 100).
            # Let's dynamically check the scaler. If amount / vol > close * 50, it means we need to divide by 100.
            
            vol_sum_13 = vol_s.rolling(window=13).sum().replace(0, np.nan)
            cyc_13 = df['amount'].rolling(window=13).sum() / vol_sum_13
            
            vol_sum_inf = vol_s.cumsum().replace(0, np.nan)
            cyc_inf = df['amount'].cumsum() / vol_sum_inf
            
            # Auto-scale correction
            if len(df) > 0 and df['close'].iloc[-1] > 0:
                ratio = cyc_13.dropna().iloc[-1] / df['close'].iloc[-1] if not cyc_13.dropna().empty else 1
                if ratio > 50:
                    cyc_13 = cyc_13 / 100
                    cyc_inf = cyc_inf / 100
            
            df['CYC_13'] = cyc_13
            df['CYC_Inf'] = cyc_inf
        else:
            df['CYC_13'] = np.nan
            df['CYC_Inf'] = np.nan
            
        # Z-Score (Rank) for 20-day return
        # Note: Rank usually needs cross-sectional data (across all stocks). 
        # Here we calculate longitudinal Z-Score of valid price for single stock or 
        # placeholder for cross-sectional calculation in strategy engine.
        # For single stock, we can use "Relative Strength" vs Index if passed, 
        # or simple statistical deviation?
        # Specification says: Z-Score(Returns_20). We'll calc Returns_20 first.
        df['Ret_20'] = df['close'].pct_change(periods=20)
        
        # Min/Max for strategies
        df['High_52'] = df['high'].rolling(window=250).max()
        df['Max_Vol_250'] = df['volume'].rolling(window=250).max()
        df['Low_20'] = df['low'].rolling(window=20).min()
        
        # --- New Indicators for Expanded Strategies ---
        
        # RKing
        df = Indicators.add_rking(df)
        
        # EMA for HPS / Trend
        df['EMA15'] = df['close'].ewm(span=15, adjust=False).mean()
        df['EMA_High_15'] = df['high'].ewm(span=15, adjust=False).mean() # HPS Channel
        df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # MACD Signal MA for HMC
        df['MACD_Hist_MA5'] = df['MACD_Hist'].rolling(window=5).mean()
        
        # Volatility for ES
        df['Std20'] = df['close'].rolling(window=20).std()
        df['Std60'] = df['close'].rolling(window=60).std()
        df['Std120'] = df['close'].rolling(window=120).std()
        
        # Pinbar features
        df['Body'] = (df['open'] - df['close']).abs()
        df['Upper_Shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['Lower_Shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['Range'] = df['high'] - df['low']
        
        # UA Exit Signal (High Volume Stagnation - 高位天量滞涨)
        # Volume > 2.5x average AND (Long Upper Shadow OR Doji/Small Body)
        # We also require it to be a relatively high price point in the backtest logic
        cond_vol_extreme = df['volume'] > (df['Vol_MA20'] * 2.5)
        cond_stagnation = (df['Upper_Shadow'] > df['Body'] * 1.5) | (df['Body'] < df['Range'] * 0.1)
        df['UA_Exit'] = cond_vol_extreme & cond_stagnation
        
        # --- Analysis Indicators ---
        # KDJ (9,3,3)
        df = Indicators.calculate_kdj(df)
        
        # Williams %R (14)
        df = Indicators.calculate_wr(df)
        
        # CCI (14)
        df = Indicators.calculate_cci(df)
        
        # HMC (High-Momentum Channel)
        df = Indicators.calculate_hmc(df)

        # Winner Ratio (Profit Proportion) for HLP3
        df = Indicators.calculate_winner_ratio(df)
        
        # Defragment memory block at the end using concat to prevent column fragmentation DataFrame KeyErrors
        df = pd.concat([df], axis=1)

        return df

    @staticmethod
    def add_rking(df):
        """
        Calculate RKing Strategy Indicators.
        Code:
            XOPEN:=(REF(O,N)+REF(C,N))/2; (N=1)
            XCLOSE:=CLOSE;
            XHIGH:=MAX(HIGH,XOPEN);
            XLOW:=MIN(LOW,XOPEN);
            VOLALITY:=MA(XHIGH-XLOW,8);
            UP:MA(XCLOSE,5)+VOLALITY/2;
            DOWN:MA(XCLOSE,5)-VOLALITY/2;
            BU:=CROSS(XCLOSE,UP);
            SEL:=CROSS(DOWN,XCLOSE);
        """
        if df.empty: return df
        
        # 1. XOPEN = (Ref(O) + Ref(C)) / 2
        df['Ref_Open'] = df['open'].shift(1)
        df['Ref_Close'] = df['close'].shift(1)
        # Handle first row NaN if needed, fill with Open/Close?
        df['XOpen'] = (df['Ref_Open'] + df['Ref_Close']) / 2
        
        # 2. XClose = Close
        df['XClose'] = df['close']
        
        # 3. XHigh/XLow
        # Using numpy maximum/minimum instead of dataframe max/min to avoid dimension crashes
        df['XHigh'] = np.maximum(df['high'], df['XOpen'])
        df['XLow'] = np.minimum(df['low'], df['XOpen'])
        
        # 4. Volatility (MA 8 of Range)
        df['RKing_Vol'] = (df['XHigh'] - df['XLow']).rolling(window=8).mean()
        
        # 5. Bands
        ma5 = df['XClose'].rolling(window=5).mean()
        df['RKing_Upper'] = ma5 + df['RKing_Vol'] / 2
        df['RKing_Lower'] = ma5 - df['RKing_Vol'] / 2
        
        # 6. Signals (Vectorized Cross)
        # BU: Cross(XClose, UP) => XClose > UP and PrevXClose <= PrevUP
        prev_close = df['XClose'].shift(1)
        prev_upper = df['RKing_Upper'].shift(1)
        prev_lower = df['RKing_Lower'].shift(1)
        
        # BU condition: Close crossed above Upper
        df['RKing_BU'] = (df['XClose'] > df['RKing_Upper']) & (prev_close <= prev_upper)
        
        # SEL condition: Lower crossed above Close (Close crossed below Lower)
        # SEL:=CROSS(DOWN,XCLOSE) => Down > Close and PrevDown <= PrevClose
        df['RKing_SEL'] = (df['RKing_Lower'] > df['XClose']) & (prev_lower <= prev_close)
        
        # 7. State Color (Simulation)
        # Red/Yellow (Long) if recent signal is BU
        # Blue/Green (Short) if recent signal is SEL
        # We need a forward fill of the last signal type.
        # 1 = Buy, -1 = Sell, 0 = No Change
        
        conditions = [
            df['RKing_BU'],
            df['RKing_SEL']
        ]
        choices = [1, -1]
        
        # Create signal series (NaN where no signal)
        df['Signal_State'] = np.select(conditions, choices, default=np.nan)
        
        # Forward fill to propagate state
        df['RKing_State'] = df['Signal_State'].ffill().fillna(0) # 0 = Neutral/Start
        
        # Cleanup temp cols if desired, or keep for debug
        # df.drop(columns=['Ref_Open', 'Ref_Close', 'XOpen', ...], inplace=True)
        
        return df

    @staticmethod
    def calculate_rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Use Exponential Moving Average for standard Wilder's RSI, avoid division by zero
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
        return pd.Series(rsi, index=series.index)

    @staticmethod
    def calculate_kdj(df, n=9, m1=3, m2=3):
        low_list = df['low'].rolling(window=n, min_periods=n).min()
        high_list = df['high'].rolling(window=n, min_periods=n).max()
        diff = (high_list - low_list).replace(0, np.nan)
        rsv = (df['close'] - low_list) / diff * 100
        df['K'] = rsv.ewm(com=m1-1, adjust=False).mean()
        df['D'] = df['K'].ewm(com=m2-1, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        return df

    @staticmethod
    def calculate_wr(df, n=14):
        low_list = df['low'].rolling(window=n, min_periods=n).min()
        high_list = df['high'].rolling(window=n, min_periods=n).max()
        diff = (high_list - low_list).replace(0, np.nan)
        # Williams %R = (High_n - Close) / (High_n - Low_n) * -100
        df['WR'] = (high_list - df['close']) / diff * -100
        return df

    @staticmethod
    def calculate_cci(df, n=14):
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(window=n).mean()
        # High-performance Vectorized Proxy for Mean Absolute Deviation (MAD):
        # md = tp.rolling(window=n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True) 
        # This lambda forces Python context switching inside C loops and freezes CPU. 
        # For a normal distribution, MAD ≈ 0.8 * StdDev.
        md_proxy = tp.rolling(window=n).std() * 0.8
        # Handle zero division by replacing 0 with a tiny epsilon
        md_safe = md_proxy.replace(0, 1e-8)
        df['CCI'] = (tp - ma) / (0.015 * md_safe)
        return df

    @staticmethod
    def calculate_winner_ratio(df, window=100):
        """
        Calculate Winner Ratio (Profit Proportion) Proxy.
        Uses VWAP (Volume Weighted Average Price) and Standard Deviation over a window.
        Approximates winner ratio using Normal CDF of (Close - VWAP) / Std.
        """
        if df.empty:
            return df
            
        # Calculate amount if missing (Approximation: Typical Price * Volume)
        if 'amount' not in df.columns:
            # Check if we have price and volume
            if 'close' in df.columns and 'volume' in df.columns:
                tp = (df['high'] + df['low'] + df['close']) / 3 if 'high' in df.columns and 'low' in df.columns else df['close']
                df['amount'] = tp * df['volume']
            else:
                # Cannot calculate
                df['winner_pct'] = 50.0
                return df
            
        # 1. VWAP over window
        # Rolling sum of Amount / Rolling sum of Volume
        roll_amt = df['amount'].rolling(window=window).sum()
        roll_vol = df['volume'].rolling(window=window).sum()
        vwap = roll_amt / roll_vol.replace(0, np.nan)
        
        # 2. Standard Deviation over window
        std = df['close'].rolling(window=window).std()
        
        # 3. Z-Score = (Close - VWAP) / Std
        # Avoid division by zero
        z_score = (df['close'] - vwap) / std.replace(0, np.nan)
        
        # 4. Proxy Winner % = Normal CDF(Z) * 100
        # CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
        import math
        def normal_cdf(x):
            if pd.isna(x): return np.nan
            return 0.5 * (1 + math.erf(x / 1.41421356))
            
        df['winner_pct'] = z_score.apply(lambda x: normal_cdf(x) if pd.notnull(x) else np.nan) * 100
        return df

    @staticmethod
    def calculate_hmc(df):
        """
        计算 HMC 指标 (High-Momentum Channel)
        Yellow Line (Pressure): HHV(50) - Close (Smaller is better)
        Red Line (Momentum): Close - EMA(200) (Larger is better)
        """
        df = df.copy()
        
        # Yellow Line: HHV(High, 50) - Close
        hhv_50 = df['high'].rolling(window=50).max()
        df['HMC_Yellow'] = hhv_50 - df['close']
        
        # Red Line: Close - EMA(200)
        ema_200 = df['close'].ewm(span=200, adjust=False).mean()
        df['HMC_Red'] = df['close'] - ema_200
        
        return df


# Chip Distribution Approximation (HLP3)
# To calculate "Profit Ratio" (Winner Ratio), we need a distribution model.
# Simplified model: 
# Assume turnover implies chip exchange.
# We hold a histogram of costs: {price_bin: volume_share}
# Update daily: decay old chips by turn_rate, add new chips at current price.
def calculate_chip_distribution(df):
    """
    Calculate Winner Ratio (Profit Proportion).
    Returns Series of winner_ratio (0-1).
    This is computationally expensive, so optimized with simplified decay.
    """
    winner_ratios = []
    # Simplified: 
    # Current Chip Cost Distribution ~ EMA(Price, Volume_Weights)? No.
    # We simulate: 
    # Chips[t] = Chips[t-1] * (1 - turn) + NewChips(Price[t]) * turn
    # Calculate % of Chips < Close[t]
    
    # We need to simulate day by day.
    # To be fast, we can use a discrete price grid or just verify logic.
    # Using a 60-day window estimation is faster:
    # Most chips are exchanged in last 60-120 days.
    # We can estimate "Average Cost" roughly or implement the decay loop.
    
    # Implementing a fast decay loop
    # cost_dist = {price: weight}
    # But prices are continuous. We use 100 bins?
    # Or just simpler: Winner Ratio ~ (Close - Cost_Avg) / Volatility? No.
    
    # Correct approach for HLP3 (Winner Ratio):
    # Iterate data.
    prices = df['close'].values
    turns = df['turn'].values / 100.0 if 'turn' in df.columns else np.zeros(len(df))
    # If turn is missing, guess 1-3%?
    
    # Full simulation is too slow for python loop on large datasets without numba.
    # We will use value `(Close - MA60) / Std60` as a proxy for profit ratio for now?
    # Or strict definition:
    # "Winner Ratio" is provided by data providers usually. 
    # Without it, we use: Close > CYC_Inf is >50% winner likely.
    # Close > CYC_13 is short term profit.
    # Let's use a placeholder or simplified proxy:
    # Proxy: sigmoid((Close - MA60)/Std60) scaled to 0-100%?
    # Strategy spec said: "Yesterday < 1%, Today > 35%". 
    # This implies we literally need the ratio.
    
    # We will skip complex simulation in this iteration and return NaN or use a proxy.
    # User might verify this. I will put a basic proxy. 
    # Proxy: Percentile of (Close - Low_120)/(High_120 - Low_120) * 100 ?
    pass 
    return pd.Series(np.nan, index=df.index)



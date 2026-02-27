"""
Royal 强势股进攻策略模块
实现 strong.md 中定义的7种强势股筛选策略
"""

import pandas as pd
import numpy as np


class StrongStrategies:
    """强势股进攻策略集合"""
    
    @staticmethod
    def calculate_z_score(df, period=20):
        """
        计算 Z-score 指标
        
        核心逻辑：(收盘价 - 20日均价) / 20日标准差
        强势标准：Z > 1.5
        过热预警：Z > 3
        
        :param df: 包含 'close' 的 DataFrame
        :param period: 周期，默认为20
        :return: 包含 Z-score 及其信号的 DataFrame
        """
        df = df.copy()
        
        df['MA20'] = df['close'].rolling(window=period).mean()
        df['STD20'] = df['close'].rolling(window=period).std()
        df['MA60'] = df['close'].rolling(window=60).mean()
        df['Vol_MA5'] = df['volume'].rolling(window=5).mean() if 'volume' in df.columns else 0
        
        # 计算 Z-score (防止除以0)
        df['Z_Score'] = (df['close'] - df['MA20']) / df['STD20'].replace(0, np.nan)
        
        # 生成信号
        # 强势信号：1.5 < Z <= 3 以及 顺势且放量突破
        base_signal = (df['Z_Score'] > 1.5) & (df['Z_Score'] <= 3)
        trend_filter = df['close'] > df['MA60']
        
        if 'volume' in df.columns:
            vol_filter = df['volume'] > (df['Vol_MA5'] * 1.5)
        else:
            vol_filter = True
            
        df['Z_Signal'] = np.where(base_signal & trend_filter & vol_filter, True, False)
        
        # 过热预警：Z > 3
        df['Z_Overheat'] = np.where(df['Z_Score'] > 3, True, False)
        
        return df
    
    @staticmethod
    def calculate_rs_strategy(stock_df, index_df, period=20, num_std=2):
        """
        计算 RS 相对强弱策略
        
        核心逻辑：(个股收盘 / 大盘收盘) * 1000，并叠加布林带（N=20, Std=2）
        信号：RS 突破 RS布林上轨
        
        :param stock_df: 个股 DataFrame (需包含 'date', 'close')
        :param index_df: 大盘指数 DataFrame (需包含 'date', 'close')
        :param period: 布林带周期，默认20
        :param num_std: 布林带标准差倍数，默认2
        :return: 包含 RS 及其布林带信号的 DataFrame
        """
        stock_df = stock_df.copy()
        
        # 确保两个df都有date列
        if 'date' not in stock_df.columns or 'date' not in index_df.columns:
            # 如果没有date列，返回空信号
            stock_df['RS_Breakout'] = False
            return stock_df
        
        # 按日期合并股票和大盘数据
        merged = pd.merge(
            stock_df[['date', 'close']],
            index_df[['date', 'close']],
            on='date',
            how='left',
            suffixes=('_stock', '_index')
        )
        
        # 前向填充大盘数据（处理缺失日期）
        merged['close_index'] = merged['close_index'].fillna(method='ffill')
        
        # 1. 计算 RS 值 (乘以1000方便显示)
        merged['RS'] = (merged['close_stock'] / merged['close_index']) * 1000
        
        # 2. 计算 RS 的布林带
        merged['RS_MA'] = merged['RS'].rolling(window=period).mean()
        merged['RS_STD'] = merged['RS'].rolling(window=period).std()
        merged['RS_Upper'] = merged['RS_MA'] + (merged['RS_STD'] * num_std)
        merged['RS_Lower'] = merged['RS_MA'] - (merged['RS_STD'] * num_std)
        
        # 2.5 计算个股趋势和量能过滤
        merged['MA60_stock'] = merged['close_stock'].rolling(window=60).mean()
        if 'volume' in stock_df.columns:
            merged['volume'] = stock_df['volume'].values
            merged['Vol_MA5'] = merged['volume'].rolling(window=5).mean()
        
        # 3. 信号：RS 突破 RS布林上轨 且处于多头排列并放量
        # 当日RS > 上轨 且 前一日RS <= 前一日上轨（避免未来函数）
        base_signal = (merged['RS'] > merged['RS_Upper']) & \
                                (merged['RS'].shift(1) <= merged['RS_Upper'].shift(1))
        trend_filter = merged['close_stock'] > merged['MA60_stock']
        
        if 'volume' in stock_df.columns:
            vol_filter = merged['volume'] > (merged['Vol_MA5'] * 1.5)
        else:
            vol_filter = True
            
        merged['RS_Breakout'] = base_signal & trend_filter & vol_filter
        
        # 将结果合并回原始df
        stock_df['RS_Breakout'] = merged['RS_Breakout'].values
        
        return stock_df
    
    @staticmethod
    def calculate_tkos(df):
        """
        计算 TKOS 股王策略
        
        核心逻辑：检测某个月的前5个交易日(近似第一周)累计涨幅是否超过50%
        信号：月涨幅 > 50%
        
        :param df: 包含 'close' 的 DataFrame
        :return: 包含 TKOS 信号的 DataFrame
        """
        df = df.copy()
        
        # 计算20日累计涨幅 (近似一月)
        # (当前收盘 - 20天前收盘) / 20天前收盘
        df['Month_Pct_Change'] = df['close'].pct_change(periods=20)
        
        # 信号：月涨幅 > 50%
        df['TKOS_Signal'] = df['Month_Pct_Change'] > 0.50
        
        return df
    
    @staticmethod
    def calculate_dtr_plus(df, ma_period=20, boll_std=2):
        """
        计算 DTR Plus 策略（高胜率共振）
        
        核心逻辑：MACD翻红 + 价格 > MA20 + 价格 >= 布林上轨
        三合一共振信号
        
        :param df: 包含 OHLC 和 volume 的 DataFrame
        :param ma_period: MA周期，默认20
        :param boll_std: 布林带标准差倍数，默认2
        :return: 包含 DTR Plus 信号的 DataFrame
        """
        df = df.copy()
        
        # 1. 计算 MACD (12, 26, 9)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = 2 * (diff - dea)
        
        # DTR翻红信号 (当前红柱，昨日绿柱)
        df['DTR_Red'] = (df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0)
        
        # 2. 计算 MA20
        df['MA20'] = df['close'].rolling(window=ma_period).mean()
        
        # 3. 计算布林上轨
        std20 = df['close'].rolling(window=ma_period).std()
        df['Boll_Upper'] = df['MA20'] + (std20 * boll_std)
        
        # 4. 计算 MA60 和 量能
        df['MA60'] = df['close'].rolling(window=60).mean()
        if 'volume' in df.columns:
            df['Vol_MA5'] = df['volume'].rolling(window=5).mean()
            
        # 5. 综合信号 (三合一 + 顺势暴量)
        # MACD是红柱状态，价格在MA20之上，价格触碰或突破上轨，在MA60之上，且激增1.5倍量
        condition1 = df['MACD_Hist'] > 0
        condition2 = df['close'] > df['MA20']
        condition3 = df['close'] >= df['Boll_Upper']
        condition4 = df['close'] > df['MA60']
        
        if 'volume' in df.columns:
            condition5 = df['volume'] > (df['Vol_MA5'] * 1.5)
        else:
            condition5 = True
            
        df['DTR_Plus_Signal'] = condition1 & condition2 & condition3 & condition4 & condition5
        
        return df
    
    @staticmethod
    def calculate_fighting_strategy(df, period=52):
        """
        计算 Fighting 策略 (三合一突破)
        
        核心逻辑：DTR翻红 + 突破52日价格新高 + 突破52日成交量新高
        
        :param df: 包含 OHLC 和 volume 的 DataFrame
        :param period: 新高周期，默认52日
        :return: 包含 Fighting 信号的 DataFrame
        """
        df = df.copy()
        
        # 1. MACD DTR
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = 2 * (diff - dea)
        
        # DTR 红柱状态
        is_dtr_red = df['MACD_Hist'] > 0
        
        # 2. 52日价格新高 (突破前52天的最高价)
        highest_price_52 = df['high'].rolling(window=period).max().shift(1)
        price_breakout = df['close'] > highest_price_52
        
        # 3. 52日成交量新高
        highest_vol_52 = df['volume'].rolling(window=period).max().shift(1)
        vol_breakout = df['volume'] > highest_vol_52
        
        # 4. Fighting 信号
        df['Fighting_Signal'] = is_dtr_red & price_breakout & vol_breakout
        
        return df
    
    @staticmethod
    def calculate_ua_strategy(df, period=250):
        """
        计算 UA 天量策略 (Ultimate Amount)
        
        核心逻辑：出现历史(或250日)天量，标记该日最高价。
        后续突破该最高价为买点。
        
        :param df: 包含 OHLC 和 volume 的 DataFrame
        :param period: 天量检测周期，默认250日
        :return: 包含 UA 信号的 DataFrame
        """
        df = df.copy()
        
        # 1. 定义天量 (250日内最大成交量)
        df['Rolling_Max_Vol'] = df['volume'].rolling(window=period).max()
        df['Is_UA'] = df['volume'] == df['Rolling_Max_Vol']
        
        # 2. 记录天量日的最高价 (UA_High)
        # 如果是UA日，记录High，否则NaN，然后向下填充
        df['UA_Target_Price'] = np.where(df['Is_UA'], df['high'], np.nan)
        df['UA_Target_Price'] = df['UA_Target_Price'].ffill()
        
        # 3. 突破信号
        # 当前收盘价突破最近一次天量的最高价
        # 且当前不是天量当日 (避免当日追高)
        df['UA_Breakout'] = (df['close'] > df['UA_Target_Price']) & (df['Is_UA'] == False)
        
        # 过滤连续信号：只看刚突破的那一天
        df['UA_Buy_Signal'] = df['UA_Breakout'] & (df['UA_Breakout'].shift(1) == False)
        
        return df
    
    @staticmethod
    def calculate_hmc_strategy(df):
        """
        计算 HMC 策略 (High-Momentum Channel)
        
        核心逻辑：
        - 黄线 = 50日最高价 - 收盘价 (越小越好，代表接近新高)
        - 红线 = 收盘价 - EMA200 (越大越好，代表强势)
        - 信号 = 红线上穿黄线 (动能强劲)
        
        :param df: 包含 OHLC 的 DataFrame
        :return: 包含 HMC 信号的 DataFrame
        """
        df = df.copy()
        
        # 1. 黄线: 50日最高价 - 收盘价
        hhv_50 = df['high'].rolling(window=50).max()
        df['HMC_Yellow'] = hhv_50 - df['close']
        
        # 2. 红线: 收盘价 - EMA200
        ema_200 = df['close'].ewm(span=200, adjust=False).mean()
        df['HMC_Red'] = df['close'] - ema_200
        
        # 3. 信号: 红线上穿黄线 且 红线 > 0 (股价在年线之上)
        # 今天红 > 黄 且 昨天 红 <= 黄 且 今天红 > 0
        df['HMC_Signal'] = (df['HMC_Red'] > df['HMC_Yellow']) & \
                           (df['HMC_Red'].shift(1) <= df['HMC_Yellow'].shift(1)) & \
                           (df['HMC_Red'] > 0)
        
        return df
    
    @staticmethod
    def calculate_cyc_max(df):
        """CYC MAX (成本突破)"""
        df = df.copy()
        if 'CYC_Inf' in df.columns and 'CYC_13' in df.columns:
            df['CYC_MAX_Signal'] = (df['close'] > df['CYC_Inf']) & (df['close'] > df['CYC_13'])
        else:
            df['CYC_MAX_Signal'] = False
        return df

    @staticmethod
    def calculate_range_break(df, period=250):
        """Range Breakout (箱体突破)"""
        df = df.copy()
        df['Box_Top'] = df['high'].rolling(window=period).max().shift(1)
        df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
        df['RangeBreak_Signal'] = (df['close'] > df['Box_Top']) & (df['volume'] > df['Vol_MA20'])
        return df

    @staticmethod
    def calculate_20vma(df):
        """20VMA (量能启动)"""
        df = df.copy()
        df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
        df['Quiet_Day'] = df['volume'] < df['Vol_MA20']
        # Past 5 days (excluding today), at least 4 quiet days
        df['Quiet_Recent'] = df['Quiet_Day'].shift(1).rolling(window=5).sum() >= 4
        df['Ignition'] = df['volume'] > df['Vol_MA20']
        df['Trend'] = (df['close'] > df['open']) & (df['close'] > df['close'].shift(1))
        df['20VMA_Signal'] = df['Quiet_Recent'] & df['Ignition'] & df['Trend']
        return df

    @staticmethod
    def calculate_obo(df):
        """OBO (Open Breakout)"""
        df = df.copy()
        df['Range_prev'] = df['high'].shift(1) - df['low'].shift(1)
        df['Target'] = df['open'] + df['Range_prev']
        df['MA250'] = df['close'].rolling(window=250).mean()
        df['OBO_Signal'] = (df['close'] > df['Target']) & (df['close'] > df['MA250'])
        return df

    @staticmethod
    def calculate_lcs(df):
        """LCS (Limit Close Super)"""
        df = df.copy()
        df['Low_5'] = df['low'].rolling(window=5).min()
        df['Deviation'] = (df['close'] - df['Low_5']) / df['Low_5']
        df['Max_250'] = df['close'].rolling(window=250).max()
        # Adjusted deviation to 0.20 for A-share suitability
        df['LCS_Signal'] = (df['Deviation'] > 0.20) & (df['close'] == df['Max_250'])
        return df
        
    @staticmethod
    def calculate_hps(df):
        """HPS 趋势系统"""
        df = df.copy()
        df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
        df['EMA_High_15'] = df['high'].ewm(span=15, adjust=False).mean()
        df['Trend'] = df['close'] > df['EMA200']
        df['Breakout'] = df['close'] > df['EMA_High_15']
        df['HPS_Signal'] = df['Trend'] & df['Breakout']
        return df

    @staticmethod
    def calculate_rking(df):
        """RKing 趋势跟随"""
        df = df.copy()
        # RKing signals usually already calculated in Indicators.add_rking.
        # But we need a simple signal column: RKing_Buy_Signal (Red bar start or valid red cross)
        if 'RKing_State' in df.columns:
            # Change from non-red to red
            df['RKing_Signal'] = (df['RKing_State'] == 'Red') & (df['RKing_State'].shift(1) != 'Red')
        else:
            df['RKing_Signal'] = False
        return df

    @staticmethod
    def calculate_ambush_calm(df):
        """
        蓄力伏击 (Ambush Calm) - T-1 波动率压缩预判
        
        基于回测数据研究：OBO+LCS/HMC+LCS等强势组合在爆发前一天，
        呈现波动率极度压缩+缩量横盘+紧贴MA20的"暴风雨前宁静"特征。
        
        严格条件：
        1. 振幅 < 2.5% (极窯幅)
        2. 连续3天振幅都 < 3% (持续压缩，而非偶然一天)
        3. 成交量 < 0.8倍20日均量 (明显缩量)
        4. 收盘价在MA20附近 (-2% ~ +4%)
        5. 收盘价 > MA60 (长期趋势向上，避免在下跌趋势中横盘)
        6. MACD红柱 > 0 (动量不能是空头)
        """
        df = df.copy()
        
        # 振幅 < 2.5%
        amplitude = (df['high'] - df['low']) / df['close'].shift(1) * 100
        cond_tight = amplitude < 2.5
        
        # 连续3天窄幅 (<3%)
        tight_3 = (amplitude < 3.0)
        cond_persist = tight_3 & tight_3.shift(1) & tight_3.shift(2)
        
        # 明显缩量: 量 < 0.8倍均量
        vol_ma20 = df['volume'].rolling(window=20).mean()
        cond_quiet = df['volume'] < vol_ma20 * 0.8
        
        # 紧贴MA20: -2% ~ +4%
        ma20 = df['close'].rolling(window=20).mean()
        dist = (df['close'] - ma20) / ma20 * 100
        cond_near_ma = (dist >= -2) & (dist <= 4)
        
        # 长期趋势向上
        ma60 = df['close'].rolling(window=60).mean()
        cond_trend = df['close'] > ma60
        
        # MACD红柱
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd_hist = 2 * (ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean())
        cond_macd = macd_hist > 0
        
        df['Ambush_Calm_Signal'] = cond_tight & cond_persist & cond_quiet & cond_near_ma & cond_trend & cond_macd
        return df

    @staticmethod
    def calculate_ambush_momentum(df):
        """
        动量伏击 (Ambush Momentum) - T-1 动量起爆预判
        
        基于回测数据研究：TKOS+Fighting+LCS等连续爆发型组合，
        在爆发前一天已经大涨+放量+MACD全红，属于追涨续航型。
        
        条件：
        1. 今日涨幅 > 0% (收红)
        2. 成交量 > 1.2倍20日均量 (量能异动)
        3. MACD红柱 > 0 且正在伸长 (动量向上)
        4. 距离MA20不超过15% (避免过度追高)
        """
        df = df.copy()
        
        # 收红
        pct_chg = df['close'].pct_change() * 100
        cond_up = pct_chg > 0
        
        # 量能异动: > 1.2倍均量
        vol_ma20 = df['volume'].rolling(window=20).mean()
        cond_vol = df['volume'] > vol_ma20 * 1.2
        
        # MACD红柱 > 0 且伸长
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd_hist = 2 * (ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean())
        cond_macd = (macd_hist > 0) & (macd_hist > macd_hist.shift(1))
        
        # 不过度追高: 距MA20 < 15%
        ma20 = df['close'].rolling(window=20).mean()
        dist = (df['close'] - ma20) / ma20 * 100
        cond_not_chase = dist < 15
        
        df['Ambush_Momentum_Signal'] = cond_up & cond_vol & cond_macd & cond_not_chase
        return df
        
    @staticmethod
    def check_all_strong_strategies(df, index_df=None, selected_strategies=None):
        """
        检查所有强势股策略
        
        :param df: 个股数据 DataFrame
        :param index_df: 大盘指数数据 DataFrame (用于RS策略)
        :param selected_strategies: 选中的策略列表
        :return: 包含所有策略信号的 DataFrame
        """
        if selected_strategies is None:
            selected_strategies = ['Z_Score', 'RS', 'TKOS', 'DTR_Plus', 'Fighting', 'UA', 'HMC', 'CYC_MAX', 'RangeBreak', '20VMA', 'OBO', 'LCS', 'HPS', 'RKing']
        
        signals = pd.DataFrame(index=df.index)
        
        if 'Z_Score' in selected_strategies:
            signals['Signal_Z_Score'] = StrongStrategies.calculate_z_score(df)['Z_Signal']
            
        if 'RS' in selected_strategies:
            if index_df is not None:
                signals['Signal_RS'] = StrongStrategies.calculate_rs_strategy(df, index_df)['RS_Breakout']
            else:
                signals['Signal_RS'] = False
            
        if 'TKOS' in selected_strategies:
            signals['Signal_TKOS'] = StrongStrategies.calculate_tkos(df)['TKOS_Signal']
            
        if 'DTR_Plus' in selected_strategies:
            signals['Signal_DTR_Plus'] = StrongStrategies.calculate_dtr_plus(df)['DTR_Signal' if 'DTR_Signal' in StrongStrategies.calculate_dtr_plus(df).columns else 'DTR_Plus_Signal']
            
        if 'Fighting' in selected_strategies:
            signals['Signal_Fighting'] = StrongStrategies.calculate_fighting_strategy(df)['Fighting_Signal']
            
        if 'UA' in selected_strategies:
            signals['Signal_UA'] = StrongStrategies.calculate_ua_strategy(df)['UA_Buy_Signal']
            
        if 'HMC' in selected_strategies:
            signals['Signal_HMC'] = StrongStrategies.calculate_hmc_strategy(df)['HMC_Signal']
            
        if 'CYC_MAX' in selected_strategies:
            signals['Signal_CYC_MAX'] = StrongStrategies.calculate_cyc_max(df)['CYC_MAX_Signal']
            
        if 'RangeBreak' in selected_strategies:
            signals['Signal_RangeBreak'] = StrongStrategies.calculate_range_break(df)['RangeBreak_Signal']
            
        if '20VMA' in selected_strategies:
            signals['Signal_20VMA'] = StrongStrategies.calculate_20vma(df)['20VMA_Signal']
            
        if 'HPS' in selected_strategies:
            signals['Signal_HPS'] = StrongStrategies.calculate_hps(df)['HPS_Signal']
            
        if 'RKing' in selected_strategies:
            signals['Signal_RKing'] = StrongStrategies.calculate_rking(df)['RKing_Signal']
            
        if 'OBO' in selected_strategies:
            signals['Signal_OBO'] = StrongStrategies.calculate_obo(df)['OBO_Signal']
            
        if 'LCS' in selected_strategies:
            signals['Signal_LCS'] = StrongStrategies.calculate_lcs(df)['LCS_Signal']
        
        if 'Ambush_Calm' in selected_strategies:
            signals['Signal_Ambush_Calm'] = StrongStrategies.calculate_ambush_calm(df)['Ambush_Calm_Signal']
            
        if 'Ambush_Momentum' in selected_strategies:
            signals['Signal_Ambush_Momentum'] = StrongStrategies.calculate_ambush_momentum(df)['Ambush_Momentum_Signal']
        
        return signals

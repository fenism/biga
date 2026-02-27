"""
Royal 股票抄底实战策略模块
实现 weak.md 中定义的8种抄底策略

核心心法：行情始于"无"（极致缩量/绝望），终于"有"（放量/贪婪）
抄底不是买在最低点，而是买在"绝望后的确认转折点"
"""

import pandas as pd
import numpy as np


class WeakStrategies:
    """弱势股抄底策略集合"""
    
    @staticmethod
    def _calculate_rsi(series, period):
        """辅助函数：计算RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    # ========== 第一阶段：扫描与初筛（寻找"绝望"与"无"） ==========
    
    @staticmethod
    def strategy_hlp3(df, winner_col='winner_pct'):
        """
        HLP3 (大慈悲点) - 筹码绝望筛查 (Price-Based Proxy)
        
        原始逻辑：昨日获利盘 < 1% (全场99%都在亏损，多头死绝)
                 今日获利盘 > 35% (主力进场扫货，大量筹码瞬间解套)
        
        由于本地数据无换手率，使用价格代理：
        - 绝望条件：股价在120日低点附近（距120日低点 < 8%），代表几乎所有人亏损
        - 启动条件：今日大幅拉升（涨幅 > 3%），且成交量放大（> 1.5倍20日均量）
                   代表主力进场扫货，大量筹码瞬间解套
        
        :param df: 包含OHLCV的DataFrame
        :param winner_col: 获利盘比例列名（本版本不使用，保留兼容性）
        :return: 包含HLP3信号的DataFrame
        """
        df = df.copy()
        
        # --- 绝望条件：股价在120日低点附近 ---
        # 120日最低价
        low_120 = df['low'].rolling(window=120, min_periods=30).min()
        # 昨日收盘距120日低点的距离 < 8%（代表几乎所有持仓者亏损）
        cond_despair = (df['close'].shift(1) - low_120.shift(1)) / low_120.shift(1) < 0.08
        
        # --- 启动条件：今日大幅拉升 + 量能放大 ---
        # 今日涨幅 > 3%
        pct_change = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100
        cond_surge = pct_change > 3.0
        
        # 今日成交量 > 20日均量 * 1.5（放量确认主力进场）
        vol_ma20 = df['volume'].rolling(window=20, min_periods=5).mean()
        cond_volume = df['volume'] > vol_ma20 * 1.5
        
        # 综合信号
        df['HLP3_Signal'] = cond_despair & cond_surge & cond_volume
        df['HLP3_Warning'] = False
        
        return df

    
    @staticmethod
    def strategy_limit(df):
        """
        Limit (极致缩量) - 量能静默筛查
        
        核心逻辑：成交量 < 20日均量 * 0.5 (市场极度死寂，变盘前夜)
        买入扳机：缩量后放量突破20日均量线 + 收阳
        
        :param df: 包含OHLCV的DataFrame
        :return: 包含Limit信号的DataFrame
        """
        df = df.copy()
        
        # 计算20日均量
        vma20 = df['volume'].rolling(window=20).mean()
        
        # 极致缩量：量 < 均量的一半
        df['Limit_Signal'] = df['volume'] < (vma20 * 0.5)
        
        # 进阶：Limit后放量突破 (Limit Breakout)
        # 过去5天内出现过Limit + 今日放量突破20日线 + 收阳
        limit_setup = df['Limit_Signal'].rolling(window=5).max() > 0
        vol_breakout = df['volume'] > vma20
        bull_candle = df['close'] > df['open']
        
        df['Limit_BO_Signal'] = limit_setup & vol_breakout & bull_candle
        
        return df
    
    @staticmethod
    def strategy_rsi_reversion(df):
        """
        RSI均值回归 - 技术极度超卖
        
        核心逻辑：价格 > EMA200 (牛市趋势) 
                 AND RSI(2) 连续2天 < 25 (短期非理性恐慌抛售)
        买入时机：第3天开盘博弈反弹
        
        :param df: 包含OHLCV的DataFrame
        :return: 包含RSI回归信号的DataFrame
        """
        df = df.copy()
        
        # 计算 EMA200 (200日均线 - 长期趋势)
        ema200 = df['close'].ewm(span=200, adjust=False).mean()
        
        # 计算 RSI(2) - 使用主引擎计算以保平滑
        from indicators import Indicators
        rsi2 = Indicators.calculate_rsi(df['close'], 2)
        
        # 条件1: 趋势向上 (收盘价需在EMA200上方)
        cond_trend = df['close'] > ema200
        
        # 条件2: RSI2 连续2天小于 25 (原文：极度恐慌超卖判定)
        cond_oversold = (rsi2.shift(1) < 25) & (rsi2 < 25)
        
        # 综合信号 (在第3天触发)
        df['RSI_Rev_Signal'] = cond_trend & cond_oversold
        
        return df
    
    # ========== 第二阶段：形态确认（寻找"诱空"与"试探"） ==========
    
    @staticmethod
    def strategy_spring(df):
        """
        Spring (弹簧) - 诱空形态
        
        核心逻辑：跌破支撑位(20日低点) + 1-3天内迅速拉回支撑上方 + 缩量下杀
        含义：主力清洗最后浮筹，测试供应
        
        :param df: 包含OHLCV的DataFrame
        :return: 包含Spring信号的DataFrame
        """
        df = df.copy()
        
        # 定义支撑：过去20天的最低点（不含今日）
        support = df['low'].rolling(window=20).min().shift(1)
        
        # 1. 最低价跌破支撑
        break_support = df['low'] < support
        
        # 2. 收盘价收回支撑上方 (Spring回抽)
        recover = df['close'] > support
        
        # 3. 缩量特征 (可选，增强信号质量)
        vma20 = df['volume'].rolling(window=20).mean()
        low_volume = df['volume'] < vma20
        
        # Spring信号：击穿且拉回
        df['Spring_Signal'] = break_support & recover & low_volume
        
        return df
    
    @staticmethod
    def strategy_pinbar(df):
        """
        Pinbar (长钉 / 单针探底) - 多头长钉
        
        核心逻辑：下影线长度 > 实体 * 3 + 放量
        含义：恐慌盘涌出被主力全盘接下（多头探底神针）
        
        :param df: 包含OHLCV的DataFrame
        :return: 包含Pinbar信号的DataFrame
        """
        df = df.copy()
        
        # 计算K线各部分
        body = abs(df['close'] - df['open'])
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
        
        # 成交量放大 (大于20日均量)
        vma20 = df['volume'].rolling(window=20).mean()
        vol_up = df['volume'] > vma20
        
        # 下影线 > 实体 * 3
        pin_shape = lower_shadow > (body * 3)
        
        # 多头长钉信号
        df['Pinbar_Signal'] = pin_shape & vol_up
        
        return df
    
    @staticmethod
    def strategy_money_flow(df):
        """
        Money Flow Divergence (资金背离)
        
        核心逻辑：股价横盘或创新低，但资金流入流出监测器显示净流入
        含义：主力在底部悄悄吸筹
        
        计算公式（来自文档）：
        D = C - REF(C,1)  # 涨跌额
        D1 = D / REF(C,1)  # 涨幅比例
        DT = (V * D1) * 100  # 上涨动能
        KT = (V * K1) * 100  # 下跌动能
        净流入 = SUM(DT - KT, 10)  # 10日累计
        
        :param df: 包含OHLCV的DataFrame
        :return: 包含资金背离信号的DataFrame
        """
        df = df.copy()
        
        # 昨日收盘
        ref_c = df['close'].shift(1)
        
        # 涨跌额
        diff = df['close'] - ref_c
        
        # 上涨部分
        d_part = np.where(diff > 0, diff, 0)
        d1 = d_part / ref_c.replace(0, np.nan)
        dt = (df['volume'] * d1) * 100
        
        # 下跌部分 (取绝对值)
        k_part = np.where(diff < 0, abs(diff), 0)
        k1 = k_part / ref_c.replace(0, np.nan)
        kt = (df['volume'] * k1) * 100
        
        # 10日累计净流入
        net_flow = pd.Series(dt - kt).rolling(window=10).sum()
        df['Money_Flow'] = net_flow
        
        # 背离逻辑：股价创20日新低 + 资金流为正
        price_low = df['close'] == df['close'].rolling(20).min()
        flow_positive = df['Money_Flow'] > 0
        
        df['Money_Flow_Signal'] = price_low & flow_positive
        
        return df
    
    # ========== 第三阶段：买入扳机（确认"有"与"启动"） ==========
    
    @staticmethod
    def strategy_ua(df, period=250):
        """
        UA (Ultimate Amount 天量) - 底部天量突破
        
        核心逻辑：底部出现历史级天量，标记最高价
        买入时机：后续价格有效突破天量日最高价（确认多头获胜）
        
        :param df: 包含OHLCV的DataFrame
        :param period: 天量检测周期，默认250日
        :return: 包含UA信号的DataFrame
        """
        df = df.copy()
        
        # 识别天量（250日内最大成交量）
        df['UA_Is_Max'] = df['volume'] == df['volume'].rolling(period).max()
        
        # 记录天量当日的最高价 (作为突破目标位)
        df['UA_Target_High'] = np.where(df['UA_Is_Max'], df['high'], np.nan)
        df['UA_Target_High'] = df['UA_Target_High'].ffill()  # 向下填充
        
        # UA突破买点：收盘价站上最近一次UA的最高价 (且当日不是UA日)
        df['UA_Breakout_Signal'] = (df['close'] > df['UA_Target_High']) & (~df['UA_Is_Max'])
        
        return df
    
    @staticmethod
    def strategy_double_volume_hold(df):
        """
        倍量不破 (Double Volume Hold)
        
        核心逻辑：今日量 > 昨日量 * 2 (倍量阳线)
        买入时机：回调不破该阳线最低价，再次启动时买入
        
        :param df: 包含OHLCV的DataFrame
        :return: 包含倍量不破信号的DataFrame
        """
        df = df.copy()
        
        # 1. 识别倍量柱
        double_vol = df['volume'] > (df['volume'].shift(1) * 2)
        
        # 2. 标记倍量柱的最低价
        df['Double_Vol_Low'] = np.where(double_vol, df['low'], np.nan)
        df['Double_Vol_Low'] = df['Double_Vol_Low'].ffill()  # 填充最近的倍量低点
        
        # 3. 检查是否守住 (当前收盘价 > 倍量低点)
        df['Is_Holding'] = df['close'] > df['Double_Vol_Low']
        
        # 4. 信号：倍量后守住低点 + 再次放量
        vma20 = df['volume'].rolling(window=20).mean()
        vol_up = df['volume'] > vma20
        
        df['Double_Vol_Signal'] = df['Is_Holding'] & vol_up & (df['close'] > df['open'])
        
        return df
    
    @staticmethod
    def strategy_wyckoff(df):
        """Wyckoff (吸筹)"""
        df = df.copy()
        vma20 = df['volume'].rolling(window=20).mean()
        low_vol_days = (df['volume'] < vma20).rolling(window=60).sum()
        cond_accum = low_vol_days >= (60 * 0.7)
        cond_effort = df['volume'] > (vma20 * 1.5)
        df['Wyckoff_Signal'] = cond_accum & cond_effort & (df['close'] > df['open'])
        return df

    @staticmethod
    def strategy_vol_min_120(df):
        """量比历史新低"""
        df = df.copy()
        vma5 = df['volume'].rolling(window=5).mean()
        vol_ratio = df['volume'] / vma5.replace(0, np.nan)
        min_ratio_120 = vol_ratio.rolling(window=120).min()
        df['Vol_Min_120_Signal'] = (vol_ratio == min_ratio_120) & (vol_ratio > 0)
        return df

    @staticmethod
    def strategy_boll_rev(df):
        """布林反转"""
        df = df.copy()
        ma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        boll_mid = ma20
        boll_upper = boll_mid + 2 * std20
        weak_zone = (df['close'] < boll_mid).rolling(window=60).sum() >= 50
        cond_breakout = (df['close'] > boll_mid) & (df['high'] >= boll_upper)
        recent_weak = weak_zone.shift(1).rolling(window=10).max() > 0 
        df['Boll_Rev_Signal'] = recent_weak & cond_breakout
        return df

    @staticmethod
    def strategy_2b(df):
        """2B 法则"""
        df = df.copy()
        prev_low = df['low'].rolling(window=20).min().shift(1)
        df['2B_Signal'] = (df['low'] < prev_low) & (df['close'] > prev_low)
        return df

    @staticmethod
    def strategy_es(df):
        """ES 波动率压缩"""
        df = df.copy()
        std20 = df['close'].rolling(window=20).std()
        std60 = df['close'].rolling(window=60).std()
        std120 = df['close'].rolling(window=120).std()
        std_long = np.minimum(std60.fillna(method='bfill'), std120.fillna(method='bfill'))
        df['ES_Signal'] = (std20 < std_long) & (df['volume'] > df['volume'].rolling(20).mean())
        return df

    @staticmethod
    def strategy_ambush_bottom(df):
        """
        底部伏击 (Ambush Bottom) - T-1 最后一跌预判
        
        基于回测数据研究：HLP3+RSI_Rev+UA_Weak等弱势抄底组合，
        在信号触发前一天呈现"最后一跌"特征。
        
        严格条件：
        1. 连续2天下跌 (确认不是偶然回调)
        2. 成交量 < 0.8倍20日均量 (卖盘真正枯竭)
        3. MACD绿柱正在缩短 (空头衰竭的关键转折)
        4. RSI(6) < 40 (短期超卖区域)
        5. 收盘价在MA20附近 (-8% ~ +3%)
        """
        df = df.copy()
        
        # 连续2天下跌
        cond_down1 = df['close'] < df['close'].shift(1)
        cond_down2 = df['close'].shift(1) < df['close'].shift(2)
        cond_down = cond_down1 & cond_down2
        
        # 明显缩量: 量 < 0.8倍均量
        vol_ma20 = df['volume'].rolling(window=20).mean()
        cond_shrink = df['volume'] < vol_ma20 * 0.8
        
        # MACD绿柱缩短
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd_hist = 2 * (ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean())
        cond_macd_turn = macd_hist > macd_hist.shift(1)
        
        # RSI(6) < 40
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
        rs = gain / loss
        rsi6 = 100 - (100 / (1 + rs))
        cond_rsi = rsi6 < 40
        
        # 在MA20附近
        ma20 = df['close'].rolling(window=20).mean()
        dist = (df['close'] - ma20) / ma20 * 100
        cond_near_ma = (dist >= -8) & (dist <= 3)
        
        df['Ambush_Bottom_Signal'] = cond_down & cond_shrink & cond_macd_turn & cond_rsi & cond_near_ma
        return df
        
    @staticmethod
    def check_all_weak_strategies(df, selected_strategies=None, winner_col='winner_pct'):
        """
        检查所有抄底策略
        
        :param df: 个股数据 DataFrame
        :param selected_strategies: 选中的策略列表
        :param winner_col: 获利盘列名（用于HLP3）
        :return: 包含所有策略信号的 DataFrame
        """
        if selected_strategies is None:
            selected_strategies = ['HLP3', 'Limit', 'RSI_Rev', 'Spring', 
                                  'Pinbar', 'Money_Flow', 'UA_Weak', 'Double_Vol',
                                  'Wyckoff', 'Vol_Min_120', 'Boll_Rev', '2B', 'ES']
        
        signals = pd.DataFrame(index=df.index)
        
        if 'HLP3' in selected_strategies:
            hlp3_result = WeakStrategies.strategy_hlp3(df, winner_col)
            signals['Signal_HLP3'] = hlp3_result['HLP3_Signal']
            signals['HLP3_Warning'] = hlp3_result['HLP3_Warning']
        
        if 'Limit' in selected_strategies:
            signals['Signal_Limit'] = WeakStrategies.strategy_limit(df)['Limit_BO_Signal']
        
        if 'RSI_Rev' in selected_strategies:
            signals['Signal_RSI_Rev'] = WeakStrategies.strategy_rsi_reversion(df)['RSI_Rev_Signal']
        
        if 'Spring' in selected_strategies:
            signals['Signal_Spring'] = WeakStrategies.strategy_spring(df)['Spring_Signal']
        
        if 'Pinbar' in selected_strategies:
            signals['Signal_Pinbar'] = WeakStrategies.strategy_pinbar(df)['Pinbar_Signal']
        
        if 'Money_Flow' in selected_strategies:
            signals['Signal_Money_Flow'] = WeakStrategies.strategy_money_flow(df)['Money_Flow_Signal']
        
        if 'UA_Weak' in selected_strategies:
            signals['Signal_UA_Weak'] = WeakStrategies.strategy_ua(df)['UA_Breakout_Signal']
        
        if 'Double_Vol' in selected_strategies:
            signals['Signal_Double_Vol'] = WeakStrategies.strategy_double_volume_hold(df)['Double_Vol_Signal']

        if 'Wyckoff' in selected_strategies:
            signals['Signal_Wyckoff'] = WeakStrategies.strategy_wyckoff(df)['Wyckoff_Signal']
            
        if 'Vol_Min_120' in selected_strategies:
            signals['Signal_Vol_Min_120'] = WeakStrategies.strategy_vol_min_120(df)['Vol_Min_120_Signal']
            
        if 'Boll_Rev' in selected_strategies:
            signals['Signal_Boll_Rev'] = WeakStrategies.strategy_boll_rev(df)['Boll_Rev_Signal']
            
        if '2B' in selected_strategies:
            signals['Signal_2B'] = WeakStrategies.strategy_2b(df)['2B_Signal']
            
        if 'ES' in selected_strategies:
            signals['Signal_ES'] = WeakStrategies.strategy_es(df)['ES_Signal']

        return signals

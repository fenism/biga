"""
Exit Signal Detection Module
根据交易SOP第四、五步，检测离场/止盈信号。

强势股离场信号:
  - 天量滞涨 (UA Reversal): 高位放巨量但收上影/十字星
  - MACD动能衰竭: MACD红柱连续缩短
  - 技术破位: 跌破MA20生命线

弱势/抄底股离场信号:
  - 突破失败: 站上参照点后又跌回
  - MACD动能衰竭: 同上
  - 技术破位: 同上
"""

import pandas as pd
import numpy as np


# Strong strategy identifiers
STRONG_STRATEGIES = {'Z_Score', 'RS', 'TKOS', 'DTR_Plus', 'Fighting', 'UA', 'HMC'}
# Weak strategy identifiers
WEAK_STRATEGIES = {'HLP3', 'Limit', 'RSI_Rev', 'Spring', 'Pinbar', 'Money_Flow', 'Double_Vol'}


class ExitSignals:
    """离场信号检测器"""

    @staticmethod
    def detect_source_type(strategies_str: str) -> str:
        """
        根据策略名称判断股票来源类型。
        
        :param strategies_str: 逗号分隔的策略字符串，如 'Z_Score, DTR_Plus'
        :return: 'strong', 'weak', 或 'both'
        """
        if not strategies_str:
            return 'both'
        
        strats = {s.strip() for s in str(strategies_str).split(',')}
        has_strong = bool(strats & STRONG_STRATEGIES)
        has_weak = bool(strats & WEAK_STRATEGIES)
        
        if has_strong and not has_weak:
            return 'strong'
        elif has_weak and not has_strong:
            return 'weak'
        else:
            return 'both'

    @staticmethod
    def detect(df, source_type='both', signal_high=None):
        """
        检测离场信号。
        
        :param df: 包含完整技术指标的 DataFrame（需 MACD_Hist, MA20, volume, Vol_MA20 等）
        :param source_type: 'strong', 'weak', 或 'both'
        :param signal_high: 参照点（高），用于抄底股的突破失败检测
        :return: DataFrame with boolean exit signal columns + description columns
        """
        exits = pd.DataFrame(index=df.index)
        
        # ===== 通用离场信号 =====
        
        # 1. MACD 动能衰竭：红柱连续3日缩短
        if 'MACD_Hist' in df.columns:
            hist = df['MACD_Hist']
            # 红柱区间内（MACD_Hist > 0），连续缩短
            declining = (hist < hist.shift(1)) & (hist.shift(1) < hist.shift(2))
            in_red_zone = (hist > 0) & (hist.shift(1) > 0) & (hist.shift(2) > 0)
            exits['Exit_MACD_Decay'] = declining & in_red_zone
        else:
            exits['Exit_MACD_Decay'] = pd.Series(False, index=df.index)
        
        # 2. 技术破位：收盘跌破MA20（生命线）
        if 'MA20' in df.columns:
            # 今日收盘 < MA20 且 前一日收盘 >= 前一日MA20（首次跌破）
            exits['Exit_Break_MA20'] = (
                (df['close'] < df['MA20']) & 
                (df['close'].shift(1) >= df['MA20'].shift(1))
            )
        else:
            exits['Exit_Break_MA20'] = pd.Series(False, index=df.index)

        # ===== 强势股专属离场信号 =====
        
        if source_type in ('strong', 'both'):
            # 3. 天量滞涨 (UA Reversal)
            # 条件：放出近250日最大量的80%以上 + 收长上影(上影 > 实体2倍) 或 十字星(实体 < 全幅10%)
            if 'volume' in df.columns and 'Vol_MA20' in df.columns:
                # 计算滚动最大量
                max_vol_250 = df['volume'].rolling(window=250, min_periods=50).max()
                vol_surge = df['volume'] >= max_vol_250 * 0.8
                
                body = (df['close'] - df['open']).abs()
                full_range = df['high'] - df['low']
                upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
                
                # 长上影：上影 > 2倍实体
                long_upper = upper_shadow > 2 * body
                # 十字星：实体 < 全幅的10%
                doji = body < full_range * 0.1
                
                exits['Exit_UA_Reversal'] = vol_surge & (long_upper | doji)
            else:
                exits['Exit_UA_Reversal'] = False
        
        # ===== 弱势/抄底股专属离场信号 =====
        
        if source_type in ('weak', 'both'):
            # 4. 突破失败：曾站上参照点后跌回
            if signal_high is not None and signal_high > 0:
                # 历史上曾突破过 signal_high，但现在又跌回去了
                ever_above = (df['high'] >= signal_high).cummax()
                now_below = df['close'] < signal_high
                exits['Exit_Fail_Breakout'] = ever_above & now_below
            else:
                exits['Exit_Fail_Breakout'] = False
            
            # 5. RSI(2) 超买回调 (RSI Reversion TP)
            # 原文要求：当 RSI 靠近超买区（大于 80）时止盈平仓
            if 'RSI' in df.columns:
                # Use RSI(2) if available, or just RSI
                rsi_col = 'RSI' if 'RSI' in df.columns else None
                if rsi_col:
                    exits['Exit_RSI_TP'] = (df[rsi_col] > 80) & (df[rsi_col].shift(1) <= 80)
            else:
                # If not in columns, calculate it on the fly if needed
                from indicators import Indicators
                rsi2 = Indicators.calculate_rsi(df['close'], 2)
                exits['Exit_RSI_TP'] = (rsi2 > 80) & (rsi2.shift(1) <= 80)
        
        # 填充NaN为False
        exits = exits.fillna(False)
        
        return exits

    @staticmethod
    def get_exit_descriptions():
        """返回各离场信号的中文描述"""
        return {
            'Exit_MACD_Decay': '⚠️ MACD动能衰竭：红柱连续缩短，上涨动能减弱',
            'Exit_Break_MA20': '⚠️ 技术破位：收盘跌破20日生命线',
            'Exit_UA_Reversal': '🚨 天量滞涨：放出巨量但无法上涨，主力可能出货',
            'Exit_Fail_Breakout': '⚠️ 突破失败：曾突破参照点后回落，反转失败',
            'Exit_RSI_TP': '💰 RSI2超买止盈：RSI(2) > 80 进入极度超买区，落袋为安',
        }

    @staticmethod
    def check_latest_exits(exits_df):
        """
        检查最新一根K线（最后一行）是否有离场信号触发。
        
        :param exits_df: detect() 返回的 DataFrame
        :return: list of (signal_name, description) tuples for active exit signals
        """
        if exits_df.empty:
            return []
        
        descs = ExitSignals.get_exit_descriptions()
        last_row = exits_df.iloc[-1]
        active = []
        for col in exits_df.columns:
            if col.startswith('Exit_') and last_row.get(col, False):
                active.append((col, descs.get(col, col)))
        return active

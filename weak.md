Royal 股票抄底实战 SOP (标准作业程序)
核心心法： 行情始于“无”（极致缩量/绝望），终于“有”（放量/贪婪）。抄底不是买在最低点，而是买在**“绝望后的确认转折点”**。
第一阶段：扫描与初筛（寻找“绝望”与“无”）
目标： 寻找处于极度恐慌或无人问津状态的标的，此时不做操作，仅加入自选池观察。
1. 筹码绝望筛查 (HLP3 Zero-Profit)
• 指标： 获利盘比例 (WINNER)。
• 标准： WINNER(Close) < 1%。
• 含义： 全场99%的人都在亏损，多头死绝，抛压衰竭。
• 动作： 标记出获利盘归零那天的K线最高价。
2. 量能静默筛查 (Limit)
• 指标： 成交量 (VOL) 与 20日均量线 (VMA20)。
• 标准： VOL < VMA20 * 0.5（成交量小于20日均量的一半）。
• 含义： 市场极度死寂，变盘前夜。
• 动作： 加入自选，等待“无中生有”。
3. 技术极度超卖 (RSI Mean Reversion)
• 指标： RSI (参数设为2)。
• 标准： RSI(2) < 25 且连续出现 2 天以上。
• 含义： 短期非理性恐慌抛售。
• 动作： 准备在第3天开盘博弈反弹（适合指数或ETF）。

--------------------------------------------------------------------------------
第二阶段：形态确认（寻找“诱空”与“试探”）
目标： 识别主力清洗最后浮筹的动作。
1. 诱空形态 (Spring 弹簧)
• 现象： 股价跌破近期重要支撑位（如箱体下沿或前低）。
• 关键： 在 1-3天内 迅速拉回支撑位上方。
• 量能： 下杀时缩量（主力测试供应），拉回时微放量。
• 动作： 确认Spring形态完成，准备入场。
2. 单针探底 (Pinbar)
• 现象： 低位出现长下影线K线，影线长度 > 实体长度 * 3。
• 关键： 必须伴随 巨量 (High Volume)。
• 含义： 恐慌盘涌出被主力全盘接下（多头探底神针）。
3. 资金背离 (Money Flow Divergence)
• 现象： 股价横盘或微跌（创新低），但 资金流入流出监测器 显示资金持续翻红（净流入）。
• 含义： 主力在底部悄悄吸筹。

--------------------------------------------------------------------------------
第三阶段：买入扳机（确认“有”与“启动”）
目标： 只有看到主力真金白银进场扫货，才跟随进场。
触发条件（满足其一即可开枪）：
• A. Limit 启动 (无中生有)：
    ◦ 在出现Limit缩量后，等待成交量重新 放量突破 20日均量线。
    ◦ 口诀： “缩量之后必放量，放量之后看方向”。放量收阳即买入。
• B. HLP3 大慈悲启动：
    ◦ 获利盘从 <1% 迅速飙升至 >35%（主力进场通吃）。
    ◦ 或者：股价有效突破之前标记的“获利盘归零K线”的最高价。
• C. UA 天量突破 (Ultimate Amount)：
    ◦ 若底部出现历史级天量 (UA)，当日不追。
    ◦ 买点： 标记天量K线的最高价，等待后续股价 有效突破该最高价 时买入（确认多头获胜）。
• D. 倍量不破：
    ◦ 出现倍量阳线（量能 > 昨日2倍）。
    ◦ 买点： 回调时不跌破该阳线的最低价，再次启动时买入。

--------------------------------------------------------------------------------
第四阶段：防守与风控（底线思维）
目标： 承认抄底是逆势交易，必须带好安全带。
1. 止损设置 (Stop Loss)：
• Spring战法： 止损设在 Spring K线的最低点。
• Pinbar战法： 止损设在长下影线的最低点。
• Rking/通用： 止损设在信号K线的低点，或本金的 -10%（硬防守）。
2. 止盈策略 (Take Profit)：
• 1:1 推保护： 当浮盈金额 = 止损金额时，将止损上移至开仓价（保本）。
• 努力无结果： 高位出现放量但滞涨（Effort vs Result 背离），离场。
• 天量见顶： 高位再次出现UA天量，且换手率过高（>30%），清仓。

--------------------------------------------------------------------------------
总结：抄底操盘流程表
步骤
观察重点
量化标准
策略动作
1. 扫描
绝望/死寂
HLP<1% 或 Limit缩量
加入自选，不动手
2. 等待
诱空/试探
Spring弹簧 或 Pinbar长钉
密切关注，准备子弹
3. 开火
启动/确认
放量突破20日均量线 OR 获利盘>35%
买入，标记关键K线
4. 防守
底线
守住启动K线最低价
破位止损，绝不补仓
5. 离场
高潮/背离
高位放量滞涨 OR 跌破均价线
止盈，落袋为安


import pandas as pd
import numpy as np

class RoyalStrategies:
    def __init__(self, df):
        """
        初始化
        :param df: 包含 'Open', 'High', 'Low', 'Close', 'Volume' 的 DataFrame
                   索引应为时间序列
        """
        self.df = df.copy()
        
    def _calculate_rsi(self, series, period):
        """辅助函数：计算RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def strategy_hlp3(self, winner_col='Winner_Pct'):
        """
        1. HLP3 (大慈悲点) [Doc 60]
        逻辑：昨日获利盘 < 1% (绝望)，今日获利盘 > 35% (主力扫货)
        注意：df中必须包含 'Winner_Pct' 列 (0-100)
        """
        if winner_col not in self.df.columns:
            return "Error: 需要包含获利盘比例数据的列"
        
        # 昨日获利盘 < 1
        cond1 = self.df[winner_col].shift(1) < 1
        # 今日获利盘 > 35
        cond2 = self.df[winner_col] > 35
        
        self.df['Signal_HLP3'] = cond1 & cond2
        return self.df[['Close', winner_col, 'Signal_HLP3']]

    def strategy_limit(self):
        """
        2. Limit (极致缩量/无中生有) [Doc 63, 76, 82]
        逻辑：成交量 < 20日均量线 * 0.5
        """
        # 计算20日均量
        vma20 = self.df['Volume'].rolling(window=20).mean()
        
        # 极致缩量：量 < 均量的一半
        self.df['Signal_Limit'] = self.df['Volume'] < (vma20 * 0.5)
        
        # 进阶：Limit后放量突破 (BO) 辅助判断
        # 今日放量突破20日线 & 过去5天内出现过Limit
        self.df['Limit_Setup_Active'] = self.df['Signal_Limit'].rolling(window=5).max() > 0
        self.df['Signal_Limit_BO'] = self.df['Limit_Setup_Active'] & (self.df['Volume'] > vma20) & (self.df['Close'] > self.df['Open'])
        
        return self.df[['Close', 'Volume', 'Signal_Limit', 'Signal_Limit_BO']]

    def strategy_rsi_reversion(self):
        """
        3. RSI均值回归 (指数策略) [Doc 20]
        逻辑：价格 > EMA200 (牛市) AND RSI(2) 连续2天 < 25 (超卖)
        """
        # 计算 EMA200
        ema200 = self.df['Close'].ewm(span=200, adjust=False).mean()
        # 计算 RSI(2)
        rsi2 = self._calculate_rsi(self.df['Close'], 2)
        
        # 条件1: 趋势向上
        cond_trend = self.df['Close'] > ema200
        
        # 条件2: RSI2 连续2天小于25
        # shift(1)代表昨天, 当前行代表今天
        cond_oversold = (rsi2.shift(1) < 25) & (rsi2 < 25)
        
        self.df['Signal_RSI_Buy'] = cond_trend & cond_oversold
        return self.df[['Close', 'Signal_RSI_Buy']]

    def strategy_spring(self):
        """
        4. Spring (弹簧) [Doc 40, 72]
        逻辑：跌破支撑(20日低点)后，快速(3日内)收回支撑上方
        """
        # 定义支撑：过去20天的最低点（不含今日）
        support = self.df['Low'].rolling(window=20).min().shift(1)
        
        # 1. 最低价跌破支撑
        break_support = self.df['Low'] < support
        
        # 2. 收盘价收回支撑上方
        recover = self.df['Close'] > support
        
        # 3. 信号：击穿且拉回 (简化版，严格版需判断前1-2日击穿今日拉回)
        self.df['Signal_Spring'] = break_support & recover
        
        return self.df[['Close', 'Low', 'Signal_Spring']]

    def strategy_pinbar(self):
        """
        5. Pinbar (长钉) [Doc 59]
        逻辑：多头长钉 = 下影线长度 > 实体 * 3 且 放量
        """
        body = abs(self.df['Close'] - self.df['Open'])
        upper_shadow = self.df['High'] - self.df[['Close', 'Open']].max(axis=1)
        lower_shadow = self.df[['Close', 'Open']].min(axis=1) - self.df['Low']
        
        # 成交量放大 (大于20日均量)
        vol_up = self.df['Volume'] > self.df['Volume'].rolling(20).mean()
        
        # 下影线 > 实体 * 3
        pin_shape = lower_shadow > (body * 3)
        
        self.df['Signal_Bull_Pinbar'] = pin_shape & vol_up
        return self.df[['Close', 'Signal_Bull_Pinbar']]

    def strategy_money_flow(self):
        """
        6. Money Flow Divergence (资金背离) [Doc 78]
        逻辑：严格翻译文档公式
        D:=C-REF(C,1);
        D1:=D/REF(C,1); (涨幅比例)
        DT:=(V*D1)*100; (上涨动能)
        ...
        SUM(DT-KT,10) (10日净流入)
        """
        # 昨日收盘
        ref_c = self.df['Close'].shift(1)
        
        # 涨跌额
        diff = self.df['Close'] - ref_c
        
        # 涨幅/跌幅比例 (处理分母为0的情况)
        # 注意：文档逻辑分开计算了涨(D)和跌(K)
        # D1: 涨幅 (若跌则为负或0，但文档意图是分离计算)
        # 这里优化实现：直接计算带符号的涨跌幅 * Volume
        
        # 还原文档逻辑：
        # D > 0 部分
        d_part = np.where(diff > 0, diff, 0)
        d1 = d_part / ref_c
        dt = (self.df['Volume'] * d1) * 100
        
        # K > 0 部分 (代表跌幅的绝对值)
        k_part = np.where(diff < 0, abs(diff), 0)
        k1 = k_part / ref_c
        kt = (self.df['Volume'] * k1) * 100
        
        # 10日累计净流入
        net_flow = pd.Series(dt - kt).rolling(window=10).sum()
        
        self.df['Money_Flow'] = net_flow
        
        # 背离逻辑：股价创新低(20日)，但资金流为正
        price_low = self.df['Close'] == self.df['Close'].rolling(20).min()
        flow_up = self.df['Money_Flow'] > 0
        
        self.df['Signal_Flow_Divergence'] = price_low & flow_up
        
        return self.df[['Close', 'Money_Flow', 'Signal_Flow_Divergence']]

    def strategy_ua(self):
        """
        7. UA (Ultimate Amount 天量) [Doc 83]
        逻辑：成交量创250日(或历史)新高
        """
        # 250日天量
        self.df['Signal_UA'] = self.df['Volume'] == self.df['Volume'].rolling(250).max()
        
        # 记录天量当日的最高价 (作为压力位)
        self.df['UA_High_Price'] = np.where(self.df['Signal_UA'], self.df['High'], np.nan)
        # 向下填充，方便后续判断突破
        self.df['UA_High_Price'] = self.df['UA_High_Price'].ffill()
        
        # UA突破买点：收盘价站上最近一次UA的最高价
        self.df['Signal_UA_Breakout'] = (self.df['Close'] > self.df['UA_High_Price']) & (~self.df['Signal_UA'])
        
        return self.df[['Close', 'Volume', 'Signal_UA', 'Signal_UA_Breakout']]

    def strategy_double_volume_hold(self):
        """
        8. 倍量不破 [Doc 62]
        逻辑：今日量 > 昨日量*2 (倍量)；后续回调不破该K线最低价
        """
        # 1. 识别倍量柱
        double_vol = self.df['Volume'] > (self.df['Volume'].shift(1) * 2)
        
        # 2. 标记倍量柱的最低价
        # 我们需要一个循环或复杂的向量化操作来检查"后续不破"
        # 这里使用简化逻辑：标记倍量日，后续N天(如3-10天)收盘价都在该最低价之上
        
        self.df['Is_Double_Vol'] = double_vol
        self.df['Double_Vol_Low'] = np.where(double_vol, self.df['Low'], np.nan)
        self.df['Double_Vol_Low'] = self.df['Double_Vol_Low'].ffill() # 填充最近的倍量低点
        
        # 检查是否跌破 (当前Low < 倍量Low)
        self.df['Broken'] = self.df['Low'] < self.df['Double_Vol_Low']
        
        # 信号：是倍量后的第3-10天，且期间没有发生过Broken
        # 此处仅返回倍量标识，实际"不破"需要回测框架支持状态保持
        
        return self.df[['Close', 'Volume', 'Is_Double_Vol', 'Double_Vol_Low']]
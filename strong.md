Royal 强势股进攻 SOP
核心逻辑： 强者恒强。不买便宜的，只买更贵的；不买缩量的，只买放量突破的。
第一阶段：海选与锁定（寻找“强”）
目标： 在数千只股票中，通过量化指标圈定“多头最强部队”。
1. 量化强度筛选（Z-score）：
    ◦ 指标： Z-score (标准分)。
    ◦ 标准： 选择 Z > 1.5 的股票。
    ◦ 含义： 股价强度跑赢平均水平 1.5 个标准差，说明该股已进入“异动/启动”状态。
    ◦ 避雷： 若 Z > 3 且换手率过高（>30%），可能过热，需谨慎。
2. 相对强弱筛选（RS）：
    ◦ 指标： RS (Relative Strength)。
    ◦ 标准： 个股的RS线必须位于RS均线甚至布林上轨之上。
    ◦ 含义： 无论大盘涨跌，该股表现都强于大盘（大盘跌它横盘，大盘涨它领涨）。
3. 超级龙头筛选（TKOS）：
    ◦ 指标： 月度涨幅。
    ◦ 标准： 每月第一周结束后，筛选月涨幅 > 50% 的股票。
    ◦ 含义： 只有敢于在一个月内涨50%的股票才具备“股王”气质，第二周择机买入，博弈主升浪。
第二阶段：确认扳机（寻找“机”）
目标： 在强势股中找到高胜率的开火点，拒绝盲目追高。
1. DTR Plus 三合一共振：
    ◦ 信号（必须同时满足）：
        1. MACD柱状图由绿变红（动能转强）。
        2. 股价站上 20日均线（生命线之上）。
        3. 股价触碰或突破布林带上轨（爆发力确认）。
    ◦ 含义： 趋势、位置、爆发力三者共振，是高胜率的右侧买点。
2. Fighting 策略（52日双突破）：
    
    ◦ 信号： DTR翻红 + 价格突破52日新高 + 成交量突破52日新高量。
    ◦ 含义： 量价齐升创阶段新高，这是最标准的“欧奈尔/达瓦斯”式突破买点。
3. UA 天量突破（Ultimate Amount）：

    ◦ 信号： 之前出现过历史级别的**“天量”**（UA）。
    ◦ 买点： 标记天量K线的最高价。当后续价格有效突破该最高价时，买入。
    ◦ 逻辑： 天量代表巨大分歧，突破天量代表多头彻底消化了分歧，前方无阻力。
第三阶段：执行与防守（寻找“利”与“命”）
目标： 机械化执行，截断亏损，让利润奔跑。
1. 进场动作：
    ◦ 试仓： 信号出现当日或次日，先买入计划仓位的 1/3（插眼战术）。
    ◦ 加仓： 如果次日价格继续创新高（确认突破有效），再加仓。
2. 止损设置（生命线）：
    ◦ 技术位： 守住 20日均线或 信号K线的最低点。
    ◦ 强势股特有： 如果是涨停板或大阳线买入，守住 阳线实体的1/2，跌破即减仓/离场。
    ◦ 硬红线： 单笔亏损不超过本金 10% 。
3. 止盈与离场：
    ◦ 动能衰竭： MACD红柱缩短，或股价跌回布林带通道内。
    ◦ HMC红黄线： 当HMC指标中代表短期爆发力的黄线开始走弱，或者红线（趋势）下穿黄线时离场。
    ◦ 天量滞涨： 高位再次出现UA天量但价格不涨，坚决离场。


1. Z-score (标准分策略)
来源文档： 49 Zscore选股 核心逻辑： (收盘价 - 20日均价) / 20日标准差。Z > 1.5 为强势。
import pandas as pd
import numpy as np

def calculate_z_score(df, period=20):
    """
    计算 Z-score 指标
    :param df: 包含 'Close' 的 DataFrame
    :param period: 周期，默认为20
    :return: 包含 Z-score 及其信号的 DataFrame
    """
    # 计算均值和标准差
    df['MA20'] = df['Close'].rolling(window=period).mean()
    df['STD20'] = df['Close'].rolling(window=period).std()
    
    # 计算 Z-score
    # 防止除以0
    df['Z_Score'] = (df['Close'] - df['MA20']) / df['STD20'].replace(0, np.nan)
    
    # 生成信号
    # 强势信号：Z > 1.5
    # 过热预警：Z > 3 (根据文档49描述)
    df['Z_Signal'] = np.where((df['Z_Score'] > 1.5) & (df['Z_Score'] <= 3), 1, 0)
    df['Z_Overheat'] = np.where(df['Z_Score'] > 3, 1, 0)
    
    return df[['Close', 'Z_Score', 'Z_Signal', 'Z_Overheat']]

--------------------------------------------------------------------------------
2. RS (相对强弱策略)
来源文档： 73 RS相对强弱策略 核心逻辑： (个股收盘 / 大盘收盘) * 100，并叠加布林带（N=20, Std=2）。
def calculate_rs_strategy(stock_df, index_df, period=20, num_std=2):
    """
    计算 RS 相对强弱策略
    :param stock_df: 个股 DataFrame
    :param index_df: 大盘指数 DataFrame (如上证指数)
    :return: 包含 RS 及其布林带信号的 DataFrame
    """
    # 确保索引对齐
    data = pd.DataFrame(index=stock_df.index)
    data['Stock_Close'] = stock_df['Close']
    data['Index_Close'] = index_df['Close']
    
    # 1. 计算 RS 值 (乘以100或1000方便显示)
    data['RS'] = (data['Stock_Close'] / data['Index_Close']) * 1000
    
    # 2. 计算 RS 的布林带
    data['RS_MA'] = data['RS'].rolling(window=period).mean()
    data['RS_STD'] = data['RS'].rolling(window=period).std()
    data['RS_Upper'] = data['RS_MA'] + (data['RS_STD'] * num_std)
    
    # 3. 信号：RS 突破 RS布林上轨 (强转极强)
    # 使用 shift(1) 避免未来函数，判断穿越
    data['RS_Breakout'] = (data['RS'] > data['RS_Upper']) & (data['RS'].shift(1) <= data['RS_Upper'].shift(1))
    
    return data

--------------------------------------------------------------------------------
3. TKOS (股王策略)
来源文档： 09 “TKOS股王”策略详解教学 核心逻辑： 月初第一周涨幅 > 50%。
def calculate_tkos(df):
    """
    计算 TKOS 股王策略
    逻辑：检测某个月的前5个交易日(近似第一周)累计涨幅是否超过50%
    注意：实际使用需严格对齐日历月
    """
    # 这是一个简化逻辑，实际需要重采样到月度数据
    # 这里演示如何计算 N 日涨幅是否 > 50%
    
    # 计算5日累计涨幅 (近似一周)
    # (当前收盘 - 5天前收盘) / 5天前收盘
    df['Week_Pct_Change'] = df['Close'].pct_change(periods=5)
    
    # 信号：涨幅 > 50%
    df['TKOS_Signal'] = df['Week_Pct_Change'] > 0.50
    
    return df[['Close', 'Week_Pct_Change', 'TKOS_Signal']]

--------------------------------------------------------------------------------
4. DTR Plus (高胜率共振)
来源文档： 84 高胜率DTR PLUS 核心逻辑： MACD翻红 + 价格 > MA20 + 价格 >= 布林上轨。
def calculate_dtr_plus(df, ma_period=20, boll_std=2):
    """
    计算 DTR Plus 策略
    """
    # 1. 计算 MACD (12, 26, 9)
    # EMA12, EMA26
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    dea = diff.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = 2 * (diff - dea)
    
    # DTR翻红信号 (当前红柱，昨日绿柱)
    df['DTR_Red'] = (df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0)
    
    # 2. 计算 MA20
    df['MA20'] = df['Close'].rolling(window=ma_period).mean()
    
    # 3. 计算布林上轨
    std20 = df['Close'].rolling(window=ma_period).std()
    df['Boll_Upper'] = df['MA20'] + (std20 * boll_std)
    
    # 4. 综合信号 (三合一)
    # 允许 DTR 是红柱状态，不一定要当天翻红，只要是红的即可，配合另外两个条件
    condition1 = df['MACD_Hist'] > 0
    condition2 = df['Close'] > df['MA20']
    condition3 = df['Close'] >= df['Boll_Upper'] # 触碰或突破上轨
    
    df['DTR_Plus_Signal'] = condition1 & condition2 & condition3
    
    return df

--------------------------------------------------------------------------------
5. Fighting (三合一突破)
来源文档： 79 DTR再升级 量价因子加入 核心逻辑： DTR翻红 + 突破52日价格新高 + 突破52日成交量新高。
def calculate_fighting_strategy(df, period=52):
    """
    计算 Fighting 策略 (DTR升级版)
    """
    # 1. MACD DTR
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    dea = diff.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = 2 * (diff - dea)
    
    # DTR 红柱状态
    is_dtr_red = df['MACD_Hist'] > 0
    
    # 2. 52日价格新高 (不包含当前K线，看收盘价是否突破前52天的最高价)
    # shift(1) 是因为我们要突破的是“之前的”高点
    highest_price_52 = df['High'].rolling(window=period).max().shift(1)
    price_breakout = df['Close'] > highest_price_52
    
    # 3. 52日成交量新高
    highest_vol_52 = df['Volume'].rolling(window=period).max().shift(1)
    vol_breakout = df['Volume'] > highest_vol_52
    
    # 4. Fighting 信号
    df['Fighting_Signal'] = is_dtr_red & price_breakout & vol_breakout
    
    return df

--------------------------------------------------------------------------------
6. UA (Ultimate Amount / 天量战法)
来源文档： 83 当UlimateAmount(天量)来临时 核心逻辑： 出现历史(或250日)天量，标记该日最高价。后续突破该最高价为买点。
def calculate_ua_strategy(df, period=250):
    """
    计算 UA 天量策略
    注意：这是一个涉及状态保持的策略
    """
    # 1. 定义天量 (250日内最大成交量)
    # shift(1) 确保是和过去比较，如果是和包含今天在内比较直接用 rolling
    # 文档指出现天量当日
    df['Rolling_Max_Vol'] = df['Volume'].rolling(window=period).max()
    df['Is_UA'] = df['Volume'] == df['Rolling_Max_Vol']
    
    # 2. 记录天量日的最高价 (UA_High)
    # 使用 pandas 的 expanding 或 循环来处理这种“记忆”逻辑比较复杂
    # 这里使用 value_fill 方法：如果是UA日，记录High，否则NaN，然后向下填充
    
    df['UA_Target_Price'] = np.where(df['Is_UA'], df['High'], np.nan)
    df['UA_Target_Price'] = df['UA_Target_Price'].ffill() # 向下填充最近一次天量的最高价
    
    # 3. 突破信号
    # 当前收盘价 突破 最近一次天量的最高价
    # 且当前不是天量当日 (避免当日追高)
    df['UA_Breakout'] = (df['Close'] > df['UA_Target_Price']) & (df['Is_UA'] == False)
    
    # 过滤连续信号：只看刚突破的那一天
    df['UA_Buy_Signal'] = df['UA_Breakout'] & (df['UA_Breakout'].shift(1) == False)
    
    return df[['Close', 'Volume', 'Is_UA', 'UA_Target_Price', 'UA_Buy_Signal']]

--------------------------------------------------------------------------------
7. HMC (High-Momentum Channel)
来源文档： 52 量化 HMC教学 核心逻辑：
• 黄线 = 50日最高价 - 收盘价 (越小越好)
• 红线 = 收盘价 - EMA200 (越大越好)
• 信号 = 红线上穿黄线
def calculate_hmc_strategy(df):
    """
    计算 HMC 策略
    """
    # 1. 黄线: 50日最高价 - 收盘价
    hhv_50 = df['High'].rolling(window=50).max()
    df['HMC_Yellow'] = hhv_50 - df['Close']
    
    # 2. 红线: 收盘价 - EMA200
    ema_200 = df['Close'].ewm(span=200, adjust=False).mean()
    df['HMC_Red'] = df['Close'] - ema_200
    
    # 3. 信号: 红线上穿黄线
    # 今天红 > 黄 且 昨天 红 < 黄
    df['HMC_Signal'] = (df['HMC_Red'] > df['HMC_Yellow']) & \
                       (df['HMC_Red'].shift(1) <= df['HMC_Yellow'].shift(1))
    
    return df[['Close', 'HMC_Yellow', 'HMC_Red', 'HMC_Signal']]
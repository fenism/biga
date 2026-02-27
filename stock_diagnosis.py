
import requests
import json
import pandas as pd
import datetime

class StockDiagnoser:
    def __init__(self, api_key):
        self.api_key = api_key
        # Gemini 3 Pro endpoint (Preview)
        # Fallback list: gemini-3.1-pro-preview, gemini-1.5-pro, gemini-2.5-pro
        self.model = "gemini-3.1-pro-preview" 
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
    def get_fundamentals(self, code):
        """
        Fetch fundamental data using Akshare.
        Returns a dictionary or string summary.
        """
        try:
            import akshare as ak
            
            # 1. Base Info (Industry, Market Cap)
            info = ak.stock_individual_info_em(symbol=code)
            # info is DF with columns 'item', 'value'
            # item: 总市值, 流通市值, 行业, 上市时间, 总股本, 流通股
            
            info_dict = dict(zip(info['item'], info['value']))
            industry = info_dict.get('行业', 'Unknown')
            market_cap = info_dict.get('总市值', 0)
            
            # 2. Financials (Revenue, Profit)
            # stock_financial_abstract returns DataFrame with columns as dates
            fin = ak.stock_financial_abstract(symbol=code)
            
            # Extract Net Profit (归母净利润)
            # Find row where '指标' == '归母净利润'
            net_profit_row = fin[fin['指标'] == '归母净利润']
            revenue_row = fin[fin['指标'] == '营业总收入']
            
            summary = {
                "Industry": industry,
                "MarketCap": market_cap,
                "PE_TTM": "N/A",
                "Revenue_Growth": "N/A",
                "Profit_Growth": "N/A",
                "Latest_Report": "N/A"
            }
            
            if not net_profit_row.empty and not revenue_row.empty:
                # Get columns that look like dates (exclude '选项', '指标')
                date_cols = [c for c in fin.columns if c not in ['选项', '指标']]
                # Sort descending (usually they are already) but let's make sure
                date_cols = sorted(date_cols, reverse=True)
                
                latest_date = date_cols[0]
                prev_year_date = None
                
                # Try to find Year-over-Year comparison (Same period last year)
                # e.g. 20240930 vs 20230930
                try:
                    latest_dt = datetime.datetime.strptime(latest_date, "%Y%m%d")
                    target_prev = (latest_dt - datetime.timedelta(days=360)).strftime("%Y%m%d") # Roughly
                    # Find closest match or exact match
                    # Simpler: just take the 5th column if quarterly (index 4)
                    if len(date_cols) > 4:
                        prev_year_date = date_cols[4] 
                except:
                    pass
                
                summary['Latest_Report'] = latest_date
                
                # Net Profit
                np_latest = float(net_profit_row[latest_date].values[0])
                if prev_year_date and prev_year_date in net_profit_row.columns:
                    np_prev = float(net_profit_row[prev_year_date].values[0])
                    if np_prev != 0:
                        summary['Profit_Growth'] = f"{((np_latest/np_prev)-1)*100:.2f}%"
                        
                # Revenue
                rev_latest = float(revenue_row[latest_date].values[0])
                if prev_year_date and prev_year_date in revenue_row.columns:
                    rev_prev = float(revenue_row[prev_year_date].values[0])
                    if rev_prev != 0:
                        summary['Revenue_Growth'] = f"{((rev_latest/rev_prev)-1)*100:.2f}%"
                        
                # PE TTM Calculation
                # Sum last 4 quarters if possible?
                # Simplify: If Q3 (0930), TTM = Q3_Current + (Year_Prev - Q3_Prev)
                # This is complex to fetch all parts.
                # Alternative: Just use Market Cap / (Latest Quarter * (365/days))? Too rough.
                # Let's just provide the raw latest profit and let AI guess or just say "Latest Profit".
                pass

            return summary

        except Exception as e:
            return {"Error": str(e)}

    def generate_report(self, df, code, name, strategy_signals):
        """
        Generate a comprehensive analysis report using Gemini.
        """
        if df.empty:
            return "数据为空，无法分析。"
            
        # 1. Prepare Data Summary
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else last_row
        
        # Trend
        ma20_trend = "Up" if last_row['MA20'] > prev_row['MA20'] else "Down"
        price_trend = "Bullish" if last_row['close'] > last_row['MA20'] else "Bearish"
        
        # Recent 5 days
        recent_data = df.tail(5)[['date', 'open', 'high', 'low', 'close', 'volume', 'MA20']].to_string()
        
        # Strategy Signals Summary
        active_strategies = []
        if strategy_signals is not None and not strategy_signals.empty:
            # Check last row signals
            last_sig = strategy_signals.iloc[-1]
            for col in last_sig.index:
                if last_sig[col]:
                    active_strategies.append(col)
        
        strategies_text = ", ".join(active_strategies) if active_strategies else "None"
        
        # Detailed Indicators
        indicators_info = f"""
        - MACD: DIF={last_row.get('DIF',0):.3f}, DEA={last_row.get('DEA',0):.3f}, Hist={last_row.get('MACD_Hist',0):.3f}
        - KDJ: K={last_row.get('K',0):.1f}, D={last_row.get('D',0):.1f}, J={last_row.get('J',0):.1f}
        - RSI: RSI6={last_row.get('RSI6',0):.1f}
        - WR: {last_row.get('WR',0):.1f}
        - CCI: {last_row.get('CCI',0):.1f}
        - Volume: Current={last_row['volume']:.0f}, MA20={last_row.get('Vol_MA20',0):.0f}
        - Position: Close={last_row['close']}, MA20={last_row['MA20']:.2f}, EMA200={last_row.get('EMA200',0):.2f}
        - RKing State: {last_row.get('RKing_State', 0)} (1=Long, -1=Short)
        """
        
        # Fetch Fundamentals
        fund_data = self.get_fundamentals(code)
        fund_text = ""
        if isinstance(fund_data, dict) and "Error" not in fund_data:
             try:
                 mcap_val = float(fund_data['MarketCap']) / 100000000 if fund_data['MarketCap'] else 0
                 fund_text = f"""
                 - **行业**: {fund_data['Industry']}
                 - **总市值**: {mcap_val:.2f} 亿
                 - **最新财报日期**: {fund_data['Latest_Report']}
                 - **营收增长 (YoY)**: {fund_data['Revenue_Growth']}
                 - **净利增长 (YoY)**: {fund_data['Profit_Growth']}
                 """
             except:
                 fund_text = str(fund_data)
        else:
             fund_text = "基本面数据获取失败或暂无。"

        # 2. Construct Prompt
        prompt_text = f"""
        你是一个专业的A股量化交易分析师。请根据以下提供的股票数据，对 [{code}] {name} 进行一份详细的 **个股诊断报告**。
        
        ### 1. 基础数据
        - **代码**: {code} - {name}
        - **最新日期**: {last_row['date']}
        - **现价**: {last_row['close']} (涨跌幅: {(last_row['close']/prev_row['close']-1)*100:.2f}%)
        - **趋势状态**: MA20 {ma20_trend}, Price vs MA20: {price_trend}
        
        ### 2. 基本面概况 (Fundamental)
        {fund_text}
        
        ### 3. 近期行情 (最后5日)
        {recent_data}
        
        ### 4. 技术指标详情
        {indicators_info}
        
        ### 5. 量化策略信号触发 (今日)
        - **触发策略**: {strategies_text}
        
        ### 分析要求
        请基于以上数据，从以下几个维度进行深度分析 (使用Markdown格式):
        1.  **基本面简评**: 结合行业地位、市值规模和最新的业绩增长情况，评估公司的基本面素质。注意：如果数据缺失则跳过。
        2.  **趋势研判**: 判断当前是处于上涨、下跌还是震荡阶段？均线系统和趋势指标 (RKing, EMA200) 指向什么方向？
        3.  **资金面分析**: 通过成交量和量价关系 (Volume, OBV等概念)，判断资金是在流入还是流出？是否有背离？
        4.  **技术面信号**: 解读 KDJ, MACD, RSI, WR, CCI 等指标的状态（超买/超卖/背离/金叉死叉）。
        5.  **策略验证**: 结合触发的量化策略 (如 {strategies_text})，评估其有效性和当前信号的可靠性。如果不涉及策略，请说明当前形态。
        6.  **综合操作建议**: 
            -   **评分**: 0-100分
            -   **建议**: 买入 / 增持 / 持有 / 减仓 / 卖出 / 观望
            -   **止损位/支撑位**: 给出具体的参考价格。
            -   **止盈位/压力位**: 给出具体的参考价格。
        
        **风格要求**: 专业、客观、逻辑清晰，避免模棱两可的废话。
        """
        
        # 3. Call API
        payload = {
            "contents": [{
                "parts": [{"text": prompt_text}]
            }]
        }
        
        headers = {'Content-Type': 'application/json'}
        
        try:
            # Gemini 3 Pro might be slow, increase timeout
            response = requests.post(self.url, headers=headers, data=json.dumps(payload), timeout=90)
            
            if response.status_code == 200:
                result = response.json()
                try:
                    return result['candidates'][0]['content']['parts'][0]['text']
                except (KeyError, IndexError):
                    return f"API 返回结构异常: {result}"
            else:
                return f"API 请求失败 (Code {response.status_code}): {response.text}"
                
        except Exception as e:
            return f"请求发生错误: {str(e)}"

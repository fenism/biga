from amarket_framework.tencent_loader import TencentLoader
from amarket_framework.macro_loader import MacroLoader
from amarket_core.ai_analyst import GeminiAnalyst
from amarket_core import indicators
import pandas as pd

class MarketAnalyzer:
    def __init__(self, api_key=None, model_name='gemini-3.1-pro-preview'):
        self.loader = TencentLoader()
        self.macro_loader = MacroLoader()
        self.ai = GeminiAnalyst(api_key=api_key, model_name=model_name)
        
    def analyze_market_status(self, skip_ai=False):
        """
        Main analysis function returning a dict of signals and AI commentary.
        """
        try:
            # Define boards (Tencent codes: sh000001=上证, sz399001=深证, sz399006=创业板)
            boards = {
                "sh": {"code": "sh000001", "name": "上证指数"},
                "sz": {"code": "sz399001", "name": "深证成指"},
                "cyb": {"code": "sz399006", "name": "创业板指"}
            }
            
            results = {}
            latest_date = None
            
            # 1. Fetch Real-time Quotes Batch
            codes = [b["code"] for b in boards.values()]
            realtime_df = self.loader.fetch_realtime_quotes(codes)
            
            # 2. Fetch Macro Data (Liquidity)
            margin_data = self.macro_loader.fetch_market_margin()
            money_supply = self.macro_loader.fetch_money_supply()
            
            # 3. Analyze per board
            for key, info in boards.items():
                code = info["code"]
                
                # A. Real-time Snapshot
                if realtime_df.empty:
                    rt_row = None
                else:
                    rt_row = realtime_df[realtime_df['code'] == code.replace("sh","").replace("sz","")]
                    if rt_row.empty: 
                         # Try full match if replace didn't work (though loader handles this)
                         rt_row = realtime_df[realtime_df['code'] == code]
                
                # B. Historical K-Line (for EMA200 - needs at least 200 days of data)
                kline_df = self.loader.fetch_k_line(code, day_count=400)
                
                if kline_df.empty:
                    results[key] = {"error": "Failed to fetch k-line data"}
                    continue
                
                # Append real-time price to k-line for latest indicator calc? 
                # Ideally yes, but for simplicity we rely on k-line end + real-time quote deviation
                # Actually, Tencent k-line usually includes today's partial candle if market is open.
                
                df = kline_df
                latest_date = df.index[-1]
                
                # Use real-time price if available and valid (market open), else fallback to K-line (market close)
                # Logic: If RT volume is effectively 0 or price is 0, assume non-trading/pre-market -> use last close
                current_price = df['close'].iloc[-1]
                current_vol = df['volume'].iloc[-1]
                
                is_realtime_valid = False
                if rt_row is not None and not rt_row.empty:
                    rt_price = float(rt_row.iloc[0]['close'])
                    rt_vol = float(rt_row.iloc[0]['volume'])
                    if rt_price > 0 and rt_vol > 0:
                        current_price = rt_price
                        current_vol = rt_vol # Note: RT vol is cumulative for the day
                        is_realtime_valid = True
                
                # If falling back to K-line (non-trading), ensure we depend on the last completed day if today is not started
                
                # --- PATCH REALTIME DATA ---
                # To ensure Relative Strength and MA calculations use the LATEST price (Real-time),
                # we must update the dataframe's last close price if we have valid real-time data.
                if is_realtime_valid:
                    # Check if K-line index matches today, or if we need to append?
                    # Tencent K-line usually contains today.
                    # We simply overwrite the last close with the accurate RT price.
                    # This ensures indicators.calculate_relative_strength uses the RT price.
                    df.at[df.index[-1], 'close'] = current_price
                    # NOTE: Do NOT overwrite volume! Real-time volume units (shares) differ from
                    # K-line volume units (hands), causing 100x inflation and chart corruption.
                    # df.at[df.index[-1], 'volume'] = current_vol  # BUGFIX: Commented out
                
                # ---------------------------
                
                # --- Analysis ---
                
                # Step 1: Funding (Water)
                vol_ma20 = df['volume'].rolling(20).mean()
                funding = {
                    "value": current_vol,
                    "ma20": vol_ma20.iloc[-1],
                    "status": "放量" if current_vol > vol_ma20.iloc[-1] else "缩量",
                    "description": "成交量 vs 20日均量"
                }
                
                # Step 2: Sentiment (NHR/Bias)
                # NHR requires scanning all stocks, which is heavy. 
                # Proxy: Use Bias as "Overheat/Panic" gauge for the index itself.
                price_ma20 = df['close'].rolling(20).mean()
                bias_20 = (current_price - price_ma20.iloc[-1]) / price_ma20.iloc[-1] * 100
                
                # Panic Index Proxy: If drop > 3% in a day or bias < -5
                panic_score = 0
                pct_chg = df['pctChg'].iloc[-1]
                if pct_chg < -3: panic_score += 50
                if bias_20 < -5: panic_score += 40
                
                sentiment_status = "过热" if bias_20 > 5 else ("极度恐慌" if bias_20 < -7 else ("恐慌" if bias_20 < -5 else "中性"))
                sentiment = {
                    "score": bias_20,
                    "panic_score": panic_score, # Proxy for "Panic Index"
                    "status": sentiment_status,
                    "description": "乖离率 & 暴跌"
                }

                # Step 3: Trend (EMA200) - The "Red Line"
                ema200 = indicators.calculate_ema(df['close'], span=200)
                trend_status = "牛市 (做多)" if current_price > ema200.iloc[-1] else "熊市 (防守)"
                trend = {
                    "current_price": current_price,
                    "ema200": ema200.iloc[-1],
                    "status": trend_status,
                    "series": ema200
                }
                
                # Step 4: Timing (Vol)
                vol_series = indicators.calculate_volatility(df)
                is_contracting, rank = indicators.detect_volatility_contraction(vol_series)
                timing = {
                    "volatility_rank": rank,
                    "is_contracting": is_contracting,
                    "status": "即将变盘" if is_contracting else "波动扩大",
                    "description": "波动率收敛"
                }
                
                results[key] = {
                    "name": info["name"],
                    "data": df,
                    "funding": funding,
                    "sentiment": sentiment,
                    "trend": trend,
                    "timing": timing
                }

            # Step 5: Style (Relative Strength)
            if "sh" in results and "cyb" in results and "data" in results["sh"] and "data" in results["cyb"]:
                df_sh = results["sh"]["data"]
                df_cyb = results["cyb"]["data"]
                rs_line, rs = indicators.calculate_relative_strength(df_cyb['close'], df_sh['close'])
                current_rs = rs_line.iloc[-1]
                prev_rs = rs_line.iloc[-2]
                
                # Logic: Suggested style based on Trend (RS vs MA20)
                # If RS is above its 20-day MA, Growth is leading.
                rs_ma20 = rs_line.rolling(20).mean()
                current_ma20 = rs_ma20.iloc[-1]
                
                is_growth_stronger = current_rs > current_ma20
                
                # Determine trend strength description
                if is_growth_stronger:
                     trend_desc = "强化 (RS > MA20)" if current_rs > prev_rs else "震荡 (RS > MA20)"
                else:
                     trend_desc = "弱势 (RS < MA20)"
                
                style = {
                   "rs_value": current_rs,
                   "trend": trend_desc, 
                   "suggestion": "成长/科技 (创业板)" if is_growth_stronger else "价值/蓝筹 (沪指)",
                   "rs_line": rs_line,
                   "rs_ma20": rs_ma20, # Pass MA20 for charting
                   "is_growth": is_growth_stronger
                }
            else:
                style = {"error": "Insufficient data"}

            # Step 6: AI Commentary
            ai_commentary = "AI 点评已跳过 (请手动点击刷新)"
            if not skip_ai:
                # Prepare context for AI
                # Use Shanghai Index as the "Market" representative
                sh_data = results.get("sh", {})
                ai_context = {
                    "margin_balance": f"{margin_data.get('margin_balance', 0):.2f}B",
                    "m1_m2_scissors": f"{money_supply.get('scissors', 0):.2f}%",
                    "nhr": "N/A (Requires Full Market Scan)", # Placeholder
                    "panic_index": sh_data.get("sentiment", {}).get("status", "N/A"),
                    "trend_status": sh_data.get("trend", {}).get("status", "N/A")
                }
                
                ai_commentary = self.ai.analyze_market(ai_context)

            return {
                "date": latest_date,
                "boards": results,
                "style": style,
                "macro": {
                    "margin": margin_data,
                    "money": money_supply
                },
                "ai_commentary": ai_commentary
            }
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}", "boards": {}, "style": {}}

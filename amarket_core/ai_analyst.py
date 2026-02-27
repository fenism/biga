import google.generativeai as genai
import os

class GeminiAnalyst:
    def __init__(self, api_key=None, model_name='gemini-3.1-pro-preview'):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None

    def list_models(self):
        """List available models for the configured API key."""
        if not self.api_key:
            return ["API Key not set"]
        try:
            return [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        except Exception as e:
            return [f"Error listing models: {str(e)}"]

    def analyze_market(self, context_data):
        """
        Generate market commentary based on structured context.
        context_data: dict containing liquidity, sentiment, and trend metrics.
        """
        if not self.model:
            return "Please configure Gemini API Key in the sidebar to enable AI analysis."

        prompt = f"""
        你是一位专业的宏观量化交易员。请根据依然下A股市场数据进行宏观点评。
        
        **分析目标**：判断当前市场是否适合开仓交易（避免在“泥沙俱下”时接飞刀）。请严格遵循以下分析框架：
        
        1. **宏观流动性监测 (Liquidity Matrix)**：
           - **融资余额**：{context_data.get('margin_balance', 'N/A')}
           - **M1/M2 剪刀差**：{context_data.get('m1_m2_scissors', 'N/A')}
           - *判断逻辑*：融资余额快速上升=散户过热风险；剪刀差负值扩大=流动性陷阱。
           
        2. **市场情绪 (Sentiment)**：
           - **NHR (新高/新低)**：{context_data.get('nhr', 'N/A')}
           - **恐慌指数 (下跌家数占比)**：{context_data.get('panic_index', 'N/A')}
           - *判断逻辑*：NHR低且新低占比>20%=底部恐慌；恐慌指数>80%=人性极值（短线博弈点）。
           
        3. **牛熊分界线 (Trend)**：
           - **指数 vs EMA200**：{context_data.get('trend_status', 'N/A')}
           - *判断逻辑*：EMA200之上=做多安全区；EMA200之下=空仓/超跌反弹区（不可逾越的红线）。

        **输出格式要求**：
        - **核心结论 (Verdict)**：[安全开仓 / 观望等待 / 接飞刀博弈]
        - **逻辑分析**：用2-3句话串联上述数据解释结论。
        - **风险提示**：一句话指出当前最大的风险点。
        
        请用中文回答，风格专业、犀利。
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg or "leaked" in error_msg.lower():
                return "🚨 **Security Alert**: Your API Key was reported as leaked/invalid by Google. Please generate a NEW key at [Google AI Studio](https://aistudio.google.com/) and update your Streamlit Secrets."
            return f"AI Analysis failed: {error_msg}"

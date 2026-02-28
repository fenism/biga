"""
Trading Manager - Portfolio & Watchlist Management with SOP Analysis
Handles JSON-based persistence and Gemini AI trading advice.
"""

import json
import os
import datetime
import requests
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRADING_DATA_FILE = os.path.join(DATA_DIR, "trading_data.json")
SOP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "交易sop.md")

# A-Share Transaction Costs
COMMISSION_RATE = 0.0003  # 0.03% (万三)
STAMP_DUTY_RATE = 0.0005  # 0.05% (印花税, Sell only)


class TradingManager:
    def __init__(self):
        self.data = self._load()

    # ── Persistence ────────────────────────────────────────────────
    def _load(self):
        if os.path.exists(TRADING_DATA_FILE):
            try:
                with open(TRADING_DATA_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"capital": 100000, "watchlist": [], "portfolio": []}

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(TRADING_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    # ── Capital ────────────────────────────────────────────────────
    def get_capital(self):
        return self.data.get("capital", 100000)

    def set_capital(self, amount):
        self.data["capital"] = amount
        self._save()

    # ── Watchlist ──────────────────────────────────────────────────
    def get_watchlist(self):
        return self.data.get("watchlist", [])

    def add_to_watchlist(self, code, name, signal_date="", strategies="",
                         signal_high=0, signal_low=0):
        wl = self.data.setdefault("watchlist", [])
        # Avoid duplicate
        if any(item["code"] == code for item in wl):
            return False
        wl.append({
            "code": code,
            "name": name,
            "added_date": datetime.date.today().isoformat(),
            "signal_date": signal_date,
            "strategies": strategies,
            "signal_high": signal_high,
            "signal_low": signal_low,
        })
        self._save()
        return True

    def remove_from_watchlist(self, code):
        wl = self.data.get("watchlist", [])
        self.data["watchlist"] = [w for w in wl if w["code"] != code]
        self._save()

    # ── Portfolio ─────────────────────────────────────────────────
    def get_portfolio(self):
        return self.data.get("portfolio", [])

    def add_position(self, code, name, buy_price, quantity, buy_date, stop_loss=0, notes=""):
        pf = self.data.setdefault("portfolio", [])
        pf.append({
            "code": code,
            "name": name,
            "buy_price": buy_price,
            "quantity": quantity,
            "buy_date": buy_date,
            "stop_loss": stop_loss,
            "notes": notes,
        })
        self._save()
        return True

    def remove_position(self, index):
        pf = self.data.get("portfolio", [])
        if 0 <= index < len(pf):
            pf.pop(index)
            self._save()
            return True
        return False

    def update_position(self, index, **kwargs):
        pf = self.data.get("portfolio", [])
        if 0 <= index < len(pf):
            pf[index].update(kwargs)
            self._save()
            return True
        return False

    # ── Portfolio Summary ─────────────────────────────────────────
    def get_portfolio_summary(self, current_prices: dict):
        """
        current_prices: {code: latest_close_price}
        Returns summary dict with net-of-fee metrics.
        """
        capital = self.get_capital()
        portfolio = self.get_portfolio()
        total_cost = 0
        total_value = 0
        nodes = []
        today = datetime.datetime.now().strftime("%Y-%m-%d")

        for p in portfolio:
            # 1. Gross metrics
            cost = p["buy_price"] * p["quantity"]
            current = current_prices.get(p["code"], p["buy_price"])
            value = current * p["quantity"]
            
            # 2. Fee Estimation
            buy_fee = cost * COMMISSION_RATE
            sell_fee = value * (COMMISSION_RATE + STAMP_DUTY_RATE)
            net_value = value - sell_fee
            net_pnl = net_value - (cost + buy_fee)
            net_pnl_pct = (net_pnl / (cost + buy_fee) * 100) if (cost + buy_fee) > 0 else 0
            
            # 3. T+1 Check
            can_sell = p.get("buy_date", "") < today

            total_cost += (cost + buy_fee)
            total_value += net_value
            
            nodes.append({
                **p,
                "current_price": current,
                "market_value": value,
                "net_value": net_value,
                "pnl": net_pnl,
                "pnl_pct": net_pnl_pct,
                "exposure_pct": (value / capital * 100) if capital > 0 else 0,
                "can_sell": can_sell,
                "is_t1_locked": not can_sell
            })

        return {
            "capital": capital,
            "total_cost": total_cost,
            "total_value": total_value,
            "total_pnl": total_value - total_cost,
            "total_pnl_pct": ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0,
            "available_cash": capital - total_cost,
            "total_exposure_pct": (total_cost / capital * 100) if capital > 0 else 0,
            "position_count": len(portfolio),
            "positions": nodes,
        }

    # ── SOP Text ──────────────────────────────────────────────────
    @staticmethod
    def _load_sop():
        try:
            with open(SOP_FILE, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return "（SOP文件未找到）"

    # ── Gemini Helpers ────────────────────────────────────────────
    @staticmethod
    def _call_gemini(api_key, prompt):
        model = "gemini-3.1-pro-preview"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            resp = requests.post(url, headers={"Content-Type": "application/json"},
                                 data=json.dumps(payload), timeout=90)
            if resp.status_code == 200:
                result = resp.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            return f"API 请求失败 (Code {resp.status_code}): {resp.text}"
        except Exception as e:
            return f"请求发生错误: {str(e)}"

    # ── Watchlist SOP Analysis (Steps 1-2) ────────────────────────
    def analyze_watchlist_stock(self, api_key, code, name, capital, df, sigs):
        """Analyse whether to open a position, based on SOP Steps 1-2."""
        sop_text = self._load_sop()

        # Find watchlist entry for signal info
        wl_entry = next((w for w in self.get_watchlist() if w["code"] == code), {})
        signal_date = wl_entry.get("signal_date", "")
        strategies = wl_entry.get("strategies", "")
        signal_high = wl_entry.get("signal_high", 0)
        signal_low = wl_entry.get("signal_low", 0)

        if df.empty:
            return "无行情数据，无法分析。"

        last = df.iloc[-1]
        recent_data = df.tail(5)[["date", "open", "high", "low", "close", "volume"]].to_string()

        # Active signals on latest bar
        active_sigs = []
        if sigs is not None and not sigs.empty:
            last_sig = sigs.iloc[-1]
            for c in sigs.columns:
                if c.startswith("Signal_") and last_sig[c]:
                    active_sigs.append(c.replace("Signal_", ""))

        # Core Facts for AI to prevent math errors
        risk_diff = signal_high - signal_low if signal_low > 0 else signal_high * 0.05
        risk_pct = (risk_diff / last['close']) * 100 if last['close'] > 0 else 0
        
        # 2% Risk Calculation (2000 for 100k capital)
        suggested_qty = int(2000 / risk_diff / 100) * 100 if risk_diff > 0 else 0
        target_1_1 = signal_high + risk_diff
        target_2_1 = signal_high + (2 * risk_diff)

        indicators = f"- 现价: {last['close']:.2f}, MA20: {last.get('MA20',0):.2f}, EMA200: {last.get('EMA200',0):.2f}\n- 成交量: {last['volume']:.0f}, Vol_MA20: {last.get('Vol_MA20',0):.0f}"

        prompt = f"""你是一个严格遵循 Royal 交易体系的A股交易员。请根据以下SOP规则和股票数据，对自选股 [{code}] {name} 进行**建仓评估**。
 
 ### Royal 交易执行 SOP
 {sop_text}
 
 ### 【核心事实数据】 (禁止AI自行计算，必须以此为准)
 1. **当前价格**: {last['close']:.2f} 元
 2. **参照点 (信号日最高价)**: {signal_high:.2f} 元
 3. **初始止损位 (信号日最低价)**: {signal_low:.2f} 元
 4. **单股风险距离**: {risk_diff:.2f} 元 ({risk_pct:.1f}%)
 5. **基于2%风险原则的建议买入股数**: {suggested_qty} 股
 6. **SOP 目标位**:
    - 保本位 (1:1): {target_1_1:.2f} 元
    - 减仓位 (2:1): {target_2_1:.2f} 元
 
 ### 信号信息
 - **信号日期**: {signal_date}
 - **触发策略**: {strategies}
 
 ### 近5日行情
 {recent_data}
 
 ### 最新技术指标
 {indicators}
 
 ### 最新量化信号
 {', '.join(active_sigs) if active_sigs else '无'}

### 分析要求（严格按SOP执行）
请依次完成以下分析，并**引用上述【核心事实数据】中的准确数值**，禁止幻觉：

1. **第一步：止损价确认** — 确认止损位 {signal_low}，计算与现价的实际安全垫。
2. **第二步：仓位计算（2%法则）** — 解释为何系统建议买入 {suggested_qty} 股，并检查总额是否超过20%仓位上限。
3. **第三步：参照点突破判断** — 当前价 {last['close']:.2f} 是否已站稳参照点 {signal_high:.2f}？
4. **综合建议** — 给出明确结论。

**总结风格**: 极其冷酷、机械化、数据驱动。"""

        return self._call_gemini(api_key, prompt)

    # ── Portfolio SOP Analysis (Steps 3-5) ────────────────────────
    def analyze_portfolio_stock(self, api_key, code, name, position_info, df, sigs):
        """Analyse whether to hold/add/reduce/exit, based on SOP Steps 3-5."""
        sop_text = self._load_sop()

        if df.empty:
            return "无行情数据，无法分析。"

        last = df.iloc[-1]
        recent_data = df.tail(10)[["date", "open", "high", "low", "close", "volume"]].to_string()

        buy_price = position_info.get("buy_price", 0)
        stop_loss = position_info.get("stop_loss", 0)
        quantity = position_info.get("quantity", 0)
        buy_date = position_info.get("buy_date", "")
        current_price = last["close"]
        cost = buy_price * quantity
        pnl = (current_price - buy_price) * quantity
        pnl_pct = ((current_price / buy_price) - 1) * 100 if buy_price > 0 else 0
        risk_per_share = buy_price - stop_loss if stop_loss > 0 else 0
        reward = current_price - buy_price
        rr_ratio = (reward / risk_per_share) if risk_per_share > 0 else 0

        # Active signals on latest bar
        active_sigs = []
        if sigs is not None and not sigs.empty:
            last_sig = sigs.iloc[-1]
            for c in sigs.columns:
                if c.startswith("Signal_") and last_sig[c]:
                    active_sigs.append(c.replace("Signal_", ""))

        indicators = f"""
- 现价: {current_price}, MA20: {last.get('MA20',0):.2f}, EMA200: {last.get('EMA200',0):.2f}
- MACD: DIF={last.get('DIF',0):.3f}, DEA={last.get('DEA',0):.3f}, Hist={last.get('MACD_Hist',0):.3f}
- KDJ: K={last.get('K',0):.1f}, D={last.get('D',0):.1f}, J={last.get('J',0):.1f}
- RSI6={last.get('RSI6',0):.1f}, WR={last.get('WR',0):.1f}
- RKing State: {last.get('RKing_State',0)} (1=多头, -1=空头)
- 成交量: {last['volume']:.0f}, Vol_MA20: {last.get('Vol_MA20',0):.0f}
- 250日最高量: {last.get('Max_Vol_250',0):.0f}
"""
        # Core Facts for AI
        risk_per_share = buy_price - stop_loss if stop_loss > 0 else buy_price * 0.05
        target_1_1 = buy_price + risk_per_share
        target_2_1 = buy_price + (2 * risk_per_share)
        
        prompt = f"""你是一个严格遵循 Royal 交易体系的A股交易员。请对持仓股 [{code}] {name} 进行**持仓管理分析**。

### Royal 交易执行 SOP
{sop_text}

### 【核心持仓事实】 (禁止AI自行计算，必须以此为准)
1. **当前价格**: {current_price:.2f} 元
2. **买入成本**: {buy_price:.2f} 元
3. **当前净盈亏**: {pnl:.2f} 元 ({pnl_pct:+.2f}%) (已扣除万三佣金与印花税)
4. **初始止损位**: {stop_loss:.2f} 元
5. **单位风险 (R)**: {risk_per_share:.2f} 元
6. **盈亏比状态**: {rr_ratio:.2f} R
7. **SOP 目标检查位**:
   - 保本位 (1:1目标): {target_1_1:.2f} 元
   - 减仓位 (2:1目标): {target_2_1:.2f} 元
8. **当前持仓状态**: {quantity} 股
9. **T+1 状态**: {"✅ 可卖出" if position_info.get("can_sell", True) else "❌ T+1 锁定中 (今日买入)"}

### 近10日行情
{recent_data}

### 最新技术指标
{indicators}

### 最新量化信号
{', '.join(active_sigs) if active_sigs else '无'}

### 分析要求（严格按 SOP 3-5 步执行）
请引用上述【核心持仓事实】数据，根据SOP规则给出操作指令：

1. **第一步：止损检查** — 现价 {current_price:.2f} 是否安全？(当前止损位 {stop_loss:.2f})。
2. **第二步：保护/减仓逻辑**
   - 是否已达保本位 {target_1_1:.2f}？如果是，明确建议止损上移。
   - 是否已达减仓位 {target_2_1:.2f}？如果是，明确建议减半仓。
3. **第三步：趋势强度评估** — 结合MA20（{last.get('MA20',0):.2f}）和量能。
4. **综合建议** — 明确给出下一步动作（持仓/推保护/减仓/离场）。

**总结风格**: 军事化风格，禁止废话和模糊辞令。"""

        return self._call_gemini(api_key, prompt)

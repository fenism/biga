import pandas as pd
import numpy as np
import datetime
from typing import List, Dict, Tuple, Optional

from indicators import Indicators
from exit_signals import ExitSignals

class BacktestEngine:
    """
    回测引擎 (Backtest Engine)
    
    用于模拟交易策略，验证策略在历史数据上的表现。
    包含资金管理、仓位控制、止盈止损逻辑。
    """
    
    def __init__(self, data_loader, initial_capital: float = 100000.0):
        """
        :param data_loader: DataLoader 实例，用于获取K线数据
        :param initial_capital: 初始资金
        """
        self.loader = data_loader
        self.initial_capital = initial_capital
        
        # State
        self.cash = initial_capital
        self.positions = {}  # {code: {'amount': int, 'cost': float, 'entry_date': date, 'high_since_entry': float}}
        self.history = []    # 每日资产记录
        self.trades = []     # 交易记录

        # Transaction Fees (A-share Defaults)
        self.commission_rate = 0.0003 # 0.03% (万三)
        self.stamp_duty_rate = 0.0005 # 0.05% (印花税, 仅卖出)
        
    def run(self, signal_df: pd.DataFrame, start_date: str, end_date: str,
            stop_loss_pct: float = None, take_profit_pct: float = None,
            max_hold_days: int = None, position_sizing: float = 0.2,
            max_positions: int = 5, breakeven_activation_pct: float = 0.05,
            breakeven_fallback_pct: float = 0.005,
            commission_rate: float = 0.0003, stamp_duty_rate: float = 0.0005,
            use_sector_filter: bool = True,
            use_ambush_filter: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        执行回测
        
        :param signal_df: 信号DataFrame (包含 code, name, date, triggered_strategies)
        :param start_date: 回测开始日期
        :param end_date: 回测结束日期
        :param stop_loss_pct: 止损百分比 (e.g. 0.05 for 5%)
        :param take_profit_pct: 止盈百分比 (e.g. 0.10 for 10%)
        :param max_hold_days: 最大持仓天数
        :param position_sizing: 单只股票仓位比例 (0.0~1.0)
        :param max_positions: 最大同时持仓数
        :return: (equity_curve, trade_log)
        """
        # Reset state
        self.cash = self.initial_capital
        self.positions = {}
        self.history = []
        self.trades = []
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        
        # 1. 准备交易日历
        # 简单起见，使用信号中的日期范围，或者用大盘指数的日期
        index_df = self.loader.get_k_data("000001", start_date, end_date) # Load SH Index for calendar
        if index_df.empty:
            print("⚠️ 无法加载交易日历 (000001)，尝试使用信号日期")
            dates = sorted(signal_df['date'].unique())
        else:
            dates = sorted(index_df['date'].unique())
            
        # Ensure dates are datetime
        dates = pd.to_datetime(dates)
        
        # Group signals by date for fast access
        signals_by_date = signal_df.groupby('date')
        
        print(f"🚀 开始回测: {start_date} ~ {end_date}, 初始资金: {self.initial_capital}")
        
        # --- 0. PRE-FETCH ALL NEEDED DATA TO AVOID LOOP IO ---
        # We only need data for stocks that ACTUALLY have signals during this period!
        codes_with_signals = signal_df['code'].unique()
        self.market_data = {} # code -> DataFrame
        
        start_fetch = pd.to_datetime(start_date) - pd.Timedelta(days=15)
        end_dt = pd.to_datetime(end_date)
        
        for code in codes_with_signals:
            df = self.loader.get_k_data(code, start_fetch, end_dt, compute_indicators=False)
            if not df.empty:
                df = df.set_index('date').sort_index()
                self.market_data[code] = df
        
        print(f"✅ Pre-fetched data arrays for {len(self.market_data)} stocks.")
                
        for i, current_date in enumerate(dates):
            if i > 0 and i % 50 == 0:
                print(f"  ... simulating date {current_date.strftime('%Y-%m-%d')} ({i}/{len(dates)})")
            # --- 1. 处理持仓 (Sell Check) ---
            self._process_holdings(current_date, stop_loss_pct, take_profit_pct, max_hold_days, 
                                   breakeven_activation_pct, breakeven_fallback_pct)
            
            # --- 2. 处理开仓 (Buy Check) ---
            if current_date in signals_by_date.groups:
                daily_signals = signals_by_date.get_group(current_date)
                self._process_entry(current_date, daily_signals, position_sizing, max_positions, use_sector_filter, use_ambush_filter)
                
            # --- 3. 记录当日资产 ---
            total_value = self.cash
            holding_value = 0.0
            
            # 更新持仓市值 (简单起见使用前一日收盘价或假设当日收盘价)
            # 准确的做法是获取当日收盘价。
            # 这里为了性能，我们可能需要批量获取当日价格，或者在 _process_holdings 里顺便更新价格
            
            for code, pos in self.positions.items():
                # Try getting price from loader (cached?)
                # Ideally we pass a price_map or fetch one by one
                # Fetching one by one is slow. 
                # Optimization: In real backtest engine, we'd load all data matrix.
                # Here we fetch current price just for valuation? Or rely on Sell Logic's price?
                
                # For simplified daily valuation, let's try to get today's price.
                # If fail, use cost (imperfect but keeps it running).
                # To do it right: use loader.
                
                # Performance Hazard: getting k_data inside loop for valuation.
                # Let's Skip precise daily MTM (Mark-to-Market) for speed if needed, 
                # BUT Sell Logic *NEEDS* price.
                # So _process_holdings ALREADY fetched price. We should cache it for this step.
                
                last_price = pos.get('last_price', pos['cost']) # Updated in _process_holdings
                holding_value += pos['amount'] * last_price
                
            total_value += holding_value
            
            self.history.append({
                'date': current_date,
                'cash': self.cash,
                'positions_value': holding_value,
                'total_assets': total_value,
                'num_positions': len(self.positions)
            })
            
        # Finish
        # Finish
        equity_df = pd.DataFrame(self.history)
        trade_df = pd.DataFrame(self.trades)
        
        # VERY IMPORTANT: Release memory explicitly! 
        # Over 510 combinations, this 5000-stock matrix will cause MacOS OOM Kernel Panic.
        self.market_data.clear()
        import gc
        gc.collect()
        
        return equity_df, trade_df

    def _process_holdings(self, current_date, stop_loss, take_profit, max_hold, 
                          breakeven_activation_pct=0.05, breakeven_fallback_pct=0.005):
        """处理持仓：止盈、止损、最大持仓天数、更新最新价"""
        todays_prices = {} # code -> {open, close, high, low}
        
        # Copy keys to modify dict during iteration
        held_codes = list(self.positions.keys())
        
        for code in held_codes:
            pos = self.positions[code]
            
            # --- T+1 Rule Check ---
            # In A-share, you cannot sell on the same day you bought (T+1)
            if current_date <= pos['entry_date']:
                continue
            
            # 1. Get Market Data for Today + History for Indicators
            # Fast in-memory lookup instead of DataLoader fetch
            if code not in getattr(self, 'market_data', {}):
                continue
                
            full_df = self.market_data[code]
            try:
                # Optimized O(1) hash lookup instead of slicing the whole dataframe
                # Indicators and Exits are already pre-computed by DataLoader!
                row = full_df.loc[current_date]
            except KeyError:
                continue # Suspended or no data for today
                
            current_close = row['close']
            current_low = row['low']
            current_high = row['high']
            current_open = row['open']
            prev_close = row.get('prev_close', pos['cost']) # Fallback to cost if prev_close missing

            # --- Price Limit Check (Limit Down) ---
            # If a stock is locked at Limit Down, we cannot sell it.
            if self._is_at_price_limit(code, current_open, current_close, current_high, current_low, prev_close, mode='down'):
                # Update valuation but skip exit logic
                self.positions[code]['last_price'] = current_close
                continue
            
            # Store price for valuation
            self.positions[code]['last_price'] = current_close
            
            # Update High since entry (for trailing stop if needed)
            pos['high_since_entry'] = max(pos['high_since_entry'], current_high)
            
            # 2. Check Exits
            sell_signal = False
            sell_price = current_close
            reason = ""
            
            # A. Time Stop (Max Hold Days)
            days_held = (current_date - pos['entry_date']).days
            
            # B. Tight Breakout Stop (假突破斩仓)
            # If held for <= 3 days and drops below entry day's low, kill it instantly
            if days_held <= 3 and current_close < pos.get('entry_low', pos['cost'] * 0.95):
                sell_signal = True
                sell_price = current_close
                reason = "Tight Breakout Stop (短期跌破入场底)"
            
            # CRITICAL FIX: Do not forcefully terminate massive trend-following wings after 10 days!
            # Only trigger Time Stop if the stock is stagnant or losing money. Let winners run until Dynamic Exit.
            elif max_hold and days_held >= max_hold:
                if current_close < pos['cost'] * 1.05: # Allow letting it run indefinitely if it's up more than 5%
                    sell_signal = True
                    sell_price = current_close
                    reason = f"Time Stop ({max_hold}d No Trend)"
            
            # C. Dynamic Exits and Scaling Out (SOP Level 4/5)
            if not sell_signal:
                # 1. Base Stop Loss = Day T Low (entry_low)
                # If not present, fallback to configured -stop_loss pct
                fallback_sl = pos['cost'] * (1 - stop_loss)
                sl_price = pos.get('entry_low', fallback_sl)
                stop_name = "Strict Initial Stop (Day T Low)"
                
                # SOP Fix: Never let the strict stop loss be below a reasonable fallback 
                if sl_price < pos['cost'] * 0.5: 
                    sl_price = pos['cost'] * (1 - stop_loss)
                    stop_name = f"Hard Stop Loss Fallback (-{stop_loss:.0%})"
                
                # 2. 1:1 Breakeven (推保护) 
                # 当浮动盈利金额 = 初始止损金额（即盈亏比达到 1:1）时，将止损位上移至 开仓成本价。
                initial_risk = pos['cost'] - pos.get('entry_low', pos['cost'] * 0.9)
                if initial_risk <= 0: initial_risk = pos['cost'] * 0.05
                
                if pos['high_since_entry'] >= pos['cost'] + initial_risk:
                    be_price = pos['cost'] * 1.005 # 起盈保本
                    if be_price > sl_price:
                        sl_price = be_price
                        stop_name = "Breakeven Stop (1:1 盈亏比推保护)"
                
                # 3. 2:1 Partial Take Profit (减仓落袋)
                # 当盈亏比达到 2:1 时，减仓 50%，锁定利润。
                if not pos.get('scaled_out', False) and current_high >= pos['cost'] + (2 * initial_risk):
                    # Use current high to trigger, but sell at a realistic level (max of open/2:1 price)
                    tp_trigger = pos['cost'] + (2 * initial_risk)
                    tp_price = max(current_open, tp_trigger)
                    
                    self._execute_sell(current_date, code, pos, tp_price, "2:1 Reward Scale-Out (减仓50%)", 0.5)
                    pos['scaled_out'] = True # Mark as scaled out
                    # After scaling out, the remaining position logic continues
                
                # 4. MA20 Final Exit (生命线离场)
                # 剩余仓位利用趋势线、20日均线...进行移动止损。
                ma20 = row.get('MA20', 0)
                if pos.get('scaled_out', False) and ma20 > 0 and current_close < ma20:
                    sell_signal = True
                    sell_price = current_close
                    reason = "Life Line Exit (跌破MA20生命线)"
                    self._execute_sell(current_date, code, pos, sell_price, reason, 1.0)
                
                # 5. UA Exit (Ultimate Stagnation - 高位天量滞涨)
                # If we have a UA_Exit signal (high volume + stall)
                if not sell_signal and row.get('UA_Exit', False):
                    # Only trigger if we are somewhat in profit or held for a few days
                    # to avoid selling on the entry day's high volume if it's already a breakout.
                    if (current_close > pos['cost'] * 1.05) or (days_held > 5):
                        sell_signal = True
                        sell_price = current_close
                        reason = "UA Stagnation Exit (高位天量滞涨)"
                        self._execute_sell(current_date, code, pos, sell_price, reason, 1.0)
                
                if not sell_signal and current_low <= sl_price:
                    sell_signal = True
                    # Execute sell at the moment price breaches the SL (or at open if gapped down)
                    sell_price = current_open if current_open < sl_price else sl_price
                    reason = stop_name
                    self._execute_sell(current_date, code, pos, sell_price, reason, 1.0)
            
            # Removed forced 50% Partial Take Profit clipping logic to allow massive 100%+ exponential running on full share counts
            
            
            # Dynamic Exit Signals (最后通牒) on remaining position
            if not sell_signal and code in self.positions:
                # Use pre-computed exits
                if 'Exits' in full_df.columns and pd.notna(row['Exits']) and row['Exits'] != '':
                    sell_signal = True
                    # Since it's EOD signals, we sell at close (or next open, using close for simplicity here)
                    sell_price = current_close
                    reason = str(row['Exits']) # e.g. 'Exit_Break_MA20'
                    self._execute_sell(current_date, code, self.positions[code], sell_price, reason, 1.0)
                    
            # 补漏：如果之前（如 Max Hold）设置了 sell_signal = True 但没执行，在这里执行
            if sell_signal and code in self.positions:
                # Avoid selling twice if the execute_sell was already triggered inside the loop block
                if reason in ["Hard Stop Loss (-10%)", "Breakeven Stop (起盈保本)"]:
                    pass
                else:
                    self._execute_sell(current_date, code, pos, sell_price, reason, 1.0)

    def _process_entry(self, current_date, signals, position_sizing, max_positions, use_sector_filter=True, use_ambush_filter=False):
        """处理开仓"""
        if len(self.positions) >= max_positions:
            return # Full
            
        # Prioritize signals by consensus (high confluence)
        if 'signal_count' in signals.columns:
            signals = signals.sort_values(by='signal_count', ascending=False)
        
        from market_env import MarketEnvironmentLoader
        env_loader = MarketEnvironmentLoader()

        for idx, row in signals.iterrows():
            if len(self.positions) >= max_positions:
                break
                
            code = row['code']
            name = row['name']
            triggered_strats_str = str(row.get('triggered_strategies', ''))
            
            # --- 🚀 MACRO ENVIRONMENT FILTER FOR STRONG STRATEGIES ---
            # If the signal was generated purely by STRONG strategies (Momentum/Breakout),
            # we must ensure the stock's Macro Board (SH/SZ/CYB) is in a healthy uptrend.
            # If it includes Weak strategies (Dip buying), we allow it to bypass because 
            # dip buying is specifically designed for bear markets/panics.
            
            # List of known Strong Strategies
            strong_strats = ['Z_Score', 'RS', 'TKOS', 'DTR_Plus', 'Fighting', 'UA', 'HMC']
            
            triggered_list = [s.strip() for s in triggered_strats_str.split(',') if s.strip()]
            
            # Consider it a "Strong-only" signal if ALL triggered strategies are inside the Strong list
            is_strong_only_signal = all(strat in strong_strats for strat in triggered_list)
            
            if is_strong_only_signal and len(triggered_list) > 0:
                # 1. Macro Board Health (SH/SZ/CYB)
                if not env_loader.is_board_healthy(code, current_date.strftime('%Y-%m-%d')):
                    continue 
                
                # 2. Sector Health (Industry Board)
                if use_sector_filter:
                    if not env_loader.is_sector_healthy(code, current_date.strftime('%Y-%m-%d')):
                        # print(f"🏭 SECTOR BLOCKED: {code} {name} (Sector in Downtrend)")
                        continue
            # ---------------------------------------------------------
            
            if code in self.positions:
                continue # Already held
                
            # Phase 4: Dynamic Entry & Sizing (T+1 Confirmation & 2% Kelly Rule)
            if code not in getattr(self, 'market_data', {}):
                continue
                
            full_df = self.market_data[code]
            
            try:
                # Get Day T data
                loc_idx = full_df.index.get_loc(current_date)
                row_T = full_df.iloc[loc_idx]
                
                # --- 🔮 AMBUSH FILTER (早鸟伏击条件) ---
                # Check Day T-1 for ambush characteristics.
                # Signal on Day T, ambush check on T-1, buy on T+1.
                # Requires at least one of: Calm(蓄力), Momentum(动量), Bottom(底部)
                if use_ambush_filter:
                    if loc_idx < 1:
                        continue  # No T-1 data available
                    t_minus_1_idx = loc_idx - 1
                    lookback = max(0, t_minus_1_idx - 120)
                    hist_df = full_df.iloc[lookback:t_minus_1_idx+1].copy()
                    hist_df = hist_df.reset_index()  # restore 'date' column for strategy funcs
                    
                    if len(hist_df) >= 30:
                        from strong_strategies import StrongStrategies
                        from weak_strategies import WeakStrategies
                        
                        has_ambush = False
                        try:
                            r1 = StrongStrategies.calculate_ambush_calm(hist_df)
                            if r1['Ambush_Calm_Signal'].iloc[-1]:
                                has_ambush = True
                        except: pass
                        
                        if not has_ambush:
                            try:
                                r2 = StrongStrategies.calculate_ambush_momentum(hist_df)
                                if r2['Ambush_Momentum_Signal'].iloc[-1]:
                                    has_ambush = True
                            except: pass
                            
                        if not has_ambush:
                            try:
                                r3 = WeakStrategies.strategy_ambush_bottom(hist_df)
                                if r3['Ambush_Bottom_Signal'].iloc[-1]:
                                    has_ambush = True
                            except: pass
                        
                        if not has_ambush:
                            continue  # Skip: no ambush on T-1
                    else:
                        continue  # Not enough history
                # --- END AMBUSH FILTER ---
                
                # Get Day T+1 data to check breakout
                if loc_idx + 1 >= len(full_df):
                    continue # No data for T+1
                    
                row_T1 = full_df.iloc[loc_idx + 1]
                t1_date = full_df.index[loc_idx + 1]
                
            except KeyError:
                continue

            t_high = row_T['high']
            t_low = row_T['low']
            t1_high = row_T1['high']
            t1_open = row_T1['open']
            
            # 1. T+1 Breakout Confirmation (不见兔子不撒鹰)
            if t1_high <= t_high:
                continue # Failed to break T's high, cancel the trade
                
            # --- Price Limit Check (Limit Up) ---
            # If a stock is locked at Limit Up, we cannot buy it on T+1.
            prev_close_T1 = row_T['close'] # Closing price of Day T
            if self._is_at_price_limit(code, t1_open, row_T1['close'], t1_high, row_T1['low'], prev_close_T1, mode='up'):
                continue

            # We buy the moment it breaches T's high (or at T+1 open if it gapped up)
            buy_price = max(t1_open, t_high)
            
            # 2. Risk-Based Position Sizing (using position_sizing as risk %)
            risk_cap = self.initial_capital * position_sizing
            stop_loss_price = t_low
            
            risk_per_share = buy_price - stop_loss_price
            if risk_per_share <= 0:
                risk_per_share = buy_price * 0.02
                stop_loss_price = buy_price * 0.98
                
            # target_shares * risk_per_share = risk_cap
            target_shares = int(risk_cap / risk_per_share / 100) * 100
            target_investment = target_shares * buy_price
            
            # Hard limit: Single position cannot exceed 20% of initial capital
            amt_cap = self.initial_capital * 0.20
            if target_investment > amt_cap:
                target_shares = int(amt_cap / buy_price / 100) * 100
                target_investment = target_shares * buy_price
                
            if self.cash < target_investment:
                target_shares = int(self.cash / buy_price / 100) * 100
                target_investment = target_shares * buy_price
                
            if target_shares <= 0 or target_investment < 1000:
                continue
            
            cost = target_shares * buy_price
            
            # Execute Buy on T+1
            # Deduct commission
            commission = cost * self.commission_rate
            self.cash -= (cost + commission)
            self.positions[code] = {
                'name': name,
                'amount': target_shares,
                'cost': buy_price,
                'entry_date': t1_date,       # Entering on T+1
                'entry_low': stop_loss_price, # Strict initial stop loss anchored to Day T low
                'high_since_entry': max(buy_price, row_T1['high']),
                'last_price': row_T1['close']
            }
            self.trades.append({
                'code': code,
                'name': name,
                'side': 'BUY',
                'date': t1_date, # Trade executed on T+1 date
                'price': buy_price,
                'amount': target_shares,
                'reason': row.get('triggered_strategies', 'Optimized Combo Entry') + " (T+1 Breakout)"
            })

    def _execute_sell(self, date, code, pos, price, reason, fraction=1.0):
        """执行卖出, fraction 表示卖出的比例 (默认 1.0 全卖)"""
        shares = pos['amount']
        revenue = shares * price * fraction
        
        # Deduct Fees: Commission + Stamp Duty (0.05%)
        commission = revenue * self.commission_rate
        stamp_duty = revenue * self.stamp_duty_rate
        net_revenue = revenue - commission - stamp_duty
        
        profit = net_revenue - (shares * fraction * pos['cost'])
        profit_pct = (price - pos['cost']) / pos['cost']
        
        self.cash += net_revenue
        
        if fraction >= 1.0:
            del self.positions[code]
        else:
            # Update position amount for partial sell
            pos['amount'] = pos['amount'] * (1.0 - fraction)
        
        self.trades.append({
            'code': code,
            'name': pos.get('name', ''),
            'side': 'SELL',
            'date': date,
            'price': price,
            'amount': int(shares * fraction),
            'reason': reason,
            'pnl': round(profit, 2),
            'pnl_pct': round(profit_pct, 4)
        })

    def _is_at_price_limit(self, code: str, open_p: float, close_p: float, high_p: float, low_p: float, 
                          prev_close: float, mode='up') -> bool:
        """
        Check if a stock is hit by price limit (10% or 20%).
        mode='up' : 涨停 (Limit Up)
        mode='down' : 跌停 (Limit Down)
        """
        # Determine multiplier (Main Board: 0.1, GEM/STAR: 0.2)
        multiplier = 0.1
        if code.startswith('30') or code.startswith('68'):
            multiplier = 0.2
            
        if mode == 'up':
            # Simplified Limit Up: If HIGH equals LIMIT price and cannot enter.
            limit_price = round(prev_close * (1 + multiplier) + 0.0001, 2)
            if high_p >= limit_price and low_p >= limit_price: # "One-word" Limit Up
                return True
            return False
        else:
            # Limit Down: Cannot SELL if price is locked at floor.
            limit_price = round(prev_close * (1 - multiplier) + 0.0001, 2)
            if low_p <= limit_price and high_p <= limit_price: # "One-word" Limit Down
                return True
            return False

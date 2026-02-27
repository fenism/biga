#!/usr/bin/env python3
"""
近两年的策略回测分析脚本
1. 重新构建信号缓存（覆盖近两年）
2. 运行策略组合回测优化
3. 输出高盈利策略组合报告
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations
import time
import json

# 确保导入路径正确
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from signal_cache import SignalCacheBuilder, SignalCacheReader
from data_loader import DataLoader
from backtest import BacktestEngine

# 配置参数
START_DATE = "2024-02-23"  # 近两年起始
END_DATE = "2026-02-23"    # 当前日期
INITIAL_CAPITAL = 100000.0  # 初始资金
DATA_DIR = "data/market_data"

# 强势策略列表
STRONG_STRATEGIES = [
    'Z_Score', 'RS', 'TKOS', 'DTR_Plus', 'Fighting', 
    'UA', 'HMC', 'CYC_MAX', 'RangeBreak', '20VMA', 
    'OBO', 'LCS', 'HPS', 'RKing'
]

# 弱势策略列表
WEAK_STRATEGIES = [
    'HLP3', 'Limit', 'RSI_Rev', 'Spring', 'Pinbar', 
    'Money_Flow', 'UA_Weak', 'Double_Vol',
    'Wyckoff', 'Vol_Min_120', 'Boll_Rev', '2B', 'ES'
]


def step1_rebuild_cache():
    """步骤1: 重新构建信号缓存"""
    print("=" * 80)
    print("步骤 1/3: 重新构建信号缓存")
    print("=" * 80)
    print(f"数据范围: {START_DATE} ~ {END_DATE}")
    print()
    
    builder = SignalCacheBuilder(data_dir=DATA_DIR)
    success = builder.build_all_signals(
        start_date=START_DATE,
        end_date=END_DATE,
        progress_callback=lambda c, t, msg: print(f"  {msg}") if c % 100 == 0 or c == t else None
    )
    
    if success:
        print("\n✅ 信号缓存构建完成！")
    else:
        print("\n❌ 信号缓存构建失败！")
    
    return success


def step2_run_backtest_optimization():
    """步骤2: 运行回测优化"""
    print("\n" + "=" * 80)
    print("步骤 2/3: 运行策略组合回测优化")
    print("=" * 80)
    
    # 回测参数
    bt_start = START_DATE
    bt_end = END_DATE
    initial_capital = INITIAL_CAPITAL
    stop_loss = 0.08
    take_profit = 0.20
    max_hold = 15
    position_sizing = 0.15
    max_positions = 5
    
    print(f"回测参数:")
    print(f"  初始资金: ¥{initial_capital:,.0f}")
    print(f"  止损比例: {stop_loss*100:.0f}%")
    print(f"  止盈比例: {take_profit*100:.0f}%")
    print(f"  最大持仓天数: {max_hold}")
    print(f"  单仓仓位: {position_sizing*100:.0f}%")
    print(f"  最大持仓数: {max_positions}")
    print()
    
    # 读取缓存
    cache_reader = SignalCacheReader()
    is_valid, msg = cache_reader.is_cache_valid()
    print(f"缓存状态: {msg}")
    print()
    
    # 准备策略组合
    combos_to_test = []
    
    # 单策略
    print("准备策略组合...")
    for s in STRONG_STRATEGIES:
        combos_to_test.append((['Z_Score'], []))  # 基准
        combos_to_test.append(([s], []))
    for w in WEAK_STRATEGIES:
        combos_to_test.append(([], [w]))
    
    # 强势策略组合 (2-3个)
    for i in range(2, 4):
        for combo in combinations(STRONG_STRATEGIES, i):
            combos_to_test.append((list(combo), []))
    
    # 弱势策略组合 (2-3个)
    for i in range(2, 4):
        for combo in combinations(WEAK_STRATEGIES[:8], i):  # 限制数量
            combos_to_test.append(([], list(combo)))
    
    # 强弱混合组合 (1强+1弱)
    for s in ['Z_Score', 'Fighting', 'DTR_Plus', 'UA', 'HMC']:
        for w in ['HLP3', 'Limit', 'Spring', 'Pinbar']:
            combos_to_test.append(([s], [w]))
    
    print(f"共 {len(combos_to_test)} 个策略组合待测试")
    print()
    
    # 预加载数据
    loader = DataLoader(DATA_DIR)
    strong_path = os.path.join(cache_reader.cache_dir, "strong_signals.parquet")
    weak_path = os.path.join(cache_reader.cache_dir, "weak_signals.parquet")
    
    df_s_full = pd.DataFrame()
    df_w_full = pd.DataFrame()
    
    if os.path.exists(strong_path):
        df_s_full = pd.read_parquet(strong_path)
        df_s_full['date'] = pd.to_datetime(df_s_full['date'])
        df_s_full = df_s_full[(df_s_full['date'] >= pd.to_datetime(bt_start)) & 
                              (df_s_full['date'] <= pd.to_datetime(bt_end))]
        print(f"强势信号: {len(df_s_full)} 条")
    
    if os.path.exists(weak_path):
        df_w_full = pd.read_parquet(weak_path)
        df_w_full['date'] = pd.to_datetime(df_w_full['date'])
        df_w_full = df_w_full[(df_w_full['date'] >= pd.to_datetime(bt_start)) & 
                              (df_w_full['date'] <= pd.to_datetime(bt_end))]
        print(f"弱势信号: {len(df_w_full)} 条")
    print()
    
    # 运行回测
    results = []
    total = len(combos_to_test)
    start_time = time.time()
    
    for idx, (s_combo, w_combo) in enumerate(combos_to_test, 1):
        result = run_single_backtest(
            s_combo, w_combo, df_s_full, df_w_full, loader,
            bt_start, bt_end, initial_capital, stop_loss, take_profit,
            max_hold, position_sizing, max_positions
        )
        
        if result:
            results.append(result)
        
        if idx % 10 == 0 or idx == total:
            elapsed = time.time() - start_time
            progress = idx / total * 100
            print(f"进度: {idx}/{total} ({progress:.1f}%) | 耗时: {elapsed:.1f}s | 已发现 {len(results)} 个有效结果")
    
    print(f"\n✅ 回测完成！共测试 {total} 个组合，{len(results)} 个产生交易")
    return results


def run_single_backtest(s_combo, w_combo, df_s_full, df_w_full, loader,
                        start_date, end_date, initial_capital, 
                        stop_loss, take_profit, max_hold, position_sizing, max_positions):
    """运行单个策略组合回测"""
    
    engine = BacktestEngine(loader, initial_capital)
    
    # 筛选信号
    signal_df = pd.DataFrame()
    df_strong = pd.DataFrame()
    df_weak = pd.DataFrame()
    
    all_selected = s_combo + w_combo
    
    if s_combo and not df_s_full.empty:
        s_cols = [f'Signal_{s}' for s in s_combo]
        avail = [c for c in s_cols if c in df_s_full.columns]
        if avail:
            mask = df_s_full[avail].any(axis=1)
            df_strong = df_s_full[mask].copy()
    
    if w_combo and not df_w_full.empty:
        w_cols = [f'Signal_{w}' for w in w_combo]
        avail = [c for c in w_cols if c in df_w_full.columns]
        if avail:
            mask = df_w_full[avail].any(axis=1)
            df_weak = df_w_full[mask].copy()
    
    # 合并强弱信号
    if not df_strong.empty and not df_weak.empty:
        signal_df = pd.merge(df_strong, df_weak, on=['code', 'name', 'date'], 
                            how='outer', suffixes=('', '_weak'))
    elif not df_strong.empty:
        signal_df = df_strong
    else:
        signal_df = df_weak
    
    if signal_df.empty:
        return None
    
    # 确保所有需要的列存在
    all_cols = [f'Signal_{s}' for s in all_selected]
    for c in all_cols:
        if c not in signal_df.columns:
            signal_df[c] = False
    
    signal_df[all_cols] = signal_df[all_cols].fillna(False)
    mask = signal_df[all_cols].any(axis=1)
    signal_df = signal_df[mask].copy()
    signal_df = signal_df.drop_duplicates(subset=['code', 'date'])
    
    if len(signal_df) == 0:
        return None
    
    # 计算信号数量
    all_signal_cols = [c for c in signal_df.columns if c.startswith('Signal_')]
    signal_df['signal_count'] = signal_df[all_signal_cols].fillna(False).astype(int).sum(axis=1)
    signal_df = signal_df.sort_values(by=['date', 'signal_count'], ascending=[True, False])
    
    # 运行回测
    try:
        equity, trades = engine.run(
            signal_df, start_date, end_date, 
            stop_loss, take_profit, max_hold, 
            position_sizing, max_positions
        )
        
        if len(trades) > 0:
            final_equity = equity.iloc[-1]['total_assets']
            ret_pct = (final_equity - initial_capital) / initial_capital
            winning_trades = trades[trades['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades)
            
            equity['high_water_mark'] = equity['total_assets'].cummax()
            drawdown = ((equity['total_assets'] - equity['high_water_mark']) / equity['high_water_mark']).min()
            
            # 计算夏普比率（简化版）
            equity['daily_return'] = equity['total_assets'].pct_change()
            sharpe = 0
            if equity['daily_return'].std() > 0:
                sharpe = (equity['daily_return'].mean() / equity['daily_return'].std()) * np.sqrt(252)
            
            combo_name = " + ".join(all_selected) if all_selected else "None"
            return {
                "strategy": combo_name,
                "strong_strategies": ",".join(s_combo),
                "weak_strategies": ",".join(w_combo),
                "return_pct": round(ret_pct * 100, 2),
                "win_rate_pct": round(win_rate * 100, 2),
                "trades": len(trades),
                "max_drawdown_pct": round(abs(drawdown) * 100, 2),
                "final_equity": round(final_equity, 2),
                "sharpe_ratio": round(sharpe, 2)
            }
    except Exception as e:
        pass
    
    return None


def step3_analyze_results(results):
    """步骤3: 分析结果并输出报告"""
    print("\n" + "=" * 80)
    print("步骤 3/3: 分析结果并输出报告")
    print("=" * 80)
    
    if not results:
        print("❌ 没有有效的回测结果！")
        return
    
    df = pd.DataFrame(results)
    
    # 保存原始结果
    results_path = "backtest_results_2years.json"
    df.to_json(results_path, orient="records", indent=2)
    print(f"✅ 原始结果已保存: {results_path}")
    print()
    
    # 1. 按收益率排序
    print("\n" + "=" * 80)
    print("🏆 TOP 20 高收益策略组合（按收益率排序）")
    print("=" * 80)
    top_return = df.sort_values(by="return_pct", ascending=False).head(20)
    for idx, row in top_return.iterrows():
        print(f"\n  排名 {top_return.index.get_loc(idx) + 1}: {row['strategy']}")
        print(f"    收益率: {row['return_pct']:+.2f}% | 胜率: {row['win_rate_pct']:.1f}% | "
              f"交易次数: {row['trades']} | 最大回撤: {row['max_drawdown_pct']:.1f}% | "
              f"夏普比率: {row['sharpe_ratio']:.2f}")
    
    # 2. 按夏普比率排序
    print("\n" + "=" * 80)
    print("📊 TOP 15 风险调整收益最佳策略（按夏普比率排序，最少10笔交易）")
    print("=" * 80)
    top_sharpe = df[df['trades'] >= 10].sort_values(by="sharpe_ratio", ascending=False).head(15)
    for idx, row in top_sharpe.iterrows():
        print(f"\n  排名 {top_sharpe.index.get_loc(idx) + 1}: {row['strategy']}")
        print(f"    夏普比率: {row['sharpe_ratio']:.2f} | 收益率: {row['return_pct']:+.2f}% | "
              f"胜率: {row['win_rate_pct']:.1f}% | 交易次数: {row['trades']}")
    
    # 3. 高胜率策略（最少10笔交易）
    print("\n" + "=" * 80)
    print("🎯 TOP 15 高胜率策略（胜率>60%，最少10笔交易）")
    print("=" * 80)
    high_win = df[(df['trades'] >= 10) & (df['win_rate_pct'] > 60)].sort_values(by="win_rate_pct", ascending=False).head(15)
    for idx, row in high_win.iterrows():
        print(f"\n  排名 {high_win.index.get_loc(idx) + 1}: {row['strategy']}")
        print(f"    胜率: {row['win_rate_pct']:.1f}% | 收益率: {row['return_pct']:+.2f}% | "
              f"交易次数: {row['trades']} | 最大回撤: {row['max_drawdown_pct']:.1f}%")
    
    # 4. 综合分析 - 最佳综合表现
    print("\n" + "=" * 80)
    print("⭐ 最佳综合表现策略（收益>20%，回撤<15%，胜率>50%）")
    print("=" * 80)
    best_combined = df[
        (df['return_pct'] > 20) & 
        (df['max_drawdown_pct'] < 15) & 
        (df['win_rate_pct'] > 50) &
        (df['trades'] >= 10)
    ].sort_values(by="return_pct", ascending=False)
    
    if len(best_combined) > 0:
        for idx, row in best_combined.head(10).iterrows():
            print(f"\n  ✓ {row['strategy']}")
            print(f"    收益率: {row['return_pct']:+.2f}% | 胜率: {row['win_rate_pct']:.1f}% | "
                  f"最大回撤: {row['max_drawdown_pct']:.1f}% | 夏普: {row['sharpe_ratio']:.2f}")
    else:
        print("  没有找到满足所有条件的策略组合")
    
    # 5. 策略类型分析
    print("\n" + "=" * 80)
    print("📈 策略类型分析")
    print("=" * 80)
    
    # 分析强势策略表现
    strong_only = df[df['weak_strategies'] == '']
    if len(strong_only) > 0:
        print(f"\n强势策略 ({len(strong_only)} 个组合):")
        print(f"  平均收益率: {strong_only['return_pct'].mean():+.2f}%")
        print(f"  平均胜率: {strong_only['win_rate_pct'].mean():.1f}%")
        print(f"  最佳策略: {strong_only.loc[strong_only['return_pct'].idxmax(), 'strategy']}")
        print(f"    收益率: {strong_only['return_pct'].max():+.2f}%")
    
    # 分析弱势策略表现
    weak_only = df[df['strong_strategies'] == '']
    if len(weak_only) > 0:
        print(f"\n弱势策略 ({len(weak_only)} 个组合):")
        print(f"  平均收益率: {weak_only['return_pct'].mean():+.2f}%")
        print(f"  平均胜率: {weak_only['win_rate_pct'].mean():.1f}%")
        print(f"  最佳策略: {weak_only.loc[weak_only['return_pct'].idxmax(), 'strategy']}")
        print(f"    收益率: {weak_only['return_pct'].max():+.2f}%")
    
    # 分析混合策略表现
    mixed = df[(df['strong_strategies'] != '') & (df['weak_strategies'] != '')]
    if len(mixed) > 0:
        print(f"\n混合策略 ({len(mixed)} 个组合):")
        print(f"  平均收益率: {mixed['return_pct'].mean():+.2f}%")
        print(f"  平均胜率: {mixed['win_rate_pct'].mean():.1f}%")
        print(f"  最佳策略: {mixed.loc[mixed['return_pct'].idxmax(), 'strategy']}")
        print(f"    收益率: {mixed['return_pct'].max():+.2f}%")
    
    # 6. 生成推荐配置
    print("\n" + "=" * 80)
    print("📝 推荐策略配置")
    print("=" * 80)
    
    # 最佳单一策略
    single_strategy = df[(df['strategy'].str.count('\+') == 0) & (df['trades'] >= 10)]
    if len(single_strategy) > 0:
        best_single = single_strategy.sort_values(by="return_pct", ascending=False).iloc[0]
        print(f"\n【推荐1: 最佳单一策略】")
        print(f"  策略: {best_single['strategy']}")
        print(f"  收益率: {best_single['return_pct']:+.2f}% | 胜率: {best_single['win_rate_pct']:.1f}%")
    
    # 最佳双策略组合
    double_strategy = df[(df['strategy'].str.count('\+') == 1) & (df['trades'] >= 10)]
    if len(double_strategy) > 0:
        best_double = double_strategy.sort_values(by="return_pct", ascending=False).iloc[0]
        print(f"\n【推荐2: 最佳双策略组合】")
        print(f"  策略: {best_double['strategy']}")
        print(f"  收益率: {best_double['return_pct']:+.2f}% | 胜率: {best_double['win_rate_pct']:.1f}%")
    
    # 最佳三策略组合
    triple_strategy = df[(df['strategy'].str.count('\+') == 2) & (df['trades'] >= 10)]
    if len(triple_strategy) > 0:
        best_triple = triple_strategy.sort_values(by="return_pct", ascending=False).iloc[0]
        print(f"\n【推荐3: 最佳三策略组合】")
        print(f"  策略: {best_triple['strategy']}")
        print(f"  收益率: {best_triple['return_pct']:+.2f}% | 胜率: {best_triple['win_rate_pct']:.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ 回测分析报告生成完成！")
    print("=" * 80)


def main():
    print("\n" + "=" * 80)
    print("A股策略回测分析系统 - 近两年回测")
    print(f"回测区间: {START_DATE} ~ {END_DATE}")
    print("=" * 80 + "\n")
    
    # 步骤1: 重建缓存
    if not step1_rebuild_cache():
        print("缓存构建失败，退出")
        return
    
    # 步骤2: 运行回测优化
    results = step2_run_backtest_optimization()
    
    # 步骤3: 分析结果
    step3_analyze_results(results)


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_loader import DataLoader
from signal_cache import SignalCacheBuilder, SignalCacheReader
from exit_signals import ExitSignals
from indicators import Indicators
from strategies import Strategies
from strong_strategies import StrongStrategies
from weak_strategies import WeakStrategies
from backtest import BacktestEngine
import datetime
import os
import requests
import json
import concurrent.futures
import pytz
import plotly.express as px
from amarket_core.market_logic import MarketAnalyzer

from amarket_framework.timezone_utils import is_trading_time

@st.cache_data(ttl=10) # Cache for 10 seconds during trading hours
def get_analysis(key=None, model="gemini-3.1-pro-preview", cache_key=None, skip_ai=False):
    """Fetch market analysis. cache_key prevents updates outside trading hours."""
    # Pass key to analyzer
    analyzer = MarketAnalyzer(api_key=key, model_name=model)
    return analyzer.analyze_market_status(skip_ai=skip_ai)



def patch_df_with_realtime(df, code):
    """
    Fetch real-time quote for 'code' and patch it into the dataframe.
    This ensures the latest candle is up-to-date even if CSV hasn't downloaded it yet.
    """
    if df.empty:
        return df
        
    from amarket_framework.timezone_utils import is_trading_time
    if not is_trading_time():
        return df
        
    try:
        from amarket_framework.tencent_loader import TencentLoader
        t_loader = TencentLoader()
        
        # Determine full symbol (shXXXXXX or szXXXXXX)
        full_code = t_loader.get_full_code(code)
        
        # Fetch RT quote
        rt_df = t_loader.fetch_realtime_quotes([full_code])
        if rt_df.empty:
            return df
            
        rt_row = rt_df.iloc[0]
        # Beijing time today
        beijing_tz = pytz.timezone('Asia/Shanghai')
        today = datetime.datetime.now(beijing_tz).date()
        
        # If the last row in df is NOT today, or if it IS today but price/vol mismatch significant
        # Append or update
        last_date = df['date'].iloc[-1].date()
        
        # Patch data
        patch_data = {
            'date': pd.to_datetime(today),
            'close': float(rt_row['close']),
            'open': float(rt_row['open']),
            'high': float(rt_row['high']),
            'low': float(rt_row['low']),
            'volume': float(rt_row['volume']),
            'amount': float(rt_row['amount'])
        }
        
        # Note: Tencent RT has high/low too, let's try to find them if parts[4] parts[5]
        # Looking at tencent_loader.py, it doesn't currently parse H/L.
        # Let's keep it simple for now or update tencent_loader.
        
        if last_date == today:
            # Update last row
            for k, v in patch_data.items():
                if k in df.columns:
                    df.iloc[-1, df.columns.get_loc(k)] = v
        else:
            # Append as new row
            new_row = pd.DataFrame([patch_data])
            # Ensure columns match
            for col in df.columns:
                if col not in new_row.columns:
                    new_row[col] = np.nan
            df = pd.concat([df, new_row], ignore_index=True)
            
        # Re-calc key indicators used in UI
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        return df
    except Exception as e:
        print(f"Error patching realtime for {code}: {e}")
        return df

st.set_page_config(layout="wide", page_title="A股全市场选股策略")

# --- Global Session State Initialization ---
# Ensure widget keys are present to avoid the "value + key" conflict warning
if 'initialized' not in st.session_state:
    default_keys = {
        # Strong Strategies
        'ss_zscore': True, 'ss_cyc': False, 'ss_rs': False, 'ss_range': False,
        'ss_tkos': False, 'ss_20vma': False, 'ss_obo': False, 'ss_lcs': False,
        'ss_dtr': True, 'ss_fighting': False, 'ss_ua': False, 'ss_hmc': False,
        'ss_hps': False, 'ss_rking': False,
        # Weak Strategies
        'ws_hlp3': False, 'ws_wyckoff': False, 'ws_limit': True, 'ws_volmin': False,
        'ws_rsi': False, 'ws_es': False, 'ws_boll': False, 'ws_spring': True,
        'ws_pinbar': False, 'ws_flow': False, 'ws_2b': False, 'ws_ua': False,
        'ws_dv': False,
        # Chart Settings
        'strong_ma': True, 'strong_ema': True, 'strong_boll': True, 'strong_sig': True,
        'weak_ma': True, 'weak_ema': True, 'weak_boll': True, 'weak_sig': True,
        'dp_ma': True, 'dp_ema': True, 'dp_boll': True, 'dp_sig': True,
        'sc_ma': True, 'sc_ema': True, 'sc_boll': True, 'sc_sig': True
    }
    for k, v in default_keys.items():
        if k not in st.session_state:
            st.session_state[k] = v
    st.session_state['initialized'] = True

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MARKET_DATA_DIR = os.path.join(DATA_DIR, 'market_data')

# Ensure directories exist
if not os.path.exists(MARKET_DATA_DIR):
    os.makedirs(MARKET_DATA_DIR)

# Title and Intro
st.title("A股全市场选股")
st.markdown("""
基于 **本地数据仓库 (Local Data Warehouse)**，覆盖全市场（剔除ST/科创/北交）。
**使用前请确保已运行数据下载脚本更新本地数据。**
""")

# --- Sidebar Configuration ---
st.sidebar.header("配置")

# --- Data Status Section ---
with st.sidebar.expander("📊 数据状态 (Data Status)", expanded=True):
    # Count stock list
    list_path = os.path.join(MARKET_DATA_DIR, "stock_list.csv")
    total_stocks = 0
    if os.path.exists(list_path):
        try:
            total_stocks = sum(1 for line in open(list_path)) - 1 # minus header
        except: pass
        
    # Count downloaded files
    downloaded_count = 0
    if os.path.exists(MARKET_DATA_DIR):
        files = [name for name in os.listdir(MARKET_DATA_DIR) if name.endswith('.csv')]
        downloaded_count = len(files)
        if "stock_list.csv" in files:
            downloaded_count -= 1
            
    if total_stocks > 0:
        progress = downloaded_count / total_stocks
        st.progress(min(progress, 1.0))
        st.write(f"已下载: **{downloaded_count}** / {total_stocks}")
    else:
        st.error("未找到股票列表")
        
    if st.button("🔄 刷新下载进度"):
        st.rerun()
    
    st.markdown("---")
    
    # --- Cache Status Section ---
    st.markdown("**📦 信号缓存状态**")
    cache_reader = SignalCacheReader()
    is_valid, cache_msg = cache_reader.is_cache_valid()
    
    if is_valid:
        st.success(f"✅ {cache_msg}")
    else:
        st.warning(f"⚠️ {cache_msg}")
    
    if st.button("🔄 重建信号缓存", help="预计算所有策略信号，加速筛选"):
        cache_builder = SignalCacheBuilder()
        
        status_container = st.status("正在构建信号缓存...", expanded=True)
        progress_bar = status_container.progress(0)
        progress_text = status_container.empty()
        
        def progress_callback(current, total, message):
            progress = current / total
            progress_bar.progress(progress)
            progress_text.write(f"{message} ({current}/{total})")
        
        with st.spinner("预计需要 10-15 分钟，请耐心等待..."):
            success = cache_builder.build_all_signals(progress_callback=progress_callback)
        
        if success:
            status_container.update(label="缓存构建完成！", state="complete", expanded=False)
            st.success("🎉 信号缓存已更新，筛选速度将大幅提升！")
            st.rerun()
        else:
            status_container.update(label="缓存构建失败", state="error",expanded=False)
            st.error("❌ 缓存构建失败，请查看日志")
        
    st.markdown("---")
    if st.button("📥 立即下载行情数据 (Download)", help="从腾讯财经下载日线数据到本地"):
        # Imports already at top
        
        status_container = st.status("正在初始化下载任务...", expanded=True)
        
        # 1. Check Stock List
        if not os.path.exists(list_path):
            status_container.write("正在尝试多种途径获取全市场股票列表...")
            
            stock_df = pd.DataFrame()
            import akshare as ak
            import time
            
            # List of methods to try in order
            methods = [
                ("Eastmoney A-Share Spot", ak.stock_zh_a_spot_em),
                ("General Code/Name Info", ak.stock_info_a_code_name),
                ("SH/SZ Combined (legacy)", ak.stock_zh_a_spot),
            ]
            
            for name, method in methods:
                if not stock_df.empty: break
                for i in range(2): # 2 retries per method
                    try:
                        status_container.write(f"正在通过 {name} 获取数据 (尝试 {i+1})...")
                        temp_df = method()
                        if temp_df is not None and not temp_df.empty:
                            stock_df = temp_df
                            break
                    except Exception as e:
                        time.sleep(1)
            
            if not stock_df.empty:
                try:
                    # Map different column names used by different AKShare methods
                    cols_to_use = []
                    if '代码' in stock_df.columns and '名称' in stock_df.columns:
                        stock_df = stock_df[['代码', '名称']]
                        stock_df.columns = ['code', 'name']
                    elif 'code' in stock_df.columns and 'name' in stock_df.columns:
                        stock_df = stock_df[['code', 'name']]
                    elif 'symbol' in stock_df.columns and 'name' in stock_df.columns:
                        stock_df = stock_df[['symbol', 'name']]
                        stock_df.columns = ['code', 'name']
                    
                    # Ensure code is string and clean
                    stock_df['code'] = stock_df['code'].astype(str).str.extract(r'(\d{6})')[0]
                    stock_df = stock_df.dropna(subset=['code'])

                    # --- Filter Logic (Exclude ST/KC/BJ) ---
                    # 1. Exclude Beijing (8xxx, 4xxx, 43, 83, etc) & Sci-Tech (688)
                    stock_df = stock_df[~stock_df['code'].str.startswith(('8', '4', '688'))]
                    # 2. Exclude ST
                    stock_df = stock_df[~stock_df['name'].str.contains('ST', na=False)]
                    
                    stock_df.to_csv(list_path, index=False)
                    status_container.write(f"已创建股票列表: {len(stock_df)} 只")
                except Exception as e:
                    status_container.error(f"处理数据失败: {e}")
                    st.stop()
            else:
                status_container.error("所有途径获取股票列表均告失败，请检查网络环境。")
                st.stop()
        else:
            stock_df = pd.read_csv(list_path, dtype={'code': str})
            
        # 2. Download Loop
        stocks = stock_df.to_dict('records')
        total_d = len(stocks)
        params_list = []
        
        # Prepare params
        for s in stocks:
             code = s['code']
             if code.startswith('6'): symbol = f"sh{code}"
             elif code.startswith('0') or code.startswith('3'): symbol = f"sz{code}"
             else: symbol = f"sz{code}" # fallback
             params_list.append((code, symbol))
        
        # Manually append the Shanghai Index, Shenzhen Index, and Chuangye Board for Macro logic
        params_list.append(('000001.SH', 'sh000001'))
        params_list.append(('399001.SZ', 'sz399001'))
        params_list.append(('399006.SZ', 'sz399006'))
             
        status_container.write("正在并发下载数据 (Tencent API)...")
        progress_bar = status_container.progress(0)
        
        def download_one(args):
            c, sym = args
            url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var=kline_dayqfq&param={sym},day,,,600,qfq"
            try:
                r = requests.get(url, timeout=2)
                if r.status_code != 200: return False
                content = r.text
                if "=" in content: json_str = content.split("=", 1)[1]
                else: json_str = content
                data = json.loads(json_str)
                k_data = data.get('data', {}).get(sym, {})
                klines = k_data.get('qfqday', []) or k_data.get('day', [])
                if not klines: return False
                
                # Save
                cols = ['date', 'open', 'close', 'high', 'low', 'volume']
                recs = []
                for k in klines:
                    if len(k) < 6: continue
                    recs.append({
                        'date': k[0], 
                        'open': k[1], 'close': k[2], 
                        'high': k[3], 'low': k[4], 'volume': k[5]
                    })
                if recs:
                    if not os.path.exists(MARKET_DATA_DIR): os.makedirs(MARKET_DATA_DIR)
                    pd.DataFrame(recs).to_csv(os.path.join(MARKET_DATA_DIR, f"{c}.csv"), index=False)
                    return True
            except:
                return False
            return False

        # Run ThreadPool
        done_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(download_one, p) for p in params_list]
            for f in concurrent.futures.as_completed(futures):
                done_count += 1
                if done_count % 50 == 0:
                    progress_bar.progress(done_count / total_d)
                    
        status_container.update(label="下载完成!", state="complete", expanded=False)
        st.success(f"下载任务结束。正在自动构建信号缓存...")
        
        # Auto-trigger cache rebuild after download
        cache_builder = SignalCacheBuilder()
        cache_status = st.status("正在构建信号缓存...", expanded=True)
        cache_progress = cache_status.progress(0)
        cache_text = cache_status.empty()
        
        def cache_progress_callback(current, total, message):
            cache_progress.progress(current / total)
            cache_text.write(f"{message} ({current}/{total})")
        
        cache_success = cache_builder.build_all_signals(progress_callback=cache_progress_callback)
        
        if cache_success:
            cache_status.update(label="缓存构建完成！", state="complete", expanded=False)
            st.success("🎉 数据下载和缓存构建全部完成！")
        else:
            cache_status.update(label="缓存构建部分完成", state="running", expanded=False)
            st.warning("数据已下载，但缓存构建遇到一些问题，部分功能可能较慢")
        
        st.rerun()
    
    # --- Golden Alliance Gallery (Phase 6 & 11) ---
    with st.sidebar.expander("👑 黄金盟军排行榜 (Top Alliances)", expanded=True):
        st.markdown("基于 **24个月 A股全回测** 的最强组合参考：")
        st.info("💡 **致胜密码**：前20名均被【弱势抄底】指标霸榜。突破确认虽然有弹性，但盈亏比往往不如极度恐慌后的首阳。")
        
        try:
            json_path = os.path.join(BASE_DIR, "optimization_results.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    all_results = json.load(f)
                top_alliances_data = sorted(all_results, key=lambda x: x['Return (%)'], reverse=True)[:20]
            else:
                top_alliances_data = []
        except:
            top_alliances_data = []

        if top_alliances_data:
            # Shared helper for leaderboard application
            def apply_strat_helper(strat_str, mode='replace'):
                # Mapping: Backtest Name -> Session State Key
                mapping = {
                    # Strong
                    'Z_Score': 'ss_zscore', 'CYC_MAX': 'ss_cyc', 'RS': 'ss_rs', 
                    'RangeBreak': 'ss_range', 'TKOS': 'ss_tkos', '20VMA': 'ss_20vma', 
                    'OBO': 'ss_obo', 'LCS': 'ss_lcs', 'DTR_Plus': 'ss_dtr', 
                    'Fighting': 'ss_fighting', 'UA': 'ss_ua', 'HMC': 'ss_hmc',
                    'HPS': 'ss_hps', 'RKing': 'ss_rking',
                    # Weak
                    'HLP3': 'ws_hlp3', 'Wyckoff': 'ws_wyckoff', 'Limit': 'ws_limit', 
                    'Vol_Min_120': 'ws_volmin', 'RSI_Rev': 'ws_rsi', 'ES': 'ws_es', 
                    'Boll_Rev': 'ws_boll', 'Spring': 'ws_spring', 'Pinbar': 'ws_pinbar', 
                    'Money_Flow': 'ws_flow', '2B': 'ws_2b', 'UA_Weak': 'ws_ua', 
                    'Double_Vol': 'ws_dv', 'Limit_Open': None # Skip logic-only placeholders
                }
                
                # Reset all if replace mode
                if mode == 'replace':
                    for k in mapping.values():
                        if k: st.session_state[k] = False
                
                # Enable selected
                parts = [p.strip() for p in strat_str.split('+')]
                for p in parts:
                    key = mapping.get(p)
                    if key: st.session_state[key] = True
                
                st.toast(f"✅ 已加载盟军: {strat_str}")

            for i, a in enumerate(top_alliances_data):
                st.markdown(f"**TOP {i+1}:** {a['Strategy']}")
                st.caption(f"收益: **{a['Return (%)']:.1f}%** | 胜率: **{a['Win Rate (%)']:.1f}%**")
                
                c1, c2 = st.columns(2)
                if c1.button("加载 (覆盖)", key=f"btn_r_{i}", help="清除当前勾选，仅加载此组合", use_container_width=True):
                    apply_strat_helper(a['Strategy'], mode='replace')
                if c2.button("叠加 (多选)", key=f"btn_a_{i}", help="保留当前勾选，叠加此组合", use_container_width=True):
                    apply_strat_helper(a['Strategy'], mode='append')
                
                st.markdown("---")
        else:
            st.info("尚未发现回测结果。")

# --- Light Mode Theme (Fixed) ---
plotly_template = 'plotly_white'
st.markdown("""
    <style>
    /* Main Area and Global Defaults */
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    
    /* Sidebar - Force Light Background and Black Text */
    section[data-testid="stSidebar"] {
        background-color: #F0F2F6 !important;
        color: #000000 !important;
    }
    
    /* Force Text Color Globally (including Sidebar) */
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, 
    .stApp label, .stApp span, .stApp div[data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }
    
    /* Force Text Color in Sidebar specifically (in case .stApp doesn't cover it) */
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4, 
    section[data-testid="stSidebar"] h5, 
    section[data-testid="stSidebar"] h6, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }
    
    /* Specific Widget Overrides */
    .stRadio div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    .stCheckbox label div[data-testid="stMarkdownContainer"] p {
         color: #000000 !important;
    }
    
    /* Inputs (Date, Select, Text) - Force White Background */
    div[data-baseweb="input"], 
    div[data-baseweb="select"] > div, 
    div[data-baseweb="base-input"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #E0E0E0 !important;
    }
    
    /* Input Text Color inside the box */
    input[type="text"], input[type="number"], input {
        color: #000000 !important;
    }
    
    /* Dropdown menu items */
    ul[data-baseweb="menu"] li {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }

    /* Buttons (Global) */
    button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #CCCCCC !important;
    }
    button p {
        color: #000000 !important;
    }
    button:hover {
        background-color: #E0E0E0 !important;
        border-color: #999999 !important;
        color: #000000 !important;
    }
    button:hover p {
        color: #000000 !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] details summary {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #E0E0E0 !important;
    }
    div[data-testid="stExpander"] details summary span,
    div[data-testid="stExpander"] details summary svg {
        color: #000000 !important;
        fill: #000000 !important;
    }
    div[data-testid="stExpander"] details {
        border-color: #E0E0E0 !important;
        color: #000000 !important;
    }

    /* Progress Bar Text */
    div[data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    
    /* Header (Top Bar) - Force Light */
    header[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }
    header[data-testid="stHeader"] button {
        background-color: transparent !important;
        border: none !important;
    }
    header[data-testid="stHeader"] svg {
        fill: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Common Date Configuration ---
today = datetime.datetime.now().date()
# Default dates (Screening window default)
default_end = today
default_start = today - datetime.timedelta(days=180)
# Calculation start date (for indicators) - derived from default_start
calc_start_date = default_start - datetime.timedelta(days=400) 

# --- Mode Selection (Top Navigation) ---
MODE_OPTIONS = [
    "🌍 宏观大盘", "📈 个股行情", "🚀 强势股进攻", "🔄 弱势股抄底",
    "🛠️ 策略回测", "📅 每日交易计划", "💰 交易管理"
]
app_mode = st.pills("功能导航", MODE_OPTIONS, default="🌍 宏观大盘")






STRATEGY_DESCRIPTIONS = {
    # 强势策略
    "Z_Score": "**Z_Score (标准化强势)**: 股价偏离20日均线的标准化程度。Z值 > 1.5 代表强势，1.5-3为最佳介入区，>3过热需警惕。",
    "RS": "**RS (相对强弱)**: 个股表现相对大盘的强弱对比。RS值突破自身布林上轨，代表无论大盘涨跌都跑赢市场。",
    "TKOS": "**TKOS (股王爆发)**: 短期爆发力极强，5日涨幅超50%，属妖股启动信号。",
    "DTR_Plus": "**DTR_Plus (三维共振)**: MACD翻红 + 股价站上MA20 + 触碰布林上轨，三重条件确认，高胜率突破信号。",
    "UA": "**UA (天量突破)**: 出现250日历史天量后，股价突破该天量日最高价，多头完全掌控，强势启动。",
    "Fighting": "**Fighting (趋势共振)**: MACD翻红+股价新高+量能放大+布林带确认。主升浪信号。",
    "CYC_MAX": "**CYC MAX (成本突破)**: 股价站上无穷成本均线，市场全获利状态。",
    "RangeBreak": "**Range Break (箱体突破)**: 突破52周(或250日)最高价，伴随放量。",
    "20VMA": "**20VMA (量能启动)**: 长期缩量后首次放量突破20日均量线，趋势启动。",
    "HMC": "**HMC (动量通道)**: MACD柱状图乖离率过大，动量强劲。",
    "HPS": "**HPS (趋势系统)**: 站上EMA200牛熊线，且突破EMA15通道。",
    "RKing": "**RKing (趋势跟随)**: 红柱代表多头趋势，绿柱代表空头趋势。此为趋势中继或启动。",
    "OBO": "**OBO (Open Breakout)**: 强势跳空高开，开盘价加上昨日振幅成为今日目标价，且必须在年线之上。",
    "LCS": "**LCS (极限收盘)**: 偏离5日极致低点超20%，且创下250日新高，短期极度强势。",
    # 弱势策略
    "Limit": "**Limit (极致缩量)**: 成交量低于20日均量的50%，市场极度死寂，变盘在即。",
    "Boll_Rev": "**Boll Rev (布林反转)**: 长期处于布林带弱势区(下轨运行)，突破中轨且最高价触碰上轨。",
    "RSI_Rev": "**RSI2 Reversion**: RSI2极度超卖(连续2天<25)后的回归买点，在长期趋势(EMA200)向上时抄底。",
    "2B": "**2B 法则**: 创新低后迅速拉回，洗盘结束，底部反转。",
    "Wyckoff": "**Wyckoff (吸筹)**: 经过长达60天以上的地量横盘，突然出现1.5倍以上的放量启动，主力吸筹完毕。",
    "Spring": "**Spring (弹簧)**: 跌破20日支撑后，1-3天内迅速收回支撑上方，且伴随缩量下杀，主力清洗最后浮筹。",
    "Pinbar": "**Pinbar (针线)**: 产生长下影线(下影线>实体3倍)，且出现放量，表示恐慌盘涌出被主力全盘接下。",
    "ES": "**ES (波动率压缩)**: 20日短期波动率(Std20)降至60日及120日长周期极值以下，预示剧烈变盘。",
    "Money_Flow": "**Money_Flow (资金背离)**: 股价创20日新低，但资金流为正(净买入)，主力在底部悄悄吸筹。",
    "UA_Weak": "**UA_Weak (底部天量)**: 底部出现历史级天量，后续价格有效站上天量日最高价确认多头获胜。",
    "Double_Vol": "**Double_Vol (倍量不破)**: 出现倍量阳线，回调不破该阳线最低价，再次放量启动。",
    "Vol_Min_120": "**Vol_Min_120 (地量群)**: 短期平均成交量(量比)创下120日新低，空头力量完全衰竭。",
    "HLP3": "**HLP3 (大慈悲点)**: 股价处于120日极低点附近(大众绝望)，今日放量(>1.5倍)大幅拉升(>3%)，主力进场大举扫货解套。",
}

def get_dyn_strategy_desc(strat_key, df_recent):
    """
    根据给定的策略 Key，动态计算并在尾部附加最新一天的指标数值，并结合参考范围反馈解读。
    格式: "**Z_Score (标准化强势)**: ... | 【当前数据】 Z=1.65 (正常范围: >1.5强势, >3过热)"
    """
    base_desc = STRATEGY_DESCRIPTIONS.get(strat_key, f"**{strat_key}**: 暂无详细说明")
    
    if df_recent is None or df_recent.empty:
        return base_desc
        
    dyn_info = ""
    
    try:
        # ---- 强势指标 ----
        if strat_key == "Z_Score":
            # Recalculate if missing
            period = 20
            ma20 = df_recent['close'].rolling(window=period).mean().iloc[-1]
            std20 = df_recent['close'].rolling(window=period).std().iloc[-1]
            if pd.notna(std20) and std20 != 0:
                z = (df_recent['close'].iloc[-1] - ma20) / std20
                dyn_info = f"<br>↳ <font color='{'red' if z>3 else 'dodgerblue'}'>**实时数值**: Z={z:.2f}</font> (参考: >1.5强势, 1.5~3绝佳介入点, >3短期过热风险)"
            
        elif strat_key == "RS":
            # Since index isn't passed here easily, we try to use it if present, otherwise skip
            if "RS" in df_recent.columns:
                rs = df_recent["RS"].iloc[-1]
                if "RS_Upper" in df_recent.columns:
                    rs_up = df_recent["RS_Upper"].iloc[-1]
                    dyn_info = f"<br>↳ <font color='dodgerblue'>**实时数值**: 当日相对强度RS={rs:.2f}</font> (参考: 突破布林上轨 {rs_up:.2f} 视为极强状态)"
                else:
                    dyn_info = f"<br>↳ <font color='dodgerblue'>**实时数值**: 当日相对强度RS={rs:.2f}</font>"
                
        elif strat_key == "TKOS":
            if len(df_recent) >= 20:
                pct = (df_recent['close'].iloc[-1] - df_recent['close'].iloc[-21]) / df_recent['close'].iloc[-21] * 100
                dyn_info = f"<br>↳ <font color='dodgerblue'>**实时数值**: 20日累计涨幅={pct:.1f}%</font> (参考: >50%才符合妖股爆发条件)"
            
        elif strat_key == "HMC":
            hhv_50 = df_recent['high'].rolling(window=50).max().iloc[-1]
            ema_200 = df_recent['close'].ewm(span=200, adjust=False).mean().iloc[-1]
            if pd.notna(hhv_50) and pd.notna(ema_200):
                yellow = hhv_50 - df_recent['close'].iloc[-1]
                red = df_recent['close'].iloc[-1] - ema_200
                dyn_info = f"<br>↳ <font color='dodgerblue'>**实时数值**: 动能红线={red:.2f}, 阻力黄线={yellow:.2f}</font> (参考: 红线必须上穿黄线，且红线>0)"
            
        elif strat_key == "LCS":
            low_5 = df_recent['low'].rolling(window=5).min().iloc[-1]
            if pd.notna(low_5) and low_5 > 0:
                dev = (df_recent['close'].iloc[-1] - low_5) / low_5 * 100
                dyn_info = f"<br>↳ <font color='dodgerblue'>**实时数值**: 此时偏离5日谷底程度={dev:.1f}%</font> (参考: >20%为极限拉升)"
                
        elif strat_key == "DTR_Plus":
            if "MACD_Hist" in df_recent.columns and "Boll_Upper" in df_recent.columns:
                 m_hist = df_recent["MACD_Hist"].iloc[-1]
                 bull_up = df_recent["Boll_Upper"].iloc[-1]
                 dyn_info = f"<br>↳ <font color='dodgerblue'>**实时数值**: MACD柱={m_hist:.3f}, 布林上轨={bull_up:.2f}</font> (参考: MACD必须翻红(>0)且股价上穿布林上轨)"
                 
        elif strat_key == "Fighting":
             if "MACD_Hist" in df_recent.columns:
                 m_hist = df_recent["MACD_Hist"].iloc[-1]
                 max_52_price = df_recent["high"].rolling(52).max().shift(1).iloc[-1]
                 max_52_vol = df_recent["volume"].rolling(52).max().shift(1).iloc[-1]
                 if pd.notna(max_52_price) and pd.notna(max_52_vol):
                     dyn_info = f"<br>↳ <font color='dodgerblue'>**实时数值**: MACD柱={m_hist:.3f}, 前高价={max_52_price:.2f}, 前天量={max_52_vol:,.0f}</font> (参考: 三重指标必须同时突破)"

        # ---- 弱势指标 ----
        elif strat_key == "Limit" and "volume" in df_recent.columns:
            vol = df_recent['volume'].iloc[-1]
            if "Vol_MA20" in df_recent.columns:
                ma20 = df_recent["Vol_MA20"].iloc[-1]
            else:
                ma20 = df_recent['volume'].rolling(window=20).mean().iloc[-1]
                
            if ma20 > 0 and pd.notna(ma20):
                ratio = vol / ma20
                dyn_info = f"<br>↳ <font color='{'orange' if ratio<0.5 else 'dodgerblue'}'>**实时数值**: 当日成交量={vol:,.0f}，约为20日均量比例的 {ratio*100:.1f}%</font> (参考: <50%为极致无量状态)"
                
        elif strat_key in ["RSI_Rev", 'RSI2_Rev']:
            # Recompute RSI2 using central engine for precision and smoothing
            from indicators import Indicators
            rsi2_series = Indicators.calculate_rsi(df_recent['close'].tail(30), 2)
            rsi_val = rsi2_series.iloc[-1]
            if pd.notna(rsi_val):
                dyn_info = f"<br>↳ <font color='dodgerblue'>**实时数值**: 当日RSI(2)={rsi_val:.1f}</font> (参考: 连续两天<25时代表过度非理性抛售极值)"
                
        elif strat_key == "ES":
             std20 = df_recent['close'].rolling(20).std().iloc[-1]
             std60 = df_recent['close'].rolling(60).std().iloc[-1]
             std120 = df_recent['close'].rolling(120).std().iloc[-1]
             if pd.notna(std20) and pd.notna(std60) and pd.notna(std120):
                 dyn_info = f"<br>↳ <font color='dodgerblue'>**实时数值**: 20日短波幅(Std)={std20:.3f}, 60日长波幅={std60:.3f}, 120日波幅={std120:.3f}</font> (参考: 必须出现 20日 < 60日 及 120日)"
                 
        elif strat_key == "2B":
             prev_low = df_recent['low'].rolling(window=20).min().shift(1).iloc[-1]
             if pd.notna(prev_low):
                 dyn_info = f"<br>↳ <font color='dodgerblue'>**实时数值**: 20日支撑低点={prev_low:.2f}</font> (参考: 股价突破新低后必须反转拉回站上该支撑位)"
                 
        elif strat_key == "Money_Flow":
             # MF is complex, if it's there use it, else skip
             if "Money_Flow" in df_recent.columns:
                 mf = df_recent["Money_Flow"].iloc[-1]
                 dyn_info = f"<br>↳ <font color='{'red' if mf>0 else 'green'}'>**实时数值**: 10日累计动能资金流入={mf:.2f}</font> (参考: 股价创新低的同时流入必须>0)"

    except Exception as e:
        # 吞掉任何计算异常，保证正常显示
        pass

    # 将动态值附加到文字中并允许Html渲染
    return f"{base_desc} {dyn_info}"

def plot_stock_chart(df_sel, code, name, show_ma, show_ema, show_boll, show_cyc, show_ema15, show_box, show_supt, show_signals, sub_chart_type, plotly_template, sigs=None, signal_dates=None, triggered_strategies=None, return_fig=False, buy_price=None, stop_loss=None, take_profit_1=None, take_profit_2=None, exit_signals=None, highlight_date=None):
    if df_sel.empty:
        st.warning("No data to plot.")
        return

    # Create a copy to avoid modifying original df
    df_plot = df_sel.copy()
    # Convert date to string for category axis (removes gaps)
    df_plot['date'] = df_plot['date'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (datetime.datetime, datetime.date)) or isinstance(x, pd.Timestamp) else x)

    # Plotly
    # 2 rows: Main(0.7) + Sub(0.3)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3])
    
    # Apply Template
    fig.update_layout(template=plotly_template)

    # --- Main Chart ---
    # Candlestick
    fig.add_trace(go.Candlestick(x=df_plot['date'],
                    open=df_plot['open'], high=df_plot['high'],
                    low=df_plot['low'], close=df_plot['close'],
                    increasing_line_color='red', decreasing_line_color='green',
                    name='K线'), row=1, col=1)
    
    # Overlays
    if show_ma and 'MA20' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['MA20'], line=dict(color='orange', width=1), name='MA20'), row=1, col=1)
    if show_ema and 'EMA200' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['EMA200'], line=dict(color='purple', width=1.5), name='EMA200'), row=1, col=1)
    if show_ema15 and 'EMA_High_15' in df_plot.columns:
         fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['EMA_High_15'], line=dict(color='blue', width=1), name='HPS Channel (EMA15 High)'), row=1, col=1)
    
    if show_cyc:
         if 'CYC_Inf' in df_plot.columns:
             fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['CYC_Inf'], line=dict(color='brown', width=1.5), name='CYC无穷'), row=1, col=1)
         if 'CYC_13' in df_plot.columns:
             fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['CYC_13'], line=dict(color='cyan', width=1, dash='dot'), name='CYC短线'), row=1, col=1)
             
    if show_boll and 'Boll_Upper' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['Boll_Upper'], line=dict(color='gray', width=1, dash='dot'), name='Boll Up'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['Boll_Lower'], line=dict(color='gray', width=1, dash='dot'), name='Boll Low'), row=1, col=1)
        
    # Strategy Specific Overlays (Box Top, Support, etc)
    if show_box and 'High_52' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['High_52'], line=dict(color='green', width=1, dash='dash'), name='Box Top (250日)'), row=1, col=1)
    if show_supt and 'Low_20' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['Low_20'], line=dict(color='red', width=1, dash='dot'), name='Support (20日)'), row=1, col=1)
    
    # RKing Main Chart Overlay REMOVED
    
    # --- Signal Display (Hover-based) ---
    # Build per-bar signal text for hover tooltip
    hover_signal_texts = []
    if show_signals and sigs is not None and not sigs.empty:
        sig_cols = [c for c in sigs.columns if c.startswith('Signal_')]
        for idx in df_plot.index:
            parts = []
            try:
                if idx in sigs.index:
                    r = sigs.loc[idx]
                    if isinstance(r, pd.DataFrame):
                        r = r.iloc[0]
                    for col in sig_cols:
                        if r.get(col, False):
                            parts.append(col.replace('Signal_', ''))
            except:
                pass
            hover_signal_texts.append(', '.join(parts) if parts else '')
    else:
        hover_signal_texts = [''] * len(df_plot)
    
    # Add exit signal info to hover too
    hover_exit_texts = []
    if exit_signals is not None and not exit_signals.empty:
        exit_cols_h = [c for c in exit_signals.columns if c.startswith('Exit_')]
        exit_desc_map = {
            'Exit_MACD_Decay': 'MACD衰竭',
            'Exit_Break_MA20': '破MA20',
            'Exit_UA_Reversal': '天量滞涨',
            'Exit_Fail_Breakout': '突破失败',
        }
        for idx in df_plot.index:
            parts = []
            try:
                for ec in exit_cols_h:
                    if exit_signals.loc[idx, ec]:
                        parts.append(exit_desc_map.get(ec, ec.replace('Exit_', '')))
            except:
                pass
            hover_exit_texts.append(', '.join(parts) if parts else '')
    else:
        hover_exit_texts = [''] * len(df_plot)
    
    # Apply custom hover text to candlestick (per-bar)
    hover_texts = []
    for i_h in range(len(df_plot)):
        d_h = df_plot.iloc[i_h]
        lines = [f"<b>{d_h['date']}</b>",
                 f"开: {d_h['open']:.2f}",
                 f"高: {d_h['high']:.2f}",
                 f"低: {d_h['low']:.2f}",
                 f"收: {d_h['close']:.2f}"]
        if hover_signal_texts[i_h]:
            lines.append(f"✅ 策略: {hover_signal_texts[i_h]}")
        if hover_exit_texts[i_h]:
            lines.append(f"⚠️ 离场: {hover_exit_texts[i_h]}")
        hover_texts.append('<br>'.join(lines))
    
    fig.data[0].hoverinfo = 'text'
    fig.data[0].text = hover_texts
    
    # Single highlight marker on the signal/added date
    if highlight_date is not None:
        hl_str = highlight_date if isinstance(highlight_date, str) else highlight_date.strftime('%Y-%m-%d') if hasattr(highlight_date, 'strftime') else str(highlight_date)
        hl_mask = df_plot['date'] == hl_str
        hl_pts = df_plot[hl_mask]
        if not hl_pts.empty:
            # Use triggered_strategies param for label (most accurate)
            hl_label = ', '.join(triggered_strategies) if triggered_strategies else ''
            if not hl_label and sigs is not None:
                for idx in hl_pts.index:
                    if idx in sigs.index:
                        r = sigs.loc[idx]
                        if isinstance(r, pd.DataFrame):
                            r = r.iloc[0]
                        sig_cols_hl = [c for c in sigs.columns if c.startswith('Signal_')]
                        hl_parts = [c.replace('Signal_', '') for c in sig_cols_hl if r.get(c, False)]
                        hl_label = ', '.join(hl_parts)
            fig.add_trace(go.Scatter(
                x=hl_pts['date'],
                y=hl_pts['low'] * 0.97,
                mode='markers+text',
                marker=dict(symbol='star', size=16, color='red', line=dict(width=1, color='darkred')),
                text=[f'★ {hl_label}' if hl_label else '★ 信号日'],
                textposition='bottom center',
                textfont=dict(size=10, color='red'),
                name='信号日',
                showlegend=True
            ), row=1, col=1)

    # --- Sub Chart ---
    if sub_chart_type == "MACD":
        fig.add_trace(go.Bar(x=df_plot['date'], y=df_plot['MACD_Hist'], name='MACD Hist', marker_color=df_plot['MACD_Hist'].apply(lambda x: 'red' if x>0 else 'green')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['DIF'], line=dict(color='black', width=1), name='DIF'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['DEA'], line=dict(color='blue', width=1), name='DEA'), row=2, col=1)
    
    elif sub_chart_type == "KDJ":
        if 'K' in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['K'], name='K'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['D'], name='D'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['J'], name='J'), row=2, col=1)
            fig.add_shape(type="line", x0=df_plot['date'].iloc[0], x1=df_plot['date'].iloc[-1], y0=20, y1=20, line=dict(color="gray", dash="dot"), row=2, col=1)
            fig.add_shape(type="line", x0=df_plot['date'].iloc[0], x1=df_plot['date'].iloc[-1], y0=80, y1=80, line=dict(color="gray", dash="dot"), row=2, col=1)

    elif sub_chart_type == "WR":
        if 'WR' in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['WR'], name='Williams %R'), row=2, col=1)
            fig.add_shape(type="line", x0=df_plot['date'].iloc[0], x1=df_plot['date'].iloc[-1], y0=-20, y1=-20, line=dict(color="gray", dash="dot"), row=2, col=1)
            fig.add_shape(type="line", x0=df_plot['date'].iloc[0], x1=df_plot['date'].iloc[-1], y0=-80, y1=-80, line=dict(color="gray", dash="dot"), row=2, col=1)

    elif sub_chart_type == "CCI":
        if 'CCI' in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['CCI'], name='CCI'), row=2, col=1)
            fig.add_shape(type="line", x0=df_plot['date'].iloc[0], x1=df_plot['date'].iloc[-1], y0=100, y1=100, line=dict(color="gray", dash="dot"), row=2, col=1)
            fig.add_shape(type="line", x0=df_plot['date'].iloc[0], x1=df_plot['date'].iloc[-1], y0=-100, y1=-100, line=dict(color="gray", dash="dot"), row=2, col=1)

    elif sub_chart_type == "Volume":
        colors = ['red' if r.close > r.open else 'green' for i, r in df_plot.iterrows()]
        fig.add_trace(go.Bar(x=df_plot['date'], y=df_plot['volume'], marker_color=colors, name='Volume'), row=2, col=1)
        if 'Vol_MA20' in df_plot.columns:
             fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['Vol_MA20'], line=dict(color='black', width=1), name='MA20 Vol'), row=2, col=1)
    
    elif sub_chart_type == "RKing (趋势)":
        # RKing is a Heikin-Ashi based system with Bands
        # Plot X-Candles
        if 'XOpen' in df_plot.columns:
            # Custom Candles
            fig.add_trace(go.Candlestick(x=df_plot['date'],
                            open=df_plot['XOpen'], high=df_plot['XHigh'],
                            low=df_plot['XLow'], close=df_plot['XClose'],
                            increasing_line_color='red', decreasing_line_color='green',
                            name='RKing HA'), row=2, col=1)
            
            # Bands
            if 'RKing_Upper' in df_plot.columns:
                fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['RKing_Upper'], line=dict(color='orange', width=1), name='RKing UP'), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['RKing_Lower'], line=dict(color='cyan', width=1), name='RKing DOWN'), row=2, col=1)

            # Signals
            bu_mask = df_plot['RKing_BU']
            sel_mask = df_plot['RKing_SEL']
            
            # Adjust marker position relative to XLow/XHigh
            fig.add_trace(go.Scatter(x=df_plot[bu_mask]['date'], y=df_plot[bu_mask]['XLow']*0.98, mode='markers', 
                                     marker=dict(symbol='triangle-up', size=10, color='red'), name='Buy'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_plot[sel_mask]['date'], y=df_plot[sel_mask]['XHigh']*1.02, mode='markers', 
                                     marker=dict(symbol='triangle-down', size=10, color='green'), name='Sell'), row=2, col=1)

    elif sub_chart_type == "RSI":
        if 'RSI6' in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['RSI6'], name='RSI6'), row=2, col=1)
        if 'RSI2' in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['RSI2'], name='RSI2'), row=2, col=1)
        fig.add_shape(type="line", x0=df_plot['date'].iloc[0], x1=df_plot['date'].iloc[-1], y0=80, y1=80, line=dict(color="gray", dash="dot"), row=2, col=1)
        fig.add_shape(type="line", x0=df_plot['date'].iloc[0], x1=df_plot['date'].iloc[-1], y0=20, y1=20, line=dict(color="gray", dash="dot"), row=2, col=1)
        
    elif sub_chart_type == "Volatility":
        if 'Std20' in df_plot.columns:
             fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['Std20'], name='Std20'), row=2, col=1)
             fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['Std60'], name='Std60'), row=2, col=1)

    elif sub_chart_type == "HMC":
        if 'HMC_Yellow' in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['HMC_Yellow'], line=dict(color='yellow', width=1.5), name='Yellow (Resist)'), row=2, col=1)
        if 'HMC_Red' in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['HMC_Red'], line=dict(color='red', width=1.5), name='Red (Momentum)'), row=2, col=1)
        # Add Zero line
        fig.add_shape(type="line", x0=df_plot['date'].iloc[0], x1=df_plot['date'].iloc[-1], y0=0, y1=0, line=dict(color="gray", width=1, dash="dot"), row=2, col=1)
    
    # --- Price Level Lines (SOP: 止损/止盈) ---
    x0_date = df_plot['date'].iloc[0]
    x1_date = df_plot['date'].iloc[-1]
    
    if buy_price is not None and buy_price > 0:
        fig.add_shape(type="line", x0=x0_date, x1=x1_date, y0=buy_price, y1=buy_price,
                      line=dict(color="dodgerblue", width=1.5, dash="dash"), row=1, col=1)
        fig.add_annotation(x=x1_date, y=buy_price, text=f"买入 {buy_price:.2f}",
                           showarrow=False, xanchor="left", font=dict(color="dodgerblue", size=10), row=1, col=1)
    
    if stop_loss is not None and stop_loss > 0:
        fig.add_shape(type="line", x0=x0_date, x1=x1_date, y0=stop_loss, y1=stop_loss,
                      line=dict(color="red", width=1.5, dash="dash"), row=1, col=1)
        fig.add_annotation(x=x1_date, y=stop_loss, text=f"止损 {stop_loss:.2f}",
                           showarrow=False, xanchor="left", font=dict(color="red", size=10), row=1, col=1)
    
    if take_profit_1 is not None and take_profit_1 > 0:
        fig.add_shape(type="line", x0=x0_date, x1=x1_date, y0=take_profit_1, y1=take_profit_1,
                      line=dict(color="goldenrod", width=1.5, dash="dash"), row=1, col=1)
        fig.add_annotation(x=x1_date, y=take_profit_1, text=f"1:1保本 {take_profit_1:.2f}",
                           showarrow=False, xanchor="left", font=dict(color="goldenrod", size=10), row=1, col=1)
    
    if take_profit_2 is not None and take_profit_2 > 0:
        fig.add_shape(type="line", x0=x0_date, x1=x1_date, y0=take_profit_2, y1=take_profit_2,
                      line=dict(color="limegreen", width=1.5, dash="dash"), row=1, col=1)
        fig.add_annotation(x=x1_date, y=take_profit_2, text=f"2:1减半 {take_profit_2:.2f}",
                           showarrow=False, xanchor="left", font=dict(color="limegreen", size=10), row=1, col=1)
    
    # --- Exit Signal Markers (last 20 trading days only) ---
    if exit_signals is not None and not exit_signals.empty:
        exit_cols = [c for c in exit_signals.columns if c.startswith('Exit_')]
        if exit_cols:
            any_exit = exit_signals[exit_cols].any(axis=1)
            # Limit to last 20 trading days
            recent_mask = pd.Series(False, index=df_plot.index)
            recent_mask.iloc[-min(20, len(recent_mask)):] = True
            exit_points = df_plot[any_exit & recent_mask]
            if not exit_points.empty:
                exit_descs = {
                    'Exit_MACD_Decay': 'MACD衰竭',
                    'Exit_Break_MA20': '破MA20',
                    'Exit_UA_Reversal': '天量滞涨',
                    'Exit_Fail_Breakout': '突破失败',
                }
                exit_hover = []
                for idx_e in exit_points.index:
                    parts = []
                    for ec in exit_cols:
                        try:
                            if exit_signals.loc[idx_e, ec]:
                                parts.append(exit_descs.get(ec, ec.replace('Exit_', '')))
                        except:
                            pass
                    exit_hover.append('⚠️ ' + ', '.join(parts))
                
                fig.add_trace(go.Scatter(
                    x=exit_points['date'],
                    y=exit_points['high'] * 1.02,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color='orange'),
                    hovertext=exit_hover,
                    hoverinfo='text',
                    name='⚠️ 离场信号',
                    showlegend=True
                ), row=1, col=1)

    
    # --- Optimize Hover Display ---
    # Disable hover for all indicators (MA, Boll, etc.) to prevent clutter/confusion
    # Only show hover for Candlestick (K线) and Signal Markers
    for trace in fig.data:
        if trace.name not in ['K线', '信号日', '⚠️ 离场信号']:
            trace.hoverinfo = 'skip'
            trace.hovertemplate = None

    fig.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0), 
                      hovermode='x', # Snap to x-axis
 
                      title=f"{code} - {name} ({df_sel['date'].iloc[-1].strftime('%Y-%m-%d')})",
                      xaxis_rangeslider_visible=False,
                      xaxis2_rangeslider_visible=False)
    
    # Use category axis to remove non-trading days gaps
    # We need to ensure X values are strings for this to work best or let Plotly handle it
    # But simply setting type='category' usually works on the dataframe index or column
    fig.update_xaxes(type='category', tickmode='auto', nticks=20)
    
    if return_fig:
        return fig
    else:
        st.plotly_chart(fig, use_container_width=True)
    
    # Indicator Explanation
    st.markdown("### 📚 指标与战法说明")
    with st.expander("点击展开查看详细说明", expanded=True if triggered_strategies else False):
        # 1. Triggered Strategies
        if triggered_strategies:
            st.markdown("#### 🎯 本次筛选触发策略")
            for strat in triggered_strategies:
                 strat_key = strat.strip()
                 desc = get_dyn_strategy_desc(strat_key, df_sel)
                 st.markdown(f"- {desc}", unsafe_allow_html=True)
            st.divider()

        st.markdown("#### 📉 当前副图指标")
        if sub_chart_type == "MACD":
            st.markdown("""
            **MACD (平滑异同移动平均线)**
            - **用法**: 
                - **Fighting**: 柱状图(Hist)翻红，DIF > DEA，且位于0轴上方，配合K线突破，为主升浪信号。
                - **底背离**: 股价创新低但 MACD 底部抬高，预示反转。
            """)
        elif sub_chart_type == "KDJ":
             st.markdown("""
            **KDJ (随机指标)**
            - **用法**:
                - **超买**: J > 100, K/D > 80.
                - **超卖**: J < 0, K/D < 20.
                - **金叉**: K 上穿 D (低位更佳).
            """)
        elif sub_chart_type == "WR":
             st.markdown("""
            **Williams %R (威廉指标)**
            - **用法**:
                - **超买**: %R > -20.
                - **超卖**: %R < -80.
            """)
        elif sub_chart_type == "CCI":
             st.markdown("""
            **CCI (顺势指标)**
            - **用法**:
                - **趋势**: > 100 强势，< -100 弱势。
                - **背离**: 股价创新高但 CCI 未创新高。
            """)
        elif sub_chart_type == "Volume":
            st.markdown("""
            **Volume (成交量)**
            - **Limit 缩量**: 当成交量低于 MA20 的一半时，为主力洗盘极致，变盘在即。
            - **20VMA 启动**: 长期缩量后，成交量首次突破 20日均量线，是趋势启动的信号。
            """)
        elif sub_chart_type == "RKing (趋势)":
            st.markdown("""
            **RKing 趋势跟随系统**
            - **核心逻辑**: 基于平均K线(Heikin-Ashi)变体与波动率通道构建的趋势系统。
            - **信号**: 
                - <font color='red'>**红色柱**</font>: 多头趋势 (Long State)。
                - <font color='green'>**绿色柱**</font>: 空头趋势 (Short State)。
                - <font color='red'>**🔺 红色买入点**</font>: 趋势由空转多，且突破上轨 (UP)。
                - <font color='green'>**🔻 绿色卖出点**</font>: 趋势由多转空，跌破下轨 (DOWN)。
            """, unsafe_allow_html=True)
        elif sub_chart_type == "RSI":
            st.markdown("""
            **RSI (相对强弱指标)**
            - **RSI2 回归**: 短期震荡策略。在上升趋势中，RSI2 < 10 (或25) 代表极度超卖，是回调买点。
            """)
        elif sub_chart_type == "Volatility":
             st.markdown("""
            **Volatility (波动率)**
            - **ES 压缩**: Std20 小于长周期波动率，代表K线形态收敛到极致（心电图），通常紧接着剧烈变盘。
            """)
        elif sub_chart_type == "HMC":
             st.markdown("""
            **HMC (High-Momentum Channel)**
            - **红线 (动能)**: 收盘价 - EMA200 (越大越强，代表趋势强度)。
            - **黄线 (阻力)**: 50日最高价 - 收盘价 (越小越好，代表离新高距离)。
            - **买入信号**: 红线上穿黄线，且红线 > 0 (年线之上)。
            - **卖出信号**: 红线下穿黄线，或黄线大幅抬头(远离新高)。
            """)


# Initialize Loader
@st.cache_resource
def get_loader():
    # Use absolute path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data/market_data")
    return DataLoader(data_dir=data_dir)
loader = get_loader()
stock_list_df = loader.get_stock_list()

# --- Main Application Logic ---

if app_mode == "🌍 宏观大盘":
    st.header("🌍 宏观大盘战法看板 (实时流动性与情绪)")
    
    @st.fragment(run_every=60 if is_trading_time() else None)
    def macro_dashboard():
        st.markdown("### 💡 智能宏观点评 (AI Insight)")
        
        # Initialize session state for AI commentary
        if 'macro_ai_commentary' not in st.session_state:
            st.session_state['macro_ai_commentary'] = None
        
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None
            
        model_name = "gemini-3.1-pro-preview"
        
        # Cache key for data stability
        beijing_tz = pytz.timezone('Asia/Shanghai')
        now = datetime.datetime.now(beijing_tz)
        if is_trading_time():
            cache_key = now.strftime("%Y-%m-%d-%H-") + str((now.minute // 5) * 5)
        else:
            cache_key = now.strftime("%Y-%m-%d-CLOSED")
            
        # 1. Fetch Macro Data (Fast, Skip AI by default)
        data = get_analysis(key=api_key, model=model_name, cache_key=cache_key, skip_ai=True)
            
        if "error" in data:
            st.error(data["error"])
            return

        # 2. AI Refresh Button
        ai_col1, ai_col2 = st.columns([1, 4])
        with ai_col1:
            if st.button("🤖 刷新 AI 点评", help="调动大模型深度分析当前盘面", type="primary"):
                with st.spinner("AI 正在深度思考中..."):
                    # Explicitly call with skip_ai=False
                    ai_data = get_analysis(key=api_key, model=model_name, cache_key=cache_key + "-FORCE-AI", skip_ai=False)
                    if "ai_commentary" in ai_data:
                        st.session_state['macro_ai_commentary'] = ai_data["ai_commentary"]
                        st.rerun()

        # 3. Display AI Commentary
        if st.session_state['macro_ai_commentary']:
            st.success(st.session_state['macro_ai_commentary'], icon="🤖")
        else:
            st.info("💡 请点击左侧按钮，让 AI 生成深度宏观逻辑解析。")

        # --- LIVE PATCHING FOR INDEX QUOTES ---
        # Ensure prices and charts in the fragment are truly live (every 60s)
        try:
            from amarket_framework.tencent_loader import TencentLoader
            rt_loader = TencentLoader()
            rt_indices = ["sh000001", "sz399001", "sz399006"]
            rt_df = rt_loader.fetch_realtime_quotes(rt_indices)
            
            if not rt_df.empty:
                for _, row in rt_df.iterrows():
                    code = row['code']
                    # Patch board info
                    for board_key, board_val in data.get('boards', {}).items():
                        if board_val['code'].endswith(code):
                            # Update current price
                            board_val['trend']['current_price'] = float(row['close'])
                            
                            # Patch the historical dataframe used for charts
                            if 'data' in board_val:
                                board_val['data'] = patch_df_with_realtime(board_val['data'], board_val['code'])
        except Exception as e:
            print(f"Error in macro live patching: {e}")

        st.divider()

        # Macro & Liquidity Section
        st.subheader("1. 宏观流动性 (Liquidity)")
        macro = data.get("macro", {})
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            margin_balance = macro.get('margin', {}).get('margin_balance', 0)
            if margin_balance is None: margin_balance = 0
            st.metric("两融余额 (Margin Balance)", f"{float(margin_balance):.2f}亿", help="融资+融券余额，代表杠杆资金情绪")
            
            # Margin History Chart
            margin_hist = macro.get('margin', {}).get('history', None)
            if margin_hist is not None and not margin_hist.empty:
                df_chart = margin_hist.copy()
                df_chart['date'] = pd.to_datetime(df_chart['date'])
                df_chart['DisplayBalance'] = df_chart['total_balance'] / 1e8
                
                fig_margin = px.area(df_chart, x='date', y='DisplayBalance', 
                                     title="两融余额趋势 (近1年)", 
                                     labels={'DisplayBalance': '余额 (亿)', 'date': '日期'},
                                     height=300)
                
                fig_margin.add_hline(y=20000, line_dash="dash", line_color="red", 
                                     annotation_text="2万亿警戒线", annotation_position="top right")
                
                fig_margin.update_layout(template='plotly_white', margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
                st.plotly_chart(fig_margin, use_container_width=True)

        with m_col2:
            cutoff = macro.get('money', {}).get('scissors', 0)
            if cutoff is None: cutoff = 0
            
            # Determine market health status based on scissors gap
            if cutoff > 0:
                health_status = "✅ 健康区域"
                health_color = "green"
                health_desc = "M1>M2，资金活跃，市场健康"
            elif cutoff > -5:
                health_status = "⚠️ 观察区域"
                health_color = "orange"
                health_desc = "剪刀差收窄，关注流动性"
            else:
                health_status = "🚨 风险区域"
                health_color = "red"
                health_desc = "剪刀差倒挂，流动性陷阱风险"
            
            st.metric("M1-M2 剪刀差", f"{float(cutoff):.2f}%", delta_color="normal" if cutoff > 0 else "inverse", help="M1同比 - M2同比。负值扩大代表流动性陷阱。")
            st.markdown(f"**当前状态**: :{health_color}[{health_status}] - {health_desc}")
            
            money_hist = macro.get('money', {}).get('history', None)
            if money_hist is not None and not money_hist.empty:
                fig_money = go.Figure()
                
                fig_money.add_hrect(y0=-20, y1=-5, fillcolor="rgba(255,0,0,0.1)", layer="below", line_width=0)
                fig_money.add_hrect(y0=-5, y1=0, fillcolor="rgba(255,255,0,0.1)", layer="below", line_width=0)
                fig_money.add_hrect(y0=0, y1=40, fillcolor="rgba(0,255,0,0.1)", layer="below", line_width=0)
                
                fig_money.add_trace(go.Scatter(x=money_hist['date'], y=money_hist['m1_yoy'], name='M1同比%', line=dict(color='blue', width=1.5)))
                fig_money.add_trace(go.Scatter(x=money_hist['date'], y=money_hist['m2_yoy'], name='M2同比%', line=dict(color='green', width=1.5)))
                fig_money.add_trace(go.Scatter(x=money_hist['date'], y=money_hist['scissors'], name='剪刀差 (M1-M2)', line=dict(color='red', width=2), fill='tozeroy'))
                
                fig_money.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7, annotation_text="0% (健康线)", annotation_position="right")
                fig_money.add_hline(y=-5, line_dash="dot", line_color="orange", opacity=0.7, annotation_text="-5% (警戒线)", annotation_position="right")
                
                fig_money.update_layout(
                    title="M1-M2 剪刀差趋势 (资金面健康度) - 区域根据红色剪刀差线判断",
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    hovermode="x unified",
                    yaxis=dict(title="%", range=[-20, 40]),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_money, use_container_width=True)
        st.caption("数据来源: 两融数据 (沪深交易所 via AkShare) / 货币供应 (中国人民银行 via AkShare)")

        st.divider()

        # Display last update time with Beijing timezone
        beijing_tz = pytz.timezone('Asia/Shanghai')
        current_time = datetime.datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')
        
        now = datetime.datetime.now(beijing_tz)
        
        if is_trading_time():
            snapshot_str = now.strftime('%Y-%m-%d %H:%M')
        else:
            if now.weekday() >= 5:
                days_since_friday = (now.weekday() - 4) % 7
                last_friday = now - pd.Timedelta(days=days_since_friday)
                snapshot_str = last_friday.strftime('%Y-%m-%d') + ' 15:00'
            elif now.time() < datetime.datetime.strptime("09:00", "%H:%M").time():
                prev_day = now - pd.Timedelta(days=1)
                while prev_day.weekday() >= 5:
                    prev_day = prev_day - pd.Timedelta(days=1)
                snapshot_str = prev_day.strftime('%Y-%m-%d') + ' 15:00'
            else:
                snapshot_str = now.strftime('%Y-%m-%d') + ' 15:00'
        
        st.subheader(f"2. 市场全景 (Snapshot) - {snapshot_str}")
        
        if is_trading_time():
            st.caption(f"🔄 最后更新: {current_time} | ✅ 交易时间 - 数据每10秒自动刷新")
        else:
            st.caption(f"🔄 最后更新: {current_time} | ⏸️ 非交易时间 - 数据已暂停刷新")
        
        cols = st.columns(3)
        for i, (key, info) in enumerate(data.get("boards", {}).items()):
            if "error" in info: continue
            
            trend = info['trend']
            price = trend['current_price']
            ema = trend['ema200']
            
            trend_color = "inverse" if price > ema else "normal"
            
            cols[i].metric(
                label=f"{info['name']}",
                value=f"{price:.2f}",
                delta=f"EMA200: {ema:.2f} ({trend['status']})",
                delta_color=trend_color
            )

        st.markdown("### 📊 核心指标矩阵")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**资金与情绪**", icon="🌊")
            i_cols = st.columns(3)
            for i, (key, info) in enumerate(data.get("boards", {}).items()):
                if "error" in info: continue
                fund = info['funding']
                fund_color = "inverse" if fund['status'] == "放量" else "normal"
                i_cols[i].metric(info['name'], fund['status'], f"Vol: {fund['value']/10000:.0f}万手", delta_color=fund_color)
                
        with col2:
            st.info("**恐慌与时机**", icon="⚡")
            i_cols = st.columns(3)
            for i, (key, info) in enumerate(data.get("boards", {}).items()):
                if "error" in info: continue
                sent = info['sentiment']
                i_cols[i].metric(info['name'], sent['status'], f"Bias: {sent['score']:.2f}%", delta_color="inverse")
            st.caption("注：Bias (乖离率) = (当前价 - MA20)/MA20。>5%为过热(风险)，<-5%为恐慌(机会)。")
        st.divider()
        
        tab1, tab2, tab3 = st.tabs(["趋势与K线", "资金成交量", "风格轮动"])
        
        def plot_board_charts(chart_func):
            valid_boards = [info for key, info in data.get('boards', {}).items() if "error" not in info]
            if not valid_boards:
                st.warning("暂无有效市场数据")
                return
                
            b_tabs = st.tabs([b['name'] for b in valid_boards])
            for i, info in enumerate(valid_boards):
                with b_tabs[i]:
                    chart_func(info['data'], info)

        with tab1:
            st.caption("蓝色线为EMA200牛熊分界线。线上做多，线下防守。")
            def chart_trend(df, info):
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index.strftime('%Y-%m-%d'), 
                                open=df['open'], high=df['high'],
                                low=df['low'], close=df['close'], name='K线',
                                increasing_line_color='red', decreasing_line_color='green'))
                                
                fig.add_trace(go.Scatter(x=df.index.strftime('%Y-%m-%d'), 
                                         y=df['close'].ewm(span=200, adjust=False).mean(), 
                                         name='EMA200', line=dict(color='blue', width=2)))
                
                fig.update_layout(template='plotly_white', xaxis_rangeslider_visible=False, height=400,
                                  xaxis=dict(type='category', nticks=10, tickangle=-45))
                st.plotly_chart(fig, use_container_width=True)
            plot_board_charts(chart_trend)
            
        with tab2:
            st.caption("橙色线为20日均量线。显示最近90日成交量。")
            def chart_funding(df, info):
                df_display = df.tail(90)
                if len(df_display) > 0:
                    vol_min, vol_max, vol_mean = df_display['volume'].min(), df_display['volume'].max(), df_display['volume'].mean()
                    zero_count, non_zero_count = (df_display['volume'] == 0).sum(), (df_display['volume'] > 0).sum()
                    
                    st.info(f"📊 数据范围: {df_display.index.min().strftime('%Y-%m-%d')} 至 {df_display.index.max().strftime('%Y-%m-%d')} | 共 {len(df_display)} 个交易日")
                    st.caption(f"🔍 成交量统计: 最小={vol_min:,.0f}, 最大={vol_max:,.0f}, 均值={vol_mean:,.0f} | 零值天数={zero_count}, 非零天数={non_zero_count}")
                    
                    if zero_count > len(df_display) * 0.9:
                        st.error("❌ 数据异常: 90%以上的交易日成交量为0，请检查数据源")
                        return
                else:
                    st.warning("⚠️ 无成交量数据")
                    return
                
                df_filtered = df_display[df_display['volume'] > 0].copy()
                if len(df_filtered) == 0:
                    st.error("❌ 所有交易日成交量均为0")
                    return
                
                st.caption(f"📈 实际绘制 {len(df_filtered)} 个非零成交量交易日")
                
                fig = go.Figure()
                colors = ['red' if c > o else 'green' for c, o in zip(df_filtered['close'], df_filtered['open'])]
                x_dates = df_filtered.index.strftime('%Y-%m-%d').tolist()
                
                fig.add_trace(go.Bar(x=x_dates, y=df_filtered['volume'].tolist(), name='成交量', marker_color=colors,
                                     hovertemplate='日期: %{x}<br>成交量: %{y:,.0f} 手<extra></extra>'))
                                     
                ma20_values = df_filtered['volume'].rolling(20, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=x_dates, y=ma20_values.tolist(), name='MA20', line=dict(color='orange', width=2),
                                         hovertemplate='日期: %{x}<br>MA20: %{y:,.0f}<extra></extra>'))
                                         
                fig.update_layout(template='plotly_white', height=400, xaxis=dict(type='category', showgrid=True, gridcolor='rgba(128,128,128,0.2)', tickangle=-45, tickmode='auto', nticks=20),
                                  yaxis=dict(title='成交量 (手)', showgrid=True, gridcolor='rgba(128,128,128,0.2)', rangemode='tozero'),
                                  title=f"成交量 (最近90日) - 单位: 手", hovermode='x unified', showlegend=True,
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
            plot_board_charts(chart_funding)
            
        with tab3:
            if "error" not in data.get('style', {}):
                style = data.get('style', {})
                if 'suggestion' in style and 'trend' in style:
                    st.metric("当前主线", style['suggestion'], f"趋势: {style['trend']}", delta_color="inverse")
                    st.caption("逻辑: 相对强弱(RS) = 创业板/沪指。当RS位于均线(MA20)上方时，视为成长风格占优。")
                    
                    rs_line = style['rs_line']
                    rs_ma20 = style.get('rs_ma20', None)
                    
                    fig_style = go.Figure()
                    fig_style.add_trace(go.Scatter(x=rs_line.index, y=rs_line, name='RS (创业板/沪指)', line=dict(color='blue')))
                    
                    if rs_ma20 is not None:
                        fig_style.add_trace(go.Scatter(x=rs_ma20.index, y=rs_ma20, name='MA20', line=dict(color='orange', width=1)))
                        
                    fig_style.update_layout(template='plotly_white', title="风格相对强弱趋势", height=350, hovermode="x unified")
                    st.plotly_chart(fig_style, use_container_width=True)
            else:
                st.write("数据不足")

    # Actually call the fragment
    macro_dashboard()

elif app_mode == "📈 个股行情":
    st.header("📈 个股行情分析/Analysis (交互式Beta)")
    st.info("💡 **新功能**: 点击 K 线图任意位置，下方将显示当日触发的策略信号。")
    
    if stock_list_df.empty:
        st.error("数据未就绪，请先下载。")
        st.stop()
        
    # 1. Stock Selection
    # Format: "000001 - 平安银行"
    stock_options = [f"{r['code']} - {r['name']}" for r in stock_list_df.to_dict('records')]
    
    selected_stock = st.selectbox("搜索/选择股票 (Search Stock)", options=stock_options)
    
    if selected_stock:
        code = selected_stock.split(" - ")[0]
        name = selected_stock.split(" - ")[1]
        
        # Controls
        col_ctrl, col_chart = st.columns([1, 3])
        with col_ctrl:
            st.subheader("图表配置")
            
            # Date Range override
            analysis_start = st.date_input("开始日期", default_start, key='ana_start')
            analysis_end = st.date_input("结束日期", default_end, key='ana_end')
            
            st.markdown("**主图层**")
            show_ma = st.checkbox("MA20 均线", value=True)
            show_ema = st.checkbox("EMA200 (牛熊线)", value=True)
            show_ema15 = st.checkbox("EMA15 (HPS通道)", value=False)
            show_cyc = st.checkbox("CYC (成本均线)", value=False)
            show_boll = st.checkbox("布林带", value=True)
            show_box = st.checkbox("Box Top (250日高点)", value=False)
            show_supt = st.checkbox("Support (20日低点)", value=False)

            st.markdown("**副图指标**")
            sub_chart_type = st.radio("选择副图:", ["MACD", "KDJ", "RSI", "WR", "CCI", "Volume", "RKing (趋势)", "Volatility", "HMC"])
            

        with col_chart:
            # Load Data
            # Need strict load range for proper indicator calc? 
            # Loader basically just loads file, we filter later.
            # But calculating indicators needs history.
            load_start = (analysis_start - datetime.timedelta(days=400)).strftime("%Y-%m-%d")
            load_end = analysis_end.strftime("%Y-%m-%d")
            
            df = loader.get_k_data(code, load_start, load_end)
            
            if not df.empty:
                df = patch_df_with_realtime(df, code)
                # --- Phase 17: Sector/Concept Information ---
                from market_env import MarketEnvironmentLoader
                env_loader = MarketEnvironmentLoader()
                sector_name = env_loader.get_stock_sector(code)
                concepts = env_loader.get_stock_concepts(code)
                
                # Latest health status
                c_date = df.iloc[-1]['date'].strftime("%Y-%m-%d") if not df.empty else analysis_end.strftime("%Y-%m-%d")
                is_healthy = env_loader.is_sector_health_healthy(code, c_date) if hasattr(env_loader, 'is_sector_health_healthy') else env_loader.is_sector_healthy(code, c_date)
                health_icon = "🟢" if is_healthy else "🔴"
                health_text = "多头趋势" if is_healthy else "空头趋势"
                
                st.markdown(f"🏭 **所属板块**: `{sector_name}` ({health_icon} {health_text})")
                if concepts:
                    # Show top 15 concepts to avoid clutter
                    concept_str = " | ".join(concepts[:15])
                    st.markdown(f"🧩 **关联概念**: <font color='gray'>{concept_str}</font>", unsafe_allow_html=True)
                st.markdown("---")

                df = Indicators.add_all_indicators(df)
                # Filter for display
                df_display = df[(df['date'].dt.date >= analysis_start) & (df['date'].dt.date <= analysis_end)]
                
                # === Load Cached Signals ===
                cache_sigs = pd.DataFrame()
                try:
                    cache_reader_ana = SignalCacheReader()
                    cache_sigs = cache_reader_ana.get_stock_signals(code)
                except Exception:
                    pass
                
                if not cache_sigs.empty:
                    # Cache has DatetimeIndex. Align with df's RangeIndex.
                    cache_sigs = cache_sigs.reset_index()
                    cache_sigs['date'] = pd.to_datetime(cache_sigs['date'])
                    merged = pd.merge(df[['date']], cache_sigs, on='date', how='left')
                    sig_cols = [c for c in merged.columns if c.startswith('Signal_')]
                    sigs_ana = merged[sig_cols].fillna(False)
                    sigs_ana.index = df.index
                else:
                    # Fallback: Real-time calculation
                    st.warning("⚠️ 未找到缓存信号，正在实时计算...")
                    try:
                        index_df_ana = None
                        index_path_ana = os.path.join(MARKET_DATA_DIR, "000001.SH.csv")
                        if os.path.exists(index_path_ana):
                            try:
                                index_df_ana = pd.read_csv(index_path_ana)
                                index_df_ana['date'] = pd.to_datetime(index_df_ana['date'])
                            except: pass
                        sigs_strong_ana = StrongStrategies.check_all_strong_strategies(df, index_df=index_df_ana)
                        sigs_weak_ana = WeakStrategies.check_all_weak_strategies(df)
                        sigs_ana = pd.concat([sigs_strong_ana, sigs_weak_ana], axis=1)
                        sigs_ana = sigs_ana.loc[:, ~sigs_ana.columns.duplicated()]
                    except Exception:
                        sigs_ana = Strategies.check_all(df)
                
                # === Determine signal_dates for chart markers ===
                sig_cols_all = [c for c in sigs_ana.columns if c.startswith('Signal_')]
                any_signal = sigs_ana[sig_cols_all].any(axis=1) if sig_cols_all else pd.Series(False, index=df.index)
                signal_dates = df[any_signal & (df['date'].dt.date >= analysis_start) & (df['date'].dt.date <= analysis_end)]['date']
                
                # === Interactive Chart with return_fig ===
                fig = plot_stock_chart(df_display, code, name, show_ma, show_ema, show_boll, show_cyc, show_ema15, show_box, show_supt, True, sub_chart_type, plotly_template, sigs_ana, signal_dates, return_fig=True)
                fig.update_layout(clickmode='event+select')
                event = st.plotly_chart(fig, on_select="rerun", use_container_width=True, key="analysis_chart_main")
                
                # === Handle Click Event ===
                if event and event.selection and event.selection.points:
                    sel_point = event.selection.points[0]
                    if 'point_index' in sel_point:
                        idx = sel_point['point_index']
                        if idx < len(df_display):
                            sel_date = df_display.iloc[idx]['date']
                            st.info(f"📅 **选中日期**: {sel_date.strftime('%Y-%m-%d')}")
                            
                            try:
                                target_idx = df[df['date'] == sel_date].index
                                if not target_idx.empty:
                                    sig_row = sigs_ana.loc[target_idx[0]]
                                    found_sigs = [c.replace('Signal_', '') for c in sig_cols_all if sig_row[c]]
                                    if found_sigs:
                                        st.success(f"🔥 当日触发信号: **{', '.join(found_sigs)}**")
                                    else:
                                        st.info("当日无策略信号触发。")
                            except Exception as e:
                                st.error(f"Error: {e}")
                
                # === Indicator Table with Triggered_Signals ===
                with st.expander("📊 指标数值详情 (Indicator Values)", expanded=True):
                    df_table = df_display.copy()
                    
                    # Add Triggered_Signals column
                    if sig_cols_all:
                        sigs_subset = sigs_ana.loc[df_table.index]
                        def get_sig_str(row):
                            names = [c.replace('Signal_', '') for c in sig_cols_all if row[c]]
                            return ", ".join(names) if names else ""
                        df_table['Triggered_Signals'] = sigs_subset.apply(get_sig_str, axis=1)
                    else:
                        df_table['Triggered_Signals'] = ""
                    
                    cols_to_show = ['date', 'close', 'volume', 'Triggered_Signals', 'MA20', 'MACD_Hist', 'K', 'D', 'J', 'RSI6', 'RKing_State']
                    cols_final = [c for c in cols_to_show if c in df_table.columns]
                    st.dataframe(
                        df_table[cols_final].sort_values(by='date', ascending=False),
                        use_container_width=True,
                        column_config={
                            "date": st.column_config.DateColumn("日期", format="YYYY-MM-DD"),
                            "Triggered_Signals": st.column_config.TextColumn("触发信号", width="large", help="当日触发的缓存策略信号"),
                            "volume": st.column_config.NumberColumn("成交量", format="%d")
                        }
                    )

                # --- AI Diagnosis ---
                st.markdown("---")
                st.subheader("🤖 AI 智能诊断 (Gemini 3 Pro)")
                
                if st.button("开始诊断 (Start Diagnosis)"):
                    try:
                        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
                    except (FileNotFoundError, KeyError):
                        st.error("未找到 API Key。请在 .streamlit/secrets.toml 中配置 GEMINI_API_KEY。")
                        st.stop()
                        
                    from stock_diagnosis import StockDiagnoser
                    diagnoser = StockDiagnoser(GEMINI_API_KEY)
                    
                    with st.spinner("正在请求 AI 模型进行深度分析... (可能需要30-60秒)"):
                        report = diagnoser.generate_report(df, code, name, sigs_ana)
                    st.markdown(report)
            else:
                st.warning("暂无数据 (No Data).")
                st.error(f"Debug: Code={code}, LoadStart={load_start}, LoadEnd={load_end}")
                # Check file existence
                file_p = os.path.join("stock_app/data/market_data", f"{code}.csv")
                if os.path.exists(file_p):
                    st.write(f"File exists at {file_p}")
                else:
                    st.write(f"File NOT found at {file_p}")

elif app_mode == "🔍 策略选股":
    # --- Strategy Screening Logic ---
    st.header("🔍 策略选股/Screening")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        chart_start = st.date_input("筛选/显示开始日期", default_start, key='scr_start')
    with col_d2:
        chart_end = st.date_input("筛选/显示结束日期", default_end, key='scr_end')
    
    # --- Strategy Configuration (Inline) ---
    with st.expander("⚙️ 策略配置", expanded=True):
        st.markdown("**强势跟随类** — 追踪趋势突破、量价共振信号")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            strat_fighting = st.checkbox("Fighting (突破)", value=True, key='scr_fighting', help="MACD翻红 + 股价创52日新高 + 量能创52日新高 + 布林带确认。主升浪启动信号。")
            strat_cyc = st.checkbox("CYC MAX (成本均线)", key='scr_cyc', help="股价站上无穷成本均线(CYC∞)，意味着市场上所有持仓者全部获利，极致多头状态。")
        with col2:
            strat_range = st.checkbox("Range Break (年度箱体)", key='scr_range', help="突破250日最高价（年度箱体上沿），伴随放量确认，新一轮主升浪的起点。")
            strat_20vma = st.checkbox("20VMA (量能觉醒)", key='scr_20vma', help="长期缩量后成交量首次突破20日均量线，配合价格突破，量能觉醒趋势启动信号。")
        with col3:
            strat_hmc = st.checkbox("HMC (动量)", key='scr_hmc', help="高动量通道：红线(收盘-EMA200)上穿黄线(50日最高-收盘)，动量从防守切换到进攻。")
            strat_hps = st.checkbox("HPS (趋势通道)", key='scr_hps', help="站上EMA200牛熊分界线，且突破EMA15高点通道，双重趋势确认。")
            strat_obo = st.checkbox("OBO (开盘突破)", key='scr_obo', help="突破开盘价加上日前震幅的区间。")
        with col4:
            strat_tkos = st.checkbox("TKOS (股王)", key='scr_tkos', help="月涨幅>50%，只有敢于在一个月内涨50%的股票才具备「股王」气质，妖股启动信号。")
            strat_rking = st.checkbox("Rking (趋势)", key='scr_rking', help="红柱=多头趋势，绿柱=空头趋势。红柱出现代表趋势中继或启动买点。")
            strat_lcs = st.checkbox("LCS (极限策略)", key='scr_lcs', help="5日极其偏离，博弈超高弹性的极限反转。")
        
        # Ambush checkboxes moved to Scoring in Daily Trading Plan
        st.markdown("**超跌底部类** — 捕捉极端超卖后的反转回归信号")
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            strat_hlp3 = st.checkbox("HLP3 (大慈悲点)", key='scr_hlp3', help="获利盘<1%后飙升>35%")
            strat_wyckoff = st.checkbox("Wyckoff (量价背离)", key='scr_wyckoff', help="价格创新低但成交量萎缩")
            strat_flow = st.checkbox("Money Flow", key='scr_flow', help="资金底背离")
        with col6:
            strat_limit = st.checkbox("Limit (极致缩量)", key='scr_limit', help="成交量低于均量50%")
            strat_vol_min = st.checkbox("量比历史新低", key='scr_volmin', help="120日量比新低")
            strat_ua_weak = st.checkbox("UA (弱势)", key='scr_ua_weak', help="底部天量突破")
        with col7:
            strat_rsi = st.checkbox("RSI2 (均值回归)", key='scr_rsi', help="RSI(2)极度超卖")
            strat_es = st.checkbox("ES (波动率压缩)", key='scr_es', help="波动积极压缩至极致")
            strat_dv = st.checkbox("倍量不破", key='scr_dv', help="倍量回调不破")
        with col8:
            strat_boll = st.checkbox("Boll Rev (布林回归)", key='scr_boll', help="布林带下轨反弹")
            strat_2b = st.checkbox("2B (法则)", key='scr_2b', help="创新低后迅速拉回前低之上")
            strat_spring = st.checkbox("Spring (弹簧)", key='scr_spring', help="跌破前低快速收回")
            strat_pinbar = st.checkbox("Pinbar (长钉)", key='scr_pinbar', help="长下影线>实体3倍+伴随放量")
    
    # Session State
    if 'scan_results' not in st.session_state:
        st.session_state['scan_results'] = None

    if st.button("🚀 开始筛选 / Run Screening", type="primary", use_container_width=True):
        if stock_list_df.empty:
            st.error("无法开始：请先下载数据。")
            st.stop()
            
        scan_start_str = chart_start.strftime("%Y-%m-%d")
        scan_end_str = chart_end.strftime("%Y-%m-%d")
        
        # Determine selected strategies
        strong_strategies = []
        weak_strategies = []
        
        strat_map_strong = [
            (strat_fighting, "Fighting"),
            (strat_cyc, "CYC_MAX"),
            (strat_range, "RangeBreak"),
            (strat_20vma, "20VMA"),
            (strat_hmc, "HMC"),
            (strat_hps, "HPS"),
            (strat_tkos, "TKOS"),
            (strat_rking, "RKing"),
        ]
        strat_map_weak = [
            (strat_hlp3, "HLP3"),
            (strat_limit, "Limit"),
            (strat_vol_min, "Vol_Min_120"),
            (strat_rsi, "RSI_Rev"),
            (strat_es, "ES"),
            (strat_boll, "Boll_Rev"),
            (strat_2b, "2B"),
            (strat_wyckoff, "Wyckoff"),
            (strat_spring, "Spring"),
            (strat_pinbar, "Pinbar"),
            (strat_flow, "Money_Flow"),
            (strat_dv, "Double_Vol")
        ]
        
        for is_checked, name in strat_map_strong:
            if is_checked:
                strong_strategies.append(name)
        for is_checked, name in strat_map_weak:
            if is_checked:
                weak_strategies.append(name)
        
        if not strong_strategies and not weak_strategies:
            st.warning("请至少选择一个策略。")
            st.stop()
        
        # Try cache-based screening first (fast!)
        cache_reader = SignalCacheReader()
        is_valid, msg = cache_reader.is_cache_valid()
        
        if is_valid:
            st.info(f"📦 使用信号缓存进行快速筛选 ({scan_start_str} ~ {scan_end_str})...")
            
            results = []
            
            try:
                if strong_strategies:
                    strong_df = cache_reader.filter_strong_stocks(strong_strategies, scan_start_str, scan_end_str)
                    if not strong_df.empty:
                        results.append(strong_df)
                
                if weak_strategies:
                    weak_df = cache_reader.filter_weak_stocks(weak_strategies, scan_start_str, scan_end_str)
                    if not weak_df.empty:
                        results.append(weak_df)
                
                if results:
                    combined = pd.concat(results, ignore_index=True)
                    
                    # Group by stock: for each stock, take the latest signal date
                    combined['date'] = pd.to_datetime(combined['date'])
                    
                    # Aggregate: last signal date per stock, merge all triggered strategies
                    agg_results = []
                    for (code, name), grp in combined.groupby(['code', 'name']):
                        last_row = grp.loc[grp['date'].idxmax()]
                        # Collect all unique strategies triggered across all dates
                        all_strats = set()
                        for s in grp['triggered_strategies']:
                            all_strats.update(s.split(', '))
                        
                        # Get latest price from data if possible
                        price = None
                        try:
                            df_price = loader.get_k_data(code, scan_start_str, scan_end_str)
                            if not df_price.empty:
                                price = df_price.iloc[-1]['close']
                        except:
                            pass
                        
                        agg_results.append({
                            "Code": code,
                            "Name": name,
                            "Price": price,
                            "Signal Date": last_row['date'].strftime("%Y-%m-%d"),
                            "Strategies": ", ".join(sorted(all_strats))
                        })
                    
                    res_df = pd.DataFrame(agg_results)
                    res_df = res_df.sort_values(by='Signal Date', ascending=False).reset_index(drop=True)
                    st.session_state['scan_results'] = res_df
                    st.success(f"⚡ 缓存筛选完成！发现 {len(agg_results)} 只符合条件的股票。")
                else:
                    st.session_state['scan_results'] = pd.DataFrame()
                    st.warning("未找到符合条件的股票。")
                    
            except Exception as e:
                st.error(f"缓存筛选出错: {e}")
                st.session_state['scan_results'] = pd.DataFrame()
        else:
            st.error(f"信号缓存不可用: {msg}。请先点击侧边栏「🔄 重建信号缓存」按钮。")

    # --- Results Display ---
    if st.session_state['scan_results'] is not None and not st.session_state['scan_results'].empty:
        res_df = st.session_state['scan_results']
        # Convert Code to string to avoid comma format
        res_df['Code'] = res_df['Code'].astype(str)
        
        # Interactive Dataframe
        st.markdown("### 📊 筛选结果 (点击表格行查看详情)")
        event = st.dataframe(
            res_df, 
            use_container_width=True,
            on_select="rerun",  # Rerun app on selection
            selection_mode="single-row" 
        )
        
        st.divider()
        
        # Determine Selected Stock
        # Priority 1: Table Selection
        # Priority 2: Selectbox (Fallback/Legacy)
        
        selected_row_index = None
        if event.selection.rows:
            selected_row_index = event.selection.rows[0]
            
        # Update session state for selection if table clicked? 
        # Actually, let's use the table selection directly if present.
        
        if selected_row_index is not None:
             # After reset_index(drop=True), iloc[i] == loc[i], both are safe
             row_data = res_df.iloc[selected_row_index]
             code_s = str(row_data['Code'])
             name_s = str(row_data['Name'])
             highlight_date_s = row_data.get('Signal Date', None)
             st.info(f"当前选中: {code_s} - {name_s} (触发日期: {highlight_date_s})")
        else:
             st.info("👆 请在上方表格中点击选择一只股票查看详情。")
             # Fallback to Selectbox if no table selection? 
             # Let's keep selectbox as valid alternative or just hide it? 
             # User asked for "click table", so table is primary.
             # We can keep selectbox consistent. 
             # If table selected, we can't easily force selectbox to update unless we use a key and session state.
             # Simple approach: If table selection exists, use it. Else show selectbox.
             
             if 'Signal Date' in res_df.columns:
                screen_options = [f"{r['Code']} - {r['Name']} (Signal: {r['Signal Date']})" for r in res_df.to_dict('records')]
             else:
                screen_options = [f"{r['Code']} - {r['Name']}" for r in res_df.to_dict('records')]
                
             selected_screen = st.selectbox("或者：从下拉列表选择", options=screen_options, index=None, placeholder="选择股票...")
             
             if selected_screen:
                 code_s = selected_screen.split(" - ")[0]
                 name_s = selected_screen.split(" - ")[1].split(" (")[0]
                 
                 # Try to find signal date from dataframe for this code
                 try:
                    match_row = res_df[res_df['Code'] == code_s]
                    if not match_row.empty and 'Signal Date' in match_row.columns:
                        highlight_date_s = match_row.iloc[0]['Signal Date']
                    else:
                        highlight_date_s = None
                 except:
                    highlight_date_s = None
             else:
                 code_s = None
                 highlight_date_s = None
        
        if code_s:
            # Show Chart
            # We need to load data again for this specific stock
            # Use strict load range?
            load_start_s = (chart_start - datetime.timedelta(days=400)).strftime("%Y-%m-%d")
            load_end_s = chart_end.strftime("%Y-%m-%d")
            
            df_s = loader.get_k_data(code_s, load_start_s, load_end_s)
            
            if not df_s.empty:
                df_s = patch_df_with_realtime(df_s, code_s)
                df_s = Indicators.add_all_indicators(df_s)
                # Filter for display
                df_disp_s = df_s[(df_s['date'].dt.date >= chart_start) & (df_s['date'].dt.date <= chart_end)]
                
                # Re-calculate or Fetch signals for display markers 
                # User requested to use CACHED signals directly for consistency.
                
                # Try to fetch from cache first
                cache_sigs = pd.DataFrame()
                try:
                    reader = SignalCacheReader()
                    # Check if cache valid? (Optional, but good practice)
                    # get_stock_signals handles reading safely
                    cache_sigs = reader.get_stock_signals(code_s)
                except Exception as e:
                    # st.error(f"Cache Read Error: {e}")
                    pass
                
                if not cache_sigs.empty:
                    # Cache has date index. Align with df_s index.
                    # df_s index is RangeIndex? No, get_k_data usually returns RangeIndex with 'date' col.
                    # We need to map cache_sigs (Date Index) to df_s (Date Col)
                    
                    # 1. Reset cache index to get 'date' col
                    cache_sigs = cache_sigs.reset_index()
                    
                    # 2. Key on date
                    # Ensure date types match
                    cache_sigs['date'] = pd.to_datetime(cache_sigs['date'])
                    
                    # 3. Merge with df_s to align rows
                    # Left join on df_s to keep df_s structure
                    # We only need the Signal columns from cache
                    merged = pd.merge(df_s[['date']], cache_sigs, on='date', how='left')
                    
                    # 4. Extract Signal Columns
                    sig_cols = [c for c in merged.columns if c.startswith('Signal_')]
                    sigs_s = merged[sig_cols].fillna(False)
                    
                    # 5. Re-index to match df_s
                    sigs_s.index = df_s.index
                    
                    st.toast(f"✅ 已加载缓存信号数据 ({len(cache_sigs)} 条记录)", icon="📦")
                    
                else:
                    # Fallback to Real-time Calculation (Full Strategy Set)
                    st.warning("⚠️ 未找到缓存信号，正在实时计算 (可能与选股结果略有差异)...")
                    
                    # 1. Load Index Data for Strong Strategies (if needed)
                    index_df = None
                    index_path = os.path.join(MARKET_DATA_DIR, "000001.SH.csv")
                    if os.path.exists(index_path):
                        try:
                            index_df = pd.read_csv(index_path)
                            index_df['date'] = pd.to_datetime(index_df['date'])
                        except: pass
                    
                    # 2. Calculate Strong & Weak Signals
                    try:
                        sigs_strong = StrongStrategies.check_all_strong_strategies(df_s, index_df=index_df)
                    except Exception:
                        sigs_strong = pd.DataFrame(index=df_s.index)
                    
                    try:
                        sigs_weak = WeakStrategies.check_all_weak_strategies(df_s)
                    except Exception:
                        sigs_weak = pd.DataFrame(index=df_s.index)
                    
                    # 3. Merge Signals
                    sigs_s = pd.concat([sigs_strong, sigs_weak], axis=1)
                    sigs_s = sigs_s.loc[:, ~sigs_s.columns.duplicated()]
                    
                    if sigs_s.empty:
                         sigs_s = Strategies.check_all(df_s)
                
                # Determine combined signal mask for visualization
                # We want to see where the selected strategies triggered
                final_sig = pd.Series(True, index=df_s.index)
                selected_any_cfg = False
                 
                checks_map = {
                    'Signal_Fighting': strat_fighting,
                    'Signal_CYC_MAX': strat_cyc,
                    'Signal_RangeBreak': strat_range,
                    'Signal_20VMA': strat_20vma,
                    'Signal_HMC': strat_hmc,
                    'Signal_HPS': strat_hps,
                    'Signal_TKOS': strat_tkos,
                    'Signal_RKing': strat_rking,
                    'Signal_OBO': strat_obo,
                    'Signal_LCS': strat_lcs,
                    'Signal_HLP3': strat_hlp3,
                    'Signal_Limit': strat_limit,
                    'Signal_Vol_Min_120': strat_vol_min,
                    'Signal_Boll_Rev': strat_boll,
                    'Signal_RSI_Rev': strat_rsi,
                    'Signal_2B': strat_2b,
                    'Signal_Wyckoff': strat_wyckoff,
                    'Signal_Spring': strat_spring,
                    'Signal_Pinbar': strat_pinbar,
                    'Signal_Money_Flow': strat_flow,
                    'Signal_UA_Weak': strat_ua_weak,
                    'Signal_Double_Vol': strat_dv,
                    'Signal_ES': strat_es,
                    'Signal_Ambush_Calm': strat_ambush_calm,
                    'Signal_Ambush_Momentum': strat_ambush_momentum,
                    'Signal_Ambush_Bottom': strat_ambush_bottom
                }
                
                for col_name, is_chk in checks_map.items():
                    if is_chk:
                        selected_any_cfg = True
                        if col_name in sigs_s.columns:
                            final_sig &= sigs_s[col_name]
                
                signal_dates = None
                if selected_any_cfg:
                    # Filter for display range signals
                    signal_dates = df_s[final_sig & (df_s['date'].dt.date >= chart_start) & (df_s['date'].dt.date <= chart_end)]['date']

                # Controls Layout
                col_c1, col_c2 = st.columns([1, 4])
                with col_c1:
                    st.subheader("图表配置")
                    show_ma = st.checkbox("MA20", key='sc_ma')
                    show_ema = st.checkbox("EMA200", key='sc_ema')
                    show_boll = st.checkbox("Boll", key='sc_boll')
                    show_signals = st.checkbox("标注信号", key='sc_sig')
                    sub_chart_type = st.radio("副图:", ["MACD", "KDJ", "RSI", "WR", "CCI", "Volume", "RKing (趋势)", "Volatility", "HMC"], key='sc_sub')
                    
                with col_c2:
                    # Parse triggered strategies from result if available (this is from screening result, not interactive)
                    triggered_strats = []
                    if 'Strategies' in row_data:
                         triggered_strats = str(row_data['Strategies']).split(', ')

                    # Plot with Return Fig
                    # Plot with Return Fig
                    # Use highlight_date_s if available. DO NOT fallback to latest (it confuses user).
                    h_date = highlight_date_s 
                    fig = plot_stock_chart(df_disp_s, code_s, name_s, show_ma, show_ema, show_boll, False, False, False, False, show_signals, sub_chart_type, plotly_template, sigs_s, signal_dates, triggered_strategies=triggered_strats, return_fig=True, highlight_date=h_date)
                    
                    # Enable click event for selection
                    fig.update_layout(clickmode='event+select')
                    
                    # Interactive Chart
                    event = st.plotly_chart(fig, on_select="rerun", use_container_width=True, key="analysis_chart")
                    
                    # Debug: Show event structure
                    # st.write("DEBUG event:", event)
                    
                    # Handle Selection
                    if event and event.selection and event.selection.points:
                        # Plotly selection gives point detail. For candlestick/scatter on shared axis, point_index should allow us to find date.
                        # Note: shared_xaxes in subplots might optimize access.
                        # event.selection.points is a list of dictionaries.
                        sel_point = event.selection.points[0]
                        # point_index refers to the trace data index. 
                        # Since we use df_disp_s for plotting, index should match iloc.
                        if 'point_index' in sel_point:
                            idx = sel_point['point_index']
                            if idx < len(df_disp_s):
                                sel_date = df_disp_s.iloc[idx]['date']
                                st.info(f"📅 **选中日期**: {sel_date.strftime('%Y-%m-%d')}")
                                
                                # Find signals for this date
                                # sigs_s aligns with df_s. We need to find by index or date match.
                                # df_disp_s is a slice, so indices "should" optionally be preserved if not reset?
                                # Let's use date match for safety.
                                # sigs_s also has same index as df_s.
                                
                                # Find row in sigs_s
                                # Assuming sigs_s has same index (timestamp) or we join.
                                # Strategies.check_all(df) returns df with same index.
                                try:
                                    # Match date
                                    # Need to ensure type match. sel_date is Timestamp.
                                    # sigs_s index might be Int64Index if df_s was reset_index?
                                    # Let's check df_s index. 
                                    # Safe bet: Filter df_s by date, get index, use that index on sigs_s.
                                    target_idx = df_s[df_s['date'] == sel_date].index
                                    if not target_idx.empty:
                                        sig_row = sigs_s.loc[target_idx[0]]
                                        
                                        found_sigs = []
                                        for col in sigs_s.columns:
                                            if col.startswith('Signal_') and sig_row[col]:
                                                 found_sigs.append(col.replace('Signal_', ''))
                                        
                                        if found_sigs:
                                            st.success(f"🔥 当日触发信号: **{', '.join(found_sigs)}**")
                                        else:
                                            st.info("当日无策略信号触发 (No Strategy Signals Triggers).")
                                            # Debug info if no signal found
                                            # st.write("Checked Columns:", [c for c in sigs_s.columns if c.startswith('Signal_')])
                                    else:
                                        st.warning("Date alignment error (Index mismatch).")

                                except Exception as e:
                                    st.error(f"Error checking signals: {e}")
                
                # --- Indicator Table (Screening) ---
                with st.expander("📊 指标数值详情 (Indicator Values)", expanded=True):
                     # Add Signal Column for Table
                     # Create a copy for display to avoid messing up logic
                     df_table = df_disp_s.copy()
                     
                     # Map signals to each row
                     # Use sigs_s. Reindex to match df_table index (which is subset of df_s)
                     # Iterate is slow but fine for display (lines < 1000)
                     
                     # Better: Vectorized approach
                     # Create a series of joined strings
                     sig_cols = [c for c in sigs_s.columns if c.startswith('Signal_')]
                     if sig_cols:
                        # Filter sigs_s to match df_table rows
                        sigs_subset = sigs_s.loc[df_table.index]
                        
                        def get_sig_str(row):
                             names = [c.replace('Signal_', '') for c in sig_cols if row[c]]
                             return ", ".join(names) if names else ""
                             
                        df_table['Triggered_Signals'] = sigs_subset.apply(get_sig_str, axis=1)
                     else:
                        df_table['Triggered_Signals'] = ""

                     cols_to_show = ['date', 'close', 'volume', 'Triggered_Signals', 'MA20', 'MACD_Hist', 'K', 'D', 'J', 'RSI6', 'RKing_State']
                     cols_final = [c for c in cols_to_show if c in df_table.columns]
                     
                     st.dataframe(
                         df_table[cols_final].sort_values(by='date', ascending=False), 
                         use_container_width=True,
                         column_config={
                             "date": st.column_config.DateColumn("日期", format="YYYY-MM-DD"),
                             "Triggered_Signals": st.column_config.TextColumn("触发信号", help="当日触发的策略信号"),
                             "volume": st.column_config.NumberColumn("成交量", format="%d")
                         }
                     )

                # --- AI Diagnosis (Screening) ---
                st.subheader("🤖 AI 智能诊断 (Gemini 3 Pro)")
                if st.button("开始诊断 (Start Diagnosis)", key='diag_scr'):
                     try:
                        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
                     except (FileNotFoundError, KeyError):
                        st.error("未找到 API Key。请在 .streamlit/secrets.toml 中配置 GEMINI_API_KEY。")
                        st.stop()

                     from stock_diagnosis import StockDiagnoser
                     diagnoser = StockDiagnoser(GEMINI_API_KEY)
                     with st.spinner("正在请求 AI 模型进行深度分析..."):
                         report = diagnoser.generate_report(df_s, code_s, name_s, sigs_s)
                     st.markdown(report)

    elif st.session_state['scan_results'] is None:
        st.info("请点击上方按钮开始筛选。")

elif app_mode == "🚀 强势股进攻":
    # --- Strong Stock Attack Mode ---
    st.header("💪 强势股进攻 / Strong Stock Attack")
    st.markdown("""
    **核心逻辑**: 强者恒强。不买便宜的，只买更贵的；不买缩量的，只买放量突破的。
    
    - **第一阶段(海选与锁定)**: Z-score, RS, TKOS
    - **第二阶段(确认扳机)**: DTR Plus, Fighting, UA
    - **第三阶段(执行与防守)**: HMC
    """)
    
    # Import strong strategies module
    from strong_strategies import StrongStrategies
    
    # Date Range
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        strong_start = st.date_input("筛选/显示开始日期", default_start, key='strong_start')
    with col_d2:
        strong_end = st.date_input("筛选/显示结束日期", default_end, key='strong_end')
    
    # --- Strategy Configuration (Inline) ---
    
    # Pre-fetch Top 1 for button labels (FILTERED: Strong strategies only)
    STRONG_KEYS = {'Z_Score', 'CYC_MAX', 'RS', 'RangeBreak', 'TKOS', '20VMA', 'OBO', 'LCS', 'DTR_Plus', 'Fighting', 'UA', 'HMC', 'HPS', 'RKing'}
    top1_strat = "Z_Score + Fighting" # Fallback
    top1_return = "0.0"
    try:
        json_path = os.path.join(BASE_DIR, "optimization_results.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                res = json.load(f)
            # Filter: only combos where ALL strategies are strong
            strong_only = [r for r in res if all(s.strip() in STRONG_KEYS for s in r['Strategy'].split('+'))]
            if strong_only:
                best = sorted(strong_only, key=lambda x: x['Return (%)'], reverse=True)[0]
                top1_strat = best['Strategy']
                top1_return = f"{best['Return (%)']:.0f}%"
    except: pass

    st.markdown(f"### 🔥 一键加载黄金盟军 ({top1_return})")
    col_pre1, col_pre2 = st.columns([1, 4])
    with col_pre1:
        if st.button(f"👑 加载回测冠军组合 ({top1_return})", help=f"当前最强组合: {top1_strat}", use_container_width=True):
            # Shared mapping (Redefined local for simplicity but should be unified)
            def apply_inline(s):
                mapping = {
                    'Z_Score': 'ss_zscore', 'CYC_MAX': 'ss_cyc', 'RS': 'ss_rs', 
                    'RangeBreak': 'ss_range', 'TKOS': 'ss_tkos', '20VMA': 'ss_20vma', 
                    'OBO': 'ss_obo', 'LCS': 'ss_lcs', 'DTR_Plus': 'ss_dtr', 
                    'Fighting': 'ss_fighting', 'UA': 'ss_ua', 'HMC': 'ss_hmc',
                    'HPS': 'ss_hps', 'RKing': 'ss_rking',
                    'HLP3': 'ws_hlp3', 'Wyckoff': 'ws_wyckoff', 'Limit': 'ws_limit', 
                    'Vol_Min_120': 'ws_volmin', 'RSI_Rev': 'ws_rsi', 'ES': 'ws_es', 
                    'Boll_Rev': 'ws_boll', 'Spring': 'ws_spring', 'Pinbar': 'ws_pinbar', 
                    'Money_Flow': 'ws_flow', '2B': 'ws_2b', 'UA_Weak': 'ws_ua', 
                    'Double_Vol': 'ws_dv'
                }
                # Reset all keys
                for k in mapping.values(): st.session_state[k] = False
                for p in [x.strip() for x in s.split('+')]:
                    if mapping.get(p): st.session_state[mapping[p]] = True
                st.toast(f"✅ 加载成功: {s}")
                st.rerun()

            apply_inline(top1_strat)
        
    with st.expander("⚙️ 强势股策略配置 (自定义)", expanded=True):
        st.markdown("**第一阶段: 海选与锁定** — 寻找强势股候选池")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            strat_zscore = st.checkbox("Z-score (标准分)", key='ss_zscore', help="股价偏离20日均线的标准化程度。")
            strat_cyc = st.checkbox("CYC MAX", key='ss_cyc', help="股价站上无穷成本均线")
        with col2:
            strat_rs = st.checkbox("RS (相对强弱)", key='ss_rs', help="个股表现相对大盘的强弱对比。")
            strat_range = st.checkbox("Range Break", key='ss_range', help="突破年度箱体上沿")
        with col3:
            strat_tkos_strong = st.checkbox("TKOS (股王)", key='ss_tkos', help="月涨幅>50%")
            strat_20vma = st.checkbox("20VMA (量能启动)", key='ss_20vma', help="长期缩量后放量")
        with col4:
            strat_obo = st.checkbox("OBO (开盘突破)", key='ss_obo', help="突破昨日振幅区间")
            strat_lcs = st.checkbox("LCS (极限策略)", key='ss_lcs', help="非常偏离五日最低价")

        st.markdown("**第二阶段: 确认扳机** — 等待精确买入信号")
        col5, col6, col7 = st.columns(3)
        with col5:
            strat_dtr = st.checkbox("DTR Plus (共振)", key='ss_dtr', help="三维共振确认")
        with col6:
            strat_fighting_strong = st.checkbox("Fighting (突破)", key='ss_fighting', help="量价齐升主升浪信号")
        with col7:
            strat_ua = st.checkbox("UA (天量)", key='ss_ua', help="突破250日历史天量最高价")

        st.markdown("**第三阶段: 执行与防守** — 持仓动量监控")
        col8, col9, col10 = st.columns(3)
        with col8:
            strat_hmc_strong = st.checkbox("HMC (动量)", key='ss_hmc', help="高动量通道")
        with col9:
            strat_hps = st.checkbox("HPS (趋势通道)", key='ss_hps', help="站稳长期牛熊线")
        with col10:
            strat_rking = st.checkbox("RKing (趋势)", key='ss_rking', help="红柱趋势波段")

        st.markdown("**🔮 伏击/早鸟** — T-1提前布局")
        col_amb1, col_amb2 = st.columns(2)
        with col_amb1:
            strat_ambush_calm_s = st.checkbox("🔮 蓄力伏击", key='ss_amb_calm', help="波动率压缩+缩量+贴MA20，预判明日突破")
        with col_amb2:
            strat_ambush_mom_s = st.checkbox("🔮 动量伏击", key='ss_amb_mom', help="收红+放量+MACD红柱伸长，预判明日连续爆发")
    
    # Session State for Strong Attack
    if 'strong_scan_results' not in st.session_state:
        st.session_state['strong_scan_results'] = None
    
    if st.button("🚀 开始强势股筛选 / Start Strong Scan", type="primary", use_container_width=True):
        if stock_list_df.empty:
            st.error("无法开始：请先下载数据。")
            st.stop()
        
        # Check if at least one strategy is selected
        selected_strats = []
        if strat_zscore: selected_strats.append('Z_Score')
        if strat_rs: selected_strats.append('RS')
        if strat_tkos_strong: selected_strats.append('TKOS')
        if strat_dtr: selected_strats.append('DTR_Plus')
        if strat_fighting_strong: selected_strats.append('Fighting')
        if strat_ua: selected_strats.append('UA')
        if strat_hmc_strong: selected_strats.append('HMC')
        if strat_cyc: selected_strats.append('CYC_MAX')
        if strat_range: selected_strats.append('RangeBreak')
        if strat_20vma: selected_strats.append('20VMA')
        if strat_obo: selected_strats.append('OBO')
        if strat_lcs: selected_strats.append('LCS')
        if strat_hps: selected_strats.append('HPS')
        if strat_rking: selected_strats.append('RKing')
        if strat_ambush_calm_s: selected_strats.append('Ambush_Calm')
        if strat_ambush_mom_s: selected_strats.append('Ambush_Momentum')
        
        if not selected_strats:
            st.warning("请至少选择一个策略!")
            st.stop()
        
        scan_start_str = strong_start.strftime("%Y-%m-%d")
        scan_end_str = strong_end.strftime("%Y-%m-%d")
        
        # Use signal cache for fast lookup
        cache_reader = SignalCacheReader()
        is_valid, msg = cache_reader.is_cache_valid()
        
        if not is_valid:
            st.error(f"信号缓存不可用: {msg}。请先点击侧边栏「🔄 重建信号缓存」按钮。")
            st.stop()
        
        st.info(f"📦 使用信号缓存快速筛选 ({scan_start_str} ~ {scan_end_str})...")
        st.write(f"已选策略: {', '.join(selected_strats)}")
        
        try:
            df_cache = cache_reader.filter_strong_stocks(selected_strats, scan_start_str, scan_end_str)
            
            if df_cache.empty:
                st.session_state['strong_scan_results'] = pd.DataFrame()
                st.warning("未找到符合条件的股票。建议放宽策略组合或扩大时间范围。")
            else:
                # Apply Alliance logic (OR): keep rows where ANY selected strategies fired
                signal_cols = [f'Signal_{s}' for s in selected_strats if f'Signal_{s}' in df_cache.columns]
                if len(signal_cols) > 0:
                    mask = df_cache[signal_cols].any(axis=1)
                    df_cache = df_cache[mask]
                
                if df_cache.empty:
                    st.session_state['strong_scan_results'] = pd.DataFrame()
                    st.warning("未找到任何符合所选策略组合的股票。建议放宽时间范围。")
                else:
                    from market_env import MarketEnvironmentLoader
                    env_loader = MarketEnvironmentLoader()
                    
                    df_cache['date'] = pd.to_datetime(df_cache['date'])
                    scan_results = []
                    
                    for _, row in df_cache.iterrows():
                        code = row['code']
                        name = row['name']
                        date_str = row['date'].strftime("%Y-%m-%d")
                        
                        # Strategies are already in triggered_strategies for this specific day
                        all_strats = [x.strip() for x in str(row['triggered_strategies']).split(',') if x.strip()]
                        score = len(all_strats)
                        
                        # Macro check
                        m_env = "⚪ 待检"
                        try:
                            if env_loader.is_board_healthy(code, date_str):
                                m_env = "🟢 允许"
                            else:
                                m_env = "🔴 风险"
                        except: pass
                        
                        # Sector check
                        s_name = env_loader.get_stock_sector(code)
                        s_env = "⚪ 待检"
                        try:
                            is_s_h = env_loader.is_sector_health_healthy(code, date_str) if hasattr(env_loader, "is_sector_health_healthy") else env_loader.is_sector_healthy(code, date_str)
                            s_env = "🟢 多头" if is_s_h else "🔴 空头"
                        except: pass

                        scan_results.append({
                            "Code": code,
                            "Name": name,
                            "所属板块": s_name,
                            "板块状态 (Sector)": s_env,
                            "Score": score,
                            "强度 (Intensity)": "🔥" * score,
                            "大盘状态 (Macro)": m_env,
                            "Signal Date": date_str,
                            "Strategies": ", ".join(sorted(all_strats))
                        })
                    
                    res_df = pd.DataFrame(scan_results)
                    # Sort by Date (newest first), then Score
                    res_df = res_df.sort_values(by=['Signal Date', 'Score'], ascending=[False, False]).reset_index(drop=True)
                    st.session_state['strong_scan_results'] = res_df
                    st.success(f"⚡ 缓存筛选完成！发现 {len(scan_results)} 个信号点 (按日期由新到旧排序)。")
        except Exception as e:
            st.error(f"缓存筛选出错: {e}")
            st.session_state['strong_scan_results'] = pd.DataFrame()
    
    # Display Results
    if st.session_state['strong_scan_results'] is not None and not st.session_state['strong_scan_results'].empty:
        res_df = st.session_state['strong_scan_results']
        res_df['Code'] = res_df['Code'].astype(str)
        
        st.markdown("### 📊 强势股进攻筛选结果 (点击表格行查看详情)")
        st.markdown("⚠️ **风险提示**: 强势股波动剧烈，务必严格设置 2% 风险止损")
        
        event = st.dataframe(
            res_df[["Code", "Name", "所属板块", "板块状态 (Sector)", "大盘状态 (Macro)", "强度 (Intensity)", "Signal Date", "Strategies"]],
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        st.divider()
        
        # Determine Selected Stock
        selected_row_index = None
        if event.selection.rows:
            selected_row_index = event.selection.rows[0]
        
        if selected_row_index is not None:
            row_data = res_df.iloc[selected_row_index]
            code_s = str(row_data['Code'])
            name_s = str(row_data['Name'])
            highlight_date_s = row_data.get('Signal Date', None)
            st.info(f"当前选中: {code_s} - {name_s} (触发日期: {highlight_date_s})")
        else:
            st.info("👆 请在上方表格中点击选择一只股票查看详情。")
            
            # Fallback selectbox
            if 'Signal Date' in res_df.columns:
                screen_options = [f"{r['Code']} - {r['Name']} (Signal: {r['Signal Date']})" 
                                for r in res_df.to_dict('records')]
            else:
                screen_options = [f"{r['Code']} - {r['Name']}" for r in res_df.to_dict('records')]
            
            selected_screen = st.selectbox("或者：从下拉列表选择", options=screen_options, 
                                          index=None, placeholder="选择股票...")
            
            if selected_screen:
                code_s = selected_screen.split(" - ")[0]
                name_s = selected_screen.split(" - ")[1].split(" (")[0]
            else:
                code_s = None
        
        if code_s:
            # Display Chart
            load_start_s = (strong_start - datetime.timedelta(days=400)).strftime("%Y-%m-%d")
            load_end_s = strong_end.strftime("%Y-%m-%d")
            
            df_s = loader.get_k_data(code_s, load_start_s, load_end_s)
            
            if not df_s.empty:
                df_s = patch_df_with_realtime(df_s, code_s)
                df_s = Indicators.add_all_indicators(df_s)
                df_disp_s = df_s[(df_s['date'].dt.date >= strong_start) & 
                                (df_s['date'].dt.date <= strong_end)]
                
                # Calculate signals for visualization
                index_data_chart = None
                if strat_rs:
                    index_code = "000001"
                    index_data_chart = loader.get_k_data(index_code, load_start_s, load_end_s)
                
                selected_strats_chart = []
                if strat_zscore: selected_strats_chart.append('Z_Score')
                if strat_rs: selected_strats_chart.append('RS')
                if strat_tkos_strong: selected_strats_chart.append('TKOS')
                if strat_dtr: selected_strats_chart.append('DTR_Plus')
                if strat_fighting_strong: selected_strats_chart.append('Fighting')
                if strat_ua: selected_strats_chart.append('UA')
                if strat_hmc_strong: selected_strats_chart.append('HMC')
                if strat_cyc: selected_strats_chart.append('CYC_MAX')
                if strat_range: selected_strats_chart.append('RangeBreak')
                if strat_20vma: selected_strats_chart.append('20VMA')
                if strat_obo: selected_strats_chart.append('OBO')
                if strat_lcs: selected_strats_chart.append('LCS')
                if strat_hps: selected_strats_chart.append('HPS')
                if strat_rking: selected_strats_chart.append('RKing')
                
                sigs_s = StrongStrategies.check_all_strong_strategies(
                    df_s, 
                    index_df=index_data_chart,
                    selected_strategies=selected_strats_chart
                )
                
                # Find signal dates
                df_s_with_sigs = df_s.copy()
                for col in sigs_s.columns:
                    df_s_with_sigs[col] = sigs_s[col]
                
                signal_cols = [f'Signal_{s}' for s in selected_strats_chart]
                combined_signal = df_s_with_sigs[signal_cols].all(axis=1)
                signal_dates = df_s_with_sigs[combined_signal & 
                    (df_s_with_sigs['date'].dt.date >= strong_start) & 
                    (df_s_with_sigs['date'].dt.date <= strong_end)]['date']
                
                # Controls
                col_c1, col_c2 = st.columns([1, 4])
                with col_c1:
                    st.subheader("图表配置")
                    show_ma = st.checkbox("MA20", key='strong_ma')
                    show_ema = st.checkbox("EMA200", key='strong_ema')
                    show_boll = st.checkbox("Boll", key='strong_boll')
                    show_signals = st.checkbox("标注信号", key='strong_sig')
                    sub_chart_type = st.radio("副图:", ["MACD", "KDJ", "RSI", "WR", "CCI", "Volume", "RKing (趋势)", "Volatility", "HMC"], key='strong_sub')
                
                with col_c2:
                    triggered_strats = str(row_data['Strategies']).split(', ')
                    plot_stock_chart(df_disp_s, code_s, name_s, show_ma, show_ema, show_boll, 
                                   False, False, False, False, show_signals, sub_chart_type, 
                                   plotly_template, sigs_s, signal_dates, 
                                   triggered_strategies=triggered_strats,
                                   highlight_date=highlight_date_s)
                
                # Indicator Table
                with st.expander("📊 指标数值详情"):
                    cols_to_show = ['date', 'close', 'volume', 'MA20', 'MACD_Hist']
                    cols_final = [c for c in cols_to_show if c in df_disp_s.columns]
                    st.dataframe(df_disp_s[cols_final].tail(10).sort_values(by='date', ascending=False)
                               .style.format({"close": "{:.2f}", "MA20": "{:.2f}"}), 
                               use_container_width=True)
                
                # --- AI Diagnosis (Strong Attack) ---
                st.subheader("🤖 AI 智能诊断 (Gemini 3 Pro)")
                if st.button("开始诊断 (Start Diagnosis)", key='diag_strong'):
                    try:
                        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
                    except (FileNotFoundError, KeyError):
                        st.error("未找到 API Key。请在 .streamlit/secrets.toml 中配置 GEMINI_API_KEY。")
                        st.stop()
                    
                    from stock_diagnosis import StockDiagnoser
                    diagnoser = StockDiagnoser(GEMINI_API_KEY)
                    with st.spinner("正在请求 AI 模型进行深度分析..."):
                        report = diagnoser.generate_report(df_s, code_s, name_s, sigs_s)
                    st.markdown(report)
    
    elif st.session_state['strong_scan_results'] is None:
        st.info("请点击上方按钮开始强势股筛选。")

elif app_mode == "🔄 弱势股抄底":
    # --- Weak Stock Reversal Mode ---
    st.header("🔄 弱势股抄底 / Weak Stock Reversal")
    st.markdown("""
    **核心心法**: 行情始于"无"（极致缩量/绝望），终于"有"（放量/贪婪）。
    
    抄底不是买在最低点，而是买在**"绝望后的确认转折点"**。
    
    - **第一阶段(扫描与初筛)**: 寻找"绝望"与"无" - HLP3, Limit, RSI回归
    - **第二阶段(形态确认)**: 寻找"诱空"与"试探" - Spring, Pinbar, 资金背离
    - **第三阶段(买入扳机)**: 确认"有"与"启动" - UA天量, 倍量不破
    """)
    
    # Import weak strategies module
    from weak_strategies import WeakStrategies
    
    # Date Range
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        weak_start = st.date_input("筛选/显示开始日期", default_start, key='weak_start')
    with col_d2:
        weak_end = st.date_input("筛选/显示结束日期", default_end, key='weak_end')
    
    # --- Strategy Configuration (Inline) ---
    # Pre-fetch Top 1 (FILTERED: Weak strategies only)
    WEAK_KEYS = {'HLP3', 'Wyckoff', 'Limit', 'Vol_Min_120', 'RSI_Rev', 'ES', 'Boll_Rev', 'Spring', 'Pinbar', 'Money_Flow', '2B', 'UA_Weak', 'Double_Vol', 'Limit_Open'}
    top1_strat = "RSI_Rev + Spring + Money_Flow"
    top1_return = "0.0"
    try:
        json_path = os.path.join(BASE_DIR, "optimization_results.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                res = json.load(f)
            # Filter: only combos where ALL strategies are weak
            weak_only = [r for r in res if all(s.strip() in WEAK_KEYS for s in r['Strategy'].split('+'))]
            if weak_only:
                best = sorted(weak_only, key=lambda x: x['Return (%)'], reverse=True)[0]
                top1_strat = best['Strategy']
                top1_return = f"{best['Return (%)']:.0f}%"
    except: pass

    st.markdown(f"### 🔥 一键加载黄金盟军 ({top1_return})")
    col_pre1, col_pre2 = st.columns([1, 4])
    with col_pre1:
        if st.button(f"👑 加载回测冠军组合 ({top1_return})", help=f"当前最强组合: {top1_strat}", use_container_width=True, key='preset_weak'):
            # Reuse mapping logic
            mapping = {
                'Z_Score': 'ss_zscore', 'CYC_MAX': 'ss_cyc', 'RS': 'ss_rs', 
                'RangeBreak': 'ss_range', 'TKOS': 'ss_tkos', '20VMA': 'ss_20vma', 
                'OBO': 'ss_obo', 'LCS': 'ss_lcs', 'DTR_Plus': 'ss_dtr', 
                'Fighting': 'ss_fighting', 'UA': 'ss_ua', 'HMC': 'ss_hmc',
                'HPS': 'ss_hps', 'RKing': 'ss_rking',
                'HLP3': 'ws_hlp3', 'Wyckoff': 'ws_wyckoff', 'Limit': 'ws_limit', 
                'Vol_Min_120': 'ws_volmin', 'RSI_Rev': 'ws_rsi', 'ES': 'ws_es', 
                'Boll_Rev': 'ws_boll', 'Spring': 'ws_spring', 'Pinbar': 'ws_pinbar', 
                'Money_Flow': 'ws_flow', '2B': 'ws_2b', 'UA_Weak': 'ws_ua', 
                'Double_Vol': 'ws_dv'
            }
            for k in mapping.values(): st.session_state[k] = False
            for p in [x.strip() for x in top1_strat.split('+')]:
                if mapping.get(p): st.session_state[mapping[p]] = True
            st.toast(f"✅ 加载成功: {top1_strat}")
            st.rerun()

    with st.expander("⚙️ 抄底策略配置 (自定义)", expanded=True):
        st.markdown("**第一阶段: 扫描与初筛** — 寻找极致缩量、超卖的「绝望」状态")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            strat_hlp3 = st.checkbox("HLP3 (大慈悲点)", key='ws_hlp3', help="获利盘<1%后飙升>35%")
            strat_wyckoff = st.checkbox("Wyckoff (量价背离)", key='ws_wyckoff', help="底背离吸筹")
        with col2:
            strat_limit = st.checkbox("Limit (极致缩量)", key='ws_limit', help="成交量<均量50%")
            strat_vol_min = st.checkbox("量比历史新低", key='ws_volmin', help="120日量比新低")
        with col3:
            strat_rsi_rev = st.checkbox("RSI2 (均值回归)", key='ws_rsi', help="RSI<25超卖反弹")
            strat_es = st.checkbox("ES (波动率压缩)", key='ws_es', help="布林带极窄变盘")
        with col4:
            strat_boll_rev = st.checkbox("Boll Rev (布林反转)", key='ws_boll', help="触碰下轨反弹")
        
        st.markdown("**第二阶段: 形态确认** — 寻找主力「诱空」洗盘的底部形态")
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            strat_spring = st.checkbox("Spring (弹簧)", key='ws_spring', help="假跌破洗盘")
        with col6:
            strat_pinbar = st.checkbox("Pinbar (长钉)", key='ws_pinbar', help="长下影线探底神针")
        with col7:
            strat_flow = st.checkbox("Money Flow", key='ws_flow', help="资金背离净流入")
        with col8:
            strat_2b = st.checkbox("2B 法则", key='ws_2b', help="破底翻")
        
        st.markdown("**第三阶段: 买入扳机** — 确认底部反转「启动」")
        col9, col10 = st.columns(2)
        with col9:
            strat_ua_weak = st.checkbox("UA (天量突破)", key='ws_ua', help="突破底部天量价")
        with col10:
            strat_dv = st.checkbox("倍量不破", key='ws_dv', help="倍量阳线回调不破")
        
        st.warning("⚠️ **风控提醒**: 抄底是逆势交易，必须严格止损（-10%硬防守）。仓位不超过总资金20%。")
    
    # Session State for Weak Reversal
    if 'weak_scan_results' not in st.session_state:
        st.session_state['weak_scan_results'] = None
    
    if st.button("📉 开始抄底筛选 / Start Reversal Scan", type="primary", use_container_width=True):
        if stock_list_df.empty:
            st.error("无法开始：请先下载数据。")
            st.stop()
        
        # Check if at least one strategy is selected
        selected_strats = []
        if strat_hlp3: selected_strats.append('HLP3')
        if strat_wyckoff: selected_strats.append('Wyckoff')
        if strat_limit: selected_strats.append('Limit')
        if strat_vol_min: selected_strats.append('Vol_Min_120')
        if strat_rsi_rev: selected_strats.append('RSI_Rev')
        if strat_es: selected_strats.append('ES')
        if strat_boll_rev: selected_strats.append('Boll_Rev')
        if strat_spring: selected_strats.append('Spring')
        if strat_pinbar: selected_strats.append('Pinbar')
        if strat_flow: selected_strats.append('Money_Flow')
        if strat_2b: selected_strats.append('2B')
        if strat_ua_weak: selected_strats.append('UA_Weak')
        if strat_dv: selected_strats.append('Double_Vol')
        
        if not selected_strats:
            st.warning("请至少选择一个策略!")
            st.stop()
        
        scan_start_str = weak_start.strftime("%Y-%m-%d")
        scan_end_str = weak_end.strftime("%Y-%m-%d")
        
        # Use signal cache for fast lookup
        cache_reader = SignalCacheReader()
        is_valid, msg = cache_reader.is_cache_valid()
        
        if not is_valid:
            st.error(f"信号缓存不可用: {msg}。请先点击侧边栏「🔄 重建信号缓存」按钮。")
            st.stop()
        
        st.info(f"📦 使用信号缓存快速筛选 ({scan_start_str} ~ {scan_end_str})...")
        st.write(f"已选策略: {', '.join(selected_strats)}")
        
        try:
            df_cache = cache_reader.filter_weak_stocks(selected_strats, scan_start_str, scan_end_str)
            
            if df_cache.empty:
                st.session_state['weak_scan_results'] = pd.DataFrame()
                st.warning("未找到符合条件的股票。抄底信号较为少见，建议放宽策略组合或扩大时间范围。")
            else:
                # Apply Alliance logic (OR): keep rows where ANY selected strategies fired
                # This matches the 'Alliance' logic in optimize_strategies.py
                signal_cols = [f'Signal_{s}' for s in selected_strats if f'Signal_{s}' in df_cache.columns]
                if len(signal_cols) > 0:
                    mask = df_cache[signal_cols].any(axis=1)
                    df_cache = df_cache[mask]
                
                if df_cache.empty:
                    st.session_state['weak_scan_results'] = pd.DataFrame()
                    st.warning("未找到任何符合所选策略组合的股票。建议放宽时间范围。")
                else:
                    from market_env import MarketEnvironmentLoader
                    env_loader = MarketEnvironmentLoader()
                    
                    df_cache['date'] = pd.to_datetime(df_cache['date'])
                    scan_results = []
                    
                    for _, row in df_cache.iterrows():
                        code = row['code']
                        name = row['name']
                        date_str = row['date'].strftime("%Y-%m-%d")
                        
                        # Strategies are already in triggered_strategies for this specific day
                        all_strats = [x.strip() for x in str(row['triggered_strategies']).split(',') if x.strip()]
                        score = len(all_strats)
                        
                        # Macro check
                        m_env = "⚪ 待检"
                        try:
                            if env_loader.is_board_healthy(code, date_str):
                                m_env = "🟢 允许"
                            else:
                                m_env = "🔴 风险"
                        except: pass
                        
                        # Sector check
                        s_name = env_loader.get_stock_sector(code)
                        s_env = "⚪ 待检"
                        try:
                            is_s_h = env_loader.is_sector_health_healthy(code, date_str) if hasattr(env_loader, "is_sector_health_healthy") else env_loader.is_sector_healthy(code, date_str)
                            s_env = "🟢 多头" if is_s_h else "🔴 空头"
                        except: pass

                        scan_results.append({
                            "Code": code,
                            "Name": name,
                            "所属板块": s_name,
                            "板块状态 (Sector)": s_env,
                            "Score": score,
                            "强度 (Intensity)": "🔥" * score,
                            "大盘状态 (Macro)": m_env,
                            "Signal Date": date_str,
                            "Strategies": ", ".join(sorted(all_strats))
                        })
                    
                    res_df = pd.DataFrame(scan_results)
                    # Sort by Date (newest first), then Score
                    res_df = res_df.sort_values(by=['Signal Date', 'Score'], ascending=[False, False]).reset_index(drop=True)
                    st.session_state['weak_scan_results'] = res_df
                    st.success(f"⚡ 缓存筛选完成！发现 {len(scan_results)} 个信号点。")
        except Exception as e:
            st.error(f"缓存筛选出错: {e}")
            st.session_state['weak_scan_results'] = pd.DataFrame()
    
    # Display Results
    if st.session_state['weak_scan_results'] is not None and not st.session_state['weak_scan_results'].empty:
        res_df = st.session_state['weak_scan_results']
        res_df['Code'] = res_df['Code'].astype(str)
        
        st.markdown("### 📊 抄底机会筛选结果 (点击表格行查看详情)")
        st.markdown("⚠️ **风险提示**: 抄底是逆势交易，务必设置止损，单笔亏损不超过本金10%")
        
        event = st.dataframe(
            res_df[["Code", "Name", "所属板块", "板块状态 (Sector)", "大盘状态 (Macro)", "强度 (Intensity)", "Signal Date", "Strategies"]],
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        st.divider()
        
        # Determine Selected Stock
        selected_row_index = None
        if event.selection.rows:
            selected_row_index = event.selection.rows[0]
        
        if selected_row_index is not None:
            row_data = res_df.iloc[selected_row_index]
            code_s = str(row_data['Code'])
            name_s = str(row_data['Name'])
            highlight_date_s = row_data.get('Signal Date', None)
            st.info(f"当前选中: {code_s} - {name_s} (触发日期: {highlight_date_s})")
        else:
            st.info("👆 请在上方表格中点击选择一只股票查看详情。")
            
            # Fallback selectbox
            if 'Signal Date' in res_df.columns:
                screen_options = [f"{r['Code']} - {r['Name']} (Signal: {r['Signal Date']})" 
                                for r in res_df.to_dict('records')]
            else:
                screen_options = [f"{r['Code']} - {r['Name']}" for r in res_df.to_dict('records')]
            
            selected_screen = st.selectbox("或者：从下拉列表选择", options=screen_options, 
                                          index=None, placeholder="选择股票...")
            
            if selected_screen:
                code_s = selected_screen.split(" - ")[0]
                name_s = selected_screen.split(" - ")[1].split(" (")[0]
            else:
                code_s = None
        
        if code_s:
            # Display Chart
            load_start_s = (weak_start - datetime.timedelta(days=400)).strftime("%Y-%m-%d")
            load_end_s = weak_end.strftime("%Y-%m-%d")
            
            df_s = loader.get_k_data(code_s, load_start_s, load_end_s)
            
            if not df_s.empty:
                df_s = patch_df_with_realtime(df_s, code_s)
                df_s = Indicators.add_all_indicators(df_s)
                df_disp_s = df_s[(df_s['date'].dt.date >= weak_start) & 
                                (df_s['date'].dt.date <= weak_end)]
                
                # Calculate signals for visualization
                selected_strats_chart = []
                if strat_hlp3: selected_strats_chart.append('HLP3')
                if strat_limit: selected_strats_chart.append('Limit')
                if strat_rsi_rev: selected_strats_chart.append('RSI_Rev')
                if strat_spring: selected_strats_chart.append('Spring')
                if strat_pinbar: selected_strats_chart.append('Pinbar')
                if strat_flow: selected_strats_chart.append('Money_Flow')
                if strat_ua_weak: selected_strats_chart.append('UA')
                if strat_dv: selected_strats_chart.append('Double_Vol')
                
                sigs_s = WeakStrategies.check_all_weak_strategies(
                    df_s,
                    selected_strategies=selected_strats_chart,
                    winner_col='winner_pct'
                )
                
                # Find signal dates
                df_s_with_sigs = df_s.copy()
                for col in sigs_s.columns:
                    if col.startswith('Signal_'):
                        df_s_with_sigs[col] = sigs_s[col]
                
                signal_cols = [f'Signal_{s}' for s in selected_strats_chart]
                signal_cols = [col for col in signal_cols if col in df_s_with_sigs.columns]
                
                if signal_cols:
                    combined_signal = df_s_with_sigs[signal_cols].all(axis=1)
                    signal_dates = df_s_with_sigs[combined_signal & 
                        (df_s_with_sigs['date'].dt.date >= weak_start) & 
                        (df_s_with_sigs['date'].dt.date <= weak_end)]['date']
                else:
                    signal_dates = pd.Series(dtype='datetime64[ns]')
                
                # Controls
                col_c1, col_c2 = st.columns([1, 4])
                with col_c1:
                    st.subheader("图表配置")
                    show_ma = st.checkbox("MA20", key='weak_ma')
                    show_ema = st.checkbox("EMA200", key='weak_ema')
                    show_boll = st.checkbox("Boll", key='weak_boll')
                    show_signals = st.checkbox("标注信号", key='weak_sig')
                    sub_chart_type = st.radio("副图:", ["MACD", "KDJ", "RSI", "WR", "CCI", "Volume", "RKing (趋势)", "Volatility", "HMC"], key='weak_sub')
                
                with col_c2:
                    triggered_strats = str(row_data['Strategies']).split(', ')
                    plot_stock_chart(df_disp_s, code_s, name_s, show_ma, show_ema, show_boll, 
                                   False, False, False, False, show_signals, sub_chart_type, 
                                   plotly_template, sigs_s, signal_dates, 
                                   triggered_strategies=triggered_strats,
                                   highlight_date=highlight_date_s)
                
                # Indicator Table
                with st.expander("📊 指标数值详情"):
                    cols_to_show = ['date', 'close', 'volume', 'MA20', 'MACD_Hist']
                    cols_final = [c for c in cols_to_show if c in df_disp_s.columns]
                    st.dataframe(df_disp_s[cols_final].tail(10).sort_values(by='date', ascending=False)
                               .style.format({"close": "{:.2f}", "MA20": "{:.2f}"}), 
                               use_container_width=True)
                
                # --- AI Diagnosis (Weak Reversal) ---
                st.subheader("🤖 AI 智能诊断 (Gemini 3 Pro)")
                if st.button("开始诊断 (Start Diagnosis)", key='diag_weak'):
                    try:
                        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
                    except (FileNotFoundError, KeyError):
                        st.error("未找到 API Key。请在 .streamlit/secrets.toml 中配置 GEMINI_API_KEY。")
                        st.stop()
                    
                    from stock_diagnosis import StockDiagnoser
                    diagnoser = StockDiagnoser(GEMINI_API_KEY)
                    with st.spinner("正在请求 AI 模型进行深度分析..."):
                        report = diagnoser.generate_report(df_s, code_s, name_s, sigs_s)
                    st.markdown(report)
    
    elif st.session_state['weak_scan_results'] is None:
        st.info("请点击上方按钮开始抄底筛选。")

# --- Backtest Mode ---
elif app_mode == "🛠️ 策略回测":
    st.header("🛠️ 策略历史回测 (Strategy Backtest)")
    st.markdown("验证策略在过去一段时间的盈亏表现，优化交易规则。")
    
    # 1. Configuration
    col_conf1, col_conf2 = st.columns([1, 1])
    
    with col_conf1:
        st.subheader("1. 基础设置")
        bt_start = st.date_input("开始日期", default_start - datetime.timedelta(days=365))
        bt_end = st.date_input("结束日期", default_end)
        initial_capital = st.number_input("初始资金", value=100000.0, step=10000.0)
        
    with col_conf2:
        st.subheader("2. 交易规则")
        position_sizing = st.slider("单股仓位 (%)", 10, 100, 20, 5) / 100.0
        max_positions = int(1.0 / position_sizing)
        st.caption(f"最大持仓数: {max_positions} 只")
        
        stop_loss = st.slider("止损比例 (%)", 0, 20, 5, 1) / 100.0
        take_profit = st.slider("止盈比例 (%)", 0, 50, 10, 5) / 100.0
        max_hold = st.number_input("最长持仓 (天)", value=20, step=1)
        if stop_loss == 0: stop_loss = None
        if take_profit == 0: take_profit = None
        if max_hold == 0: max_hold = None

    # 2. Strategy Selection (Select cached signals to use)
    st.subheader("3. 信号来源")
    cache_reader = SignalCacheReader()
    is_valid, msg = cache_reader.is_cache_valid()
    
    if not is_valid:
        st.error(f"无法回测: {msg}")
    else:
        st.success(msg)
        
        st.markdown("选择要测试的策略（并集关系）：")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**强势策略**")
            s_zscore = st.checkbox("Z_Score (突破)", value=True, key='bt_z')
            s_rs = st.checkbox("RS (相对强度)", value=True, key='bt_rs')
            s_tkos = st.checkbox("TKOS (潜伏)", key='bt_tkos')
            s_dtr = st.checkbox("DTR++ (趋势)", value=True, key='bt_dtr')
            s_fight = st.checkbox("Fighting", key='bt_fight')
            s_hmc = st.checkbox("HMC (动量)", key='bt_hmc')
            s_ua = st.checkbox("UA (天量)", key='bt_ua')
            s_obo = st.checkbox("OBO (一阴包一阳)", key='bt_obo')
            s_lcs = st.checkbox("LCS (缩量回调)", key='bt_lcs')
        with col_s2:
            st.markdown("**弱势策略**")
            s_hlp3 = st.checkbox("HLP3 (高低位)", key='bt_hlp3')
            s_limit = st.checkbox("Limit (跌停)", key='bt_limit')
            s_rsi = st.checkbox("RSI2 (回归)", key='bt_rsi')
            s_spring = st.checkbox("Spring (弹簧)", key='bt_spring')
            s_pin = st.checkbox("Pinbar (长钉)", key='bt_pin')
            s_flow = st.checkbox("Money Flow (资金)", key='bt_flow')
            s_ua_weak = st.checkbox("UA (天量突破)", key='bt_ua_w')
            s_dv = st.checkbox("Double Vol (倍量)", key='bt_dv')
            s_limit_open = st.checkbox("Limit Open (撬板)", key='bt_limit_o')
        
        # Collect selected
        selected_bt_strategies = []
        if s_zscore: selected_bt_strategies.append('Z_Score')
        if s_rs: selected_bt_strategies.append('RS')
        if s_tkos: selected_bt_strategies.append('TKOS')
        if s_dtr: selected_bt_strategies.append('DTR_Plus')
        if s_fight: selected_bt_strategies.append('Fighting')
        if s_hmc: selected_bt_strategies.append('HMC')
        if s_ua: selected_bt_strategies.append('UA')
        if s_obo: selected_bt_strategies.append('OBO')
        if s_lcs: selected_bt_strategies.append('LCS')
        
        if s_hlp3: selected_bt_strategies.append('HLP3')
        if s_limit: selected_bt_strategies.append('Limit')
        if s_rsi: selected_bt_strategies.append('RSI_Rev')
        if s_spring: selected_bt_strategies.append('Spring')
        if s_pin: selected_bt_strategies.append('Pinbar')
        if s_flow: selected_bt_strategies.append('Money_Flow')
        if s_ua_weak: selected_bt_strategies.append('UA_Weak')
        if s_dv: selected_bt_strategies.append('Double_Vol')
        if s_limit_open: selected_bt_strategies.append('Limit_Open')
        
        if st.button("🚀 开始回测 (Run Backtest)", type="primary"):
            if not selected_bt_strategies:
                st.warning("请至少选择一个策略")
            else:
                with st.spinner("正在加载信号并模拟交易..."):
                    # 1. Load Signals
                    # Optimize: Load Strong and Weak and combine
                    
                    df_strong = pd.DataFrame()
                    df_weak = pd.DataFrame()
                    
                    # Identify which are strong/weak types
                    strong_types = ['Z_Score', 'RS', 'TKOS', 'DTR_Plus', 'Fighting', 'UA', 'HMC', 'OBO', 'LCS']
                    weak_types = ['HLP3', 'Limit', 'RSI_Rev', 'Spring', 'Pinbar', 'Money_Flow', 'Double_Vol', 'UA_Weak', 'DBL_VOL', 'Limit_Open']
                    
                    sel_strong = [s for s in selected_bt_strategies if s in strong_types]
                    sel_weak = [s for s in selected_bt_strategies if s in weak_types]
                    
                    if sel_strong:
                        try:
                            df_strong = cache_reader.filter_strong_stocks(sel_strong, str(bt_start), str(bt_end))
                        except Exception:
                            pass
                    if sel_weak:
                        try:
                            df_weak = cache_reader.filter_weak_stocks(sel_weak, str(bt_start), str(bt_end))
                        except Exception:
                            pass
                            
                    # Combine and Apply AND Logic
                    signal_df = pd.DataFrame()
                    
                    if df_strong.empty and df_weak.empty:
                        st.warning("所选范围内无信号触发")
                        st.stop()
                    elif df_strong.empty:
                        signal_df = df_weak
                    elif df_weak.empty:
                        signal_df = df_strong
                    else:
                        # Merge Strong and Weak (Outer join to get all dates/codes first, then filter)
                        # We need to align on code, name, date
                        signal_df = pd.merge(df_strong, df_weak, on=['code', 'name', 'date'], how='outer', suffixes=('', '_weak'))
                        # Note: Merging might duplicate columns if names collide, but suffixes handles it.
                        # Ideally, signal columns don't collide.
                    
                    # Force ALL requested signals to exist in the dataframe. If they don't, it implies NO stock matched them.
                    signal_cols = [f'Signal_{s}' for s in selected_bt_strategies]
                    
                    for s_col in signal_cols:
                        if s_col not in signal_df.columns:
                            # If a requested strategy isn't even in the merged dataframe, 
                            # it means 0 stocks triggered it during this period.
                            # So we fill the column with False.
                            signal_df[s_col] = False
                            
                    # -------------------------------------------------------------
                    # ALLIANCE LOGIC (OR): As long as ANY requested strategy fired, 
                    # keep the row. (This perfectly aligns with optimize_strategies.py)
                    # -------------------------------------------------------------
                    # Now all signal_cols are guaranteed to exist. Fill NaNs with False.
                    signal_df[signal_cols] = signal_df[signal_cols].fillna(False)
                    
                    # Alliance (OR) Logic: Any selected strategy must be True
                    or_mask = signal_df[signal_cols].any(axis=1)
                    signal_df = signal_df[or_mask].copy()
                    
                    valid_cols = signal_cols # For the display string function below
                    
                    # Re-generate triggered_strategies string for display
                    def combine_strategies(row):
                        trig = []
                        for c in valid_cols:
                            if row[c]:
                                trig.append(c.replace('Signal_', ''))
                        return ', '.join(trig)
                    
                    if not signal_df.empty:
                        signal_df['triggered_strategies'] = signal_df.apply(combine_strategies, axis=1)

                    if signal_df.empty:
                        st.warning("没有找到同时满足所有选中策略的信号。")
                        st.stop()
                    
                    # Deduplicate on code/date just in case
                    signal_df = signal_df.drop_duplicates(subset=['code', 'date'])
                    
                    # -------------------------------------------------------------
                    # PRIORITY SORTING: Calculate Confluence Score (signal_count)
                    # Align with optimize_strategies.py to ensure the highest 
                    # probability targets are picked first when hitting max_positions
                    # -------------------------------------------------------------
                    full_cols = [c for c in (df_strong.columns.tolist() + df_weak.columns.tolist()) if c.startswith('Signal_')]
                    available_cols = list(set([c for c in full_cols if c in signal_df.columns]))
                    if available_cols:
                        signal_df['signal_count'] = signal_df[available_cols].fillna(False).astype(int).sum(axis=1)
                    else:
                        signal_df['signal_count'] = 1
                        
                    # Sort signals by date ASC, then by signal_count DESC (highest score first)
                    signal_df = signal_df.sort_values(by=['date', 'signal_count'], ascending=[True, False])
                    
                    st.write(f"共加载信号: {len(signal_df)} 个 (Alliance/OR Logic, Confluence Sorted)")
                    
                    # 2. Run Engine
                    engine = BacktestEngine(DataLoader(MARKET_DATA_DIR), initial_capital)
                    equity, trades = engine.run(signal_df, str(bt_start), str(bt_end), 
                                              stop_loss, take_profit, max_hold, position_sizing, max_positions)
                    
                    # 3. Visualize
                    st.divider()
                    
                    # Summary Metrics
                    final_equity = equity.iloc[-1]['total_assets']
                    ret_pct = (final_equity - initial_capital) / initial_capital
                    winning_trades = trades[trades['pnl'] > 0]
                    win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("最终权益", f"{final_equity:,.0f}", f"{ret_pct*100:.1f}%")
                    m2.metric("交易次数", len(trades))
                    m3.metric("胜率", f"{win_rate*100:.1f}%")
                    
                    # Max Drawdown
                    equity['high_water_mark'] = equity['total_assets'].cummax()
                    equity['drawdown'] = (equity['total_assets'] - equity['high_water_mark']) / equity['high_water_mark']
                    max_dd = equity['drawdown'].min()
                    m4.metric("最大回撤", f"{max_dd*100:.1f}%")
                    
                    # Charts
                    st.subheader("📈 资金曲线 (Equity Curve)")
                    
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                      vertical_spacing=0.03, row_heights=[0.7, 0.3])
                    
                    # Equity
                    fig.add_trace(go.Scatter(x=equity['date'], y=equity['total_assets'], 
                                           mode='lines', name='Total Assets',
                                           line=dict(color='#00C805', width=2),
                                           fill='tozeroy', fillcolor='rgba(0, 200, 5, 0.1)'), 
                                 row=1, col=1)
                    
                    # Benchmark (沪深300 - sh000300)
                    try:
                        from amarket_framework.tencent_loader import TencentLoader
                        t_loader = TencentLoader()
                        # Fetch enough days to cover the backtest period
                        df_bench = t_loader.fetch_k_line('sh000300', day_count=800)
                        
                        if not df_bench.empty:
                            df_bench = df_bench.reset_index()
                            # Align dates with equity curve
                            df_bench['date'] = pd.to_datetime(df_bench['date'])
                            df_bench = df_bench[df_bench['date'].isin(equity['date'])]
                            
                            if not df_bench.empty:
                                # Normalize benchmark to start at initial_capital
                                first_close = df_bench.iloc[0]['close']
                                df_bench['norm_close'] = (df_bench['close'] / first_close) * initial_capital
                                
                                fig.add_trace(go.Scatter(x=df_bench['date'], y=df_bench['norm_close'], 
                                                       mode='lines', name='沪深300 (Benchmark)',
                                                       line=dict(color='gray', width=1.5, dash='dash')), 
                                             row=1, col=1)
                    except Exception as e:
                        pass # Silently skip benchmark if fetch fails
                    
                    # Drawdown
                    fig.add_trace(go.Scatter(x=equity['date'], y=equity['drawdown'], 
                                           mode='lines', name='Drawdown',
                                           line=dict(color='#FF4B4B', width=1),
                                           fill='tozeroy'), 
                                 row=2, col=1)
                                 
                    fig.update_layout(template='plotly_white', height=500, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trade Log
                    st.subheader("📝 交易明细 (Trade Log)")
                    if not trades.empty:
                        # Format
                        disp_trades = trades.copy()
                        disp_trades['date'] = pd.to_datetime(disp_trades['date']).dt.date
                        # Colorize PnL
                        st.dataframe(disp_trades.style.format({
                            'price': '{:.2f}', 
                            'pnl': '{:+.2f}',
                            'pnl_pct': '{:+.2%}'
                        }), use_container_width=True)
                    else:
                        st.info("无交易记录")

# --- Daily Plan Mode ---
elif app_mode == "📅 每日交易计划":
    st.header("📅 每日交易计划 (Daily Trade Plan)")
    st.markdown("基于最新收盘数据，生成明日的操作计划。")
    
    # Check latest data date
    cache_reader = SignalCacheReader()
    is_valid, msg = cache_reader.is_cache_valid()
    
    if not is_valid:
        st.error(f"无法生成计划: {msg}")
    else:
        # Get metadata to find latest date
        meta = cache_reader.get_metadata()
        if meta:
            last_date_str = meta.get('data_date_range', [None, None])[1]
            try:
                default_date = datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
            except:
                default_date = datetime.date.today()
            
            st.info(f"数据截至日期: **{last_date_str}**")
            
            # Date Picker
            plan_date = st.date_input("📅 选择生成计划的日期", default_date)
            plan_date_str = plan_date.strftime("%Y-%m-%d")

            # Initialize session state for plan results if not exists
            if 'plan_results' not in st.session_state:
                st.session_state['plan_results'] = None
                
            # If date updates, maybe clear results? Optional. 
            # For now, let user manually click generate.

            if st.button("🔄 生成交易计划 (Generate Plan)", type="primary"):
                with st.spinner(f"正在扫描 {plan_date_str} 的全市场信号..."):
                    # Scan for signals on the SELECTED date
                    
                    # 1. Define Strategies to Watch (ALL available strategies)
                    strong_strats = ['Z_Score', 'RS', 'TKOS', 'DTR_Plus', 'Fighting', 'UA', 'HMC']
                    weak_strats = ['HLP3', 'Limit', 'RSI_Rev', 'Spring', 'Pinbar', 'Money_Flow', 'UA', 'Double_Vol']
                    
                    # 2. Query Cache for Selected Date
                    df_strong = cache_reader.filter_strong_stocks(strong_strats, plan_date_str, plan_date_str)
                    df_weak = pd.DataFrame() 
                    try:
                        df_weak = cache_reader.filter_weak_stocks(weak_strats, plan_date_str, plan_date_str)
                    except: pass
                    
                    # Store in session state
                    st.session_state['plan_results'] = {
                        'date': plan_date_str,
                        'strong': df_strong,
                        'weak': df_weak
                    }
                    
            # Check if results exist in session state
            if st.session_state['plan_results']:
                res = st.session_state['plan_results']
                # Check if date matches current selection? 
                # If user changes date but doesn't click generate, showing old results might be confusing.
                # But let's show what we have, maybe add a warning if date mismatch?
                # Or just show "Plan for {res['date']}"
                
                st.markdown(f"### 📅 交易计划: {res['date']}")
                
                df_strong = res['strong']
                df_weak = res['weak']
                
                # 过滤掉只有单个策略的标的 (用户需求: 必须产生共振 >= 2)
                if not df_strong.empty:
                    df_strong['score'] = df_strong['triggered_strategies'].apply(lambda x: len(str(x).split(',')))
                    df_strong = df_strong[df_strong['score'] >= 2]
                if not df_weak.empty:
                    df_weak['score'] = df_weak['triggered_strategies'].apply(lambda x: len(str(x).split(',')))
                    df_weak = df_weak[df_weak['score'] >= 2]
                    
                # 3. Process & Display
                
                # --- Chart Integration ---
                st.header("📈 个股详情 (Chart Analysis)")
                
                # Merge lists for selection (Pre-calculate for index finding)
                combined_candidates = []
                if not df_strong.empty:
                    for _, r in df_strong.iterrows():
                        combined_candidates.append(f"{r['code']} | {r['name']} (Strong)")
                if not df_weak.empty:
                    for _, r in df_weak.iterrows():
                        combined_candidates.append(f"{r['code']} | {r['name']} (Weak)")

                # Initialize selection state
                if 'selected_candidate' not in st.session_state:
                    st.session_state['selected_candidate'] = None

                # 3. Process & Display
                def process_display_df(df, type_name, key_suffix, label_suffix):
                    if df.empty:
                        st.write(f"今日无强共振{type_name}信号 (需同时触发至少2个策略)。")
                        return

                    # Score already calculated implicitly above, but re-calculating is safe
                    if 'score' not in df.columns:
                        df['score'] = df['triggered_strategies'].apply(lambda x: len(str(x).split(',')))
                        
                    df = df.sort_values(by='score', ascending=False)
                    
                    def get_intensity(s):
                        return "🔥" * int(s)
                    df['强度 (Intensity)'] = df['score'].apply(get_intensity)
                    
                    from market_env import MarketEnvironmentLoader
                    env_loader = MarketEnvironmentLoader()
                    
                    # Add Macro Environment Check
                    def get_macro_env(code):
                        # Strong strategies require Marco OK. Weak strategies are always OK.
                        if "Strong" in label_suffix:
                            if env_loader.is_board_healthy(code, res['date']):
                                return "🟢 允许"
                            else:
                                return "🔴 风险"
                        else:
                            return "⚪ 无视 (弱势抄底)"
                            
                    df['大盘状态 (Macro)'] = df['code'].apply(get_macro_env)
                    
                    # --- Phase 17: Industry Resonance Display ---
                    df['所属板块'] = df['code'].apply(env_loader.get_stock_sector)
                    def get_sector_status(code):
                        is_h = env_loader.is_sector_health_healthy(code, res['date']) if hasattr(env_loader, 'is_sector_health_healthy') else env_loader.is_sector_healthy(code, res['date'])
                        return "🟢 多头" if is_h else "🔴 空头"
                    df['板块状态 (Sector)'] = df['code'].apply(get_sector_status)
                    
                    # --- 🏎️ PERFORMANCE OPTIMIZATION START ---
                    import datetime
                    import pytz
                    beijing_tz = pytz.timezone('Asia/Shanghai')
                    today = datetime.datetime.now(beijing_tz).date()
                    today_str = today.strftime("%Y-%m-%d")
                    
                    plan_date_dt = datetime.datetime.strptime(res['date'], "%Y-%m-%d")
                    plan_date_str = res['date']
                    lookback_dt = plan_date_dt - datetime.timedelta(days=150)
                    
                    # 1. Shared DataLoader instance to reuse in-memory indicators cache
                    # Doing this prevents re-computing 60+ indicators per stock multiple times.
                    p_loader = DataLoader(MARKET_DATA_DIR)
                    
                    # 2. Batch fetch real-time quotes to avoid N synchronous HTTP requests (only during trading hours)
                    from amarket_framework.tencent_loader import TencentLoader
                    from amarket_framework.timezone_utils import is_trading_time
                    
                    rt_dict = {}
                    if is_trading_time():
                        t_loader = TencentLoader()
                        unique_codes = df['code'].unique().tolist()
                        full_codes = [t_loader.get_full_code(c) for c in unique_codes]
                        rt_df = t_loader.fetch_realtime_quotes(full_codes)
                        
                        if not rt_df.empty:
                            rt_df = rt_df.set_index('code')
                            for c in unique_codes:
                                if c in rt_df.index:
                                    rt_dict[c] = rt_df.loc[c].to_dict()

                    # Helper to quick patch using local dict instead of HTTP
                    def fast_patch_realtime(k_df, code):
                        if k_df.empty or code not in rt_dict:
                            return k_df
                            
                        rt_row = rt_dict[code]
                        last_dt = k_df['date'].iloc[-1].date()
                        
                        patch_data = {
                            'date': pd.to_datetime(today),
                            'close': float(rt_row['close']),
                            'open': float(rt_row['open']),
                            'high': float(rt_row['high']),
                            'low': float(rt_row['low']),
                            'volume': float(rt_row['volume']),
                            'amount': float(rt_row['amount'])
                        }
                        
                        if last_dt == today:
                            for k, v in patch_data.items():
                                if k in k_df.columns:
                                    k_df.iloc[-1, k_df.columns.get_loc(k)] = v
                        else:
                            new_row = pd.DataFrame([patch_data])
                            k_df = pd.concat([k_df, new_row], ignore_index=True)
                            
                        return k_df
                        
                    # --- SOP PHASE 4: ENTRY & SIZING COLUMNS ---
                    def get_sop_info(code):
                        try:
                            # Use shared loader, fetch required window (we need enough history for indicators anyway, loader handles caching)
                            p_df = p_loader.get_k_data(code, plan_date_str, today_str)
                            
                            if not p_df.empty:
                                p_df = fast_patch_realtime(p_df, code)
                                t_row = p_df[p_df['date'].dt.strftime('%Y-%m-%d') == plan_date_str]
                                if not t_row.empty:
                                    h = float(t_row.iloc[0]['high'])
                                    l = float(t_row.iloc[0]['low'])
                                    
                                    # Check T+1 status
                                    t1_status = "⚪ 盘前埋伏 (今日信号)"
                                    post_t_df = p_df[p_df['date'] > t_row.iloc[0]['date']]
                                    if not post_t_df.empty:
                                        max_h_since = float(post_t_df['high'].max())
                                        if max_h_since > h:
                                            t1_status = "🟢 已触发买点"
                                        else:
                                            diff_pct = (h - max_h_since) / h * 100
                                            t1_status = f"🟡 尚未突破 (差{diff_pct:.1f}%)"

                                    # Calculate 2% Risk Sizing (100k capital baseline)
                                    risk_capital = 2000 
                                    risk_diff = h - l
                                    if risk_diff <= 0: risk_diff = h * 0.05
                                    s_qty = int(risk_capital / risk_diff / 100) * 100
                                    target_2_1 = h + (2 * risk_diff)
                                    return h, l, s_qty, target_2_1, t1_status
                        except: pass
                        return 0.0, 0.0, 0, 0.0, "⚪ 未知获取"
                    
                    sop_data = df['code'].apply(get_sop_info)
                    df['买入触发 (T高)'] = sop_data.apply(lambda x: x[0])
                    df['初始止损 (T低)'] = sop_data.apply(lambda x: x[1])
                    df['建议股数 (2%风险)'] = sop_data.apply(lambda x: x[2])
                    df['减仓目标 (2:1)'] = sop_data.apply(lambda x: x[3])
                    df['买点确认 (T+1状态)'] = sop_data.apply(lambda x: x[4])
                    
                    # --- NEW: T-1 Ambush Characteristics Scoring ---
                    def get_ambush_score(code):
                        try:
                            # Use shared loader again! Instant cache hit.
                            a_df = p_loader.get_k_data(code, lookback_dt.strftime("%Y-%m-%d"), plan_date_str)
                            if not a_df.empty:
                                from strong_strategies import StrongStrategies
                                from weak_strategies import WeakStrategies
                                r1 = StrongStrategies.calculate_ambush_calm(a_df)
                                r2 = StrongStrategies.calculate_ambush_momentum(a_df)
                                r3 = WeakStrategies.strategy_ambush_bottom(a_df)
                                
                                scores = []
                                if r1['Ambush_Calm_Signal'].iloc[-1]: scores.append("蓄力(缩量横盘)")
                                if r2['Ambush_Momentum_Signal'].iloc[-1]: scores.append("动量(放量起爆)")
                                if r3['Ambush_Bottom_Signal'].iloc[-1]: scores.append("底部(绝望杀跌)")
                                
                                if not scores: return "⚪ 无"
                                return "🔮 " + "+".join(scores)
                        except: pass
                        return "⚪ 无"
                        
                    df['伏击特征 (T-1)'] = df['code'].apply(get_ambush_score)
                    
                    # --- Add Backtest Metrics for Triggered Strategies ---
                    import json
                    import os
                    def get_strategy_performance(triggered_strats_str):
                        try:
                            # Load optimization results
                            json_path = os.path.join(BASE_DIR, "optimization_results.json")
                            if os.path.exists(json_path):
                                with open(json_path, 'r') as f:
                                    all_results = json.load(f)
                                
                                # Process the triggered strategy string into a comparable format
                                strats = [s.strip() for s in str(triggered_strats_str).split(',') if s.strip()]
                                if not strats:
                                    return "N/A"
                                
                                # Basic matching: try to find an exact match or highest match
                                # For simplicity, we create the standard string representation
                                target_combo_str = " + ".join(sorted(strats))
                                
                                # Look for exact match first
                                for r in all_results:
                                    res_strats = [s.strip() for s in r['Strategy'].split('+')]
                                    res_combo_str = " + ".join(sorted(res_strats))
                                    if res_combo_str == target_combo_str:
                                        return f"胜率 {r['Win Rate (%)']:.1f}% | 收益 {r['Return (%)']:.0f}%"
                                        
                                # If no exact match (e.g. 3 strategies triggered but we only optimized pairs),
                                # find the best sub-combination
                                best_match = None
                                best_match_len = 0
                                for r in all_results:
                                    res_strats = set([s.strip() for s in r['Strategy'].split('+')])
                                    tgt_strats = set(strats)
                                    if res_strats.issubset(tgt_strats):
                                        if len(res_strats) > best_match_len:
                                            best_match = r
                                            best_match_len = len(res_strats)
                                        elif len(res_strats) == best_match_len and best_match:
                                            if r['Return (%)'] > best_match['Return (%)']:
                                                best_match = r
                                                
                                if best_match:
                                    return f"(部分匹配 {best_match['Strategy']}) 胜率 {best_match['Win Rate (%)']:.1f}% | 收益 {best_match['Return (%)']:.0f}%"
                        except:
                            pass
                        return "暂无数据"

                    df['历史绩效预估'] = df['triggered_strategies'].apply(get_strategy_performance)

                    cols = ['code', 'name', '所属板块', '板块状态 (Sector)', '大盘状态 (Macro)', '强度 (Intensity)', 
                            '伏击特征 (T-1)', '买点确认 (T+1状态)', '买入触发 (T高)', '初始止损 (T低)', '减仓目标 (2:1)', '建议股数 (2%风险)', 'triggered_strategies', '历史绩效预估']
                    
                    # --- NEW: Explicit Multi-Column Sorting UI ---
                    st.markdown("##### 🔀 自定义多维排序")
                    col_s1, col_s2 = st.columns([3, 1])
                    
                    # Define custom defaults based on Strong vs Weak category
                    if "strong" in key_suffix:
                        default_sort = ['伏击特征 (T-1)', '买点确认 (T+1状态)', '强度 (Intensity)', '板块状态 (Sector)']
                    else:
                        default_sort = ['强度 (Intensity)', '买点确认 (T+1状态)', '伏击特征 (T-1)', '板块状态 (Sector)']
                        
                    with col_s1:
                        sort_cols = st.multiselect(
                            "选择排序列（按选择先后决定主次优先级）:",
                            options=cols,
                            default=default_sort,
                            key=f"sort_cols_{key_suffix}"
                        )
                    with col_s2:
                        sort_order = st.radio(
                            "排序方向:",
                            options=["降序 (优先看强)", "升序 (优先看弱)"],
                            horizontal=True,
                            key=f"sort_order_{key_suffix}"
                        )
                        
                    if sort_cols:
                        ascending = True if "升序" in sort_order else False
                        try:
                            df = df.sort_values(by=sort_cols, ascending=ascending)
                        except Exception as e:
                            st.error(f"排序失败: {e}")
                    
                    # Interactive Table
                    event = st.dataframe(
                        df[cols], 
                        use_container_width=True,
                        on_select="rerun",
                        selection_mode="single-row",
                        column_config={
                            "所属板块": st.column_config.TextColumn("所属板块", width="medium"),
                            "板块状态 (Sector)": st.column_config.TextColumn("板块状态", width="small", help="🟢=多头(Close>EMA20/200), 🔴=空头"),
                            "大盘状态 (Macro)": st.column_config.TextColumn("大盘状态", width="small"),
                            "强度 (Intensity)": st.column_config.TextColumn("合力", width="small"),
                            "triggered_strategies": st.column_config.TextColumn("触发策略", width="large")
                        },
                        key=f"df_{key_suffix}"
                    )
                    
                    # Handle Selection
                    if event.selection.rows:
                        idx = event.selection.rows[0]
                        row = df.iloc[idx]
                        candidate_str = f"{row['code']} | {row['name']} ({label_suffix})"
                        # Update global selection if changed
                        if st.session_state['selected_candidate'] != candidate_str:
                            st.session_state['selected_candidate'] = candidate_str

                        # Add to Watchlist button
                        if st.button(f"📥 添加至自选: {row['code']} {row['name']}", key=f"dp_wl_{key_suffix}_{idx}"):
                            from trading_manager import TradingManager
                            if 'trading_mgr' not in st.session_state:
                                st.session_state['trading_mgr'] = TradingManager()
                            dp_mgr = st.session_state['trading_mgr']
                            # Load signal K-line high/low
                            sig_high, sig_low = 0, 0
                            try:
                                dp_loader = DataLoader(MARKET_DATA_DIR)
                                df_sig = dp_loader.get_k_data(row['code'],
                                    (datetime.datetime.strptime(res['date'], "%Y-%m-%d") - datetime.timedelta(days=5)).strftime("%Y-%m-%d"),
                                    res['date'])
                                if not df_sig.empty:
                                    df_sig = patch_df_with_realtime(df_sig, row['code'])
                                    sig_row = df_sig[df_sig['date'].dt.strftime('%Y-%m-%d') == res['date']]
                                    if not sig_row.empty:
                                        sig_high = float(sig_row.iloc[0]['high'])
                                        sig_low = float(sig_row.iloc[0]['low'])
                            except:
                                pass
                            ok = dp_mgr.add_to_watchlist(
                                row['code'], row['name'],
                                signal_date=res['date'],
                                strategies=str(row.get('triggered_strategies', '')),
                                signal_high=sig_high, signal_low=sig_low
                            )
                            if ok:
                                st.success(f"✅ {row['code']} {row['name']} 已添加至自选（信号高:{sig_high:.2f} 低:{sig_low:.2f}）")
                            else:
                                st.info(f"{row['code']} 已在自选中")

                        # --- NEW: Execute Buy Button ---
                        if st.button(f"🚀 执行买入: {row['code']} {row['name']}", key=f"dp_buy_{key_suffix}_{idx}", type="primary"):
                            from trading_manager import TradingManager
                            if 'trading_mgr' not in st.session_state:
                                st.session_state['trading_mgr'] = TradingManager()
                            dp_mgr = st.session_state['trading_mgr']
                            
                            # Get SOP metrics calculated in process_display_df
                            h, l, qty = row['买入触发 (T高)'], row['初始止损 (T低)'], row['建议股数 (2%风险)']
                            
                            if h > 0 and qty > 0:
                                ok = dp_mgr.add_position(
                                    row['code'], row['name'],
                                    buy_price=h,
                                    quantity=qty,
                                    buy_date=res['date'],
                                    stop_loss=l,
                                    notes=f"来自交易计划 [{res['date']}] | 触发策略: {row.get('triggered_strategies', '')}"
                                )
                                if ok:
                                    st.success(f"💰 {row['code']} {row['name']} 已加入持仓 (价格:{h:.2f} 股数:{qty} 止损:{l:.2f})")
                                    st.balloons()
                            else:
                                st.warning("无法获取有效的买入价格或计算出的股数为0")

                st.subheader("🚀 强势突围机会 (Strong Buy)")
                process_display_df(df_strong, "强势", "strong", "Strong")
                    
                st.divider()
                
                st.subheader("⚓ 弱势反转机会 (Weak Reversal)")
                process_display_df(df_weak, "弱势", "weak", "Weak")
                    
                st.divider()
                st.caption("注：🔥越多代表同时触发的策略越多，共振越强。点击表格行可直接查看下方图表。")
                    
                if combined_candidates:
                    # Determine Index
                    sel_index = 0
                    if st.session_state['selected_candidate'] in combined_candidates:
                        sel_index = combined_candidates.index(st.session_state['selected_candidate'])
                        
                    selected_str = st.selectbox("选择要查看的股票", combined_candidates, index=sel_index)
                    
                    # Update state if manually changed via selectbox
                    if selected_str != st.session_state['selected_candidate']:
                         st.session_state['selected_candidate'] = selected_str
                    if selected_str:
                        sel_code = selected_str.split(" | ")[0]
                        sel_name = selected_str.split(" | ")[1].split(" (")[0]
                        is_strong = selected_str.endswith("(Strong)")
                        
                        # Get triggered strategies from table data
                        source_df = df_strong if is_strong else df_weak
                        stock_row = source_df[source_df['code'] == sel_code]
                        if not stock_row.empty:
                            triggered_strats_str = stock_row.iloc[0].get('triggered_strategies', '')
                            triggered_strats = [s.strip() for s in str(triggered_strats_str).split(',') if s.strip()]
                        else:
                            triggered_strats = []
                        
                        # Load Data for Chart
                        st.write(f"正在加载 {sel_name} ({sel_code}) ...")
                        
                        local_loader = DataLoader(MARKET_DATA_DIR)
                        c_start = (datetime.datetime.strptime(plan_date_str, "%Y-%m-%d") - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
                        c_end = datetime.date.today().strftime("%Y-%m-%d") # Changed to view up to today
                        
                        df_chart = local_loader.get_k_data(sel_code, c_start, c_end)
                            
                        if not df_chart.empty:
                            df_chart = patch_df_with_realtime(df_chart, sel_code)
                            # Calculate Indicators
                            df_chart = Indicators.add_all_indicators(df_chart)
                            
                            # Calculate Signals using the CORRECT strategy checker
                            if is_strong:
                                sigs = StrongStrategies.check_all_strong_strategies(
                                    df_chart, 
                                    selected_strategies=triggered_strats if triggered_strats else ['Z_Score', 'TKOS', 'DTR_Plus', 'Fighting', 'UA', 'HMC']
                                )
                            else:
                                sigs = WeakStrategies.check_all_weak_strategies(
                                    df_chart,
                                    selected_strategies=triggered_strats if triggered_strats else ['HLP3', 'Limit', 'RSI_Rev', 'Spring', 'Pinbar', 'Money_Flow', 'UA', 'Double_Vol']
                                )
                            
                            # Determine signal dates
                            sig_cols = [c for c in sigs.columns if c.startswith('Signal_')]
                            if sig_cols:
                                is_sig = sigs[sig_cols].any(axis=1)
                                signal_dates = df_chart[is_sig & (df_chart['date'].dt.strftime('%Y-%m-%d') == plan_date_str)]['date']
                            else:
                                signal_dates = None
                            
                            # Controls Layout
                            col_c1, col_c2 = st.columns([1, 4])
                            with col_c1:
                                st.markdown("##### 图表设置")
                                show_ma = st.checkbox("MA20", key='dp_ma')
                                show_ema = st.checkbox("EMA200", key='dp_ema')
                                show_boll = st.checkbox("Boll", key='dp_boll')
                                show_signals = st.checkbox("标注信号", key='dp_sig')
                                sub_chart_type = st.radio("副图:", ["MACD", "KDJ", "RSI", "WR", "CCI", "Volume", "RKing (趋势)", "Volatility", "HMC"], key='dp_sub')
                                
                            with col_c2:
                                # Plot with correct signals and triggered strategies
                                plot_stock_chart(df_chart, sel_code, sel_name, show_ma, show_ema, show_boll, False, False, False, False, show_signals, sub_chart_type, "plotly_white", sigs=sigs, signal_dates=signal_dates, triggered_strategies=triggered_strats, highlight_date=plan_date_str)
                else:
                    st.info("暂无股票可供查看")

# ================================================================
# 💰 交易管理 (Trading Management)
# ================================================================
elif app_mode == "💰 交易管理":
    from trading_manager import TradingManager
    import plotly.graph_objects as go

    st.header("💰 交易管理 (Trading Management)")
    st.markdown("基于 Royal 交易执行 SOP，管理您的自选股与持仓。")

    # Initialize manager
    if 'trading_mgr' not in st.session_state:
        st.session_state['trading_mgr'] = TradingManager()
    mgr = st.session_state['trading_mgr']

    # ── Tabs ──────────────────────────────────────────────────
    tab_overview, tab_watchlist, tab_portfolio = st.tabs([
        "💰 账户总览", "👁️ 自选股 (Watchlist)", "📊 持仓管理 (Portfolio)"
    ])

    # ════════════════════════════════════════════════════════════
    # TAB 1: Account Overview
    # ════════════════════════════════════════════════════════════
    with tab_overview:
        st.subheader("💰 账户总览")
        st.info("💡 **A股实盘基准**：万三佣金 + 千五印花税 (卖出) + T+1 交易限制。")

        col_cap1, col_cap2 = st.columns([2, 1])
        with col_cap1:
            new_capital = st.number_input(
                "总本金 (元)", min_value=0, value=mgr.get_capital(),
                step=10000, key="capital_input"
            )
        with col_cap2:
            if st.button("💾 保存本金", key="save_capital"):
                mgr.set_capital(new_capital)
                st.success(f"本金已更新为 {new_capital:,.0f} 元")

        st.divider()

        # Get current prices for all portfolio stocks
        portfolio = mgr.get_portfolio()
        current_prices = {}
        for p in portfolio:
            try:
                df_p = loader.get_k_data(p["code"],
                    (datetime.datetime.now() - datetime.timedelta(days=10)).strftime("%Y-%m-%d"),
                    datetime.datetime.now().strftime("%Y-%m-%d"))
                if not df_p.empty:
                    df_p = patch_df_with_realtime(df_p, p["code"])
                    current_prices[p["code"]] = df_p.iloc[-1]["close"]
            except:
                pass

        summary = mgr.get_portfolio_summary(current_prices)

        # Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总本金", f"¥{summary['capital']:,.0f}")
        with col2:
            st.metric("持仓市值", f"¥{summary['total_value']:,.0f}",
                      delta=f"{summary['total_pnl']:+,.0f}")
        with col3:
            st.metric("可用资金", f"¥{summary['available_cash']:,.0f}")
        with col4:
            st.metric("持仓数量", f"{summary['position_count']} 只",
                      delta=f"仓位 {summary['total_exposure_pct']:.1f}%")

        st.divider()

        # Risk Alert
        if summary['total_exposure_pct'] > 80:
            st.error("⚠️ 总仓位超过80%，风险较高！建议控制仓位。")
        elif summary['total_exposure_pct'] > 60:
            st.warning("⚠️ 总仓位超过60%，请注意风险控制。")

        # Position Distribution Pie Chart
        if summary['positions']:
            st.subheader("📊 持仓分布")
            labels = [f"{p['code']}-{p['name']}" for p in summary['positions']]
            values = [p['market_value'] for p in summary['positions']]
            # Add available cash
            labels.append("可用资金")
            values.append(max(summary['available_cash'], 0))

            fig_pie = go.Figure(data=[go.Pie(
                labels=labels, values=values,
                hole=0.4, textinfo='label+percent',
                marker=dict(colors=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4',
                                   '#FFEAA7','#DDA0DD','#98D8C8','#C0C0C0'])
            )])
            fig_pie.update_layout(template='plotly_white', height=350, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Position Detail Table
        if summary['positions']:
            st.subheader("📋 持仓明细")
            pos_df = pd.DataFrame(summary['positions'])
            display_cols = ['code', 'name', 'buy_price', 'current_price',
                           'quantity', 'market_value', 'pnl', 'pnl_pct', 'exposure_pct']
            col_names = {'code': '代码', 'name': '名称', 'buy_price': '买入价',
                        'current_price': '现价', 'quantity': '数量',
                        'market_value': '市值', 'pnl': '盈亏', 'pnl_pct': '盈亏%',
                        'exposure_pct': '仓位%'}
            display_cols = [c for c in display_cols if c in pos_df.columns]
            st.dataframe(
                pos_df[display_cols].rename(columns=col_names)
                    .style.format({
                        '买入价': '{:.2f}', '现价': '{:.2f}',
                        '市值': '{:,.0f}', '盈亏': '{:+,.0f}',
                        '盈亏%': '{:+.2f}%', '仓位%': '{:.1f}%'
                    })
                    .applymap(lambda x: 'color: red' if isinstance(x, (int, float)) and x > 0 else
                              ('color: green' if isinstance(x, (int, float)) and x < 0 else ''),
                              subset=['盈亏', '盈亏%']),
                use_container_width=True
            )

    # ════════════════════════════════════════════════════════════
    # TAB 2: Watchlist
    # ════════════════════════════════════════════════════════════
    with tab_watchlist:
        st.subheader("👁️ 自选股管理")

        # Add Stock Form
        with st.expander("➕ 添加自选股", expanded=False):
            if not stock_list_df.empty:
                wl_options = [f"{r['code']} - {r['name']}" for r in stock_list_df.to_dict('records')]
                wl_selected = st.selectbox("选择股票", wl_options, key="wl_add_sel")

                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    wl_sig_date = st.date_input("信号日期", datetime.date.today(), key="wl_sig_date")
                    wl_strategies = st.text_input("触发策略 (逗号分隔)", "", key="wl_strats")
                with col_a2:
                    wl_sig_high = st.number_input("信号K线最高价", value=0.0, step=0.01, key="wl_high")
                    wl_sig_low = st.number_input("信号K线最低价 (止损位)", value=0.0, step=0.01, key="wl_low")

                if st.button("📥 添加至自选", type="primary", key="wl_add_btn"):
                    wl_code = wl_selected.split(" - ")[0]
                    wl_name = wl_selected.split(" - ")[1]
                    ok = mgr.add_to_watchlist(
                        wl_code, wl_name,
                        signal_date=wl_sig_date.isoformat(),
                        strategies=wl_strategies,
                        signal_high=wl_sig_high,
                        signal_low=wl_sig_low
                    )
                    if ok:
                        st.success(f"✅ {wl_code} {wl_name} 已添加至自选")
                        st.rerun()
                    else:
                        st.warning("该股票已在自选中")
            else:
                st.warning("股票列表为空，请先下载数据。")

        # Display Watchlist
        watchlist = mgr.get_watchlist()
        if watchlist:
            st.markdown(f"**共 {len(watchlist)} 只自选股**")

            for i, w in enumerate(watchlist):
                code = w['code']
                name = w['name']
                sig_date_str = w.get('signal_date', '')
                sig_high = w.get('signal_high', 0)
                sig_low = w.get('signal_low', 0)

                # 1. Fetch data and calculate status (Logic ported from Daily Trade Plan)
                t1_status_text = "⚪ 待观察"
                t1_status_emoji = "⚪"
                ambush_info = "⚪ 无"
                latest_price_str = "—"
                
                try:
                    # Date window: enough for T-1 check and T+1 check
                    if sig_date_str:
                        sig_dt = datetime.datetime.strptime(sig_date_str, "%Y-%m-%d")
                        start_fetch_dt = sig_dt - datetime.timedelta(days=150)
                        df_w = loader.get_k_data(code, start_fetch_dt.strftime("%Y-%m-%d"), datetime.date.today().isoformat())
                        
                        if not df_w.empty:
                            df_w = patch_df_with_realtime(df_w, code)
                            latest_close = df_w.iloc[-1]['close']
                            latest_price_str = f"{latest_close:.2f}"
                            
                            # A. T+1 Breakout Check
                            t_row_mask = df_w['date'].dt.strftime('%Y-%m-%d') == sig_date_str
                            if t_row_mask.any():
                                t_idx = df_w[t_row_mask].index[0]
                                post_t_df = df_w.loc[t_idx+1:]
                                if not post_t_df.empty:
                                    max_h_since = float(post_t_df['high'].max())
                                    if max_h_since > sig_high:
                                        t1_status_text = "🟢 已触发买点"
                                        t1_status_emoji = "🟢"
                                    else:
                                        diff_pct = (sig_high - max_h_since) / sig_high * 100
                                        t1_status_text = f"🟡 尚未突破 (差{diff_pct:.1f}%)"
                                        t1_status_emoji = "🟡"
                                else:
                                    t1_status_text = "⚪ 信号日(待T+1)"
                            
                            # B. T-1 Ambush Check
                            if t_row_mask.any():
                                t_idx = df_w[t_row_mask].index[0]
                                if t_idx > 0:
                                    lookback_idx = max(0, t_idx - 121)
                                    a_df = df_w.iloc[lookback_idx:t_idx].copy()
                                    if len(a_df) >= 30:
                                        from strong_strategies import StrongStrategies
                                        from weak_strategies import WeakStrategies
                                        r1 = StrongStrategies.calculate_ambush_calm(a_df)
                                        r2 = StrongStrategies.calculate_ambush_momentum(a_df)
                                        r3 = WeakStrategies.strategy_ambush_bottom(a_df)
                                        
                                        scores = []
                                        if r1['Ambush_Calm_Signal'].iloc[-1]: scores.append("蓄力")
                                        if r2['Ambush_Momentum_Signal'].iloc[-1]: scores.append("动量")
                                        if r3['Ambush_Bottom_Signal'].iloc[-1]: scores.append("底部")
                                        if scores: ambush_info = "🔮 " + "+".join(scores)
                except:
                    pass

                # 2. Render Expander with Status Badge in Title
                expander_title = f"**{t1_status_emoji} {code} - {name}** | 现价: {latest_price_str} | 状态: {t1_status_text}"
                with st.expander(expander_title, expanded=False):
                    # Info row
                    st.markdown(f"""
- 信号日期: **{sig_date_str}** | 触发策略: **{w.get('strategies', '—')}**
- 参照高点: **{sig_high:.2f}** | 止损低点: **{sig_low:.2f}**
- **买点确认**: {t1_status_text}
- **伏击特征 (T-1)**: {ambush_info}
                    """)

                    # Action buttons
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1:
                        show_chart = st.checkbox("📈 K线走势", value=False, key=f"wl_chart_{i}")
                    with col_b2:
                        if st.button("🤖 SOP分析", key=f"wl_ai_{i}"):
                            st.session_state[f'wl_analyzing_{i}'] = True
                    with col_b3:
                        if st.button("🗑️ 删除", key=f"wl_del_{i}", type="secondary"):
                            mgr.remove_from_watchlist(w['code'])
                            st.rerun()

                    # --- Execution: Use 100k capital baseline for quantity suggestion ---
                    risk_capital = 2000 # 2% of 100k
                    risk_diff = sig_high - sig_low
                    if risk_diff <= 0: risk_diff = sig_high * 0.05
                    qty = int(risk_capital / risk_diff / 100) * 100 if risk_diff > 0 else 0
                    
                    if st.button(f"🚀 执行买入 (建议股数: {qty})", key=f"wl_buy_{i}", type="primary", use_container_width=True):
                        if sig_high > 0 and qty > 0:
                            ok = mgr.add_position(
                                code, name,
                                buy_price=sig_high,
                                quantity=qty,
                                buy_date=datetime.date.today().isoformat(),
                                stop_loss=sig_low,
                                notes=f"来自自选股 | 信号日: {sig_date_str} | 伏击: {ambush_info}"
                            )
                            if ok:
                                st.success(f"💰 {code} {name} 已加入持仓 (价格:{sig_high:.2f} 股数:{qty})")
                                st.balloons()
                                st.rerun()
                        else:
                            st.warning("信号数据不完整，无法自动计算仓位。")

                    # K-Line Chart
                    if show_chart:
                        from indicators import Indicators
                        st.divider()
                        # Chart controls
                        col_ctrl, col_cv = st.columns([1, 3])
                        with col_ctrl:
                            st.markdown("**主图层**")
                            wl_ma = st.checkbox("MA20", value=True, key=f"wl_ma_{i}")
                            wl_ema = st.checkbox("EMA200", value=True, key=f"wl_ema_{i}")
                            wl_boll = st.checkbox("布林带", value=True, key=f"wl_boll_{i}")
                            wl_sig = True
                            st.markdown("**副图**")
                            wl_sub = st.radio("副图:", ["MACD", "KDJ", "RSI", "WR", "Volume", "RKing (趋势)"], key=f"wl_sub_{i}")

                        with col_cv:
                            df_chart = loader.get_k_data(w['code'],
                                (datetime.datetime.now() - datetime.timedelta(days=400)).strftime("%Y-%m-%d"),
                                datetime.datetime.now().strftime("%Y-%m-%d"))
                            if not df_chart.empty:
                                df_chart = patch_df_with_realtime(df_chart, w['code'])
                                df_chart = Indicators.add_all_indicators(df_chart)
                                # Use correct strategy checker based on source type
                                wl_source = ExitSignals.detect_source_type(w.get('strategies', ''))
                                if wl_source == 'strong':
                                    sigs_chart = StrongStrategies.check_all_strong_strategies(df_chart)
                                elif wl_source == 'weak':
                                    sigs_chart = WeakStrategies.check_all_weak_strategies(df_chart)
                                else:
                                    sigs_chart = StrongStrategies.check_all_strong_strategies(df_chart)
                                    sigs_weak = WeakStrategies.check_all_weak_strategies(df_chart)
                                    for c in sigs_weak.columns:
                                        if c not in sigs_chart.columns:
                                            sigs_chart[c] = sigs_weak[c]
                                # Signal dates
                                sig_cols_wl = [c for c in sigs_chart.columns if c.startswith('Signal_')]
                                any_sig_wl = sigs_chart[sig_cols_wl].any(axis=1) if sig_cols_wl else pd.Series(False, index=df_chart.index)
                                sig_dates_wl = df_chart[any_sig_wl]['date']
                                wl_triggered = [s.strip() for s in str(w.get('strategies', '')).split(',') if s.strip()]
                                
                                # Exit signals detection
                                wl_source = ExitSignals.detect_source_type(w.get('strategies', ''))
                                wl_exit_sigs = ExitSignals.detect(df_chart, source_type=wl_source, signal_high=w.get('signal_high', 0))
                                
                                # Show exit warnings
                                wl_active_exits = ExitSignals.check_latest_exits(wl_exit_sigs)
                                if wl_active_exits:
                                    for _, exit_desc in wl_active_exits:
                                        st.warning(exit_desc)
                                
                                plot_stock_chart(df_chart, w['code'], w['name'],
                                    wl_ma, wl_ema, wl_boll, False, False, False, False,
                                    wl_sig, wl_sub, "plotly_white",
                                    sigs=sigs_chart, signal_dates=sig_dates_wl,
                                    triggered_strategies=wl_triggered,
                                    stop_loss=w.get('signal_low', 0) if w.get('signal_low', 0) > 0 else None,
                                    exit_signals=wl_exit_sigs)
                            else:
                                st.warning("无法加载行情数据")

                    # SOP Analysis
                    if st.session_state.get(f'wl_analyzing_{i}', False):
                        with st.spinner(f"🤖 正在分析 {w['code']} {w['name']} ..."):
                            try:
                                GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
                            except (FileNotFoundError, KeyError):
                                st.error("未找到 API Key。请在 .streamlit/secrets.toml 中配置 GEMINI_API_KEY。")
                                GEMINI_API_KEY = None

                            if GEMINI_API_KEY:
                                from indicators import Indicators
                                df_wl = loader.get_k_data(w['code'],
                                    (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
                                    datetime.datetime.now().strftime("%Y-%m-%d"))
                                if not df_wl.empty:
                                    df_wl = patch_df_with_realtime(df_wl, w['code'])
                                    df_wl = Indicators.add_all_indicators(df_wl)
                                    sigs_wl = Strategies.check_all(df_wl)
                                    report = mgr.analyze_watchlist_stock(
                                        GEMINI_API_KEY, w['code'], w['name'],
                                        mgr.get_capital(), df_wl, sigs_wl
                                    )
                                    st.markdown("---")
                                    st.markdown(report)
                                else:
                                    st.warning("无法加载行情数据")
                        st.session_state[f'wl_analyzing_{i}'] = False

        else:
            st.info("📭 自选股列表为空。您可以通过上方表单添加，或在「每日交易计划」中点击「添加至自选」。")

    # ════════════════════════════════════════════════════════════
    # TAB 3: Portfolio Management
    # ════════════════════════════════════════════════════════════
    with tab_portfolio:
        st.subheader("📊 持仓管理")

        # Add Position Form
        with st.expander("➕ 添加持仓", expanded=False):
            if not stock_list_df.empty:
                pf_options = [f"{r['code']} - {r['name']}" for r in stock_list_df.to_dict('records')]
                pf_selected = st.selectbox("选择股票", pf_options, key="pf_add_sel")

                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    pf_buy_price = st.number_input("买入价", value=0.0, step=0.01, key="pf_price")
                    pf_quantity = st.number_input("买入数量 (股)", value=100, step=100, key="pf_qty")
                with col_p2:
                    pf_buy_date = st.date_input("买入日期", datetime.date.today(), key="pf_date")
                    pf_stop_loss = st.number_input("止损价", value=0.0, step=0.01, key="pf_sl")

                pf_notes = st.text_input("备注", "", key="pf_notes")

                if st.button("📥 添加持仓", type="primary", key="pf_add_btn"):
                    pf_code = pf_selected.split(" - ")[0]
                    pf_name = pf_selected.split(" - ")[1]
                    if pf_buy_price > 0 and pf_quantity > 0:
                        mgr.add_position(
                            pf_code, pf_name,
                            buy_price=pf_buy_price,
                            quantity=pf_quantity,
                            buy_date=pf_buy_date.isoformat(),
                            stop_loss=pf_stop_loss,
                            notes=pf_notes
                        )
                        st.success(f"✅ {pf_code} {pf_name} 已添加至持仓")
                        st.rerun()
                    else:
                        st.warning("请输入有效的买入价和数量")
            else:
                st.warning("股票列表为空，请先下载数据。")

        # Display Portfolio
        # 1. Fetch live prices for all positions
        portfolio_raw = mgr.get_portfolio()
        current_prices = {}
        for p in portfolio_raw:
            try:
                df_p = loader.get_k_data(p["code"],
                    (datetime.datetime.now() - datetime.timedelta(days=10)).strftime("%Y-%m-%d"),
                    datetime.datetime.now().strftime("%Y-%m-%d"))
                if not df_p.empty:
                    df_p = patch_df_with_realtime(df_p, p["code"])
                    current_prices[p["code"]] = df_p.iloc[-1]["close"]
            except:
                pass
        
        # 2. Get fee-aware summary
        summary = mgr.get_portfolio_summary(current_prices)
        positions = summary['positions']

        if positions:
            st.markdown(f"**共 {len(positions)} 只持仓** | 净资产市值: ¥{summary['total_value']:,.2f}")

            for i, p in enumerate(positions):
                # Fetch MA20 for status check
                ma20_val = 0.0
                try:
                    df_pp = loader.get_k_data(p['code'],
                        (datetime.datetime.now() - datetime.timedelta(days=60)).strftime("%Y-%m-%d"),
                        datetime.datetime.now().strftime("%Y-%m-%d"))
                    if not df_pp.empty:
                        df_pp = patch_df_with_realtime(df_pp, p['code'])
                        df_pp['MA20'] = df_pp['close'].rolling(window=20).mean()
                        ma20_val = df_pp.iloc[-1]['MA20'] if pd.notna(df_pp.iloc[-1]['MA20']) else p['current_price']
                except:
                    pass

                # SOP Metrics (Risk per share)
                risk_per_share = p['buy_price'] - p['stop_loss'] if p['stop_loss'] > 0 else p['buy_price'] * 0.05
                target_1_1 = p['buy_price'] + risk_per_share
                target_2_1 = p['buy_price'] + (2 * risk_per_share)
                rr = ((p['current_price'] - p['buy_price']) / risk_per_share) if risk_per_share > 0 else 0

                # Status check 
                status = "✅ 持仓中"
                if p['stop_loss'] > 0 and p['current_price'] <= p['stop_loss']:
                    status = "🚨 触达初始止损"
                elif p['current_price'] < ma20_val:
                    status = "🔴 跌破MA20离场"
                elif p['pnl'] > 0:
                    if rr >= 2:
                        status = "🎯 盈亏比≥2 (建议减仓)"
                    elif rr >= 1:
                        status = "🛡️ 1:1 已达 (建议推手)"
                    else:
                        status = "🟢 浮盈中"
                elif p['pnl'] < 0:
                    status = "📉 浮亏中"

                # T+1 Badge
                t1_label = " 🟢 可交易" if p['can_sell'] else " ❌ T+1 锁定"
                
                header_str = f"**{p['code']} - {p['name']}** | 现价: ¥{p['current_price']:.2f} | 净盈亏: {p['pnl_pct']:+.2f}% | {status}{t1_label}"
                with st.expander(header_str, expanded=False):
                    col_d1, col_d2, col_d3 = st.columns(3)
                    with col_d1:
                        st.metric("买入价 (Cost)", f"¥{p['buy_price']:.2f}")
                        st.metric("净市值 (Net Value)", f"¥{p['net_value']:,.0f}", help="已预扣万三佣金与千五印花税")
                    with col_d2:
                        st.metric("现价 (Current)", f"¥{p['current_price']:.2f}", delta=f"{p['pnl_pct']:+.2f}% 净盈亏")
                        st.metric("净盈亏 (Net PNL)", f"¥{p['pnl']:+,.0f}")
                    with col_d3:
                        st.metric("生命线 (MA20)", f"¥{ma20_val:.2f}",
                                  delta=f"距离 {(p['current_price'] - ma20_val) / ma20_val * 100:+.1f}%" if ma20_val > 0 else None)
                        st.metric("交易状态", "T+1 锁定" if not p['can_sell'] else "允许卖出", 
                                  delta="今日买入" if not p['can_sell'] else "已过禁售期", delta_color="inverse")

                    st.caption(f"**止损位**: ¥{p['stop_loss']:.2f} | **买入日期**: {p.get('buy_date', '未知')} | **备注**: {p.get('notes', '')}")

                    # Action buttons
                    col_a1, col_a2, col_a3 = st.columns(3)
                    with col_a1:
                        pf_show_chart = st.checkbox("📈 K线走势", value=False, key=f"pf_chart_{i}")
                    with col_a2:
                        if st.button("🤖 SOP 持仓分析", key=f"pf_ai_{i}"):
                            st.session_state[f'pf_analyzing_{i}'] = True
                    with col_a3:
                        if st.button("🗑️ 删除持仓", key=f"pf_del_{i}", type="secondary"):
                            mgr.remove_position(i)
                            st.rerun()

                    # K-Line Chart
                    if pf_show_chart:
                        from indicators import Indicators
                        st.divider()
                        col_ctrl, col_cv = st.columns([1, 3])
                        with col_ctrl:
                            st.markdown("**主图层**")
                            pf_ma = st.checkbox("MA20", value=True, key=f"pf_ma_{i}")
                            pf_ema = st.checkbox("EMA200", value=True, key=f"pf_ema_{i}")
                            pf_boll = st.checkbox("布林带", value=True, key=f"pf_boll_{i}")
                            st.markdown("**副图**")
                            pf_sub = st.radio("副图:", ["MACD", "KDJ", "RSI", "WR", "Volume", "RKing (趋势)"], key=f"pf_sub_{i}")

                        with col_cv:
                            df_chart = loader.get_k_data(p['code'],
                                (datetime.datetime.now() - datetime.timedelta(days=400)).strftime("%Y-%m-%d"),
                                datetime.datetime.now().strftime("%Y-%m-%d"))
                            if not df_chart.empty:
                                df_chart = patch_df_with_realtime(df_chart, p['code'])
                                df_chart = Indicators.add_all_indicators(df_chart)
                                # Use correct strategy checker based on source type
                                pf_source = ExitSignals.detect_source_type(p.get('strategies', ''))
                                if pf_source == 'strong':
                                    sigs_chart = StrongStrategies.check_all_strong_strategies(df_chart)
                                elif pf_source == 'weak':
                                    sigs_chart = WeakStrategies.check_all_weak_strategies(df_chart)
                                else:
                                    sigs_chart = StrongStrategies.check_all_strong_strategies(df_chart)
                                    sigs_weak = WeakStrategies.check_all_weak_strategies(df_chart)
                                    for c in sigs_weak.columns:
                                        if c not in sigs_chart.columns:
                                            sigs_chart[c] = sigs_weak[c]
                                sig_cols_pf = [c for c in sigs_chart.columns if c.startswith('Signal_')]
                                any_sig_pf = sigs_chart[sig_cols_pf].any(axis=1) if sig_cols_pf else pd.Series(False, index=df_chart.index)
                                sig_dates_pf = df_chart[any_sig_pf]['date']
                                pf_triggered = [s.strip() for s in str(p.get('strategies', '')).split(',') if s.strip()]
                                
                                # Calculate take-profit levels
                                pf_buy = p['buy_price'] if p['buy_price'] > 0 else None
                                pf_sl = p['stop_loss'] if p['stop_loss'] > 0 else None
                                pf_tp1 = None
                                pf_tp2 = None
                                if pf_buy and pf_sl and pf_buy > pf_sl:
                                    risk = pf_buy - pf_sl
                                    pf_tp1 = pf_buy + risk       # 1:1 保本
                                    pf_tp2 = pf_buy + 2 * risk   # 2:1 减半
                                
                                # Exit signals detection
                                pf_source = ExitSignals.detect_source_type(p.get('strategies', ''))
                                pf_exit_sigs = ExitSignals.detect(df_chart, source_type=pf_source)
                                
                                # Show exit warnings
                                pf_active_exits = ExitSignals.check_latest_exits(pf_exit_sigs)
                                if pf_active_exits:
                                    for _, exit_desc in pf_active_exits:
                                        st.warning(exit_desc)
                                
                                plot_stock_chart(df_chart, p['code'], p['name'],
                                    pf_ma, pf_ema, pf_boll, False, False, False, False,
                                    True, pf_sub, "plotly_white",
                                    sigs=sigs_chart, signal_dates=sig_dates_pf,
                                    triggered_strategies=pf_triggered,
                                    buy_price=pf_buy, stop_loss=pf_sl,
                                    take_profit_1=pf_tp1, take_profit_2=pf_tp2,
                                    exit_signals=pf_exit_sigs)
                            else:
                                st.warning("无法加载行情数据")

                    # SOP Analysis
                    if st.session_state.get(f'pf_analyzing_{i}', False):
                        with st.spinner(f"🤖 正在分析 {p['code']} {p['name']} ..."):
                            try:
                                GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
                            except (FileNotFoundError, KeyError):
                                st.error("未找到 API Key。请在 .streamlit/secrets.toml 中配置 GEMINI_API_KEY。")
                                GEMINI_API_KEY = None

                            if GEMINI_API_KEY:
                                from indicators import Indicators
                                df_pf = loader.get_k_data(p['code'],
                                    (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
                                    datetime.datetime.now().strftime("%Y-%m-%d"))
                                if not df_pf.empty:
                                    df_pf = patch_df_with_realtime(df_pf, p['code'])
                                    df_pf = Indicators.add_all_indicators(df_pf)
                                    sigs_pf = Strategies.check_all(df_pf)
                                    report = mgr.analyze_portfolio_stock(
                                        GEMINI_API_KEY, p['code'], p['name'],
                                        p, df_pf, sigs_pf
                                    )
                                    st.markdown("---")
                                    st.markdown(report)
                                else:
                                    st.warning("无法加载行情数据")
                        st.session_state[f'pf_analyzing_{i}'] = False

        else:
            st.info("📭 持仓列表为空。请通过上方表单添加持仓。")

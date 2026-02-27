import pandas as pd
import os
import glob
from indicators import Indicators

class MarketEnvironmentLoader:
    """
    Loads and caches Macro Index and Industry Sector data to evaluate
    if a specific board or sector is currently in a systemic "Bull" or "Panic" state.
    Used by BacktestEngine and Daily Trading Plan to filter out false breakouts.
    """
    _instance = None
    
    def __new__(cls, data_dir=None):
        if cls._instance is None:
            cls._instance = super(MarketEnvironmentLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self, data_dir=None):
        if self._initialized:
            return
            
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if data_dir is None:
            self.data_dir = os.path.join(base_dir, 'data', 'market_data')
        else:
            self.data_dir = data_dir
        
        self.sector_data_dir = os.path.join(os.path.dirname(self.data_dir), 'sector_data')
            
        # Dictionary to store pre-calculated health metrics per board per date
        # Format: {'sh': {date: bool}, 'sz': {date: bool}, 'cyb': {date: bool}}
        self.board_health_cache = {'sh': {}, 'sz': {}, 'cyb': {}}
        
        # Sector health cache: {sector_name: {date: bool}}
        self.sector_health_cache = {}
        
        # Stock to sector mapping: {stock_code: sector_name}
        self.stock_sector_map = {}
        
        # Stock to concept mapping: {stock_code: [concepts]}
        self.stock_concept_map = {}
        
        self.board_mapping = {
            'sh': '000001.SH',
            'sz': '399001.SZ',
            'cyb': '399006.SZ'
        }
        
        self._load_board_health()
        self._load_sector_mapping()
        self._load_concept_mapping()
        self._load_sector_health()
        
        self._initialized = True
        print(f"🌍 MarketEnvironmentLoader initialized. Macro & Sector Trends loaded.")

    def _load_board_health(self):
        """Loads index CSVs and calculates EMA200 and Bias."""
        for board_key, file_name in self.board_mapping.items():
            file_path = os.path.join(self.data_dir, f"{file_name}.csv")
            if not os.path.exists(file_path):
                print(f"⚠️ Warning: Macro index file {file_name}.csv not found. {board_key} will always pass.")
                continue
                
            try:
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                
                # Calculate EMA200 (Trend) natively
                df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
                
                # Calculate MA20 Bias (Sentiment Panic)
                ma20 = df['close'].rolling(20).mean()
                df['Bias20'] = (df['close'] - ma20) / ma20 * 100
                
                # Evaluate Daily Health
                # Rule: Must be ABOVE EMA200 (Bull Trend) 
                # AND Bias20 > -5% (Not in violent panic drop)
                df['is_healthy'] = (df['close'] > df['EMA200']) & (df['Bias20'] > -5.0)
                
                # Map to cache for fast O(1) lookup
                cache = dict(zip(df['date'], df['is_healthy']))
                self.board_health_cache[board_key] = cache
                
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    def _load_sector_mapping(self):
        """Loads the mapping of stocks to their sectors."""
        mapping_path = os.path.join(self.data_dir, 'stock_sector_map.csv')
        if os.path.exists(mapping_path):
            try:
                df = pd.read_csv(mapping_path)
                # Keep code as string with leading zeros if necessary
                df['code'] = df['code'].astype(str).str.zfill(6)
                self.stock_sector_map = dict(zip(df['code'], df['sector_name']))
            except Exception as e:
                print(f"Error loading sector mapping: {e}")

    def _load_sector_health(self):
        """Calculates health for all downloaded sector indices."""
        if not os.path.exists(self.sector_data_dir):
            return
            
        sector_files = glob.glob(os.path.join(self.sector_data_dir, "*.csv"))
        for file_path in sector_files:
            sector_name = os.path.basename(file_path).replace(".csv", "")
            try:
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                
                # Calculate Sector Health: Close > EMA20 (Short-term Strength) AND Close > EMA200 (Long-term Trend)
                df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
                df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
                
                # Rule: Uptrend confirmation
                df['is_healthy'] = (df['close'] > df['EMA20']) & (df['close'] > df['EMA200'])
                
                self.sector_health_cache[sector_name] = dict(zip(df['date'], df['is_healthy']))
            except Exception as e:
                pass # Silently skip malformed sector files

    def _get_board_for_stock(self, code):
        """Maps an individual stock code to its governing Macro Board."""
        code_str = str(code)
        code_clean = code_str.replace('sh', '').replace('sz', '').replace('.SH', '').replace('.SZ', '')
        
        if code_clean.startswith('6'):
            return 'sh'
        elif code_clean.startswith('30'):
            return 'cyb'
        elif code_clean.startswith('00'):
            return 'sz'
        elif code_clean.startswith('688'):
            return 'sh' 
        elif code_clean.startswith('8') or code_clean.startswith('4'):
            return 'sz'
        else:
            return 'sh'

    def _load_concept_mapping(self):
        """Loads the mapping of stocks to their concepts."""
        mapping_path = os.path.join(self.data_dir, 'stock_concept_map.csv')
        if os.path.exists(mapping_path):
            try:
                df = pd.read_csv(mapping_path)
                df['code'] = df['code'].astype(str).str.zfill(6)
                # Store as list of concepts
                self.stock_concept_map = {row['code']: str(row['concepts']).split(',') for _, row in df.iterrows()}
            except Exception as e:
                print(f"Error loading concept mapping: {e}")

    def get_stock_sector(self, code):
        """Returns the industry sector name for a stock."""
        code_str = str(code).replace('sh', '').replace('sz', '').replace('.SH', '').replace('.SZ', '').zfill(6)
        return self.stock_sector_map.get(code_str, "未知板块")

    def get_stock_concepts(self, code):
        """Returns a list of associated concepts for a stock."""
        code_str = str(code).replace('sh', '').replace('sz', '').replace('.SH', '').replace('.SZ', '').zfill(6)
        return self.stock_concept_map.get(code_str, [])

    def is_board_healthy(self, code, date_str):
        """Returns True if the stock's corresponding board is healthy on that date."""
        board_key = self._get_board_for_stock(code)
        board_cache = self.board_health_cache.get(board_key, {})
        
        if not board_cache:
            return True
            
        if date_str in board_cache:
            return board_cache[date_str]
            
        # Optimization: Use pre-cached sorted dates or memoize
        if not hasattr(self, '_board_dates_cache'):
            self._board_dates_cache = {}
        
        if board_key not in self._board_dates_cache:
            self._board_dates_cache[board_key] = sorted(board_cache.keys())
            
        available_dates = self._board_dates_cache[board_key]
        prior_dates = [d for d in available_dates if d <= date_str]
        return board_cache[prior_dates[-1]] if prior_dates else True

    def is_sector_healthy(self, code, date_str):
        """
        Returns True if the stock's industry sector is healthy on that date.
        If mapping or sector data is missing, defaults to True to allow trading.
        """
        code_str = str(code).replace('sh', '').replace('sz', '').replace('.SH', '').replace('.SZ', '').zfill(6)
        sector_name = self.stock_sector_map.get(code_str)
        
        if not sector_name:
            return True # Mapping missing
            
        sector_cache = self.sector_health_cache.get(sector_name)
        if not sector_cache:
            return True # Sector data missing
            
        if date_str in sector_cache:
            return sector_cache[date_str]
            
        # Optimization: Cache sorted dates per sector
        if not hasattr(self, '_sector_dates_cache'):
            self._sector_dates_cache = {}
            
        if sector_name not in self._sector_dates_cache:
            self._sector_dates_cache[sector_name] = sorted(sector_cache.keys())
            
        available_dates = self._sector_dates_cache[sector_name]
        prior_dates = [d for d in available_dates if d <= date_str]
        
        if prior_dates:
            return sector_cache[prior_dates[-1]]
            
        return True # Default fallback

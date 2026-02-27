import pandas as pd
import datetime
import traceback
from data_loader import DataLoader
from indicators import Indicators
from strategies import Strategies

# Initialize Loader once per process if possible, or per call?
# Loader is lightweight (just paths), so per call is fine or global in worker.

def scan_single_stock(args):
    """
    Worker function for multiprocessing.
    args: (code, name, load_start_str, load_end_str, scan_start_str, scan_end_str, checks_config)
    """
    code, name, load_start_str, load_end_str, scan_start_str, scan_end_str, checks_config = args
    
    loader = DataLoader()
    
    try:
        # Load Data (Load enough history for indicators, typically load_start is earlier than scan_start)
        df = loader.get_k_data(code, load_start_str, load_end_str)
        
        if df.empty or len(df) < 120:
            return None
            
        # Add Indicators
        df = Indicators.add_all_indicators(df)
        
        # Check Strategies
        sigs = Strategies.check_all(df)
        
        # Create mask for SCAN range
        # Ensure we are looking at the window user requested
        mask_scan = (df['date'].dt.strftime('%Y-%m-%d') >= scan_start_str) & \
                    (df['date'].dt.strftime('%Y-%m-%d') <= scan_end_str)
        
        if not mask_scan.any():
            return None
            
        # Filter signals within the scan window
        sigs_window = sigs[mask_scan]
        
        # Check if ANY day in window meets ALL selected conditions
        # But wait, "AND Logic" usually applies to a SINGLE day.
        # So we check row by row in the window.
        
        valid_dates = []
        
        # Iterate over rows in the window (usually not too many if scanning recent)
        # Vectorized check:
        # 1. Combine all selected strategy columns with AND
        
        final_sig = pd.Series(True, index=sigs_window.index)
        selected_any = False
        
        for is_checked, col_str, disp_name in checks_config:
            if is_checked:
                selected_any = True
                if col_str in sigs_window.columns:
                    final_sig &= sigs_window[col_str]
                else:
                    # Strategy col missing? treat as False
                    final_sig = False 
        
        if selected_any:
            # Get dates where final_sig is True
            valid_dates = df.loc[sigs_window[final_sig].index, 'date']
            
            if not valid_dates.empty:
                # Found match(es)
                last_date = valid_dates.iloc[-1] # Newest date
                last_row = df[df['date'] == last_date].iloc[0]
                
                # Determine which strategies were active on that LAST date
                last_sig_row = sigs.loc[df['date'] == last_date].iloc[0]
                triggered = [disp_name for chk, col, disp_name in checks_config if chk and last_sig_row.get(col, False)]

                return {
                    "Code": code, 
                    "Name": name, 
                    "Price": last_row['close'],
                    "Signal Date": last_date.strftime("%Y-%m-%d"),
                    "Strategies": ", ".join(triggered)
                }
            
    except Exception:
        # print(f"Error scanning {code}: {traceback.format_exc()}")
        return None
        
    return None

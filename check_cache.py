import os
import pandas as pd
cache_dir = "data/signal_cache"
files = [f for f in os.listdir(cache_dir) if f.startswith("strong_")]
if files:
    df = pd.read_parquet(os.path.join(cache_dir, files[0]))
    print("Strong Columns:", [c for c in df.columns if c.startswith("Signal_")])
files_w = [f for f in os.listdir(cache_dir) if f.startswith("weak_")]
if files_w:
    df = pd.read_parquet(os.path.join(cache_dir, files_w[0]))
    print("Weak Columns:", [c for c in df.columns if c.startswith("Signal_")])

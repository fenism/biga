import pandas as pd
from data_loader import DataLoader
from indicators import Indicators
from exit_signals import ExitSignals
from app import plot_stock_chart
import plotly.io as pio

# We use 511880 on 2026-02-13
loader = DataLoader("data/market_data")
df = loader.get_k_data("511880", "2025-08-01", "2026-02-15")
df = Indicators.add_all_indicators(df)
exits = ExitSignals.detect(df, source_type='strong')

import os
fig = plot_stock_chart(
    df, code="511880", name="Exit Signal Demo",
    show_ma=True, show_ema=True, show_boll=True, show_cyc=False, show_ema15=False, show_box=False,
    show_supt=False, show_signals=True, sub_chart_type="Volume", plotly_template="plotly_white",
    exit_signals=exits, return_fig=True
)
os.makedirs("/Users/swag/.gemini/antigravity/brain/c5c92b22-16f2-4656-b57d-90b5a4a2a6a8", exist_ok=True)
fig.write_image("/Users/swag/.gemini/antigravity/brain/c5c92b22-16f2-4656-b57d-90b5a4a2a6a8/exit_signal_demo.png")
print("Chart generated successfully at exit_signal_demo.png")

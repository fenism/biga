import pandas as pd
from data_loader import DataLoader
from indicators import Indicators
from exit_signals import ExitSignals
import plotly.io as pio
from app import plot_stock_chart
import os

loader = DataLoader("data/market_data")
df = loader.get_k_data("600935", "2025-08-01", "2026-02-15")
df = Indicators.add_all_indicators(df)
exits = ExitSignals.detect(df, source_type='strong')

fig = plot_stock_chart(
    df, code="600935", name="HuaHui Technology",
    show_ma=True, show_ema=True, show_boll=False, show_cyc=False, show_ema15=False, show_box=False,
    show_supt=False, show_signals=True, sub_chart_type="Volume", plotly_template="plotly_white",
    exit_signals=exits, return_fig=True
)

output_path = "/Users/swag/.gemini/antigravity/brain/c5c92b22-16f2-4656-b57d-90b5a4a2a6a8/exit_signals_test_1771402427690.webp"
fig.write_image(output_path, engine="kaleido")
print(f"Chart generated at {output_path}")


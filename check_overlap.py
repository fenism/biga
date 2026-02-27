import pandas as pd
from eval_utils import SignalCacheReader

reader = SignalCacheReader()
df_s, df_w = reader.load_selected_strategies(['TKOS', 'DTR_Plus'], [])
print('总计强策略池大小:', len(df_s))

if 'Signal_TKOS' in df_s.columns:
    print('TKOS 单独触发次数:', df_s['Signal_TKOS'].sum())
if 'Signal_DTR_Plus' in df_s.columns:
    print('DTR++ 单独触发次数:', df_s['Signal_DTR_Plus'].sum())

if 'Signal_TKOS' in df_s.columns and 'Signal_DTR_Plus' in df_s.columns:
    both = df_s[(df_s['Signal_TKOS'] == True) & (df_s['Signal_DTR_Plus'] == True)]
    print('TKOS 跟 DTR++ 严格同日触发 (AND Logic) 次数:', len(both))
    
    union = df_s[(df_s['Signal_TKOS'] == True) | (df_s['Signal_DTR_Plus'] == True)]
    print('TKOS 或者 DTR++ (OR Logic) 触发次数:', len(union))

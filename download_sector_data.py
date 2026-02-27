import akshare as ak
import pandas as pd
import os
import time

def download_sector_data():
    mapping_path = 'data/market_data/stock_sector_map.csv'
    if not os.path.exists(mapping_path):
        print(f"❌ 找不到映射文件: {mapping_path}")
        return
        
    df_map = pd.read_csv(mapping_path)
    unique_sectors = df_map[['sector_name', 'sector_code']].drop_duplicates()
    
    print(f">>> 🚀 开始下载 {len(unique_sectors)} 个行业板块的历史数据 <<<")
    
    save_dir = 'data/sector_data'
    os.makedirs(save_dir, exist_ok=True)
    
    success_count = 0
    total = len(unique_sectors)
    
    for i, (idx, row) in enumerate(unique_sectors.iterrows()):
        name = row['sector_name']
        code = row['sector_code']
        
        file_path = os.path.join(save_dir, f"{name}.csv")
        
        if i % 10 == 0:
            print(f"进度: {i}/{total} | 正在下载: {name}...")
            
        try:
            # 修正参数：移除 period，直接使用 start_date 和 end_date
            df_hist = ak.stock_board_industry_hist_em(
                symbol=name, 
                start_date="20220101", 
                end_date="20260224"
            )
            
            if not df_hist.empty:
                # 统一列名映射，方便以后读取
                col_map = {
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount'
                }
                df_hist = df_hist.rename(columns=col_map)
                df_hist.to_csv(file_path, index=False)
                success_count += 1
            
            # 延时防止限流
            time.sleep(0.05)
        except Exception as e:
            # print(f"下载板块 [{name}] 失败: {e}")
            pass

    print(f"✅ 下载完成! 成功: {success_count}/{total}")

if __name__ == "__main__":
    download_sector_data()

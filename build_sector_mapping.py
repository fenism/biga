import akshare as ak
import pandas as pd
import os
import time

def build_mapping():
    print(">>> 🚀 开始构建个股-行业映射表 <<<")
    
    # 1. 获取所有行业板块列表
    try:
        industry_boards = ak.stock_board_industry_name_em()
        print(f"找到 {len(industry_boards)} 个行业板块。")
    except Exception as e:
        print(f"获取板块列表失败: {e}")
        return

    mapping_data = []
    total = len(industry_boards)
    
    for i, row in industry_boards.iterrows():
        board_name = row['板块名称']
        board_code = row['板块代码']
        
        if i % 10 == 0:
            print(f"进度: {i}/{total} | 正在抓取: {board_name}...")
            
        try:
            members = ak.stock_board_industry_cons_em(symbol=board_name)
            for _, m_row in members.iterrows():
                mapping_data.append({
                    'code': m_row['代码'],
                    'name': m_row['名称'],
                    'sector_name': board_name,
                    'sector_code': board_code
                })
            # 适当延时防止限流
            time.sleep(0.1)
        except Exception as e:
            print(f"抓取板块 [{board_name}] 失败: {e}")

    df = pd.DataFrame(mapping_data)
    # 去重（有些股票属于多个子板块，我们取最后抓到的或主板块）
    df = df.drop_duplicates(subset=['code'], keep='first')
    
    output_path = 'data/market_data/stock_sector_map.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ 映射表构建完成: {len(df)} 只股票归宗。")
    print(f"文件保存至: {output_path}")

if __name__ == "__main__":
    build_mapping()

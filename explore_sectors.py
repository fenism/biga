import akshare as ak
import pandas as pd

def test_sectors():
    print(">>> 探测行业板块数据 <<<")
    
    # 1. 获取行业板块分类 (证监会行业分类)
    try:
        industry_df = ak.stock_industry_category_cn()
        print(f"证监会行业分类: {len(industry_df)} 个行业")
        print(industry_df.head())
    except Exception as e:
        print(f"无法获取证监会行业: {e}")

    # 2. 获取东财行业板块列表
    try:
        eastmoney_industries = ak.stock_board_industry_name_em()
        print(f"东财行业板块: {len(eastmoney_industries)} 个")
        print(eastmoney_industries.head())
    except Exception as e:
        print(f"无法获取东财行业: {e}")

    # 3. 获取个股行业归属 (以 600519 为例)
    try:
        # 这个接口可能不存在，尝试常见的
        stock_info = ak.stock_individual_info_em(symbol="600519")
        print("个股信息:")
        print(stock_info)
    except Exception as e:
        print(f"无法获取个股行业: {e}")

    # 4. 获取板块成分股
    try:
        # 获取第一个东财行业的成分股
        first_board = eastmoney_industries.iloc[0]['板块名称']
        members = ak.stock_board_industry_cons_em(symbol=first_board)
        print(f"板块 [{first_board}] 成分股: {len(members)} 只")
        print(members.head())
    except Exception as e:
        print(f"无法获取板块成分股: {e}")

if __name__ == "__main__":
    test_sectors()

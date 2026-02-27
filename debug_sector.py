import akshare as ak
import pandas as pd

def debug_sector():
    name = "半导体"
    print(f">>> 调试板块: {name} <<<")
    try:
        # 尝试不带 period 参数，或根据搜索结果使用 start_date
        df = ak.stock_board_industry_hist_em(symbol=name, start_date="20220101", end_date="20260224")
        print(f"成功获取数据! 行数: {len(df)}")
        print(f"数据列: {df.columns.tolist()}")
        print(f"前几行:\n{df.head()}")
    except Exception as e:
        print(f"直接调用失败: {e}")
        try:
            print("尝试带 adjust='' 参数...")
            df = ak.stock_board_industry_hist_em(symbol=name, start_date="20220101", end_date="20260224", adjust="")
            print(f"成功获取数据 (adjust='')! 行数: {len(df)}")
        except Exception as e2:
            print(f"带 adjust='' 依然失败: {e2}")

if __name__ == "__main__":
    debug_sector()

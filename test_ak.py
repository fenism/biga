import akshare as ak
try:
    print("Testing stock_board_industry_name_em...")
    # Fetch all industry boards
    industry_boards = ak.stock_board_industry_name_em()
    print(industry_boards.head())
    print("\nTotal industries:", len(industry_boards))
except Exception as e:
    print("Error:", e)

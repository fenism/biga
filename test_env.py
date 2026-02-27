from market_env import MarketEnvironmentLoader
env = MarketEnvironmentLoader()
# Sample checks for 2024-04-12
print("sz300059 cyb healthy?", env.is_board_healthy('300059', '2024-04-12'))
print("sh600519 sh healthy?", env.is_board_healthy('600519', '2024-04-12'))
print("sz000001 sz healthy?", env.is_board_healthy('000001', '2024-04-12'))

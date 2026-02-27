import os

path = '/Users/swag/Library/CloudStorage/GoogleDrive-fenism@gmail.com/其他计算机/我的 Mac Pro/文档/analyse/stock_app/app.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Line numbers are 1-indexed in the tool, so iterate accordingly
# Strong Stocks Table (around 2259)
for i in range(len(lines)):
    if 'res_df[["Code", "Name", "大盘状态 (Macro)"' in lines[i] and 'event = st.dataframe' in lines[i-1]:
        lines[i] = '            res_df[["Code", "Name", "所属板块", "板块状态 (Sector)", "大盘状态 (Macro)", "强度 (Intensity)", "Signal Date", "Strategies"]],\n'
        print(f"Updated Strong Scan table at line {i+1}")
        
    # Weak Stocks Table (around 2598)
    if 'res_df[["Code", "Name", "大盘状态 (Macro)"' in lines[i] and 'event = st.dataframe' in lines[i-1] and i > 2400:
        lines[i] = '            res_df[["Code", "Name", "所属板块", "板块状态 (Sector)", "大盘状态 (Macro)", "强度 (Intensity)", "Signal Date", "Strategies"]],\n'
        print(f"Updated Weak Scan table at line {i+1}")

    # Strong results collection (around 2231-2239)
    if 'agg_results.append({' in lines[i] and '"Code": code,' in lines[i+1] and '"Name": name,' in lines[i+2] and i < 2500 and i > 2000:
         # Check if already updated
         if '所属板块' not in lines[i+3]:
             insert_lines = [
                 '                            "所属板块": s_name,\n',
                 '                            "板块状态 (Sector)": s_env,\n'
             ]
             lines[i+3:i+3] = insert_lines
             print(f"Updated Strong agg_results at line {i+1}")

    # Weak results collection (around 2570-2578)
    if 'agg_results.append({' in lines[i] and '"Code": code,' in lines[i+1] and '"Name": name,' in lines[i+2] and i > 2500:
         # Check if already updated
         if '所属板块' not in lines[i+3]:
             insert_lines = [
                 '                            "所属板块": s_name,\n',
                 '                            "板块状态 (Sector)": s_env,\n'
             ]
             lines[i+3:i+3] = insert_lines
             print(f"Updated Weak agg_results at line {i+1}")

    # Sector check block in Strong (around 2230)
    if 'm_env = "🔴 风险"' in lines[i] and 'except: pass' in lines[i+1] and i < 2500:
        if 's_name = env_loader' not in lines[i+2]:
            insert_sector = [
                '                        # Sector check\n',
                '                        s_name = env_loader.get_stock_sector(code)\n',
                '                        s_env = "⚪ 待检"\n',
                '                        try:\n',
                '                            is_s_h = env_loader.is_sector_health_healthy(code, last_row["date"].strftime("%Y-%m-%d")) if hasattr(env_loader, "is_sector_health_healthy") else env_loader.is_sector_healthy(code, last_row["date"].strftime("%Y-%m-%d"))\n',
                '                            s_env = "🟢 多头" if is_s_h else "🔴 空头"\n',
                '                        except: pass\n'
            ]
            lines[i+2:i+2] = insert_sector
            print(f"Inserted Sector check in Strong at line {i+1}")

    # Sector check block in Weak (around 2568)
    if 'm_env = "🔴 风险"' in lines[i] and 'except: pass' in lines[i+1] and i > 2500:
        if 's_name = env_loader' not in lines[i+2]:
            insert_sector = [
                '                        # Sector check\n',
                '                        s_name = env_loader.get_stock_sector(code)\n',
                '                        s_env = "⚪ 待检"\n',
                '                        try:\n',
                '                            is_s_h = env_loader.is_sector_health_healthy(code, last_row["date"].strftime("%Y-%m-%d")) if hasattr(env_loader, "is_sector_health_healthy") else env_loader.is_sector_healthy(code, last_row["date"].strftime("%Y-%m-%d"))\n',
                '                            s_env = "🟢 多头" if is_s_h else "🔴 空头"\n',
                '                        except: pass\n'
            ]
            lines[i+2:i+2] = insert_sector
            print(f"Inserted Sector check in Weak at line {i+1}")

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

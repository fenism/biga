import akshare as ak
import pandas as pd
import os
import concurrent.futures
from tqdm import tqdm

def build_concept_map():
    print("🚀 Fetching concept list from Eastmoney...")
    try:
        concept_df = ak.stock_board_concept_name_em()
    except Exception as e:
        print(f"❌ Error fetching concept list: {e}")
        return

    concepts = concept_df['板块名称'].tolist()
    print(f"📦 Found {len(concepts)} concepts. Starting constituent harvest...")

    stock_to_concepts = {}

    def fetch_concept_cons(concept_name):
        try:
            cons_df = ak.stock_board_concept_cons_em(symbol=concept_name)
            if not cons_df.empty:
                return concept_name, cons_df['代码'].tolist()
        except:
            pass
        return concept_name, []

    # Use ThreadPool to speed up network requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_concept_cons, name): name for name in concepts}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(concepts), desc="Harvesting"):
            concept_name, codes = future.result()
            for code in codes:
                if code not in stock_to_concepts:
                    stock_to_concepts[code] = []
                stock_to_concepts[code].append(concept_name)

    # Convert to DataFrame
    print("📊 Processing results...")
    data = []
    for code, concept_list in stock_to_concepts.items():
        data.append({
            'code': code,
            'concepts': ",".join(concept_list)
        })

    df = pd.DataFrame(data)
    
    # Save results
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, 'data', 'market_data', 'stock_concept_map.csv')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Concept mapping saved to {output_path}")
    print(f"✨ Total stocks mapped: {len(df)}")

if __name__ == "__main__":
    build_concept_map()

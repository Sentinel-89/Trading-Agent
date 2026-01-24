import pandas as pd
import os

# Define paths relative to the script's location in backend/utils/
# '..' goes up to 'backend', the second '..' goes up to the project root
base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
master_dir = os.path.join(base_dir, "backend", "data", "processed")
backup_dir = os.path.join(base_dir, "backend", "data", "processed-backup-until2024")

symbols = [
    "HCLTECH", "HDFCBANK", "HINDUNILVR", "ICICIBANK", "INFY", 
    "ITC", "JSWSTEEL", "MARUTI", "NTPC", "RELIANCE", 
    "SBIN", "TATASTEEL", "TCS", "TMPV"
]

for symbol in symbols:
    file_master = os.path.join(master_dir, f"{symbol}_labeled.csv")
    file_backup = os.path.join(backup_dir, f"{symbol}_labeled.csv")
    
    if os.path.exists(file_master) and os.path.exists(file_backup):
        # 1. Load data
        df_2025 = pd.read_csv(file_master)
        df_2024 = pd.read_csv(file_backup)
        
        # 2. Concatenate
        combined_df = pd.concat([df_2024, df_2025], axis=0)
        
        # 3. Clean and Sort
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df = combined_df.sort_values('Date')
        
        # Keep the latest calculations from the 2025 file for overlaps
        combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
        
        # 4. Save back to the master processed folder
        combined_df.to_csv(file_master, index=False)
        print(f"Successfully merged {symbol}. Range: {combined_df['Date'].min().date()} to {combined_df['Date'].max().date()}")
    else:
        print(f"Warning: Missing files for {symbol} at {file_master} or {file_backup}")
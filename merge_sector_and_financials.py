import pandas as pd
import numpy as np
import os
from datetime import datetime

def extract_gvkey_from_id(company_id):
    """Extract gvkey from id format like '100022_01W' -> 100022"""
    if pd.isna(company_id):
        return None
    parts = str(company_id).split('_')
    if len(parts) >= 1:
        try:
            return int(parts[0])
        except ValueError:
            return None
    return None

def convert_date_to_datetime(date_val):
    """Convert date to datetime for comparison"""
    if pd.isna(date_val):
        return None
    
    # Handle string format "YYYY-MM-DD"
    if isinstance(date_val, str) and '-' in date_val:
        return pd.to_datetime(date_val)
    
    # Handle integer format YYYYMMDD
    if isinstance(date_val, (int, float)):
        date_str = str(int(date_val))
        if len(date_str) == 8:
            return pd.to_datetime(date_str, format='%Y%m%d')
    
    return pd.to_datetime(date_val)

def extract_sector_from_gics(gics_code):
    """Extract sector from GICS code (first 2 digits of 8-digit code)"""
    if pd.isna(gics_code):
        return None
    
    # Convert to string
    gics_str = str(int(gics_code)) if isinstance(gics_code, (int, float)) else str(gics_code)
    
    # Pad to 8 digits if needed
    if len(gics_str) < 8:
        gics_str = gics_str.zfill(8)
    
    # Extract first 2 digits as sector
    if len(gics_str) >= 2:
        sector_code = gics_str[:2]
        return sector_code
    
    return None

def load_and_prepare_financial_data(parquet_file):
    """Load parquet file and prepare it for merging"""
    print(f"Loading {parquet_file}...")
    
    # Read parquet file (in chunks if too large)
    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        print(f"Error loading parquet: {e}")
        return None
    
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)[:15]}... (total: {len(df.columns)})")
    
    # Extract sector from gics
    if 'gics' in df.columns:
        print("  Extracting sector from gics column...")
        df['sector'] = df['gics'].apply(extract_sector_from_gics)
        print(f"  Sectors extracted: {df['sector'].notna().sum()}")
    else:
        print("  Warning: 'gics' column not found!")
        df['sector'] = None
    
    # Compute log(market_cap) - check for market_cap, me, or market_equity
    market_cap_col = None
    for col in ['market_cap', 'me', 'market_equity']:
        if col in df.columns:
            market_cap_col = col
            break
    
    if market_cap_col:
        print(f"  Computing log(market_cap) from '{market_cap_col}' column...")
        df['log_market_cap'] = np.log1p(df[market_cap_col].fillna(0))  # log1p handles zeros better
        # Replace log(1) with NaN for zero/negative values
        df.loc[df[market_cap_col] <= 0, 'log_market_cap'] = np.nan
        print(f"  log(market_cap) computed: {df['log_market_cap'].notna().sum()} non-null values")
    else:
        print("  Warning: No market cap column found (checked: market_cap, me, market_equity)")
        df['log_market_cap'] = None
    
    # Check for required columns
    required_cols = ['ret_12_1', 'niq_su', 'ivol_ff3_21d', 'niq_at']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  Warning: Missing columns: {missing_cols}")
    
    # Check for excntry (country)
    country_col = None
    for col in ['excntry', 'excntry', 'country', 'excntry_code']:
        if col in df.columns:
            country_col = col
            break
    
    if country_col:
        print(f"  Found country column: '{country_col}'")
        df['excntry'] = df[country_col]  # Standardize to excntry
    else:
        print("  Warning: No country column found")
        df['excntry'] = None
    
    # Convert date to datetime
    if 'date' in df.columns:
        df['date_dt'] = df['date'].apply(convert_date_to_datetime)
        print(f"  Date range: {df['date_dt'].min()} to {df['date_dt'].max()}")
    else:
        print("  Warning: 'date' column not found!")
        return None
    
    # Ensure gvkey is numeric
    if 'gvkey' in df.columns:
        df['gvkey'] = pd.to_numeric(df['gvkey'], errors='coerce')
    else:
        print("  Error: 'gvkey' column not found!")
        return None
    
    # Sort by gvkey and date for efficient lookup
    df = df.sort_values(['gvkey', 'date_dt'])
    
    return df

def find_matching_financial_data(df_financial, gvkey, target_date):
    """Find most recent financial data for gvkey with date <= target_date"""
    if gvkey is None or target_date is None:
        return None, None, None, None, None, None, None
    
    target_dt = convert_date_to_datetime(target_date)
    if target_dt is None:
        return None, None, None, None, None, None, None
    
    # Filter by gvkey and date <= target_date
    matches = df_financial[
        (df_financial['gvkey'] == gvkey) & 
        (df_financial['date_dt'] <= target_dt)
    ]
    
    if len(matches) == 0:
        return None, None, None, None, None, None, None
    
    # Get most recent match (last due to sorting)
    match = matches.iloc[-1]
    
    return (
        match.get('sector', None),
        match.get('ret_12_1', None),
        match.get('log_market_cap', None),
        match.get('niq_su', None),
        match.get('ivol_ff3_21d', None),
        match.get('niq_at', None),
        match.get('excntry', None)
    )

def merge_data_to_transitions(transitions_dir, financial_df, output_dir):
    """Merge financial data columns into transition files"""
    
    # Get all transition files
    transition_files = sorted([f for f in os.listdir(transitions_dir) 
                             if f.startswith('transitions_') and f.endswith('.csv')])
    
    print(f"\nProcessing {len(transition_files)} transition files...")
    
    for filename in transition_files:
        print(f"  Processing {filename}...", end=' ')
        
        # Load transition file
        df_trans = pd.read_csv(os.path.join(transitions_dir, filename))
        
        # Extract gvkey from id column
        df_trans['gvkey_extracted'] = df_trans['id'].apply(extract_gvkey_from_id)
        
        # Convert date to datetime
        df_trans['date_dt'] = df_trans['date'].apply(convert_date_to_datetime)
        
        # Find matching financial data for each row
        sectors = []
        ret_12_1_vals = []
        log_market_cap_vals = []
        niq_su_vals = []
        ivol_ff3_21d_vals = []
        niq_at_vals = []
        excntry_vals = []
        
        for idx, row in df_trans.iterrows():
            gvkey = row['gvkey_extracted']
            target_date = row['date_dt']
            
            sector, ret_12_1, log_market_cap, niq_su, ivol_ff3_21d, niq_at, excntry = find_matching_financial_data(
                financial_df, gvkey, target_date
            )
            
            sectors.append(sector)
            ret_12_1_vals.append(ret_12_1)
            log_market_cap_vals.append(log_market_cap)
            niq_su_vals.append(niq_su)
            ivol_ff3_21d_vals.append(ivol_ff3_21d)
            niq_at_vals.append(niq_at)
            excntry_vals.append(excntry)
        
        # Add new columns
        df_trans['sector'] = sectors
        df_trans['ret_12_1'] = ret_12_1_vals
        df_trans['log_market_cap'] = log_market_cap_vals
        df_trans['niq_su'] = niq_su_vals
        df_trans['ivol_ff3_21d'] = ivol_ff3_21d_vals
        df_trans['niq_at'] = niq_at_vals
        df_trans['excntry'] = excntry_vals
        
        # Remove temporary columns
        df_trans = df_trans.drop(columns=['gvkey_extracted', 'date_dt'])
        
        # Count matches
        matched = pd.Series(ret_12_1_vals).notna().sum()
        print(f"{matched}/{len(df_trans)} matched ({matched/len(df_trans)*100:.1f}%)")
        
        # Save enhanced file
        output_path = os.path.join(output_dir, filename)
        df_trans.to_csv(output_path, index=False)
    
    print(f"\nCompleted! Enhanced files saved to {output_dir}")

if __name__ == "__main__":
    # Set paths
    transitions_dir = "Position_Changes_With_Links"
    parquet_file = "ret_sample_preprocessed_indneu.parquet"
    output_dir = "Position_Changes_With_Links"
    
    # Validate files exist
    if not os.path.exists(transitions_dir):
        print(f"Error: {transitions_dir} directory not found!")
        exit(1)
    
    if not os.path.exists(parquet_file):
        print(f"Error: {parquet_file} not found!")
        exit(1)
    
    # Load financial data
    financial_df = load_and_prepare_financial_data(parquet_file)
    
    if financial_df is None:
        print("Error: Failed to load financial data!")
        exit(1)
    
    # Merge into transitions
    merge_data_to_transitions(transitions_dir, financial_df, output_dir)





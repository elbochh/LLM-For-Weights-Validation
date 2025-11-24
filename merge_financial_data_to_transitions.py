import pandas as pd
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

def find_matching_financial_data(df_financial, gvkey, target_date):
    """Find most recent financial data for gvkey with date <= target_date"""
    if gvkey is None or target_date is None:
        return None, None, None, None, None, None
    
    target_dt = convert_date_to_datetime(target_date)
    if target_dt is None:
        return None, None, None, None, None, None
    
    # Filter by gvkey and date <= target_date
    matches = df_financial[
        (df_financial['gvkey'] == gvkey) & 
        (df_financial['date_dt'] <= target_dt)
    ]
    
    if len(matches) == 0:
        return None, None, None, None, None, None
    
    # Get most recent match (last due to sorting)
    match = matches.iloc[-1]
    
    return (
        match.get('ret_12_1', None),
        match.get('sale_gr1', None),
        match.get('ivol_ff3_21d', None),
        match.get('niq_su', None),
        match.get('excntry', None),
        match.get('industry', None)  # Will be None if column doesn't exist
    )

def load_and_prepare_financial_data(financial_file):
    """Load and prepare financial data for efficient lookup"""
    print(f"Loading {financial_file}...")
    df = pd.read_csv(financial_file)
    
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)[:10]}... (total: {len(df.columns)})")
    
    # Convert date to datetime
    df['date_dt'] = df['date'].apply(convert_date_to_datetime)
    
    # Ensure gvkey is integer
    df['gvkey'] = pd.to_numeric(df['gvkey'], errors='coerce')
    
    # Sort by gvkey and date for efficient lookup
    df = df.sort_values(['gvkey', 'date_dt'])
    
    print(f"  Date range: {df['date_dt'].min()} to {df['date_dt'].max()}")
    
    return df

def merge_financial_data_to_transitions(transitions_dir, financial_df, output_dir):
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
        ret_12_1_vals = []
        sale_gr1_vals = []
        ivol_ff3_21d_vals = []
        niq_su_vals = []
        excntry_vals = []
        industry_vals = []
        
        for idx, row in df_trans.iterrows():
            gvkey = row['gvkey_extracted']
            target_date = row['date_dt']
            
            ret_12_1, sale_gr1, ivol_ff3_21d, niq_su, excntry, industry = find_matching_financial_data(
                financial_df, gvkey, target_date
            )
            
            ret_12_1_vals.append(ret_12_1)
            sale_gr1_vals.append(sale_gr1)
            ivol_ff3_21d_vals.append(ivol_ff3_21d)
            niq_su_vals.append(niq_su)
            excntry_vals.append(excntry)
            industry_vals.append(industry)
        
        # Add new columns
        df_trans['ret_12_1'] = ret_12_1_vals
        df_trans['sale_gr1'] = sale_gr1_vals
        df_trans['ivol_ff3_21d'] = ivol_ff3_21d_vals
        df_trans['niq_su'] = niq_su_vals
        df_trans['excntry'] = excntry_vals
        df_trans['industry'] = industry_vals
        
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
    financial_file = "ret_sample_update.csv"
    output_dir = "Position_Changes_With_Links"
    
    # Validate files exist
    if not os.path.exists(transitions_dir):
        print(f"Error: {transitions_dir} directory not found!")
        exit(1)
    
    if not os.path.exists(financial_file):
        print(f"Error: {financial_file} not found!")
        exit(1)
    
    # Load financial data
    financial_df = load_and_prepare_financial_data(financial_file)
    
    # Merge into transitions
    merge_financial_data_to_transitions(transitions_dir, financial_df, output_dir)


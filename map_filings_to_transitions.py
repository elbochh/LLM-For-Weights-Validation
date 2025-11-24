import pickle
import pandas as pd
import os
import numpy as np
from datetime import datetime
import shutil
import re

def convert_date_to_datetime(date_str):
    """Convert date string to datetime for comparison"""
    if isinstance(date_str, str):
        # Handle YYYY-MM-DD format
        if '-' in date_str:
            return pd.to_datetime(date_str)
        # Handle YYYYMMDD format
        elif len(date_str) == 8:
            return pd.to_datetime(date_str, format='%Y%m%d')
    return pd.to_datetime(date_str)

def build_filing_index(text_data_dir):
    """Build consolidated index of all filings from pkl files"""
    print("Building filing index from pkl files...")
    
    all_filings = []
    pkl_files = sorted([f for f in os.listdir(text_data_dir) if f.endswith('.pkl')])
    
    for pkl_file in pkl_files:
        year = int(pkl_file.replace('text_us_', '').replace('.pkl', ''))
        filepath = os.path.join(text_data_dir, pkl_file)
        
        print(f"  Loading {pkl_file}...", end=' ')
        with open(filepath, 'rb') as f:
            df = pickle.load(f)
        
        # Add year column and reset index to preserve filing_id
        df = df.reset_index()
        df['filing_year'] = year
        df['filing_id'] = df['index']  # Original index becomes filing_id
        df = df.drop(columns=['index'])
        
        # Convert date to datetime for comparison
        df['filing_date_dt'] = df['date'].apply(convert_date_to_datetime)
        
        # Select relevant columns
        df_filings = df[['filing_id', 'filing_date_dt', 'filing_year', 'date', 'cik', 'gvkey', 'file_type']].copy()
        
        # Convert cik and gvkey to numeric for matching
        df_filings['cik'] = pd.to_numeric(df_filings['cik'], errors='coerce')
        df_filings['gvkey'] = pd.to_numeric(df_filings['gvkey'], errors='coerce')
        
        all_filings.append(df_filings)
        print(f"{len(df_filings)} filings")
    
    # Concatenate all filings
    print("  Consolidating all filings...")
    consolidated = pd.concat(all_filings, ignore_index=True)
    
    # Sort by date for efficient lookup
    consolidated = consolidated.sort_values(['filing_date_dt', 'filing_id'], ascending=[True, True])
    
    print(f"Total filings indexed: {len(consolidated)}")
    print(f"Date range: {consolidated['filing_date_dt'].min()} to {consolidated['filing_date_dt'].max()}")
    
    return consolidated

def find_most_recent_filing(df_index, cik, gvkey, target_date):
    """Find most recent filing for a company with date <= target_date
    Returns: (gvkey_from_filing, filing_date) or (None, None) if not found
    """
    # Convert target_date to datetime if needed
    target_dt = convert_date_to_datetime(target_date)
    
    # Try to match by cik first (preferred)
    if pd.notna(cik):
        cik_matches = df_index[(df_index['cik'] == cik) & (df_index['filing_date_dt'] <= target_dt)]
        if len(cik_matches) > 0:
            # Get most recent (last due to sorting)
            match = cik_matches.iloc[-1]
            return match['gvkey'], match['date']
    
    # Fallback to gvkey if cik not available or no match
    if pd.notna(gvkey):
        gvkey_matches = df_index[(df_index['gvkey'] == gvkey) & (df_index['filing_date_dt'] <= target_dt)]
        if len(gvkey_matches) > 0:
            # Get most recent (last due to sorting)
            match = gvkey_matches.iloc[-1]
            return match['gvkey'], match['date']
    
    # No match found
    return None, None

def extract_gvkey_iid(company_id):
    """Extract gvkey and iid from company ID format (e.g., '1690_01' or '1690_01W')"""
    parts = company_id.split('_')
    if len(parts) >= 2:
        gvkey_str = parts[0]
        iid_str = parts[1].rstrip('WC')  # Remove suffix if present
        try:
            gvkey = int(gvkey_str)
            iid = iid_str.zfill(2)  # Ensure 2-digit format
            return gvkey, iid
        except ValueError:
            return None, None
    return None, None

def load_linktable(linktable_path):
    """Load the linktable and prepare it for efficient lookup"""
    print("Loading linktable...")
    df_link = pd.read_csv(linktable_path)
    
    # Convert datadate to datetime for comparison
    df_link['datadate'] = pd.to_datetime(df_link['datadate'])
    
    # Ensure gvkey is integer and iid is string
    df_link['gvkey'] = df_link['gvkey'].astype(int)
    df_link['iid'] = df_link['iid'].astype(str).str.zfill(2)  # Ensure 2-digit format
    
    # Sort by gvkey, iid, datadate for efficient lookup
    df_link = df_link.sort_values(['gvkey', 'iid', 'datadate'])
    
    print(f"Loaded {len(df_link)} linktable entries")
    print(f"Date range: {df_link['datadate'].min()} to {df_link['datadate'].max()}")
    
    return df_link

def find_linktable_match(df_link, gvkey, iid, target_date):
    """Find the most recent linktable entry for gvkey+iid with datadate <= target_date"""
    if gvkey is None or iid is None:
        return None, None
    
    target_dt = convert_date_to_datetime(target_date)
    
    # Filter linktable: matching gvkey, iid, and datadate <= target_date
    matches = df_link[
        (df_link['gvkey'] == gvkey) & 
        (df_link['iid'] == iid) & 
        (df_link['datadate'] <= target_dt)
    ]
    
    if len(matches) == 0:
        return None, None
    
    # Get the most recent match (last one due to sorting)
    most_recent = matches.iloc[-1]
    return most_recent['gvkey'], most_recent['cik']

def process_transitions_with_filings(transitions_dir, filing_index, linktable, output_dir):
    """Process all transition files and add cik/gvkey and filing identifiers"""
    
    # Delete old output directory if it exists
    if os.path.exists(output_dir):
        print(f"\nDeleting old {output_dir} folder...")
        shutil.rmtree(output_dir)
    
    # Create new output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all transition files
    transition_files = sorted([f for f in os.listdir(transitions_dir) 
                             if f.startswith('transitions_') and f.endswith('.csv')])
    
    print(f"\nProcessing {len(transition_files)} transition files...")
    
    for filename in transition_files:
        print(f"  Processing {filename}...", end=' ')
        
        # Load transition file
        df_trans = pd.read_csv(os.path.join(transitions_dir, filename))
        
        # First, add cik and gvkey from linktable
        gvkeys = []
        ciks = []
        
        target_date = df_trans['date'].iloc[0]
        
        for company_id in df_trans['id']:
            gvkey, iid = extract_gvkey_iid(company_id)
            matched_gvkey, matched_cik = find_linktable_match(linktable, gvkey, iid, target_date)
            gvkeys.append(matched_gvkey if matched_gvkey is not None else None)
            ciks.append(matched_cik if matched_cik is not None else None)
        
        df_trans['gvkey'] = gvkeys
        df_trans['cik'] = ciks
        
        # Convert cik and gvkey to numeric for matching
        df_trans['cik'] = pd.to_numeric(df_trans['cik'], errors='coerce')
        df_trans['gvkey'] = pd.to_numeric(df_trans['gvkey'], errors='coerce')
        
        # Find filing for each row and get date from pickle data
        filing_dates = []
        
        for idx, row in df_trans.iterrows():
            cik = row['cik'] if pd.notna(row['cik']) else None
            gvkey = row['gvkey'] if pd.notna(row['gvkey']) else None
            target_date = row['date']
            
            _, filing_date = find_most_recent_filing(
                filing_index, cik, gvkey, target_date
            )
            
            filing_dates.append(filing_date)
        
        # Add filing_date column from pickle data
        df_trans['filing_date'] = filing_dates
        
        # Remove cik column (only needed for matching, not for output)
        # Keep only: id, weight_change, date, gvkey, filing_date
        df_trans = df_trans[['id', 'weight_change', 'date', 'gvkey', 'filing_date']]
        
        # Count matches
        filing_matched = pd.Series(filing_dates).notna().sum()
        print(f"{filing_matched}/{len(df_trans)} filings matched ({filing_matched/len(df_trans)*100:.1f}%)")
        
        # Save enhanced file
        output_path = os.path.join(output_dir, filename)
        df_trans.to_csv(output_path, index=False)
    
    print(f"\nCompleted! Enhanced files saved to {output_dir}")

if __name__ == "__main__":
    # Set paths
    text_data_dir = "text_data"
    transitions_dir = "Position_Changes"
    linktable_path = "cik_gvkey_linktable_USA_only.csv"
    output_dir = "Position_Changes_With_Links"
    
    # Validate directories exist
    if not os.path.exists(text_data_dir):
        print(f"Error: {text_data_dir} directory not found!")
        exit(1)
    
    if not os.path.exists(transitions_dir):
        print(f"Error: {transitions_dir} directory not found!")
        exit(1)
    
    if not os.path.exists(linktable_path):
        print(f"Error: {linktable_path} not found!")
        exit(1)
    
    # Load linktable
    linktable = load_linktable(linktable_path)
    
    # Build filing index
    filing_index = build_filing_index(text_data_dir)
    
    # Process transitions
    process_transitions_with_filings(transitions_dir, filing_index, linktable, output_dir)


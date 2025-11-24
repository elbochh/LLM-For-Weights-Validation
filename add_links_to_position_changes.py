import pandas as pd
import os
import re
from datetime import datetime

def parse_month_from_filename(filename):
    """Extract YYYY-MM from filename like transitions_2017-01_to_2017-02.csv"""
    match = re.search(r'(\d{4}-\d{2})_to_(\d{4}-\d{2})', filename)
    if match:
        return match.group(1), match.group(2)  # month_t, month_t1
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

def extract_gvkey_iid(company_id):
    """Extract gvkey and iid from company ID format (e.g., '1690_01' or '1690_01W')"""
    # Handle suffixes like 'W', 'C' at the end (e.g., '1690_01W' -> gvkey=1690, iid='01')
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

def find_linktable_match(df_link, gvkey, iid, target_date):
    """Find the most recent linktable entry for gvkey+iid with datadate <= target_date"""
    if gvkey is None or iid is None:
        return None, None
    
    # Filter linktable: matching gvkey, iid, and datadate <= target_date
    matches = df_link[
        (df_link['gvkey'] == gvkey) & 
        (df_link['iid'] == iid) & 
        (df_link['datadate'] <= target_date)
    ]
    
    if len(matches) == 0:
        return None, None
    
    # Get the most recent match (last one due to sorting)
    most_recent = matches.iloc[-1]
    return most_recent['gvkey'], most_recent['cik']

def add_links_to_transitions(transitions_dir, linktable_path, output_dir):
    """Add cik and gvkey columns to all transition files"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load linktable once
    df_link = load_linktable(linktable_path)
    
    # Get all transition files
    transition_files = [f for f in os.listdir(transitions_dir) 
                       if f.startswith('transitions_') and f.endswith('.csv')]
    
    transition_files.sort()  # Sort alphabetically (which is chronologically)
    
    print(f"\nFound {len(transition_files)} transition files to process")
    
    # Process each transition file
    for filename in transition_files:
        month_t, month_t1 = parse_month_from_filename(filename)
        if month_t is None:
            print(f"  Skipping {filename} (could not parse date)")
            continue
        
        print(f"Processing {filename} ({month_t} -> {month_t1})")
        
        # Load transition file
        df_trans = pd.read_csv(os.path.join(transitions_dir, filename))
        
        # Convert date to datetime for comparison
        target_date = pd.to_datetime(df_trans['date'].iloc[0])
        
        # Extract gvkey and iid from id, then find matches
        gvkeys = []
        ciks = []
        
        for company_id in df_trans['id']:
            gvkey, iid = extract_gvkey_iid(company_id)
            matched_gvkey, matched_cik = find_linktable_match(df_link, gvkey, iid, target_date)
            
            gvkeys.append(matched_gvkey if matched_gvkey is not None else None)
            ciks.append(matched_cik if matched_cik is not None else None)
        
        # Add columns to DataFrame
        df_trans['gvkey'] = gvkeys
        df_trans['cik'] = ciks
        
        # Count matches
        matched_count = df_trans['gvkey'].notna().sum()
        print(f"  Matched {matched_count}/{len(df_trans)} companies ({matched_count/len(df_trans)*100:.1f}%)")
        
        # Save enhanced file
        output_path = os.path.join(output_dir, filename)
        df_trans.to_csv(output_path, index=False)
    
    print(f"\nCompleted! Enhanced files saved to {output_dir}")

if __name__ == "__main__":
    # Set paths
    transitions_dir = "Position_Changes"
    linktable_path = "cik_gvkey_linktable_USA_only.csv"
    output_dir = "Position_Changes_With_Links"
    
    # Validate directories exist
    if not os.path.exists(transitions_dir):
        print(f"Error: {transitions_dir} directory not found!")
        exit(1)
    
    if not os.path.exists(linktable_path):
        print(f"Error: {linktable_path} not found!")
        exit(1)
    
    # Add links to transitions
    add_links_to_transitions(transitions_dir, linktable_path, output_dir)





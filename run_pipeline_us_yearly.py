import pandas as pd
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Import existing modules
import sys
sys.path.append('.')

from map_filings_to_transitions import (
    convert_date_to_datetime, build_filing_index, 
    find_most_recent_filing, load_linktable, find_linktable_match
)
import re
from merge_sector_and_financials import (
    load_and_prepare_financial_data, merge_data_to_transitions
)
from pair_stocks import pair_stocks_by_similarity
from filing_retriever import FilingRetriever
from macro_loader import load_macro_report
from rag_scorer import RAGScorer

def extract_gvkey_from_id_us(company_id):
    """Extract gvkey from US id format like 'comp_018182_01' -> 18182 or 'crsp_10232' -> 10232"""
    if pd.isna(company_id):
        return None
    parts = str(company_id).split('_')
    if len(parts) >= 2:
        # Format: comp_018182_01 or crsp_10232
        # gvkey is the second part (index 1)
        try:
            # Convert directly - int() will handle leading zeros
            return int(parts[1])
        except ValueError:
            return None
    return None

def extract_gvkey_iid_us(company_id):
    """Extract gvkey and iid from US company ID format (e.g., 'comp_018182_01' or 'crsp_10232')"""
    if pd.isna(company_id):
        return None, None
    parts = str(company_id).split('_')
    if len(parts) >= 2:
        try:
            # gvkey is the second part (int() handles leading zeros)
            gvkey = int(parts[1])
            
            # iid is the third part if it exists (for comp_ format), otherwise None
            if len(parts) >= 3:
                iid_str = parts[2].rstrip('WC')  # Remove C/W suffix if present
                iid = iid_str.zfill(2) if iid_str else None
            else:
                iid = None  # crsp format doesn't have iid
            
            return gvkey, iid
        except ValueError:
            return None, None
    return None, None

def parse_month_from_filename(filename):
    """Parse month from filename like 'weights_2017-01.csv'"""
    match = re.search(r'(\d{4}-\d{2})', filename)
    if match:
        return match.group(1)
    return None

def get_sorted_weight_files(weights_dir):
    """Get sorted list of weight files"""
    weight_files = []
    for file in os.listdir(weights_dir):
        if file.startswith('weights_') and file.endswith('.csv'):
            month = parse_month_from_filename(file)
            if month:
                weight_files.append((month, file))
    weight_files.sort(key=lambda x: x[0])
    return weight_files

def calculate_transitions(weights_dir, output_dir):
    """Calculate position changes between consecutive months"""
    os.makedirs(output_dir, exist_ok=True)
    weight_files = get_sorted_weight_files(weights_dir)
    print(f"Found {len(weight_files)} weight files")
    
    for i in range(len(weight_files) - 1):
        month_t, file_t = weight_files[i]
        month_t1, file_t1 = weight_files[i + 1]
        print(f"Processing transition: {month_t} -> {month_t1}")
        
        df_t = pd.read_csv(os.path.join(weights_dir, file_t))
        df_t1 = pd.read_csv(os.path.join(weights_dir, file_t1))
        
        weights_t = dict(zip(df_t['id'], df_t['weight']))
        weights_t1 = dict(zip(df_t1['id'], df_t1['weight']))
        
        all_ids = set(weights_t.keys()) | set(weights_t1.keys())
        date_t1 = df_t1['date'].iloc[0]
        
        transitions = []
        for company_id in sorted(all_ids):
            weight_t = weights_t.get(company_id, 0.0)
            weight_t1 = weights_t1.get(company_id, 0.0)
            weight_change = weight_t1 - weight_t
            transitions.append({
                'id': company_id,
                'weight_change': weight_change,
                'date': date_t1
            })
        
        df_transitions = pd.DataFrame(transitions)
        output_filename = f"transitions_{month_t}_to_{month_t1}.csv"
        output_path = os.path.join(output_dir, output_filename)
        df_transitions.to_csv(output_path, index=False)
        print(f"  Created {output_filename} with {len(df_transitions)} companies")
    
    print(f"\nCompleted! Created {len(weight_files) - 1} transition files in {output_dir}")

def prepare_us_weights(weights_us_file, output_dir="Weights_US"):
    """Convert weights_US.csv into monthly weight files"""
    print("=" * 80)
    print("STEP 1: Preparing US weights files")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {weights_us_file}...")
    df = pd.read_csv(weights_us_file)
    print(f"  Loaded {len(df)} rows")
    
    # Convert date and group by month
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    
    grouped = df.groupby('month')
    
    print(f"\nCreating monthly weight files...")
    for month, group in grouped:
        month_str = str(month)
        filename = f"weights_{month_str}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Select only id, weight, date columns
        output_df = group[['id', 'weight', 'date']].copy()
        output_df['date'] = output_df['date'].dt.strftime('%Y-%m-%d')
        output_df.to_csv(filepath, index=False)
        print(f"  Created {filename} with {len(output_df)} stocks")
    
    print(f"\n✓ Prepared US weights files in {output_dir}/")
    return output_dir

def process_transitions_with_filings_us(transitions_dir, filing_index, linktable, output_dir):
    """Process transitions with filings - US version with updated gvkey extraction"""
    import shutil
    
    if os.path.exists(output_dir):
        print(f"\nDeleting old {output_dir} folder...")
        shutil.rmtree(output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    transition_files = sorted([f for f in os.listdir(transitions_dir) 
                                 if f.startswith('transitions_') and f.endswith('.csv')])
    print(f"\nProcessing {len(transition_files)} transition files...")
    
    for filename in transition_files:
        print(f"  Processing {filename}...", end=' ')
        df_trans = pd.read_csv(os.path.join(transitions_dir, filename))
        
        gvkeys = []
        ciks = []
        target_date = df_trans['date'].iloc[0]
        
        for company_id in df_trans['id']:
            gvkey, iid = extract_gvkey_iid_us(company_id)
            matched_gvkey, matched_cik = find_linktable_match(linktable, gvkey, iid, target_date)
            gvkeys.append(matched_gvkey if matched_gvkey is not None else None)
            ciks.append(matched_cik if matched_cik is not None else None)
        
        df_trans['gvkey'] = gvkeys
        df_trans['cik'] = ciks
        df_trans['cik'] = pd.to_numeric(df_trans['cik'], errors='coerce')
        df_trans['gvkey'] = pd.to_numeric(df_trans['gvkey'], errors='coerce')
        
        filing_dates = []
        for idx, row in df_trans.iterrows():
            cik = row['cik'] if pd.notna(row['cik']) else None
            gvkey = row['gvkey'] if pd.notna(row['gvkey']) else None
            target_date = row['date']
            _, filing_date = find_most_recent_filing(
                filing_index, cik, gvkey, target_date
            )
            filing_dates.append(filing_date)
        
        df_trans['filing_date'] = filing_dates
        df_trans = df_trans[['id', 'weight_change', 'date', 'gvkey', 'filing_date']]
        
        filing_matched = pd.Series(filing_dates).notna().sum()
        print(f"{filing_matched}/{len(df_trans)} filings matched ({filing_matched/len(df_trans)*100:.1f}%)")
        
        output_path = os.path.join(output_dir, filename)
        df_trans.to_csv(output_path, index=False)
    
    print(f"\nCompleted! Enhanced files saved to {output_dir}")

def merge_data_to_transitions_us(transitions_dir, financial_df, output_dir):
    """Merge data to transitions - US version with updated gvkey extraction"""
    from merge_sector_and_financials import convert_date_to_datetime, find_matching_financial_data
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    transition_files = sorted([f for f in os.listdir(transitions_dir) 
                             if f.startswith('transitions_') and f.endswith('.csv')])
    
    print(f"\nProcessing {len(transition_files)} transition files...")
    
    for filename in transition_files:
        print(f"  Processing {filename}...", end=' ')
        
        df_trans = pd.read_csv(os.path.join(transitions_dir, filename))
        
        df_trans['gvkey_extracted'] = df_trans['id'].apply(extract_gvkey_from_id_us)
        df_trans['date_dt'] = df_trans['date'].apply(convert_date_to_datetime)
        
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
        
        df_trans['sector'] = sectors
        df_trans['ret_12_1'] = ret_12_1_vals
        df_trans['log_market_cap'] = log_market_cap_vals
        df_trans['niq_su'] = niq_su_vals
        df_trans['ivol_ff3_21d'] = ivol_ff3_21d_vals
        df_trans['niq_at'] = niq_at_vals
        df_trans['excntry'] = excntry_vals
        
        df_trans = df_trans.drop(columns=['gvkey_extracted', 'date_dt'])
        
        matched = pd.Series(sectors).notna().sum()
        print(f"{matched}/{len(df_trans)} matched ({matched/len(df_trans)*100:.1f}%)")
        
        output_path = os.path.join(output_dir, filename)
        df_trans.to_csv(output_path, index=False)
    
    print(f"\nCompleted! Enhanced files saved to {output_dir}")

def process_year(year, transitions_dir, text_data_dir, macro_reports_dir, 
                 api_key, output_dir, pairs_per_month=4):
    """Process all transition files for a specific year and output CSV"""
    print(f"\n{'=' * 80}")
    print(f"Processing year {year}")
    print(f"{'=' * 80}")
    
    # Get all transition files for this year
    transition_files = sorted([
        f for f in os.listdir(transitions_dir) 
        if f.startswith('transitions_') and f.endswith('.csv') and str(year) in f
    ])
    
    if not transition_files:
        print(f"No transition files found for {year}")
        return []
    
    print(f"Found {len(transition_files)} transition files for {year}")
    
    all_recommendations = []
    
    # Initialize retrievers once per year
    filing_retriever = FilingRetriever(text_data_dir, api_key=api_key, use_summarization=True)
    rag_scorer = RAGScorer(api_key)
    
    for transition_file in transition_files:
        # Load transition file
        df = pd.read_csv(os.path.join(transitions_dir, transition_file))
        
        # Filter for 2017 and onwards
        df['date_dt'] = pd.to_datetime(df['date'])
        df = df[df['date_dt'] >= pd.to_datetime('2017-01-01')]
        
        if len(df) == 0:
            continue
        
        transition_date = df['date'].iloc[0]
        
        print(f"\nProcessing {transition_file}")
        print(f"  Date: {transition_date}")
        print(f"  Stocks: {len(df)}")
        
        # Create pairings
        pairs = pair_stocks_by_similarity(df)
        print(f"  Pairs created: {len(pairs)}")
        
        if len(pairs) == 0:
            print("  No pairs found, skipping...")
            continue
        
        # Limit to pairs_per_month
        pairs = pairs[:pairs_per_month]
        print(f"  Processing only first {len(pairs)} pairs")
        
        # Load macro report
        macro_report = load_macro_report(macro_reports_dir, transition_date)
        if macro_report is None:
            macro_report = "No macro report available."
        
        # Process pairs
        for i, (pos_stock, neg_stock) in enumerate(pairs):
            try:
                print(f"  Processing pair {i+1}/{len(pairs)}: {pos_stock['id']} vs {neg_stock['id']}...", end=' ')
                
                gvkey_a = pos_stock.get('gvkey')
                gvkey_b = neg_stock.get('gvkey')
                
                mgmt_a = None
                mgmt_b = None
                
                if pd.notna(gvkey_a):
                    mgmt_a = filing_retriever.get_mgmt_text(gvkey_a, transition_date, summarize=True)
                if pd.notna(gvkey_b):
                    mgmt_b = filing_retriever.get_mgmt_text(gvkey_b, transition_date, summarize=True)
                
                company_a = pos_stock.to_dict()
                company_b = neg_stock.to_dict()
                
                result = rag_scorer.score_pair(company_a, company_b, mgmt_a, mgmt_b, macro_report)
                
                if result is None:
                    print("Failed to get score")
                    continue
                
                recommendation = {
                    'transition_file': transition_file,
                    'transition_date': transition_date,
                    'company_a_id': company_a.get('id'),
                    'company_b_id': company_b.get('id'),
                    'swap_decision': result.get('swap_decision', 0),
                    'company_a_scores': result.get('company_a_scores', {}),
                    'company_b_scores': result.get('company_b_scores', {}),
                }
                
                all_recommendations.append(recommendation)
                print(f"Swap: {result.get('swap_decision', 0)} ({result.get('urgency', 'optional')})")
                
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"Error processing pair: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save CSV for this year
    if all_recommendations:
        score_keys = ['long_term_growth', 'cash_generation', 'future_earning_yields', 'downside_risks', 'macro_sensitivity']
        
        df_recs = []
        for rec in all_recommendations:
            scores_a = rec.get('company_a_scores', {})
            scores_b = rec.get('company_b_scores', {})
            
            row = {
                'company_a_id': rec['company_a_id'],
                'company_b_id': rec['company_b_id'],
                'swap_decision': rec['swap_decision']
            }
            
            for i, score_key in enumerate(score_keys, 1):
                row[f'A_score{i}'] = scores_a.get(score_key)
            
            for i, score_key in enumerate(score_keys, 1):
                row[f'B_score{i}'] = scores_b.get(score_key)
            
            df_recs.append(row)
        
        df = pd.DataFrame(df_recs)
        csv_file = os.path.join(output_dir, f"recommendations_{year}.csv")
        df.to_csv(csv_file, index=False)
        print(f"\n✓ Saved {len(all_recommendations)} recommendations to {csv_file}")
        
        return all_recommendations
    
    return []

def main():
    """Main pipeline execution"""
    # Configuration
    weights_us_file = "weights_US.csv"
    weights_dir = "Weights_US"
    position_changes_dir = "Position_Changes_US"
    transitions_dir = "Position_Changes_With_Links_US"
    text_data_dir = "text_data"
    macro_reports_dir = "macro reports"
    linktable_path = "cik_gvkey_linktable_USA_only.csv"
    financial_parquet_file = "ret_sample_preprocessed_indneu.parquet"
    output_dir = "pair_recommendations_US"
    api_key = "yMa8EFXtJfc48PXZdVn1rt7moLUpjCTM"
    pairs_per_month = 4
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Prepare US weights files
    if not os.path.exists(weights_dir) or len(os.listdir(weights_dir)) == 0:
        prepare_us_weights(weights_us_file, weights_dir)
    else:
        print(f"Using existing {weights_dir}/ directory")
    
    # Step 2: Calculate position changes
    if not os.path.exists(position_changes_dir) or len(os.listdir(position_changes_dir)) == 0:
        print(f"\n{'=' * 80}")
        print("STEP 2: Calculating position changes")
        print(f"{'=' * 80}")
        calculate_transitions(weights_dir, position_changes_dir)
    else:
        print(f"Using existing {position_changes_dir}/ directory")
    
    # Step 3: Map filings to transitions
    if not os.path.exists(transitions_dir) or len(os.listdir(transitions_dir)) == 0:
        print(f"\n{'=' * 80}")
        print("STEP 3: Mapping filings to transitions")
        print(f"{'=' * 80}")
        from map_filings_to_transitions import build_filing_index, load_linktable
        filing_index = build_filing_index(text_data_dir)
        linktable = load_linktable(linktable_path)
        process_transitions_with_filings_us(position_changes_dir, filing_index, linktable, transitions_dir)
    else:
        print(f"Using existing {transitions_dir}/ directory")
    
    # Step 4: Merge sector and financials
    print(f"\n{'=' * 80}")
    print("STEP 4: Merging sector and financial data")
    print(f"{'=' * 80}")
    financial_df = load_and_prepare_financial_data(financial_parquet_file)
    merge_data_to_transitions_us(transitions_dir, financial_df, transitions_dir)
    
    # Step 5: Process pairs year by year
    print(f"\n{'=' * 80}")
    print("STEP 5: Processing pairs (4 per month, output by year)")
    print(f"{'=' * 80}")
    
    # Get unique years from transition files
    transition_files = sorted([
        f for f in os.listdir(transitions_dir) 
        if f.startswith('transitions_') and f.endswith('.csv')
    ])
    
    years = set()
    for f in transition_files:
        # Extract year from filename like "transitions_2017-01_to_2017-02.csv"
        parts = f.split('_')
        if len(parts) >= 2:
            year = parts[1].split('-')[0]
            try:
                years.add(int(year))
            except:
                pass
    
    years = sorted([y for y in years if y >= 2017])
    
    print(f"Processing years: {years}")
    
    for year in years:
        try:
            # Check if year already processed
            csv_file = os.path.join(output_dir, f"recommendations_{year}.csv")
            if os.path.exists(csv_file):
                print(f"\nYear {year} already processed, skipping...")
                continue
            
            recommendations = process_year(
                year, transitions_dir, text_data_dir, macro_reports_dir,
                api_key, output_dir, pairs_per_month
            )
            print(f"\n✓ Completed year {year}: {len(recommendations)} recommendations")
        except Exception as e:
            print(f"\n✗ Error processing year {year}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Continuing with next year...")
            continue
    
    print(f"\n{'=' * 80}")
    print("PIPELINE COMPLETE!")
    print(f"{'=' * 80}")
    print(f"Yearly CSV files saved to: {output_dir}/")

if __name__ == "__main__":
    main()


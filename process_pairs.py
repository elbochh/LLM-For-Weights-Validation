import pandas as pd
import os
import json
import time
from datetime import datetime
from filing_retriever import FilingRetriever
from macro_loader import load_macro_report
from pair_stocks import pair_stocks_by_similarity
from rag_scorer import RAGScorer

def process_transition_file(transition_file, transitions_dir, text_data_dir, 
                           macro_reports_dir, api_key, output_dir):
    """Process a single transition file and generate pair recommendations"""
    
    # Load transition file
    df = pd.read_csv(os.path.join(transitions_dir, transition_file))
    
    # Filter for 2017 and onwards
    df['date_dt'] = pd.to_datetime(df['date'])
    df = df[df['date_dt'] >= pd.to_datetime('2017-01-01')]
    
    if len(df) == 0:
        return []
    
    # Get transition date
    transition_date = df['date'].iloc[0]
    
    print(f"\nProcessing {transition_file}")
    print(f"  Date: {transition_date}")
    print(f"  Stocks: {len(df)}")
    
    # Create pairings
    pairs = pair_stocks_by_similarity(df)
    print(f"  Pairs created: {len(pairs)}")
    
    if len(pairs) == 0:
        print("  No pairs found, skipping...")
        return []
    
    # Initialize retrievers with summarization enabled
    filing_retriever = FilingRetriever(text_data_dir, api_key=api_key, use_summarization=True)
    rag_scorer = RAGScorer(api_key)
    
    print(f"  Using summarized management text (full-text processing enabled)")
    
    # Load macro report
    macro_report = load_macro_report(macro_reports_dir, transition_date)
    if macro_report is None:
        print(f"  Warning: No macro report found for {transition_date}")
        macro_report = "No macro report available."
    else:
        print(f"  Macro report loaded")
    
    # Process each pair
    recommendations = []
    
    for i, (pos_stock, neg_stock) in enumerate(pairs):
        print(f"  Processing pair {i+1}/{len(pairs)}: {pos_stock['id']} vs {neg_stock['id']}...", end=' ')
        
        # Get gvkeys
        gvkey_a = pos_stock.get('gvkey')
        gvkey_b = neg_stock.get('gvkey')
        
        # Retrieve summarized mgmt text (summarization happens automatically via FilingRetriever)
        mgmt_a = None
        mgmt_b = None
        
        if pd.notna(gvkey_a):
            mgmt_a = filing_retriever.get_mgmt_text(gvkey_a, transition_date, summarize=True)
        if pd.notna(gvkey_b):
            mgmt_b = filing_retriever.get_mgmt_text(gvkey_b, transition_date, summarize=True)
        
        # Convert stock rows to dicts for easier handling
        company_a = pos_stock.to_dict()
        company_b = neg_stock.to_dict()
        
        # Score pair
        result = rag_scorer.score_pair(company_a, company_b, mgmt_a, mgmt_b, macro_report)
        
        if result is None:
            print("Failed to get score")
            continue
        
        # Store recommendation
        recommendation = {
            'transition_file': transition_file,
            'transition_date': transition_date,
            'company_a_id': company_a.get('id'),
            'company_b_id': company_b.get('id'),
            'company_a_weight': company_a.get('weight_change'),
            'company_b_weight': company_b.get('weight_change'),
            'company_a_scores': result.get('company_a_scores', {}),
            'company_b_scores': result.get('company_b_scores', {}),
            'swap_decision': result.get('swap_decision', 0),
            'urgency': result.get('urgency', 'optional'),
            'justification': result.get('justification', ''),
            'has_mgmt_a': mgmt_a is not None,
            'has_mgmt_b': mgmt_b is not None
        }
        
        recommendations.append(recommendation)
        print(f"Swap: {result.get('swap_decision', 0)} ({result.get('urgency', 'optional')})")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    return recommendations

def main():
    # Configuration
    transitions_dir = "Position_Changes_With_Links"
    text_data_dir = "text_data"
    macro_reports_dir = "macro reports"
    output_dir = "pair_recommendations"
    api_key = "yMa8EFXtJfc48PXZdVn1rt7moLUpjCTM"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all transition files starting from 2017
    all_files = sorted([f for f in os.listdir(transitions_dir) 
                       if f.startswith('transitions_') and f.endswith('.csv')])
    
    # Filter for 2017 onwards
    transition_files = [f for f in all_files if '2017' in f]  
                       # any(year in f for year in ['2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025'])]
    
    print(f"Found {len(transition_files)} transition files from 2017 onwards")
    
    all_recommendations = []
    #
    # Process each file
    for transition_file in transition_files:
        try:
            recommendations = process_transition_file(
                transition_file, transitions_dir, text_data_dir,
                macro_reports_dir, api_key, output_dir
            )
            all_recommendations.extend(recommendations)
        except Exception as e:
            print(f"Error processing {transition_file}: {e}")
            continue
    
    # Save all recommendations
    if all_recommendations:
        # Save as JSON
        output_file = os.path.join(output_dir, "all_recommendations.json")
        with open(output_file, 'w') as f:
            json.dump(all_recommendations, f, indent=2)
        
        # Save as CSV with simplified format
        # Columns: company_a_id, company_b_id, swap_decision, A_score1-5, B_score1-5
        # Score mapping: 1=long_term_growth, 2=cash_generation, 3=future_earning_yields, 4=downside_risks, 5=macro_sensitivity
        df_recs = []
        for rec in all_recommendations:
            # Get scores for both companies
            scores_a = rec.get('company_a_scores', {})
            scores_b = rec.get('company_b_scores', {})
            
            # Define score order: long_term_growth, cash_generation, future_earning_yields, downside_risks, macro_sensitivity
            score_keys = ['long_term_growth', 'cash_generation', 'future_earning_yields', 'downside_risks', 'macro_sensitivity']
            
            row = {
                'company_a_id': rec['company_a_id'],
                'company_b_id': rec['company_b_id'],
                'swap_decision': rec['swap_decision']
            }
            
            # Add Company A scores as A_score1, A_score2, etc.
            for i, score_key in enumerate(score_keys, 1):
                row[f'A_score{i}'] = scores_a.get(score_key)
            
            # Add Company B scores as B_score1, B_score2, etc.
            for i, score_key in enumerate(score_keys, 1):
                row[f'B_score{i}'] = scores_b.get(score_key)
            
            df_recs.append(row)
        
        df = pd.DataFrame(df_recs)
        csv_file = os.path.join(output_dir, "all_recommendations.csv")
        df.to_csv(csv_file, index=False)
        
        print(f"\n\nCompleted! Saved {len(all_recommendations)} recommendations")
        print(f"  JSON: {output_file}")
        print(f"  CSV: {csv_file}")
        
        # Summary statistics
        swaps = sum(1 for r in all_recommendations if r['swap_decision'] == 1)
        required = sum(1 for r in all_recommendations if r['urgency'] == 'required')
        print(f"\nSummary:")
        print(f"  Total pairs: {len(all_recommendations)}")
        print(f"  Swaps recommended: {swaps} ({swaps/len(all_recommendations)*100:.1f}%)")
        print(f"  Required swaps: {required}")
    else:
        print("\nNo recommendations generated.")

if __name__ == "__main__":
    main()


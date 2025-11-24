import pandas as pd
import os
import re
from pathlib import Path


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
    """
    Calculate position changes between consecutive months.
    
    Args:
        weights_dir: Directory containing monthly weight CSV files (e.g., weights_2017-01.csv)
        output_dir: Directory to save transition CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    weight_files = get_sorted_weight_files(weights_dir)
    
    if len(weight_files) < 2:
        print(f"Error: Need at least 2 weight files to calculate transitions. Found {len(weight_files)}")
        return
    
    print(f"Found {len(weight_files)} weight files")
    print(f"Processing {len(weight_files) - 1} transitions...\n")
    
    for i in range(len(weight_files) - 1):
        month_t, file_t = weight_files[i]
        month_t1, file_t1 = weight_files[i + 1]
        print(f"Processing transition: {month_t} -> {month_t1}")
        
        # Load weight files
        df_t = pd.read_csv(os.path.join(weights_dir, file_t))
        df_t1 = pd.read_csv(os.path.join(weights_dir, file_t1))
        
        # Create dictionaries for fast lookup
        weights_t = dict(zip(df_t['id'], df_t['weight']))
        weights_t1 = dict(zip(df_t1['id'], df_t1['weight']))
        
        # Get all unique IDs from both months
        all_ids = set(weights_t.keys()) | set(weights_t1.keys())
        
        # Get date from t+1 (the transition date)
        date_t1 = df_t1['date'].iloc[0]
        
        # Calculate transitions
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
        
        # Save transition file
        df_transitions = pd.DataFrame(transitions)
        output_filename = f"transitions_{month_t}_to_{month_t1}.csv"
        output_path = os.path.join(output_dir, output_filename)
        df_transitions.to_csv(output_path, index=False)
        print(f"  ✓ Created {output_filename} with {len(df_transitions)} companies")
    
    print(f"\n{'='*60}")
    print(f"Completed! Created {len(weight_files) - 1} transition files in {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Default directories
    weights_dir = "Weights"
    output_dir = "Position_Changes"
    
    # Check if weights directory exists
    if not os.path.exists(weights_dir):
        print(f"Error: {weights_dir} directory not found!")
        print(f"Please ensure the weights directory exists and contains files like 'weights_YYYY-MM.csv'")
        exit(1)
    
    # Calculate transitions
    calculate_transitions(weights_dir, output_dir)
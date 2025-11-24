import os
from datetime import datetime

def load_macro_report(macro_reports_dir, transition_date):
    """
    Load macro report for a given transition date
    
    Args:
        macro_reports_dir: Directory containing macro reports
        transition_date: Date in format YYYY-MM-DD or YYYY-MM
    
    Returns:
        Macro report text content or None if not found
    """
    # Extract YYYY-MM from date
    if isinstance(transition_date, str):
        if '-' in transition_date:
            parts = transition_date.split('-')
            if len(parts) >= 2:
                year_month = f"{parts[0]}-{parts[1]}"
            else:
                return None
        else:
            return None
    else:
        # If it's a datetime object
        year_month = transition_date.strftime('%Y-%m')
    
    # Construct file path
    report_file = os.path.join(macro_reports_dir, f"{year_month}.md")
    
    if not os.path.exists(report_file):
        # Try to find most recent available report
        return _find_most_recent_report(macro_reports_dir, year_month)
    
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Error loading macro report {report_file}: {e}")
        return None

def _find_most_recent_report(macro_reports_dir, target_year_month):
    """Find the most recent macro report available"""
    try:
        # Get all markdown files
        all_files = [f for f in os.listdir(macro_reports_dir) if f.endswith('.md')]
        
        if not all_files:
            return None
        
        # Parse dates and find closest one before or equal to target
        target_dt = datetime.strptime(target_year_month, '%Y-%m')
        
        best_file = None
        best_diff = None
        
        for filename in all_files:
            try:
                date_str = filename.replace('.md', '')
                file_dt = datetime.strptime(date_str, '%Y-%m')
                
                # Only consider reports on or before target date
                if file_dt <= target_dt:
                    diff = (target_dt - file_dt).days
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        best_file = filename
            except:
                continue
        
        if best_file:
            report_file = os.path.join(macro_reports_dir, best_file)
            with open(report_file, 'r', encoding='utf-8') as f:
                return f.read()
        
        return None
    except Exception as e:
        print(f"Error finding most recent report: {e}")
        return None





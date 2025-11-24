import pickle
import pandas as pd
import os
from datetime import datetime

# Add this import at the top
try:
    from mgmt_summarizer import MgmtSummarizer
    HAS_SUMMARIZER = True
except ImportError:
    HAS_SUMMARIZER = False

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

class FilingRetriever:
    """Retrieve management discussion text from pickle files"""
    
    def __init__(self, text_data_dir, api_key=None, use_summarization=True):
        self.text_data_dir = text_data_dir
        self.filing_cache = {}  # Cache loaded pickle files by year
        self.use_summarization = use_summarization and HAS_SUMMARIZER
        
        # Initialize summarizer if API key provided
        if self.use_summarization and api_key and HAS_SUMMARIZER:
            self.summarizer = MgmtSummarizer(api_key)
        else:
            self.summarizer = None
            if use_summarization and not HAS_SUMMARIZER:
                print("Warning: mgmt_summarizer not available, summarization disabled")
        
    def _load_pickle_for_year(self, year):
        """Load pickle file for a specific year"""
        if year in self.filing_cache:
            return self.filing_cache[year]
        
        pkl_file = f"text_us_{year}.pkl"
        filepath = os.path.join(self.text_data_dir, pkl_file)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'rb') as f:
                df = pickle.load(f)
            self.filing_cache[year] = df
            return df
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            return None
    
    def get_mgmt_text(self, gvkey, target_date, filing_date=None, summarize=None):
        """
        Retrieve management discussion text for a company
        
        Args:
            gvkey: Company gvkey identifier
            target_date: Transition date (YYYY-MM-DD format)
            filing_date: Optional specific filing date (YYYYMMDD format)
            summarize: Whether to summarize (None = use default from __init__)
        
        Returns:
            mgmt text string (summarized if requested) or None if not found
        """
        if pd.isna(gvkey):
            return None
        
        # Convert target_date to datetime
        target_dt = convert_date_to_datetime(target_date)
        if target_dt is None:
            return None
        
        # Determine which year's pickle file to load
        year = target_dt.year
        
        # Load pickle file for that year
        df = self._load_pickle_for_year(year)
        if df is None:
            return None
        
        # Convert gvkey to numeric for matching
        df['gvkey'] = pd.to_numeric(df['gvkey'], errors='coerce')
        
        # Filter by gvkey and date <= target_date
        matches = df[
            (df['gvkey'] == gvkey) & 
            (df['date'].apply(convert_date_to_datetime) <= target_dt)
        ]
        
        if len(matches) == 0:
            # Try previous year if no match
            if year > 2005:
                df_prev = self._load_pickle_for_year(year - 1)
                if df_prev is not None:
                    df_prev['gvkey'] = pd.to_numeric(df_prev['gvkey'], errors='coerce')
                    matches = df_prev[
                        (df_prev['gvkey'] == gvkey) & 
                        (df_prev['date'].apply(convert_date_to_datetime) <= target_dt)
                    ]
        
        if len(matches) == 0:
            return None
        
        # Get most recent match
        match = matches.iloc[-1]
        
        # Return mgmt text
        mgmt_text = match.get('mgmt', None)
        
        if pd.isna(mgmt_text) or mgmt_text == '':
            return None
        
        mgmt_text = str(mgmt_text)
        
        # Determine if we should summarize
        should_summarize = summarize if summarize is not None else self.use_summarization
        
        # Summarize if requested and summarizer is available
        if should_summarize and self.summarizer:
            summarized = self.summarizer.summarize(mgmt_text)
            return summarized if summarized else mgmt_text
        
        return mgmt_text


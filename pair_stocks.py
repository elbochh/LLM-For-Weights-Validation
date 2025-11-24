import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def pair_stocks_by_similarity(df_transition):
    """
    Pair positive-weight and negative-weight stocks by financial similarity
    within same country and sector
    
    Args:
        df_transition: DataFrame with transition data including columns:
            id, weight_change, excntry, sector, ret_12_1, log_market_cap,
            niq_su, ivol_ff3_21d, niq_at
    
    Returns:
        List of pairs: [(pos_stock_row, neg_stock_row), ...]
    """
    pairs = []
    
    # Financial features to use for similarity
    feature_cols = ['ret_12_1', 'log_market_cap', 'niq_su', 'ivol_ff3_21d', 'niq_at']
    
    # Group by country and sector
    for (country, sector), group_df in df_transition.groupby(['excntry', 'sector']):
        # Skip if not enough stocks
        if len(group_df) < 2:
            continue
        
        # Separate positive and negative weight stocks
        positives = group_df[group_df['weight_change'] > 0].copy()
        negatives = group_df[group_df['weight_change'] < 0].copy()
        
        if len(positives) == 0 or len(negatives) == 0:
            continue
        
        # Filter out rows with missing features
        pos_features = positives[feature_cols].copy()
        neg_features = negatives[feature_cols].copy()
        
        # Remove rows with any NaN in features
        pos_valid = ~pos_features.isna().any(axis=1)
        neg_valid = ~neg_features.isna().any(axis=1)
        
        positives_clean = positives[pos_valid].copy()
        negatives_clean = negatives[neg_valid].copy()
        
        if len(positives_clean) == 0 or len(negatives_clean) == 0:
            continue
        
        pos_features_clean = positives_clean[feature_cols].values
        neg_features_clean = negatives_clean[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        
        # Fit on combined data
        all_features = np.vstack([pos_features_clean, neg_features_clean])
        scaler.fit(all_features)
        
        pos_features_scaled = scaler.transform(pos_features_clean)
        neg_features_scaled = scaler.transform(neg_features_clean)
        
        # Use KNN to find nearest negative for each positive
        if len(neg_features_scaled) > 0:
            nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn.fit(neg_features_scaled)
            
            # Find nearest negative for each positive
            distances, indices = nn.kneighbors(pos_features_scaled)
            
            # Create pairs (using greedy matching - one-to-one)
            used_negatives = set()
            
            for i, pos_idx in enumerate(positives_clean.index):
                # Get the nearest negative
                nearest_neg_idx_in_array = indices[i][0]
                nearest_neg_row = negatives_clean.iloc[nearest_neg_idx_in_array]
                
                # Check if this negative is already paired
                neg_original_idx = negatives_clean.index[nearest_neg_idx_in_array]
                
                if neg_original_idx not in used_negatives:
                    pos_row = positives_clean.loc[pos_idx]
                    pairs.append((pos_row, nearest_neg_row))
                    used_negatives.add(neg_original_idx)
    
    return pairs





import pandas as pd
import numpy as np

print("=" * 80)
print("MERGING TRANSITIONS WITH CHARACTERISTICS")
print("=" * 80)

# Load transitions file
print("\n1. Loading transitions file...")
transitions = pd.read_csv('Position_Changes_With_Links/transitions_2017-01_to_2017-02.csv')
print(f"   Loaded {len(transitions)} rows")

# Prepare transitions data for merge
print("\n2. Preparing transitions data...")
transitions_merge = transitions.copy()

# Convert date format: "2017-02-28" -> 20170228
def convert_date(date_str):
    """Convert YYYY-MM-DD to YYYYMMDD integer"""
    parts = date_str.split('-')
    return int(f"{parts[0]}{parts[1]}{parts[2]}")

transitions_merge['date_int'] = transitions_merge['date'].apply(convert_date)

# Convert id format: "100022_01W" -> "comp_100022_01W"
def convert_id(orig_id):
    """Convert transitions id to ret_sample id format"""
    parts = orig_id.split('_')
    gvkey = parts[0]
    iid = parts[1]
    try:
        gvkey_int = int(float(gvkey))
        # Zero-pad to 6 digits
        gvkey_padded = str(gvkey_int).zfill(6)
        return f"comp_{gvkey_padded}_{iid}"
    except (ValueError, TypeError):
        return None

transitions_merge['id_ret'] = transitions_merge['id'].apply(convert_id)
print(f"   Successfully converted {transitions_merge['id_ret'].notna().sum()} IDs")
print(f"   Failed conversions: {transitions_merge['id_ret'].isna().sum()}")

# Show sample conversions
print("\n   Sample conversions:")
for i in range(min(5, len(transitions_merge))):
    orig = transitions_merge.iloc[i]['id']
    new = transitions_merge.iloc[i]['id_ret']
    date_orig = transitions_merge.iloc[i]['date']
    date_new = transitions_merge.iloc[i]['date_int']
    print(f"     {orig} ({date_orig}) -> {new} ({date_new})")

# Load ret_sample data (only for 2017-02-28 to save memory)
print("\n3. Loading ret_sample data for 2017-02-28...")
print("   This may take a moment...")

# Read only the columns we need
cols_needed = ['id', 'date', 'gvkey', 'iid', 
               'ret_12_1', 'ret_12_7',  # Momentum
               'niq_su',  # EPS
               'sale_gr1',  # Growth
               'ivol_ff3_21d', 'rvol_21d']  # Volatility

# Read in chunks to filter by date efficiently
chunks = []
chunk_iter = pd.read_csv('ret_sample_update.csv', 
                         usecols=cols_needed,
                         dtype={'date': int},
                         chunksize=500000,
                         low_memory=False)

for chunk in chunk_iter:
    chunk_filtered = chunk[chunk['date'] == 20170228]
    if len(chunk_filtered) > 0:
        chunks.append(chunk_filtered)
        print(f"   Found {len(chunk_filtered)} rows in this chunk...")

if chunks:
    ret_sample = pd.concat(chunks, ignore_index=True)
    print(f"   Total loaded: {len(ret_sample)} rows for 2017-02-28")
else:
    print("   No data found for 2017-02-28")
    ret_sample = pd.DataFrame(columns=cols_needed)

# Merge
print("\n4. Merging data...")
merged = transitions_merge.merge(
    ret_sample,
    left_on=['id_ret', 'date_int'],
    right_on=['id', 'date'],
    how='left',
    suffixes=('', '_ret')
)

print(f"   Merged {len(merged)} rows")
print(f"   Successful matches: {merged['ret_12_1'].notna().sum()}")
print(f"   Failed matches: {merged['ret_12_1'].isna().sum()}")

# Show merge statistics
print("\n5. MERGE STATISTICS:")
print("=" * 80)
print(f"   Total transitions rows: {len(transitions)}")
print(f"   Matched rows: {merged['ret_12_1'].notna().sum()}")
print(f"   Unmatched rows: {merged['ret_12_1'].isna().sum()}")
print(f"   Match rate: {100*merged['ret_12_1'].notna().sum()/len(merged):.1f}%")

# Column coverage
print("\n6. CHARACTERISTIC COLUMNS COVERAGE:")
print("=" * 80)
for col in ['ret_12_1', 'ret_12_7', 'niq_su', 'sale_gr1', 'ivol_ff3_21d', 'rvol_21d']:
    if col in merged.columns:
        non_null = merged[col].notna().sum()
        pct = 100 * non_null / len(merged)
        print(f"   {col:20}: {non_null:3}/{len(merged)} ({pct:5.1f}%)")

# Clean up merged dataframe
print("\n7. Preparing final output...")
final_cols = ['id', 'weight_change', 'date', 'gvkey', 'filing_date',
              'ret_12_1', 'ret_12_7', 'niq_su', 'sale_gr1', 'ivol_ff3_21d', 'rvol_21d']
final_cols = [col for col in final_cols if col in merged.columns]

merged_final = merged[final_cols].copy()

# Save output
output_file = 'Position_Changes_With_Links/transitions_2017-01_to_2017-02_with_characteristics.csv'
merged_final.to_csv(output_file, index=False)
print(f"   Saved to: {output_file}")

print("\n8. SAMPLE OF MERGED DATA:")
print("=" * 80)
print(merged_final.head(10).to_string())

print("\n" + "=" * 80)
print("MERGE COMPLETE!")
print("=" * 80)


#!/usr/bin/env python3
"""
Investigate the "duplicate" issue - why 1000 stored but only 928 loaded
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Setup paths
sys.path.insert(0, 'stage_01_data_engine/collectors')
sys.path.append('pyiqfeed_orig')
sys.path.append('.')

from iqfeed_collector import IQFeedCollector

# Import TickStore directly
import importlib.util
spec = importlib.util.spec_from_file_location("tick_store", "stage_01_data_engine/storage/tick_store.py")
tick_store_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tick_store_module)
TickStore = tick_store_module.TickStore

print("="*80)
print("DUPLICATE INVESTIGATION")
print("="*80)

# Initialize components
collector = IQFeedCollector()
tick_store = TickStore()

# Connect and fetch
if not collector.ensure_connection():
    print("Failed to connect to IQFeed")
    sys.exit(1)

# Get ticks
print("\n1. FETCHING 1000 TICKS...")
print("-"*40)
ticks = collector.get_tick_data('AAPL', num_days=1, max_ticks=1000)
print(f"NumPy array length: {len(ticks)}")

# Check for duplicates in raw data
tick_ids = ticks['tick_id']
unique_tick_ids = np.unique(tick_ids)
print(f"Unique tick_ids: {len(unique_tick_ids)}")
if len(unique_tick_ids) < len(tick_ids):
    print(f"WARNING: Duplicates found in raw data! {len(tick_ids) - len(unique_tick_ids)} duplicates")

# Check timestamps
dates = ticks['date']
times = ticks['time']
print(f"\nDate range: {np.min(dates)} to {np.max(dates)}")
print(f"Unique dates: {len(np.unique(dates))}")

# Convert to DataFrame
print("\n2. CONVERTING TO DATAFRAME...")
print("-"*40)
df = tick_store._numpy_to_enhanced_dataframe(ticks, 'AAPL')
print(f"DataFrame rows: {len(df)}")
print(f"DataFrame columns: {len(df.columns)}")

# Check for duplicates in DataFrame
print("\n3. CHECKING FOR DUPLICATES IN DATAFRAME...")
print("-"*40)

# Check timestamp duplicates
duplicate_timestamps = df[df.duplicated('timestamp', keep=False)]
print(f"Rows with duplicate timestamps: {len(duplicate_timestamps)}")
if len(duplicate_timestamps) > 0:
    print("\nFirst 5 duplicate timestamp groups:")
    dup_groups = duplicate_timestamps.groupby('timestamp').size().sort_values(ascending=False)
    for ts, count in dup_groups.head().items():
        print(f"  {ts}: {count} rows with same timestamp")

# Check if deduplication is enabled
print("\n4. CHECKING STORE CONFIGURATION...")
print("-"*40)
print(f"TickStore mode: {tick_store.mode}")
# Check dedup from config
if hasattr(tick_store, 'config'):
    print(f"Deduplication setting: {tick_store.config.get('dedup', 'Not found')}")
else:
    print("Deduplication setting: Unknown")

# Store the data
print("\n5. STORING DATA...")
print("-"*40)
today = datetime.now().strftime('%Y-%m-%d')
test_date = '2025-01-16'  # Use a fixed test date

try:
    # First clear any existing data
    try:
        tick_store.delete_ticks('AAPL', test_date)
        print(f"Cleared existing data for AAPL {test_date}")
    except:
        pass

    # Store the DataFrame
    success = tick_store.store_ticks(
        symbol='AAPL',
        date=test_date,
        tick_df=df,
        metadata={'test': 'duplicate_investigation', 'input_rows': len(df)},
        overwrite=True
    )
    print(f"Store result: {success}")
    print(f"Stored {len(df)} rows")

except Exception as e:
    print(f"ERROR during storage: {e}")

# Load back the data
print("\n6. LOADING DATA BACK...")
print("-"*40)
try:
    loaded_df = tick_store.load_ticks('AAPL', test_date)

    if loaded_df is not None:
        print(f"Loaded {len(loaded_df)} rows")

        # Compare counts
        print(f"\nCOMPARISON:")
        print(f"  Input:  {len(df)} rows")
        print(f"  Output: {len(loaded_df)} rows")
        print(f"  Difference: {len(df) - len(loaded_df)} rows")

        if len(loaded_df) < len(df):
            print("\n7. INVESTIGATING MISSING ROWS...")
            print("-"*40)

            # Find what's missing
            # Compare by timestamp
            input_timestamps = set(df['timestamp'].astype(str))
            output_timestamps = set(loaded_df['timestamp'].astype(str))

            missing_timestamps = input_timestamps - output_timestamps
            print(f"Missing timestamps: {len(missing_timestamps)}")

            if missing_timestamps:
                # Show first few missing
                print("\nFirst 5 missing timestamps:")
                for ts in list(missing_timestamps)[:5]:
                    missing_rows = df[df['timestamp'].astype(str) == ts]
                    print(f"  {ts}: {len(missing_rows)} rows")

                    # Check if this timestamp has duplicates
                    duplicates = df[df['timestamp'] == missing_rows.iloc[0]['timestamp']]
                    if len(duplicates) > 1:
                        print(f"    -> This timestamp had {len(duplicates)} duplicate rows in input")

            # Check if ArcticDB deduplicated
            print("\n8. DEDUPLICATION ANALYSIS...")
            print("-"*40)

            # Count unique timestamps in input vs output
            input_unique_ts = df['timestamp'].nunique()
            output_unique_ts = loaded_df['timestamp'].nunique()

            print(f"Unique timestamps in input:  {input_unique_ts}")
            print(f"Unique timestamps in output: {output_unique_ts}")

            if output_unique_ts == input_unique_ts:
                print("CONCLUSION: ArcticDB kept one row per timestamp (deduplication by timestamp)")
            else:
                print("CONCLUSION: Something else is happening")
    else:
        print(f"Failed to load data")

except Exception as e:
    print(f"ERROR loading data: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("INVESTIGATION COMPLETE")
print("="*80)
#!/usr/bin/env python3
"""
Test deduplication issue - why we're losing legitimate trades
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
import pytz

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
print("DEDUPLICATION ISSUE ANALYSIS")
print("="*80)

# Initialize components
collector = IQFeedCollector()
tick_store = TickStore()

# Connect and fetch
if not collector.ensure_connection():
    print("Failed to connect to IQFeed")
    sys.exit(1)

# Get some ticks
print("\n1. FETCHING TICKS WITH POTENTIAL DUPLICATES...")
print("-"*40)
ticks = collector.get_tick_data('AAPL', num_days=1, max_ticks=5000)
print(f"Fetched {len(ticks)} ticks from IQFeed")

# Convert to DataFrame with all 41 fields
print("\n2. CONVERTING TO DATAFRAME (41 FIELDS)...")
print("-"*40)
df = tick_store._numpy_to_enhanced_dataframe(ticks, 'AAPL')
print(f"DataFrame has {len(df)} rows")

# Analyze timestamp duplicates
print("\n3. ANALYZING TIMESTAMP DUPLICATES...")
print("-"*40)

# Find duplicate timestamps
dup_mask = df.duplicated('timestamp', keep=False)
duplicates = df[dup_mask]
print(f"Rows with duplicate timestamps: {len(duplicates)}")

if len(duplicates) > 0:
    # Group by timestamp to see how many trades per timestamp
    dup_groups = duplicates.groupby('timestamp')

    print(f"\nFound {len(dup_groups)} unique timestamps with multiple trades")

    # Show first 5 examples
    print("\nFirst 5 examples of multiple trades at same microsecond:")
    print("-"*40)

    for i, (ts, group) in enumerate(dup_groups):
        if i >= 5:
            break
        print(f"\nTimestamp: {ts}")
        print(f"  Number of trades: {len(group)}")

        # Show the different values for these trades
        for idx, row in group.iterrows():
            print(f"    Trade {idx}: price={row['price']:.2f}, volume={row['volume']}, "
                  f"exchange={row['exchange']}, trade_sign={row['trade_sign']}")

        # Check if these are truly different trades
        unique_prices = group['price'].nunique()
        unique_volumes = group['volume'].nunique()
        unique_exchanges = group['exchange'].nunique()

        if unique_prices > 1 or unique_volumes > 1 or unique_exchanges > 1:
            print(f"  -> These are DIFFERENT trades (not true duplicates)")
        else:
            print(f"  -> These appear to be true duplicates")

print("\n4. IMPACT OF CURRENT DEDUPLICATION...")
print("-"*40)

# Apply current deduplication
original_count = len(df)
deduped_df = tick_store._remove_duplicates(df)
removed_count = original_count - len(deduped_df)

print(f"Original rows: {original_count}")
print(f"After deduplication: {len(deduped_df)}")
print(f"Rows removed: {removed_count} ({removed_count/original_count*100:.1f}%)")

if removed_count > 0:
    print("\n5. WHAT WE'RE LOSING...")
    print("-"*40)

    # Find what was removed
    removed_mask = ~df.index.isin(deduped_df.index)
    removed_trades = df[removed_mask]

    # Analyze removed trades
    total_volume_lost = removed_trades['volume'].sum()
    total_dollar_volume_lost = removed_trades['dollar_volume'].sum()

    print(f"Lost volume: {total_volume_lost:,} shares")
    print(f"Lost dollar volume: ${total_dollar_volume_lost:,.2f}")

    # Check trade signs of removed trades
    trade_signs = removed_trades['trade_sign'].value_counts()
    print(f"\nLost trades by type:")
    print(f"  Buyer-initiated: {trade_signs.get(1, 0)}")
    print(f"  Seller-initiated: {trade_signs.get(-1, 0)}")
    print(f"  Midpoint: {trade_signs.get(0, 0)}")

print("\n6. PROPOSED SOLUTION: SEQUENCE NUMBERS...")
print("-"*40)

# Add sequence numbers for trades at same timestamp
df_with_seq = df.copy()
df_with_seq['tick_sequence'] = df_with_seq.groupby('timestamp').cumcount()

print(f"Added tick_sequence column")
print(f"Max sequences at any timestamp: {df_with_seq['tick_sequence'].max() + 1}")

# Now each row is unique by (timestamp, tick_sequence)
unique_combos = df_with_seq[['timestamp', 'tick_sequence']].drop_duplicates()
print(f"Unique (timestamp, sequence) combinations: {len(unique_combos)}")
print(f"Original rows: {len(df)}")
print(f"No data loss: {len(unique_combos) == len(df)}")

print("\n" + "="*80)
print("CONCLUSION")
print("-"*40)
print(f"Current approach loses {removed_count/original_count*100:.1f}% of legitimate trades")
print("These are real trades at the same microsecond, not duplicates")
print("Solution: Add sequence numbers to preserve all trades")
print("="*80)
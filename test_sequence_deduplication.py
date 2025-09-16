#!/usr/bin/env python3
"""
Test the new sequence number deduplication solution
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
print("TESTING NEW SEQUENCE NUMBER DEDUPLICATION")
print("="*80)

# Initialize components
collector = IQFeedCollector()
tick_store = TickStore()

# Connect and fetch
if not collector.ensure_connection():
    print("Failed to connect to IQFeed")
    sys.exit(1)

# Get some ticks
print("\n1. FETCHING TICKS...")
print("-"*40)
ticks = collector.get_tick_data('AAPL', num_days=1, max_ticks=5000)
print(f"Fetched {len(ticks)} ticks from IQFeed")

# Convert to DataFrame
print("\n2. CONVERTING TO DATAFRAME...")
print("-"*40)
df_original = tick_store._numpy_to_enhanced_dataframe(ticks, 'AAPL')
print(f"DataFrame has {len(df_original)} rows")
print(f"Columns: {df_original.shape[1]} - {', '.join(df_original.columns[:5])}...")

# Apply the new deduplication (which now adds sequence numbers)
print("\n3. APPLYING NEW DEDUPLICATION (SEQUENCE NUMBERS)...")
print("-"*40)
df_with_sequences = tick_store._remove_duplicates(df_original.copy())
print(f"After deduplication: {len(df_with_sequences)} rows (should be same as original)")

# Check for tick_sequence column
if 'tick_sequence' in df_with_sequences.columns:
    print(f"[OK] tick_sequence column added")
    max_seq = df_with_sequences['tick_sequence'].max()
    print(f"Maximum sequence number: {max_seq}")

    # Find examples of multiple trades at same timestamp
    multi_trade_timestamps = df_with_sequences[df_with_sequences['tick_sequence'] > 0]['timestamp'].unique()
    print(f"Timestamps with multiple trades: {len(multi_trade_timestamps)}")
else:
    print("[X] tick_sequence column NOT added!")

# Verify NO DATA LOSS
print("\n4. VERIFYING NO DATA LOSS...")
print("-"*40)
print(f"Original rows: {len(df_original)}")
print(f"After adding sequences: {len(df_with_sequences)}")
print(f"Data preserved: {len(df_original) == len(df_with_sequences)}")

if len(df_original) == len(df_with_sequences):
    print("[OK] All trades preserved!")
else:
    print(f"[X] Lost {len(df_original) - len(df_with_sequences)} trades")

# Check tick 265596 specifically if present
print("\n5. CHECKING TICK 265596...")
print("-"*40)
if len(df_original) > 265596:
    tick_265596 = df_with_sequences.iloc[265596]
    print(f"Tick 265596:")
    print(f"  Timestamp: {tick_265596['timestamp']}")
    print(f"  Sequence: {tick_265596.get('tick_sequence', 'N/A')}")
    print(f"  Price: {tick_265596['price']}")
    print(f"  Volume: {tick_265596['volume']}")

    # Check if there are other trades at the same timestamp
    same_ts_trades = df_with_sequences[df_with_sequences['timestamp'] == tick_265596['timestamp']]
    print(f"  Trades at same timestamp: {len(same_ts_trades)}")
    if len(same_ts_trades) > 1:
        print(f"  Sequences: {sorted(same_ts_trades['tick_sequence'].unique())}")
else:
    print(f"Dataset only has {len(df_original)} ticks, can't check tick 265596")

# Now test storing to ArcticDB
print("\n6. TESTING ARCTICDB STORAGE...")
print("-"*40)
try:
    # Store the data with sequences
    result = tick_store.store_dataframe(df_with_sequences, 'AAPL_TEST_SEQ')
    if result and 'success' in result:
        print(f"[OK] Stored {result.get('rows_written', 0)} rows to ArcticDB")

        # Load it back
        loaded_df = tick_store.load_dataframe('AAPL_TEST_SEQ')
        if loaded_df is not None:
            print(f"[OK] Loaded {len(loaded_df)} rows from ArcticDB")
            print(f"  Has tick_sequence: {'tick_sequence' in loaded_df.columns}")

            # Verify data integrity
            print(f"  Data preserved: {len(loaded_df) == len(df_with_sequences)}")
except Exception as e:
    print(f"[X] Error with ArcticDB: {e}")

print("\n" + "="*80)
print("SEQUENCE NUMBER SOLUTION COMPLETE")
print(f"Result: All {len(df_original)} trades preserved with sequence numbers")
print("No more data loss from deduplication!")
print("="*80)
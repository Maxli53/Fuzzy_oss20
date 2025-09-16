#!/usr/bin/env python3
"""
Test DataFrame → ArcticDB storage step
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
print("TEST: DataFrame -> ArcticDB Storage")
print("="*80)

# Initialize components
collector = IQFeedCollector()
tick_store = TickStore()

# Connect and fetch
if not collector.ensure_connection():
    print("Failed to connect to IQFeed")
    sys.exit(1)

# Get small batch of ticks
print("\n1. FETCHING TEST DATA...")
print("-"*40)
ticks = collector.get_tick_data('AAPL', num_days=1, max_ticks=1000)
print(f"Fetched {len(ticks)} ticks")

print("\n2. CONVERTING TO DATAFRAME (41 FIELDS)...")
print("-"*40)
# Convert using our new vectorized method
df = tick_store._numpy_to_enhanced_dataframe(ticks, 'AAPL')
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {len(df.columns)}")

print("\n3. ATTEMPTING DIRECT DATAFRAME STORAGE...")
print("-"*40)
# Try to store the DataFrame directly using store_ticks (bypassing store_numpy_ticks)
today = datetime.now().strftime('%Y-%m-%d')
try:
    success = tick_store.store_ticks(
        symbol='AAPL',
        date=today,
        tick_df=df,
        metadata={'test': 'direct_df_storage', 'count': len(df)},
        overwrite=True
    )

    if success:
        print(f"[OK] Successfully stored {len(df)} ticks to ArcticDB")
    else:
        print(f"[ERROR] Failed to store ticks to ArcticDB")

except Exception as e:
    print(f"ERROR during storage: {e}")
    import traceback
    traceback.print_exc()

print("\n4. VERIFYING STORAGE...")
print("-"*40)
try:
    # Try to read back the data
    loaded_df = tick_store.load_ticks('AAPL', today)

    if loaded_df is not None:
        print(f"[OK] Successfully loaded {len(loaded_df)} ticks from ArcticDB")
        print(f"  Columns: {len(loaded_df.columns)}")

        # Verify all 41 fields are present
        if len(loaded_df.columns) == 41:
            print(f"[OK] All 41 fields preserved in storage")
        else:
            print(f"[ERROR] Field count mismatch: {len(loaded_df.columns)} != 41")

        # Check first tick
        first_tick = loaded_df.iloc[0]
        print(f"\n  Sample tick:")
        print(f"    timestamp: {first_tick['timestamp']}")
        print(f"    price: {first_tick['price']}")
        print(f"    volume: {first_tick['volume']}")
        print(f"    trade_sign: {first_tick['trade_sign']}")
    else:
        print(f"[ERROR] Failed to load data from ArcticDB")

except Exception as e:
    print(f"ERROR loading data: {e}")

print("\n5. CHECKING ARCTICDB LIBRARY...")
print("-"*40)
try:
    # List all symbols in the library
    symbols = tick_store.library.list_symbols()
    print(f"Symbols in library: {len(symbols)}")

    # Check for our AAPL data
    aapl_symbols = [s for s in symbols if 'AAPL' in s]
    if aapl_symbols:
        print(f"AAPL entries found: {aapl_symbols[:5]}")  # Show first 5

        # Get metadata for one
        if aapl_symbols:
            symbol_info = tick_store.library.get_info(aapl_symbols[0])
            print(f"\nMetadata for {aapl_symbols[0]}:")
            print(f"  Rows: {symbol_info['rows']}")
            print(f"  Columns: {symbol_info['columns']}")
    else:
        print("No AAPL data found in library")

except Exception as e:
    print(f"ERROR checking library: {e}")

print("\n" + "="*80)
print("STORAGE TEST COMPLETE")
print("="*80)
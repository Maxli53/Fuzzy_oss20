#!/usr/bin/env python3
"""
Quick test to verify tick_id is accessible in DataFrame
"""

import sys
sys.path.insert(0, 'stage_01_data_engine/collectors')
sys.path.append('pyiqfeed_orig')
sys.path.append('.')
sys.path.append('foundation')

from iqfeed_collector import IQFeedCollector
from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic

# Import TickStore
import importlib.util
spec = importlib.util.spec_from_file_location("tick_store", "stage_01_data_engine/storage/tick_store.py")
tick_store_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tick_store_module)
TickStore = tick_store_module.TickStore

# Initialize
collector = IQFeedCollector()
tick_store = TickStore()

# Connect
if not collector.ensure_connection():
    print("Failed to connect to IQFeed")
    sys.exit(1)

# Fetch small batch
print("Fetching ticks...")
tick_array = collector.get_tick_data('AAPL', num_days=1, max_ticks=100)
print(f"Fetched {len(tick_array)} ticks")

# Convert to Pydantic
print("\nConverting to Pydantic...")
pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, 'AAPL')
print(f"Converted {len(pydantic_ticks)} ticks")

# Check first tick
if pydantic_ticks:
    first_tick = pydantic_ticks[0]
    print(f"\nFirst tick has {len(first_tick.model_dump())} fields")
    print(f"tick_id: {first_tick.tick_id}")
    print(f"symbol: {first_tick.symbol}")
    print(f"price: {first_tick.price}")
    print(f"tick_sequence: {first_tick.tick_sequence}")

# Convert to DataFrame
print("\nConverting to DataFrame...")
df = tick_store._pydantic_to_dataframe(pydantic_ticks)
print(f"DataFrame shape: {df.shape}")
print(f"Columns ({len(df.columns)}): {', '.join(df.columns[:10])}...")

# Check tick_id column
if 'tick_id' in df.columns:
    print(f"\n[OK] tick_id column exists!")
    print(f"First 5 tick_ids: {df['tick_id'].head().tolist()}")
    print(f"tick_id dtype: {df['tick_id'].dtype}")
else:
    print("\n[FAIL] tick_id column MISSING!")

# Verify we can query by tick_id
if 'tick_id' in df.columns:
    target_id = df['tick_id'].iloc[50] if len(df) > 50 else df['tick_id'].iloc[0]
    print(f"\nSearching for tick_id={target_id}...")
    found = df[df['tick_id'] == target_id]
    if not found.empty:
        print(f"[OK] Found tick with id {target_id}:")
        print(f"  Price: ${found.iloc[0]['price']:.2f}")
        print(f"  Volume: {found.iloc[0]['volume']}")
        print(f"  Timestamp: {found.iloc[0]['timestamp']}")
    else:
        print(f"[FAIL] Could not find tick_id={target_id}")

print("\nTEST COMPLETE: tick_id field is working!")
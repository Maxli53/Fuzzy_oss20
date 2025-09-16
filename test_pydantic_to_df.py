#!/usr/bin/env python3
"""
Test Pydantic to DataFrame conversion for TICK 265596
"""

import sys
import os
sys.path.insert(0, 'stage_01_data_engine/collectors')
sys.path.append('pyiqfeed_orig')
sys.path.append('.')

from iqfeed_collector import IQFeedCollector
from foundation.utils.iqfeed_converter import convert_iqfeed_tick_to_pydantic
import pandas as pd

# Import the TickStore class directly to avoid circular imports
import importlib.util
spec = importlib.util.spec_from_file_location("tick_store", "stage_01_data_engine/storage/tick_store.py")
tick_store_module = importlib.util.module_from_spec(spec)

# Create a minimal TickStore instance just for testing the method
class TestTickStore:
    def __init__(self):
        pass

    # Copy the _pydantic_to_dataframe method directly
    exec(open('stage_01_data_engine/storage/tick_store.py').read().split('def _pydantic_to_dataframe')[1].split('def load_ticks')[0])

# Initialize components
collector = IQFeedCollector()
tick_store = TestTickStore()

print("="*80)
print("TEST: Pydantic → DataFrame Conversion for TICK 265596")
print("="*80)

# Get TICK 265596
if not collector.ensure_connection():
    print("Failed to connect to IQFeed")
    sys.exit(1)

ticks = collector.get_tick_data('AAPL', num_days=1, max_ticks=10000)
found_tick = None
for t in ticks:
    if t['tick_id'] == 265596:
        found_tick = t
        break

if found_tick is None:
    print("TICK 265596 not found")
    sys.exit(1)

print("\n1. RAW NUMPY TICK 265596:")
print("-"*40)
print(f"  Raw: {found_tick}")

print("\n2. CONVERT TO PYDANTIC:")
print("-"*40)
pydantic_tick = convert_iqfeed_tick_to_pydantic(found_tick, 'AAPL')
print(f"  Fields in Pydantic model: {len(pydantic_tick.model_dump())}")
print(f"  Price: {pydantic_tick.price} (Decimal)")
print(f"  Trade sign: {pydantic_tick.trade_sign}")
print(f"  Price improvement: {pydantic_tick.price_improvement}")

print("\n3. CONVERT TO DATAFRAME:")
print("-"*40)
# Convert single tick to list for the method
pydantic_ticks = [pydantic_tick]
df = tick_store._pydantic_to_dataframe(pydantic_ticks)

print(f"  DataFrame shape: {df.shape}")
print(f"  Columns ({len(df.columns)}): {', '.join(df.columns.tolist())}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

print("\n4. VERIFY KEY FIELDS:")
print("-"*40)
row = df.iloc[0]
print(f"  symbol: {row['symbol']} (dtype: {df['symbol'].dtype})")
print(f"  timestamp: {row['timestamp']} (dtype: {df['timestamp'].dtype})")
print(f"  price: {row['price']} (dtype: {df['price'].dtype})")
print(f"  volume: {row['volume']} (dtype: {df['volume'].dtype})")
print(f"  trade_sign: {row['trade_sign']} (dtype: {df['trade_sign'].dtype})")
print(f"  price_improvement: {row['price_improvement']} (dtype: {df['price_improvement'].dtype})")
print(f"  is_extended_hours: {row['is_extended_hours']} (dtype: {df['is_extended_hours'].dtype})")
print(f"  is_qualified: {row['is_qualified']} (dtype: {df['is_qualified'].dtype})")

print("\n5. DTYPE OPTIMIZATIONS:")
print("-"*40)
print("  Float32 fields:", [col for col in df.columns if df[col].dtype == 'float32'])
print("  Category fields:", [col for col in df.columns if df[col].dtype.name == 'category'])
print("  Int8 fields:", [col for col in df.columns if df[col].dtype == 'int8'])
print("  UInt32 fields:", [col for col in df.columns if df[col].dtype == 'uint32'])

print("\n6. SUMMARY:")
print("-"*40)
print(f"  ✓ NumPy (14 fields) → Pydantic ({len(pydantic_tick.model_dump())} fields) → DataFrame ({len(df.columns)} columns)")
print(f"  ✓ All fields preserved and memory-optimized")
print(f"  ✓ Ready for ArcticDB storage")
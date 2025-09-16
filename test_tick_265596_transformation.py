#!/usr/bin/env python3
"""
Show EXACT transformation for TICK 265596
NumPy (14 fields) -> Enhanced DataFrame (41 fields)
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
print("EXACT TRANSFORMATION FOR TICK 265596")
print("="*80)

# Initialize components
collector = IQFeedCollector()
tick_store = TickStore()

# Connect and fetch
if not collector.ensure_connection():
    print("Failed to connect to IQFeed")
    sys.exit(1)

# Get ticks including 265596
print("\n1. FETCHING TICK 265596 FROM IQFEED...")
print("-"*40)
ticks = collector.get_tick_data('AAPL', num_days=1, max_ticks=300000)

# Find tick 265596
found_tick = None
found_index = -1
for i, t in enumerate(ticks):
    if t['tick_id'] == 265596:
        found_tick = t
        found_index = i
        break

if found_tick is None:
    print("ERROR: TICK 265596 not found")
    sys.exit(1)

print(f"Found at index: {found_index}")
print(f"Raw NumPy record: {found_tick}")

print("\n2. RAW NUMPY FIELDS (14 fields):")
print("-"*40)
for field in found_tick.dtype.names:
    value = found_tick[field]
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    print(f"  {field:12} = {value}")

print("\n3. VECTORIZED TRANSFORMATION TO 41 FIELDS:")
print("-"*40)

# Create single-element array for transformation
single_tick_array = np.array([found_tick], dtype=found_tick.dtype)

# Apply the vectorized transformation
df = tick_store._numpy_to_enhanced_dataframe(single_tick_array, 'AAPL')

# Get the transformed row
row = df.iloc[0]

print("\nCORE FIELDS (10):")
print(f"  symbol         = '{row['symbol']}'")
print(f"  timestamp      = {row['timestamp']}")
print(f"  price          = {row['price']:.2f}")
print(f"  volume         = {row['volume']}")
print(f"  exchange       = '{row['exchange']}'")
print(f"  market_center  = {row['market_center']}")
print(f"  total_volume   = {row['total_volume']}")
print(f"  bid            = {row['bid']:.2f}")
print(f"  ask            = {row['ask']:.2f}")
print(f"  conditions     = '{row['conditions']}'")

print("\nSPREAD METRICS (5):")
print(f"  spread         = {row['spread']:.4f}")
print(f"  midpoint       = {row['midpoint']:.4f}")
print(f"  spread_bps     = {row['spread_bps']:.2f}")
print(f"  spread_pct     = {row['spread_pct']:.6f}")
print(f"  effective_spread = {row['effective_spread']:.4f}")

print("\nTRADE ANALYSIS (3):")
print(f"  trade_sign     = {row['trade_sign']} ({'buyer-initiated' if row['trade_sign'] == 1 else 'seller-initiated' if row['trade_sign'] == -1 else 'midpoint'})")
print(f"  dollar_volume  = ${row['dollar_volume']:.2f}")
print(f"  price_improvement = {row['price_improvement']:.4f}")

print("\nADDITIONAL METRICS (7):")
print(f"  tick_direction = {row['tick_direction']}")
print(f"  participant_type = '{row['participant_type']}'")
print(f"  volume_rate    = {row['volume_rate']:.2f} shares/min")
print(f"  trade_pct_of_day = {row['trade_pct_of_day']:.4f}%")
print(f"  log_return     = {row['log_return']:.8f}" if pd.notna(row['log_return']) else "  log_return     = NaN (first tick)")
print(f"  price_change   = {row['price_change']:.4f}" if pd.notna(row['price_change']) else "  price_change   = NaN (first tick)")
print(f"  price_change_bps = {row['price_change_bps']:.2f}" if pd.notna(row['price_change_bps']) else "  price_change_bps = NaN (first tick)")

print("\nCONDITION FLAGS (7):")
print(f"  is_regular     = {row['is_regular']}")
print(f"  is_extended_hours = {row['is_extended_hours']}")
print(f"  is_odd_lot     = {row['is_odd_lot']}")
print(f"  is_intermarket_sweep = {row['is_intermarket_sweep']}")
print(f"  is_derivatively_priced = {row['is_derivatively_priced']}")
print(f"  is_qualified   = {row['is_qualified']}")
print(f"  is_block_trade = {row['is_block_trade']}")

print("\nMETADATA FIELDS (4):")
print(f"  id             = '{row['id']}'")
print(f"  created_at     = {row['created_at']}")
print(f"  updated_at     = {row['updated_at']}")
print(f"  metadata       = {row['metadata']}")

print("\nTIMESTAMP FIELDS (2):")
print(f"  processed_at   = {row['processed_at']}")
print(f"  source_timestamp = {row['source_timestamp']}")

print("\nENUM FIELDS (3):")
print(f"  trade_sign_enum = '{row['trade_sign_enum']}'")
print(f"  tick_direction_enum = '{row['tick_direction_enum']}'")
print(f"  participant_type_enum = '{row['participant_type_enum']}'")

print("\n4. TRANSFORMATION SUMMARY:")
print("-"*40)
print(f"NumPy fields:    14")
print(f"DataFrame fields: {len(df.columns)}")
print(f"Fields added:     {len(df.columns) - 14}")

print("\n5. KEY TRANSFORMATIONS APPLIED:")
print("-"*40)
print("  • Date + Time -> Timestamp (ET timezone)")
print("  • last -> price")
print("  • last_sz -> volume")
print("  • last_type -> exchange (decoded)")
print("  • mkt_ctr -> market_center")
print("  • tot_vlm -> total_volume")
print("  • Computed spread from bid/ask")
print("  • Computed midpoint from bid/ask")
print("  • Computed trade_sign using Lee-Ready algorithm")
print("  • Parsed condition codes into boolean flags")
print("  • Added UUID for unique identification")
print("  • Added timestamps for processing audit")

print("\n6. DATA TYPES OPTIMIZATION:")
print("-"*40)
dtype_map = {}
for col in df.columns:
    dtype = str(df[col].dtype)
    if dtype not in dtype_map:
        dtype_map[dtype] = []
    dtype_map[dtype].append(col)

for dtype, cols in sorted(dtype_map.items()):
    print(f"  {dtype}: {len(cols)} fields")

print("\n" + "="*80)
print("TRANSFORMATION COMPLETE")
print(f"TICK 265596: 14 NumPy fields -> 41 DataFrame fields")
print("Using VECTORIZED operations (no per-tick Pydantic validation)")
print("="*80)
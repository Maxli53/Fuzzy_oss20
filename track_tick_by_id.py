#!/usr/bin/env python3
"""
Track tick with tick_id=265596 through the PRODUCTION pipeline.
NOT array index 265596!
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Setup paths
sys.path.insert(0, 'stage_01_data_engine/collectors')
sys.path.append('pyiqfeed_orig')
sys.path.append('.')
sys.path.append('foundation')

from iqfeed_collector import IQFeedCollector
from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic
from foundation.models.market import TickData

# Import TickStore
import importlib.util
spec = importlib.util.spec_from_file_location("tick_store", "stage_01_data_engine/storage/tick_store.py")
tick_store_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tick_store_module)
TickStore = tick_store_module.TickStore

print("=" * 100)
print("TRACKING TICK WITH tick_id=265596 THROUGH PRODUCTION PIPELINE")
print("=" * 100)

# Initialize
collector = IQFeedCollector()
tick_store = TickStore()

# Connect
if not collector.ensure_connection():
    print("Failed to connect to IQFeed")
    sys.exit(1)

# Fetch data - need enough to hopefully get tick_id 265596
print("\nFetching ticks to find tick_id=265596...")
tick_array = collector.get_tick_data('AAPL', num_days=2, max_ticks=500000)
print(f"Fetched {len(tick_array)} ticks")

# ============================================================================
# STAGE 1: Find tick_id=265596 in NumPy array
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 1: Finding tick_id=265596 in NumPy Array")
print("=" * 80)

# Find the tick with tick_id=265596
target_tick_id = 265596
found_index = None
found_tick = None

for i, tick in enumerate(tick_array):
    if tick['tick_id'] == target_tick_id:
        found_index = i
        found_tick = tick
        break

if found_tick is None:
    print(f"[FAIL] tick_id={target_tick_id} not found in {len(tick_array)} ticks")
    print("Tick IDs range from", tick_array['tick_id'].min(), "to", tick_array['tick_id'].max())
    sys.exit(1)

print(f"[OK] Found tick_id={target_tick_id} at array index {found_index}")
print("\nNumPy Data for tick_id=265596:")
print("-" * 40)
print(f"  Array index: {found_index}")
print(f"  tick_id:     {found_tick['tick_id']}")
print(f"  date:        {found_tick['date']}")
print(f"  time:        {found_tick['time']}")
print(f"  last:        ${found_tick['last']:.4f}")
print(f"  last_sz:     {found_tick['last_sz']}")
print(f"  last_type:   {found_tick['last_type'].decode() if isinstance(found_tick['last_type'], bytes) else found_tick['last_type']}")
print(f"  mkt_ctr:     {found_tick['mkt_ctr']}")
print(f"  tot_vlm:     {found_tick['tot_vlm']:,}")
print(f"  bid:         ${found_tick['bid']:.4f}")
print(f"  ask:         ${found_tick['ask']:.4f}")
print(f"  conditions:  [{found_tick['cond1']}, {found_tick['cond2']}, {found_tick['cond3']}, {found_tick['cond4']}]")

# Calculate timestamp
date_val = np.datetime64(found_tick['date'], 'D')
time_delta = found_tick['time']
if isinstance(time_delta, np.timedelta64):
    time_microseconds = time_delta / np.timedelta64(1, 'us')
else:
    time_microseconds = int(time_delta)
timestamp_numpy = pd.Timestamp(date_val) + pd.Timedelta(microseconds=time_microseconds)
print(f"  timestamp:   {timestamp_numpy}")

# Store the original values for comparison
original_price = found_tick['last']
original_volume = found_tick['last_sz']
original_timestamp = timestamp_numpy

# ============================================================================
# STAGE 2: Convert to Pydantic and find our tick
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 2: Converting to Pydantic and finding tick_id=265596")
print("=" * 80)

print("Converting all ticks to Pydantic...")
pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, 'AAPL')
print(f"Converted {len(pydantic_ticks)} ticks (dropped {len(tick_array) - len(pydantic_ticks)} invalid ticks)")

# Find our tick in Pydantic list - need to match by timestamp and price since tick_id might not be preserved
pydantic_tick = None
pydantic_index = None

# First try to match by exact timestamp and price
for i, tick in enumerate(pydantic_ticks):
    # Check if timestamp matches (within 1 second tolerance)
    tick_ts = pd.Timestamp(tick.timestamp).tz_localize(None)
    if abs((tick_ts - timestamp_numpy).total_seconds()) < 1:
        if abs(float(tick.price) - original_price) < 0.01:  # Price within 1 cent
            if tick.size == original_volume:  # Volume matches
                pydantic_tick = tick
                pydantic_index = i
                print(f"[OK] Found matching tick at Pydantic index {i}")
                break

if pydantic_tick is None:
    print(f"[WARN] Could not find exact match for tick_id={target_tick_id}")
    print(f"Looking for closest match by timestamp...")
    # Find closest by timestamp
    for i, tick in enumerate(pydantic_ticks):
        tick_ts = pd.Timestamp(tick.timestamp).tz_localize(None)
        if abs((tick_ts - timestamp_numpy).total_seconds()) < 1:
            pydantic_tick = tick
            pydantic_index = i
            print(f"[OK] Found close match at Pydantic index {i}")
            break

if pydantic_tick:
    print("\nPydantic Data (41 fields):")
    print("-" * 40)
    print(f"  Pydantic index: {pydantic_index}")
    print(f"  symbol:       {pydantic_tick.symbol}")
    print(f"  timestamp:    {pydantic_tick.timestamp}")
    print(f"  price:        ${pydantic_tick.price}")
    print(f"  size:         {pydantic_tick.size}")
    print(f"  exchange:     {pydantic_tick.exchange}")
    print(f"  bid:          ${pydantic_tick.bid}")
    print(f"  ask:          ${pydantic_tick.ask}")
    print(f"  spread_bps:   {pydantic_tick.spread_bps:.2f}")
    print(f"  trade_sign:   {pydantic_tick.trade_sign}")
    print(f"  dollar_volume: ${pydantic_tick.dollar_volume:.2f}")
else:
    print("[FAIL] Could not find tick in Pydantic list")
    sys.exit(1)

# ============================================================================
# STAGE 3: Convert to DataFrame and find our tick
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 3: Converting to DataFrame and finding our tick")
print("=" * 80)

df = tick_store._pydantic_to_dataframe(pydantic_ticks)
print(f"Created DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")

# Find our tick in DataFrame by matching values, not index
# Match by timestamp, price, and volume since these should be unique
df_tick = None
df_index = None

# Convert Pydantic timestamp to naive for comparison
pydantic_ts_naive = pd.Timestamp(pydantic_tick.timestamp).tz_localize(None)

# Find matching row in DataFrame
for idx, row in df.iterrows():
    row_ts = pd.Timestamp(row['timestamp']).tz_localize(None) if pd.Timestamp(row['timestamp']).tz else pd.Timestamp(row['timestamp'])

    # Check timestamp match (within 1 microsecond)
    if abs((row_ts - pydantic_ts_naive).total_seconds()) < 0.000001:
        # Check price and volume match
        if abs(row['price'] - float(pydantic_tick.price)) < 0.001:
            if row['volume'] == pydantic_tick.size:
                df_tick = row
                df_index = idx
                print(f"[OK] Found matching tick at DataFrame index {idx}")
                break

if df_tick is not None:
    print(f"\nDataFrame Data:")
    print("-" * 40)
    print(f"  DataFrame index: {df_index}")
    print(f"  timestamp:    {df_tick['timestamp']}")
    print(f"  price:        ${df_tick['price']:.4f}")
    print(f"  volume:       {df_tick['volume']}")
    print(f"  bid:          ${df_tick['bid']:.4f}")
    print(f"  ask:          ${df_tick['ask']:.4f}")
    print(f"  spread_bps:   {df_tick['spread_bps']:.2f}")
    print(f"  trade_sign:   {df_tick['trade_sign']}")
    print(f"  dollar_volume: ${df_tick['dollar_volume']:.2f}")
else:
    print("[FAIL] Could not find matching tick in DataFrame")
    print(f"Looking for: timestamp={pydantic_ts_naive}, price={pydantic_tick.price}, volume={pydantic_tick.size}")

    # Show first few rows for debugging
    print("\nFirst 5 DataFrame rows:")
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        print(f"  {i}: ts={row['timestamp']}, price={row['price']:.4f}, vol={row['volume']}")
    sys.exit(1)

# ============================================================================
# STAGE 4: Store to ArcticDB and retrieve
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 4: Storing to ArcticDB and retrieving")
print("=" * 80)

# Apply deduplication
df_with_seq = tick_store._remove_duplicates(df.copy())
print(f"Applied deduplication: {df_with_seq.shape}")

# Store
today = datetime.now().strftime('%Y-%m-%d')
success = tick_store.store_ticks(
    symbol='AAPL_TICKID_TEST',
    date=today,
    tick_df=df_with_seq,
    metadata={'test': 'tick_id_265596'},
    overwrite=True
)

if success:
    print("[OK] Stored to ArcticDB")

    # Load back
    storage_key = f"AAPL_TICKID_TEST/{today}"
    result = tick_store.tick_data_lib.read(storage_key)
    loaded_df = result.data

    # Find our tick using composite key (timestamp, tick_sequence)
    # Get the tick_sequence from our Pydantic tick
    target_sequence = pydantic_tick.tick_sequence if hasattr(pydantic_tick, 'tick_sequence') else 0

    # Convert Pydantic timestamp to compare with loaded data
    # Include microseconds for precise matching
    pydantic_ts_str = str(pydantic_tick.timestamp)[:26]  # Get YYYY-MM-DD HH:MM:SS.ffffff

    # Find matching row in loaded DataFrame
    loaded_tick = None
    loaded_index = None
    for idx, row in loaded_df.iterrows():
        row_ts_str = str(row['timestamp'])[:26]
        if row_ts_str == pydantic_ts_str:
            # Check tick_sequence if present
            if 'tick_sequence' in row and row['tick_sequence'] == target_sequence:
                loaded_tick = row
                loaded_index = idx
                print(f"[OK] Found matching tick at ArcticDB index {idx} using composite key")
                break
            elif 'tick_sequence' not in loaded_df.columns:
                # Fallback: match by timestamp only
                loaded_tick = row
                loaded_index = idx
                print(f"[OK] Found matching tick at ArcticDB index {idx} using timestamp")
                break

    if loaded_tick is not None:
        print(f"\nArcticDB Data:")
        print("-" * 40)
        print(f"  ArcticDB index: {loaded_index}")
        print(f"  timestamp:    {loaded_tick['timestamp']}")
        print(f"  price:        ${loaded_tick['price']:.4f}")
        print(f"  volume:       {loaded_tick['volume']}")
        if 'tick_sequence' in loaded_tick:
            print(f"  tick_sequence: {loaded_tick['tick_sequence']}")
        print(f"  spread_bps:   {loaded_tick['spread_bps']:.2f}")
        print(f"  dollar_volume: ${loaded_tick['dollar_volume']:.2f}")
    else:
        print("[FAIL] Could not find matching tick in ArcticDB")
        print(f"Looking for: timestamp={pydantic_ts_str}, sequence={target_sequence}")

    # Clean up
    tick_store.tick_data_lib.delete(storage_key)
    print("\n[OK] Cleaned up test data")

# ============================================================================
# FINAL VERIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("DATA INTEGRITY VERIFICATION FOR tick_id=265596")
print("=" * 80)

if loaded_tick is not None:
    print(f"\nPRICE:")
    print(f"  NumPy:     ${original_price:.4f}")
    if pydantic_tick:
        print(f"  Pydantic:  ${float(pydantic_tick.price):.4f}")
        print(f"  DataFrame: ${df_tick['price']:.4f}")
        print(f"  ArcticDB:  ${loaded_tick['price']:.4f}")
        price_match = abs(original_price - loaded_tick['price']) < 0.001
        print(f"  Match:     {'[OK]' if price_match else '[FAIL]'}")

    print(f"\nVOLUME:")
    print(f"  NumPy:     {original_volume}")
    if pydantic_tick:
        print(f"  Pydantic:  {pydantic_tick.size}")
        print(f"  DataFrame: {df_tick['volume']}")
        print(f"  ArcticDB:  {loaded_tick['volume']}")
        volume_match = original_volume == loaded_tick['volume']
        print(f"  Match:     {'[OK]' if volume_match else '[FAIL]'}")

    print(f"\nTIMESTAMP:")
    print(f"  NumPy:     {original_timestamp}")
    if pydantic_tick:
        print(f"  Pydantic:  {pydantic_tick.timestamp}")
        print(f"  DataFrame: {df_tick['timestamp']}")
        print(f"  ArcticDB:  {loaded_tick['timestamp']}")
        # Check timestamp match (ignoring timezone)
        ts_match = str(original_timestamp)[:19] == str(loaded_tick['timestamp'])[:19]
        print(f"  Match:     {'[OK]' if ts_match else '[FAIL]'}")

    if 'tick_sequence' in loaded_tick:
        print(f"\nTICK_SEQUENCE:")
        print(f"  Pydantic:  {pydantic_tick.tick_sequence if hasattr(pydantic_tick, 'tick_sequence') else 'N/A'}")
        print(f"  DataFrame: {df_tick['tick_sequence'] if 'tick_sequence' in df_tick else 'N/A'}")
        print(f"  ArcticDB:  {loaded_tick['tick_sequence']}")

    print("\n" + "=" * 100)
    if price_match and volume_match and ts_match:
        print("SUCCESS: tick_id=265596 preserved correctly through all stages!")
    else:
        print("FAIL: Data corruption detected for tick_id=265596")
else:
    print("FAIL: Could not retrieve tick from ArcticDB for verification")
print("=" * 100)
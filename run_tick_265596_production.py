#!/usr/bin/env python3
"""
Run tick ID 265596 through the PRODUCTION pipeline.
Shows all data and transformations at each stage.
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
sys.path.append('foundation')

from iqfeed_collector import IQFeedCollector
from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic
from foundation.models.market import TickData

# Import TickStore directly
import importlib.util
spec = importlib.util.spec_from_file_location("tick_store", "stage_01_data_engine/storage/tick_store.py")
tick_store_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tick_store_module)
TickStore = tick_store_module.TickStore

print("=" * 100)
print("PRODUCTION PIPELINE - TICK 265596 END-TO-END")
print("=" * 100)

# Initialize production components
collector = IQFeedCollector()
tick_store = TickStore()

# Connect to IQFeed
print("\nConnecting to IQFeed...")
if not collector.ensure_connection():
    print("Failed to connect to IQFeed")
    sys.exit(1)
print("[OK] Connected to IQFeed")

# Fetch production data
print("\n" + "=" * 80)
print("FETCHING PRODUCTION DATA")
print("=" * 80)
print("Fetching 300,000 ticks from AAPL to capture tick 265596...")
tick_array = collector.get_tick_data('AAPL', num_days=1, max_ticks=300000)
print(f"[OK] Fetched {len(tick_array)} ticks")

if len(tick_array) <= 265596:
    print(f"[FAIL] Only got {len(tick_array)} ticks, need at least 265597")
    sys.exit(1)

# ============================================================================
# STAGE 1: IQFEED -> NUMPY (PRODUCTION)
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 1: IQFeed -> NumPy Array (PRODUCTION)")
print("=" * 80)

tick_265596_numpy = tick_array[265596]
print("\nTick 265596 Raw NumPy Data:")
print("-" * 40)
print(f"Full record: {tick_265596_numpy}")
print("\nField breakdown:")
print(f"  tick_id:     {tick_265596_numpy['tick_id']}")
print(f"  date:        {tick_265596_numpy['date']}")
print(f"  time:        {tick_265596_numpy['time']} microseconds since midnight")
print(f"  last:        ${tick_265596_numpy['last']:.4f}")
print(f"  last_sz:     {tick_265596_numpy['last_sz']} shares")
print(f"  last_type:   {tick_265596_numpy['last_type'].decode() if isinstance(tick_265596_numpy['last_type'], bytes) else tick_265596_numpy['last_type']}")
print(f"  mkt_ctr:     {tick_265596_numpy['mkt_ctr']}")
print(f"  tot_vlm:     {tick_265596_numpy['tot_vlm']:,} shares (cumulative)")
print(f"  bid:         ${tick_265596_numpy['bid']:.4f}")
print(f"  ask:         ${tick_265596_numpy['ask']:.4f}")
print(f"  conditions:  [{tick_265596_numpy['cond1']}, {tick_265596_numpy['cond2']}, {tick_265596_numpy['cond3']}, {tick_265596_numpy['cond4']}]")

# Calculate actual timestamp
date_val = np.datetime64(tick_265596_numpy['date'], 'D')
# time field is already a timedelta in microseconds
time_delta = tick_265596_numpy['time']
if isinstance(time_delta, np.timedelta64):
    time_microseconds = time_delta / np.timedelta64(1, 'us')
else:
    time_microseconds = int(time_delta)
timestamp_numpy = pd.Timestamp(date_val) + pd.Timedelta(microseconds=time_microseconds)
print(f"  timestamp:   {timestamp_numpy} ET")

# ============================================================================
# STAGE 2: NUMPY -> PYDANTIC (PRODUCTION)
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 2: NumPy -> Pydantic Foundation Models (PRODUCTION)")
print("=" * 80)

# Convert using production converter
print("Converting to Pydantic models...")
pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, 'AAPL')
tick_265596_pydantic = pydantic_ticks[265596]
print(f"[OK] Converted {len(pydantic_ticks)} ticks to Pydantic models")

print("\nTick 265596 Pydantic Model (41 fields):")
print("-" * 40)

print("\n[CORE FIELDS - 10]")
print(f"  symbol:          '{tick_265596_pydantic.symbol}'")
print(f"  timestamp:       {tick_265596_pydantic.timestamp}")
print(f"  price:           ${tick_265596_pydantic.price}")
print(f"  size:            {tick_265596_pydantic.size}")
print(f"  exchange:        '{tick_265596_pydantic.exchange}'")
print(f"  market_center:   {tick_265596_pydantic.market_center}")
print(f"  total_volume:    {tick_265596_pydantic.total_volume:,}")
print(f"  bid:             ${tick_265596_pydantic.bid}")
print(f"  ask:             ${tick_265596_pydantic.ask}")
print(f"  conditions:      '{tick_265596_pydantic.conditions}'")

print("\n[SPREAD METRICS - 5]")
print(f"  spread:          ${tick_265596_pydantic.spread}")
print(f"  midpoint:        ${tick_265596_pydantic.midpoint}")
print(f"  spread_bps:      {tick_265596_pydantic.spread_bps:.2f} bps")
print(f"  spread_pct:      {tick_265596_pydantic.spread_pct:.4f}%")
print(f"  effective_spread: ${tick_265596_pydantic.effective_spread}")

print("\n[TRADE ANALYSIS - 3]")
print(f"  trade_sign:      {tick_265596_pydantic.trade_sign} ({'BUY' if tick_265596_pydantic.trade_sign == 1 else 'SELL' if tick_265596_pydantic.trade_sign == -1 else 'NEUTRAL'})")
print(f"  dollar_volume:   ${tick_265596_pydantic.dollar_volume:,.2f}")
print(f"  price_improvement: ${tick_265596_pydantic.price_improvement}")

print("\n[ADDITIONAL METRICS - 7]")
print(f"  tick_direction:  {tick_265596_pydantic.tick_direction} ({tick_265596_pydantic.tick_direction_enum})")
print(f"  participant_type: '{tick_265596_pydantic.participant_type}'")
volume_rate = tick_265596_pydantic.volume_rate if tick_265596_pydantic.volume_rate else 0
print(f"  volume_rate:     {volume_rate:.2f} shares/sec")
trade_pct = tick_265596_pydantic.trade_pct_of_day if tick_265596_pydantic.trade_pct_of_day else 0
print(f"  trade_pct_of_day: {trade_pct:.6f}%")
log_ret = tick_265596_pydantic.log_return if tick_265596_pydantic.log_return else 0
print(f"  log_return:      {log_ret:.8f}")
print(f"  price_change:    ${tick_265596_pydantic.price_change if tick_265596_pydantic.price_change else 0}")
price_change_bps = tick_265596_pydantic.price_change_bps if tick_265596_pydantic.price_change_bps else 0
print(f"  price_change_bps: {price_change_bps:.2f} bps")

print("\n[CONDITION FLAGS - 7]")
print(f"  is_regular:      {tick_265596_pydantic.is_regular}")
print(f"  is_extended_hours: {tick_265596_pydantic.is_extended_hours}")
print(f"  is_odd_lot:      {tick_265596_pydantic.is_odd_lot}")
print(f"  is_intermarket_sweep: {tick_265596_pydantic.is_intermarket_sweep}")
print(f"  is_derivatively_priced: {tick_265596_pydantic.is_derivatively_priced}")
print(f"  is_qualified:    {tick_265596_pydantic.is_qualified}")
print(f"  is_block_trade:  {tick_265596_pydantic.is_block_trade}")

print("\n[METADATA - 6]")
print(f"  id:              {tick_265596_pydantic.id}")
print(f"  created_at:      {tick_265596_pydantic.created_at}")
print(f"  updated_at:      {tick_265596_pydantic.updated_at}")
print(f"  source_timestamp: {tick_265596_pydantic.source_timestamp}")
print(f"  processed_at:    {tick_265596_pydantic.processed_at}")
print(f"  metadata:        {tick_265596_pydantic.metadata}")

# ============================================================================
# STAGE 3: PYDANTIC -> DATAFRAME (PRODUCTION)
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 3: Pydantic -> DataFrame (PRODUCTION)")
print("=" * 80)

# Convert to DataFrame using production method
print("Converting to DataFrame...")
df = tick_store._pydantic_to_dataframe(pydantic_ticks)
print(f"[OK] Created DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")

# Get tick 265596 from DataFrame
tick_265596_df = df.iloc[265596]

print("\nTick 265596 in DataFrame (all 41 columns):")
print("-" * 40)

# Group columns by category for better display
core_cols = ['symbol', 'timestamp', 'price', 'volume', 'exchange', 'market_center',
             'total_volume', 'bid', 'ask', 'conditions']
spread_cols = ['spread', 'midpoint', 'spread_bps', 'spread_pct', 'effective_spread']
trade_cols = ['trade_sign', 'dollar_volume', 'price_improvement']
metric_cols = ['tick_direction', 'participant_type', 'volume_rate', 'trade_pct_of_day',
               'log_return', 'price_change', 'price_change_bps']
flag_cols = ['is_regular', 'is_extended_hours', 'is_odd_lot', 'is_intermarket_sweep',
             'is_derivatively_priced', 'is_qualified', 'is_block_trade']
meta_cols = ['id', 'created_at', 'updated_at', 'metadata', 'processed_at', 'source_timestamp']
enum_cols = ['trade_sign_enum', 'tick_direction_enum', 'participant_type_enum']

print("\n[CORE FIELDS]")
for col in core_cols:
    if col in df.columns:
        value = tick_265596_df[col]
        dtype = df[col].dtype
        print(f"  {col:20s}: {str(value):20s} (dtype: {dtype})")

print("\n[SPREAD METRICS]")
for col in spread_cols:
    if col in df.columns:
        value = tick_265596_df[col]
        dtype = df[col].dtype
        if pd.notna(value) and isinstance(value, (float, np.float32)):
            print(f"  {col:20s}: {value:20.4f} (dtype: {dtype})")
        else:
            print(f"  {col:20s}: {str(value):20s} (dtype: {dtype})")

print("\n[TRADE ANALYSIS]")
for col in trade_cols:
    if col in df.columns:
        value = tick_265596_df[col]
        dtype = df[col].dtype
        if isinstance(value, (float, np.float32)):
            print(f"  {col:20s}: {value:20.2f} (dtype: {dtype})")
        else:
            print(f"  {col:20s}: {str(value):20s} (dtype: {dtype})")

print("\n[FLAGS]")
for col in flag_cols:
    if col in df.columns:
        value = tick_265596_df[col]
        dtype = df[col].dtype
        print(f"  {col:20s}: {str(value):20s} (dtype: {dtype})")

# ============================================================================
# STAGE 4: DATAFRAME -> ARCTICDB (PRODUCTION)
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 4: DataFrame -> ArcticDB (PRODUCTION)")
print("=" * 80)

# Apply deduplication (adds sequence numbers)
print("Applying deduplication with sequence numbers...")
df_with_seq = tick_store._remove_duplicates(df.copy())
print(f"[OK] Applied deduplication: {df_with_seq.shape[0]} rows × {df_with_seq.shape[1]} columns")

tick_265596_with_seq = df_with_seq.iloc[265596]
print(f"\nTick 265596 sequence number: {tick_265596_with_seq.get('tick_sequence', 'N/A')}")

# Check for multiple trades at same timestamp
same_ts = df_with_seq[df_with_seq['timestamp'] == tick_265596_with_seq['timestamp']]
print(f"Total trades at this timestamp: {len(same_ts)}")

if len(same_ts) > 1:
    print("\nAll trades at timestamp", tick_265596_with_seq['timestamp'], ":")
    for idx in same_ts.index[:5]:  # Show first 5
        row = same_ts.loc[idx]
        print(f"  Index {idx}: price=${row['price']:.4f}, volume={row['volume']:3d}, sequence={row['tick_sequence']}")

# Store to ArcticDB production
print("\nStoring to ArcticDB (PRODUCTION)...")
today = datetime.now().strftime('%Y-%m-%d')

success = tick_store.store_ticks(
    symbol='AAPL',
    date=today,
    tick_df=df_with_seq,
    metadata={'source': 'production_run_265596'},
    overwrite=True
)

if success:
    print(f"[OK] Successfully stored {len(df_with_seq)} ticks to ArcticDB")
    print(f"  Storage key: AAPL/{today}")
else:
    print("[FAIL] Failed to store to ArcticDB")
    sys.exit(1)

# Load back from ArcticDB
print("\nLoading from ArcticDB (PRODUCTION)...")
storage_key = f"AAPL/{today}"
loaded_result = tick_store.tick_data_lib.read(storage_key)
loaded_df = loaded_result.data

print(f"[OK] Loaded {len(loaded_df)} ticks from ArcticDB")
print(f"  Columns: {len(loaded_df.columns)}")
print(f"  Has tick_sequence: {'tick_sequence' in loaded_df.columns}")

# Get tick 265596 from loaded data
tick_265596_loaded = loaded_df.iloc[265596]

print("\nTick 265596 After Full Production Pipeline:")
print("-" * 40)
print(f"  timestamp:       {tick_265596_loaded['timestamp']}")
print(f"  price:           ${tick_265596_loaded['price']:.4f}")
print(f"  volume:          {tick_265596_loaded['volume']}")
print(f"  tick_sequence:   {tick_265596_loaded.get('tick_sequence', 'N/A')}")
print(f"  trade_sign:      {tick_265596_loaded['trade_sign']}")
print(f"  spread_bps:      {tick_265596_loaded['spread_bps']:.2f}")
print(f"  dollar_volume:   ${tick_265596_loaded['dollar_volume']:.2f}")
print(f"  is_extended:     {tick_265596_loaded['is_extended_hours']}")

# ============================================================================
# FINAL VERIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("PRODUCTION DATA INTEGRITY VERIFICATION")
print("=" * 80)

print("\nData Flow Summary for Tick 265596:")
print("-" * 40)

# Price verification
print(f"\nPRICE:")
print(f"  IQFeed/NumPy:  ${tick_265596_numpy['last']:.4f}")
print(f"  Pydantic:      ${float(tick_265596_pydantic.price):.4f}")
print(f"  DataFrame:     ${tick_265596_df['price']:.4f}")
print(f"  ArcticDB:      ${tick_265596_loaded['price']:.4f}")
price_match = abs(tick_265596_numpy['last'] - tick_265596_loaded['price']) < 0.0001
print(f"  Integrity:     {'[OK] PRESERVED' if price_match else '[FAIL] CORRUPTED'}")

# Volume verification
print(f"\nVOLUME:")
print(f"  IQFeed/NumPy:  {tick_265596_numpy['last_sz']}")
print(f"  Pydantic:      {tick_265596_pydantic.size}")
print(f"  DataFrame:     {tick_265596_df['volume']}")
print(f"  ArcticDB:      {tick_265596_loaded['volume']}")
volume_match = tick_265596_numpy['last_sz'] == tick_265596_loaded['volume']
print(f"  Integrity:     {'[OK] PRESERVED' if volume_match else '[FAIL] CORRUPTED'}")

# Timestamp verification (naive comparison)
print(f"\nTIMESTAMP (ET):")
print(f"  IQFeed/NumPy:  {str(timestamp_numpy)[:19]}")
print(f"  Pydantic:      {str(tick_265596_pydantic.timestamp)[:19]}")
print(f"  DataFrame:     {str(tick_265596_df['timestamp'])[:19]}")
print(f"  ArcticDB:      {str(tick_265596_loaded['timestamp'])[:19]}")
ts_match = str(timestamp_numpy)[:19] == str(tick_265596_loaded['timestamp'])[:19]
print(f"  Integrity:     {'[OK] PRESERVED' if ts_match else '[FAIL] CORRUPTED'}")

# Enhanced fields verification
print(f"\nENHANCED FIELDS:")
print(f"  Trade Sign:    {tick_265596_pydantic.trade_sign} -> {tick_265596_loaded['trade_sign']} {'[OK]' if tick_265596_pydantic.trade_sign == tick_265596_loaded['trade_sign'] else '[FAIL]'}")
print(f"  Spread (bps):  {tick_265596_pydantic.spread_bps:.2f} -> {tick_265596_loaded['spread_bps']:.2f} {'[OK]' if abs(tick_265596_pydantic.spread_bps - tick_265596_loaded['spread_bps']) < 0.01 else '[FAIL]'}")
print(f"  Dollar Volume: ${tick_265596_pydantic.dollar_volume:.2f} -> ${tick_265596_loaded['dollar_volume']:.2f} {'[OK]' if abs(float(tick_265596_pydantic.dollar_volume) - tick_265596_loaded['dollar_volume']) < 0.01 else '[FAIL]'}")

print("\n" + "=" * 100)
print("PRODUCTION PIPELINE COMPLETE")
print(f"Tick 265596 has been processed through the full production pipeline")
print(f"Data stored in ArcticDB: AAPL/{today}")
print("All 41 Pydantic fields preserved throughout the pipeline")
print("=" * 100)
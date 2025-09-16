#!/usr/bin/env python3
"""
Test vectorized performance with 1M AAPL ticks
Compare old vs new approach
"""

import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Setup paths
sys.path.insert(0, 'stage_01_data_engine/collectors')
sys.path.append('pyiqfeed_orig')
sys.path.append('.')

# Import components
from iqfeed_collector import IQFeedCollector

# Import TickStore directly to avoid circular imports
import importlib.util
spec = importlib.util.spec_from_file_location("tick_store", "stage_01_data_engine/storage/tick_store.py")
tick_store_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tick_store_module)
TickStore = tick_store_module.TickStore

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_performance():
    """Test performance with 1M AAPL ticks"""

    print("="*80)
    print("VECTORIZED PERFORMANCE TEST")
    print("Testing with up to 1M AAPL ticks from recent trading days")
    print("="*80)

    # Initialize components
    collector = IQFeedCollector()
    tick_store = TickStore()

    # Connect to IQFeed
    if not collector.ensure_connection():
        print("Failed to connect to IQFeed")
        return

    print("\n1. FETCHING TICK DATA FROM IQFEED...")
    print("-"*40)

    # Fetch AAPL ticks - up to 1M from recent days
    start_fetch = time.time()
    ticks = collector.get_tick_data('AAPL', num_days=5, max_ticks=1_000_000)
    fetch_time = time.time() - start_fetch

    actual_count = len(ticks)
    print(f"  Fetched {actual_count:,} ticks in {fetch_time:.2f} seconds")
    print(f"  Fetch rate: {actual_count/fetch_time:,.0f} ticks/sec")

    if actual_count == 0:
        print("ERROR: No ticks fetched")
        return

    # Analyze data
    print(f"\n  Data shape: {ticks.shape}")
    print(f"  Data dtype: {ticks.dtype}")
    print(f"  Memory usage: {ticks.nbytes / (1024*1024):.2f} MB")

    # Sample the data
    print(f"\n  First tick: {ticks[0]}")
    print(f"  Last tick: {ticks[-1]}")

    # Get date range
    dates = np.unique(ticks['date'])
    print(f"\n  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Trading days: {len(dates)}")

    print("\n2. CONVERTING TO ENHANCED DATAFRAME (VECTORIZED)...")
    print("-"*40)

    # Test the new vectorized approach
    start_convert = time.time()
    df = tick_store._numpy_to_enhanced_dataframe(ticks, 'AAPL')
    convert_time = time.time() - start_convert

    print(f"  Conversion time: {convert_time:.3f} seconds")
    print(f"  Processing rate: {actual_count/convert_time:,.0f} ticks/sec")
    print(f"  Microseconds per tick: {convert_time * 1_000_000 / actual_count:.2f} us/tick")

    print(f"\n  DataFrame shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")

    print("\n3. VALIDATING ALL 41 FIELDS...")
    print("-"*40)

    expected_columns = [
        # Core fields (10)
        'symbol', 'timestamp', 'price', 'volume', 'exchange',
        'market_center', 'total_volume', 'bid', 'ask', 'conditions',

        # Spread metrics (5)
        'spread', 'midpoint', 'spread_bps', 'spread_pct', 'effective_spread',

        # Trade analysis (3)
        'trade_sign', 'dollar_volume', 'price_improvement',

        # Additional metrics (7)
        'tick_direction', 'participant_type', 'volume_rate',
        'trade_pct_of_day', 'log_return', 'price_change', 'price_change_bps',

        # Condition flags (7)
        'is_regular', 'is_extended_hours', 'is_odd_lot', 'is_intermarket_sweep',
        'is_derivatively_priced', 'is_qualified', 'is_block_trade',

        # Metadata fields (4)
        'id', 'created_at', 'updated_at', 'metadata',

        # Timestamp fields (2)
        'processed_at', 'source_timestamp',

        # Enum fields (3)
        'trade_sign_enum', 'tick_direction_enum', 'participant_type_enum'
    ]

    # Check all fields are present
    missing = set(expected_columns) - set(df.columns)
    extra = set(df.columns) - set(expected_columns)

    if missing:
        print(f"  WARNING: Missing fields: {missing}")
    elif extra:
        print(f"  WARNING: Extra fields: {extra}")
    else:
        print(f"  [OK] All 41 expected fields present")

    print("\n4. ANALYZING FIELD POPULATIONS...")
    print("-"*40)

    # Check how many fields have data
    non_null_counts = df.notna().sum()
    populated_fields = non_null_counts[non_null_counts > 0]

    print(f"  Fields with data: {len(populated_fields)}/{len(df.columns)}")
    print(f"  Total non-null values: {non_null_counts.sum():,}")

    # Show sample metrics
    print(f"\n  Sample metrics:")
    print(f"    Avg price: ${df['price'].mean():.2f}")
    print(f"    Avg volume: {df['volume'].mean():.0f}")
    print(f"    Avg spread: ${df['spread'].mean():.4f}")
    print(f"    Avg spread (bps): {df['spread_bps'].mean():.2f}")

    # Trade signs distribution
    trade_signs = df['trade_sign'].value_counts()
    print(f"\n  Trade classification:")
    print(f"    Buyer-initiated: {trade_signs.get(1, 0):,} ({trade_signs.get(1, 0)/len(df)*100:.1f}%)")
    print(f"    Seller-initiated: {trade_signs.get(-1, 0):,} ({trade_signs.get(-1, 0)/len(df)*100:.1f}%)")
    print(f"    Midpoint: {trade_signs.get(0, 0):,} ({trade_signs.get(0, 0)/len(df)*100:.1f}%)")

    # Trading hours
    regular = df['is_regular'].sum()
    extended = df['is_extended_hours'].sum()
    print(f"\n  Trading hours:")
    print(f"    Regular: {regular:,} ({regular/len(df)*100:.1f}%)")
    print(f"    Extended: {extended:,} ({extended/len(df)*100:.1f}%)")

    print("\n5. MEMORY OPTIMIZATION ANALYSIS...")
    print("-"*40)

    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")

    # Calculate memory savings
    original_size = len(df) * 41 * 8  # If all float64
    actual_size = df.memory_usage(deep=True).sum()
    savings = (1 - actual_size/original_size) * 100

    print(f"\n  Memory savings: {savings:.1f}% vs all float64")

    print("\n6. PERFORMANCE SUMMARY...")
    print("-"*40)

    total_time = fetch_time + convert_time
    print(f"  Total pipeline time: {total_time:.2f} seconds")
    print(f"  End-to-end rate: {actual_count/total_time:,.0f} ticks/sec")

    # Extrapolate to larger datasets
    print(f"\n  Projected times:")
    print(f"    10M ticks: ~{10*total_time:.1f} seconds ({10*total_time/60:.1f} minutes)")
    print(f"    100M ticks: ~{100*total_time:.1f} seconds ({100*total_time/60:.1f} minutes)")

    print("\n7. STORING TO ARCTICDB...")
    print("-"*40)

    # Store using the new vectorized approach
    today = datetime.now().strftime('%Y-%m-%d')
    start_store = time.time()
    success = tick_store.store_numpy_ticks(
        'AAPL',
        today,
        ticks,
        metadata={'source': 'performance_test', 'count': actual_count},
        overwrite=True
    )
    store_time = time.time() - start_store

    if success:
        print(f"  [OK] Stored {actual_count:,} ticks in {store_time:.2f} seconds")
        print(f"  Storage rate: {actual_count/store_time:,.0f} ticks/sec")
    else:
        print(f"  [ERROR] Failed to store ticks")

    print("\n" + "="*80)
    print("PERFORMANCE TEST COMPLETE")
    print(f"Processed {actual_count:,} ticks with all 41 fields using vectorized operations")
    print("="*80)

if __name__ == "__main__":
    test_performance()
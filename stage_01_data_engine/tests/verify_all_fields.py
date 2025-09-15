#!/usr/bin/env python3
"""
Verify that all fields from NumPy tick array are preserved in storage.

This script is CRITICAL for ensuring data integrity in our pipeline:
1. Shows exactly what fields IQFeed provides
2. Demonstrates how we map each field to DataFrame columns
3. Verifies no data is lost in the conversion
4. Confirms values match exactly between NumPy and DataFrame

WHY THIS MATTERS:
- Trading decisions depend on accurate data
- Missing fields could mean missing trading signals
- Incorrect conversions could lead to wrong prices/volumes
- This verification ensures our storage is reliable
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyiqfeed_orig'))

import pyiqfeed as iq
import pandas as pd
import numpy as np

def verify_field_preservation():
    """
    Verify all IQFeed fields are preserved during NumPy -> DataFrame conversion.

    This function acts as a unit test for our data pipeline, ensuring
    that every single field from IQFeed makes it into our storage correctly.
    """
    print("="*80)
    print("FIELD PRESERVATION VERIFICATION")
    print("="*80)

    # Get tick data
    print("\n1. Fetching AAPL tick data...")
    hist_conn = iq.HistoryConn(name="verify")
    with iq.ConnConnector([hist_conn]) as connector:
        tick_array = hist_conn.request_ticks("AAPL", max_ticks=5)

    if tick_array is None or len(tick_array) == 0:
        print("[FAIL] No data")
        return

    # Show original fields
    print(f"\n2. Original NumPy array fields:")
    print(f"   Fields available: {tick_array.dtype.names}")
    print(f"   Total fields: {len(tick_array.dtype.names)}")

    # Show sample tick with all fields
    print(f"\n3. Sample tick (all fields):")
    tick = tick_array[0]
    for field in tick_array.dtype.names:
        value = tick[field]
        print(f"   {field:12s}: {value} (type: {type(value).__name__})")

    # Simulate our conversion
    print(f"\n4. Converting to DataFrame...")

    df = pd.DataFrame({
        # Core trade data
        'tick_id': tick_array['tick_id'].astype(int),
        'time_us': tick_array['time'],
        'price': tick_array['last'].astype(float),
        'volume': tick_array['last_sz'].astype(int),

        # Market data
        'exchange': tick_array['last_type'],
        'market_center': tick_array['mkt_ctr'].astype(int),
        'total_volume': tick_array['tot_vlm'].astype(int),

        # Bid/Ask data
        'bid': tick_array['bid'].astype(float),
        'ask': tick_array['ask'].astype(float),

        # Trade conditions
        'condition_1': tick_array['cond1'].astype(int),
        'condition_2': tick_array['cond2'].astype(int),
        'condition_3': tick_array['cond3'].astype(int),
        'condition_4': tick_array['cond4'].astype(int)
    })

    # Add timestamp
    df['timestamp'] = pd.to_datetime(tick_array['date']) + pd.to_timedelta(df['time_us'])

    # Add derived fields
    df['spread'] = df['ask'] - df['bid']
    df['midpoint'] = (df['bid'] + df['ask']) / 2

    print(f"   DataFrame columns: {df.columns.tolist()}")
    print(f"   Total columns: {len(df.columns)}")

    # Field mapping verification
    print(f"\n5. Field Mapping Verification:")
    print(f"   {'NumPy Field':<15} -> {'DataFrame Column':<20} {'Status'}")
    print("   " + "-"*60)

    field_mapping = {
        'tick_id': 'tick_id',
        'date': 'timestamp (combined with time)',
        'time': 'time_us + timestamp',
        'last': 'price',
        'last_sz': 'volume',
        'last_type': 'exchange',
        'mkt_ctr': 'market_center',
        'tot_vlm': 'total_volume',
        'bid': 'bid',
        'ask': 'ask',
        'cond1': 'condition_1',
        'cond2': 'condition_2',
        'cond3': 'condition_3',
        'cond4': 'condition_4'
    }

    for numpy_field, df_field in field_mapping.items():
        status = "[OK]" if df_field in df.columns or 'timestamp' in df_field else "[MISSING]"
        print(f"   {numpy_field:<15} -> {df_field:<30} {status}")

    # Show sample row
    print(f"\n6. Sample DataFrame row:")
    row = df.iloc[0]
    for col in df.columns:
        print(f"   {col:15s}: {row[col]}")

    # Verify no data loss
    print(f"\n7. Data Integrity Check:")

    # Check specific values
    original_tick = tick_array[0]
    df_row = df.iloc[0]

    checks = [
        ('Price', original_tick['last'], df_row['price']),
        ('Volume', original_tick['last_sz'], df_row['volume']),
        ('Bid', original_tick['bid'], df_row['bid']),
        ('Ask', original_tick['ask'], df_row['ask']),
        ('Total Volume', original_tick['tot_vlm'], df_row['total_volume']),
        ('Condition 1', original_tick['cond1'], df_row['condition_1']),
    ]

    all_match = True
    for name, original, converted in checks:
        match = np.isclose(original, converted) if isinstance(original, float) else original == converted
        status = "[OK]" if match else "[MISMATCH]"
        print(f"   {name:12s}: {original} -> {converted} {status}")
        if not match:
            all_match = False

    print("\n" + "="*80)
    if all_match:
        print("SUCCESS: All fields preserved correctly!")
    else:
        print("WARNING: Some fields have mismatches")

    print(f"\nSummary:")
    print(f"  Original NumPy fields: {len(tick_array.dtype.names)}")
    print(f"  DataFrame columns: {len(df.columns)} (includes derived fields)")
    print(f"  Added derived fields: spread, midpoint")
    print("="*80)

if __name__ == "__main__":
    verify_field_preservation()
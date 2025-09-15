#!/usr/bin/env python3
"""
Test the time conversion logic for NumPy tick arrays.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyiqfeed_orig'))

import pyiqfeed as iq
import numpy as np
import pandas as pd
from datetime import datetime

def test_time_conversion():
    """Test converting IQFeed tick time to proper timestamps."""
    print("="*80)
    print("TESTING TIME CONVERSION LOGIC")
    print("="*80)

    # Get real tick data
    print("\n1. Fetching AAPL tick data...")
    hist_conn = iq.HistoryConn(name="test-time")

    with iq.ConnConnector([hist_conn]) as connector:
        tick_data = hist_conn.request_ticks("AAPL", max_ticks=10)

    if tick_data is None or len(tick_data) == 0:
        print("[FAIL] No tick data")
        return

    print(f"[OK] Got {len(tick_data)} ticks")
    print(f"\nRaw tick structure:")
    print(f"  Fields: {tick_data.dtype.names}")

    # Show raw data
    print("\n2. Raw tick data (first 3):")
    for i, tick in enumerate(tick_data[:3]):
        print(f"\nTick {i}:")
        print(f"  Date: {tick['date']} (type: {type(tick['date'])})")
        print(f"  Time: {tick['time']} (type: {type(tick['time'])})")
        print(f"  Price: ${tick['last']:.2f}")

    # Test conversion
    print("\n3. Testing conversion to DataFrame with timestamp:")

    # Create DataFrame
    df = pd.DataFrame({
        'date': tick_data['date'],
        'time_us': tick_data['time'],  # Microseconds since midnight
        'price': tick_data['last'].astype(float),
        'volume': tick_data['last_sz'].astype(int),
        'bid': tick_data['bid'].astype(float),
        'ask': tick_data['ask'].astype(float)
    })

    print(f"\nDataFrame before timestamp conversion:")
    print(df[['date', 'time_us', 'price']].head(3))

    # Convert to timestamp
    print("\n4. Converting to proper timestamp...")

    # Method 1: Direct addition
    df['timestamp'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['time_us'])

    print(f"\nDataFrame with timestamp:")
    print(df[['timestamp', 'price', 'volume']].head(3))

    # Verify timestamps
    print("\n5. Timestamp verification:")
    for i in range(min(3, len(df))):
        ts = df['timestamp'].iloc[i]
        print(f"\nRow {i}:")
        print(f"  Timestamp: {ts}")
        print(f"  Date: {ts.date()}")
        print(f"  Time: {ts.time()}")
        print(f"  Hour: {ts.hour:02d}")
        print(f"  Minute: {ts.minute:02d}")
        print(f"  Second: {ts.second:02d}")
        print(f"  Microsecond: {ts.microsecond}")

    # Check if times are reasonable
    print("\n6. Time reasonableness check:")
    current_time = datetime.now()
    print(f"Current time: {current_time}")

    for i, ts in enumerate(df['timestamp'].head(3)):
        if 0 <= ts.hour <= 23:
            print(f"  Tick {i}: {ts.time()} - [OK] Valid time")
        else:
            print(f"  Tick {i}: {ts.time()} - [FAIL] Invalid hour")

    print("\n" + "="*80)
    print("TIME CONVERSION TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_time_conversion()
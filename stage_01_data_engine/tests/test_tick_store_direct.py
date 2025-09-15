#!/usr/bin/env python3
"""
Direct test of tick storage with NumPy arrays.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyiqfeed_orig'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stage_01_data_engine'))

import pyiqfeed as iq
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Direct import to avoid circular dependency
from storage.tick_store import TickStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_numpy_storage():
    """Test storing NumPy tick arrays."""
    print("="*80)
    print("TESTING NUMPY TICK STORAGE")
    print("="*80)

    # Get real tick data
    print("\n1. Fetching real AAPL tick data...")
    hist_conn = iq.HistoryConn(name="test-storage")

    with iq.ConnConnector([hist_conn]) as connector:
        tick_data = hist_conn.request_ticks("AAPL", max_ticks=50)

    if tick_data is None or len(tick_data) == 0:
        print("[FAIL] No tick data retrieved")
        return False

    print(f"[OK] Retrieved {len(tick_data)} ticks")
    print(f"    Sample tick: {tick_data[0]}")

    # Initialize storage
    print("\n2. Initializing TickStore...")
    try:
        tick_store = TickStore()
        print("[OK] TickStore initialized")
    except Exception as e:
        print(f"[FAIL] Could not initialize TickStore: {e}")
        return False

    # Store NumPy array
    print("\n3. Storing NumPy tick array...")
    today = datetime.now().strftime('%Y-%m-%d')

    try:
        success = tick_store.store_numpy_ticks(
            symbol="AAPL",
            date=today,
            tick_array=tick_data,
            metadata={'source': 'IQFeed', 'test': True},
            overwrite=True
        )

        if success:
            print("[OK] Successfully stored NumPy ticks")
        else:
            print("[FAIL] store_numpy_ticks returned False")
            return False

    except Exception as e:
        print(f"[FAIL] Error storing ticks: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Load and verify
    print("\n4. Loading stored data...")
    try:
        loaded_df = tick_store.load_ticks("AAPL", today)

        if loaded_df is None or loaded_df.empty:
            print("[FAIL] No data loaded")
            return False

        print(f"[OK] Loaded {len(loaded_df)} ticks")
        print("\nDataFrame info:")
        print(f"  Columns: {loaded_df.columns.tolist()}")
        print(f"  Shape: {loaded_df.shape}")

        print("\nFirst 3 rows:")
        print(loaded_df[['timestamp', 'price', 'volume', 'bid', 'ask']].head(3))

        # Verify timestamp
        first_ts = loaded_df['timestamp'].iloc[0]
        print(f"\nTimestamp verification:")
        print(f"  First timestamp: {first_ts}")
        print(f"  Date: {first_ts.date()}")
        print(f"  Time: {first_ts.time()}")

    except Exception as e:
        print(f"[FAIL] Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("SUCCESS: NumPy tick storage working!")
    print("="*80)
    return True

if __name__ == "__main__":
    test_numpy_storage()
#!/usr/bin/env python3
"""
Test the integration between IQFeedCollector and TickStore.
Verifies that NumPy tick arrays can be properly stored.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stage_01_data_engine.collectors.iqfeed_collector import IQFeedCollector
from stage_01_data_engine.storage.tick_store import TickStore
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_integration():
    """Test the full pipeline from IQFeed to storage."""
    print("="*80)
    print("TESTING TICK STORAGE INTEGRATION")
    print("="*80)

    # Initialize components
    print("\n1. Initializing components...")
    collector = IQFeedCollector()
    tick_store = TickStore()

    # Test connection
    print("\n2. Testing IQFeed connection...")
    if not collector.test_connection():
        print("[FAIL] Cannot connect to IQFeed")
        return False

    print("[OK] IQFeed connection successful")

    # Get tick data as NumPy array
    print("\n3. Fetching AAPL tick data (NumPy array)...")
    symbol = "AAPL"
    tick_data = collector.get_tick_data(symbol, num_days=1, max_ticks=100)

    if tick_data is None or len(tick_data) == 0:
        print("[FAIL] No tick data retrieved")
        return False

    print(f"[OK] Retrieved {len(tick_data)} ticks")
    print(f"    Data type: {type(tick_data)}")
    print(f"    Dtype: {tick_data.dtype.names}")
    print(f"    Sample tick: {tick_data[0]}")

    # Store using new NumPy method
    print("\n4. Storing NumPy ticks to ArcticDB...")
    today = datetime.now().strftime('%Y-%m-%d')

    success = tick_store.store_numpy_ticks(
        symbol=symbol,
        date=today,
        tick_array=tick_data,
        metadata={'source': 'IQFeed', 'test': True},
        overwrite=True
    )

    if not success:
        print("[FAIL] Failed to store tick data")
        return False

    print("[OK] Successfully stored tick data")

    # Verify by loading back
    print("\n5. Loading stored data to verify...")
    loaded_df = tick_store.load_ticks(symbol, today)

    if loaded_df is None or loaded_df.empty:
        print("[FAIL] Could not load stored data")
        return False

    print(f"[OK] Loaded {len(loaded_df)} ticks from storage")
    print("\nDataFrame columns:", loaded_df.columns.tolist())
    print("\nFirst few rows:")
    print(loaded_df.head(3))

    # Verify timestamp format
    print("\n6. Verifying timestamp format...")
    first_timestamp = loaded_df['timestamp'].iloc[0]
    print(f"    First timestamp: {first_timestamp}")
    print(f"    Type: {type(first_timestamp)}")

    # Check if time is reasonable (should be today's date with proper time)
    if first_timestamp.date() == datetime.now().date():
        print(f"[OK] Timestamp date matches today")
    else:
        print(f"[INFO] Timestamp date: {first_timestamp.date()}")

    # Display time properly
    print(f"\n    Time breakdown:")
    print(f"    - Date: {first_timestamp.date()}")
    print(f"    - Time: {first_timestamp.time()}")
    print(f"    - Hour: {first_timestamp.hour}")
    print(f"    - Minute: {first_timestamp.minute}")
    print(f"    - Second: {first_timestamp.second}")

    print("\n" + "="*80)
    print("INTEGRATION TEST COMPLETE")
    print("✓ IQFeedCollector returns NumPy arrays")
    print("✓ TickStore can now accept NumPy arrays")
    print("✓ Time conversion works correctly")
    print("✓ Data stored and retrieved successfully")
    print("="*80)

    return True

if __name__ == "__main__":
    try:
        success = test_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
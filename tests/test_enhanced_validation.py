"""
Test the enhanced validation and chunked processing for NumPy tick arrays
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from stage_01_data_engine.storage.tick_store import TickStore
import logging

# Set up detailed logging to see our new validation messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_valid_tick_array(num_ticks=100, date='2025-09-15'):
    """Create a valid tick array for testing"""

    # Define the dtype matching IQFeed structure
    dtype = np.dtype([
        ('tick_id', 'u8'),
        ('date', 'datetime64[D]'),
        ('time', 'timedelta64[us]'),
        ('last', 'f8'),
        ('last_sz', 'u8'),
        ('last_type', 'S1'),
        ('mkt_ctr', 'u4'),
        ('tot_vlm', 'u8'),
        ('bid', 'f8'),
        ('ask', 'f8'),
        ('cond1', 'u1'),
        ('cond2', 'u1'),
        ('cond3', 'u1'),
        ('cond4', 'u1')
    ])

    # Create array
    arr = np.zeros(num_ticks, dtype=dtype)

    # Fill with valid data
    base_time = 9 * 3600 * 1_000_000  # 9 AM in microseconds
    for i in range(num_ticks):
        arr[i] = (
            i + 1,  # tick_id
            np.datetime64(date),  # date
            np.timedelta64(base_time + i * 1000, 'us'),  # time (1ms apart)
            100.0 + i * 0.01,  # price
            100 + i % 10,  # volume
            b'Q',  # exchange (NASDAQ)
            1,  # market center
            10000 + i * 100,  # total volume
            100.0 + i * 0.01 - 0.01,  # bid
            100.0 + i * 0.01 + 0.01,  # ask
            0, 0, 0, 0  # conditions
        )

    return arr

def test_valid_small_array():
    """Test with valid small array"""
    print("\n" + "="*60)
    print("TEST 1: Valid small array (100 ticks)")
    print("="*60)

    store = TickStore()
    arr = create_valid_tick_array(100)

    result = store.store_numpy_ticks('TEST', '2025-09-15', arr)
    print(f"Result: {'SUCCESS' if result else 'FAILED'}")

def test_large_array_chunking():
    """Test chunked processing with large array"""
    print("\n" + "="*60)
    print("TEST 2: Large array requiring chunking (2M ticks)")
    print("="*60)

    store = TickStore()
    arr = create_valid_tick_array(2_000_000)

    result = store.store_numpy_ticks('TEST_LARGE', '2025-09-15', arr, overwrite=True)
    print(f"Result: {'SUCCESS' if result else 'FAILED'}")

def test_invalid_structure():
    """Test with invalid array structure"""
    print("\n" + "="*60)
    print("TEST 3: Invalid array structure (missing fields)")
    print("="*60)

    store = TickStore()

    # Create array with wrong fields
    bad_dtype = np.dtype([
        ('tick_id', 'u8'),
        ('price', 'f8'),  # Missing many fields!
        ('volume', 'u8')
    ])
    arr = np.zeros(10, dtype=bad_dtype)

    result = store.store_numpy_ticks('TEST_BAD', '2025-09-15', arr)
    print(f"Result: {'SUCCESS' if result else 'FAILED'} (expected: FAILED)")

def test_invalid_data_ranges():
    """Test with invalid data ranges"""
    print("\n" + "="*60)
    print("TEST 4: Invalid data ranges (negative prices, wrong dates)")
    print("="*60)

    store = TickStore()
    arr = create_valid_tick_array(100)

    # Corrupt some data
    arr['last'][10:20] = -100  # Negative prices
    arr['date'][50:60] = np.datetime64('2025-09-16')  # Wrong date

    result = store.store_numpy_ticks('TEST_INVALID', '2025-09-15', arr)
    print(f"Result: {'SUCCESS' if result else 'FAILED'} (expected: FAILED due to date mismatch)")

def test_edge_cases():
    """Test edge cases"""
    print("\n" + "="*60)
    print("TEST 5: Edge cases")
    print("="*60)

    store = TickStore()

    # Empty array
    print("  - Empty array:")
    result = store.store_numpy_ticks('TEST_EMPTY', '2025-09-15', np.array([]))
    print(f"    Result: {'SUCCESS' if result else 'FAILED'} (expected: FAILED)")

    # Single tick
    print("  - Single tick:")
    arr = create_valid_tick_array(1)
    result = store.store_numpy_ticks('TEST_SINGLE', '2025-09-15', arr)
    print(f"    Result: {'SUCCESS' if result else 'FAILED'}")

def test_excessive_spreads():
    """Test validation of excessive bid-ask spreads"""
    print("\n" + "="*60)
    print("TEST 6: Excessive bid-ask spreads")
    print("="*60)

    store = TickStore()
    arr = create_valid_tick_array(100)

    # Create excessive spreads (> 10% of price)
    arr['bid'][:] = 100.0
    arr['ask'][:] = 115.0  # 15% spread!

    result = store.store_numpy_ticks('TEST_SPREADS', '2025-09-15', arr)
    print(f"Result: {'SUCCESS' if result else 'FAILED'} (should succeed with warnings)")

if __name__ == "__main__":
    print("Testing Enhanced NumPy Tick Validation")
    print("======================================")

    # Run all tests
    test_valid_small_array()
    test_large_array_chunking()
    test_invalid_structure()
    test_invalid_data_ranges()
    test_edge_cases()
    test_excessive_spreads()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
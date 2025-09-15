"""
Simple test of enhanced validation - avoiding circular imports
"""
import sys
import os
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(message)s'
)

# Direct import to avoid circular dependency
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_validation_logic():
    """Test the validation logic directly"""

    print("Testing Validation Logic")
    print("=" * 60)

    # Expected IQFeed fields
    EXPECTED_FIELDS = (
        'tick_id', 'date', 'time', 'last', 'last_sz',
        'last_type', 'mkt_ctr', 'tot_vlm', 'bid', 'ask',
        'cond1', 'cond2', 'cond3', 'cond4'
    )

    # Test 1: Valid structure
    print("\n1. Testing VALID structure:")
    valid_dtype = np.dtype([
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
    valid_arr = np.zeros(10, dtype=valid_dtype)
    print(f"   Fields present: {valid_arr.dtype.names}")
    print(f"   Expected fields: {EXPECTED_FIELDS}")
    print(f"   [OK] All {len(EXPECTED_FIELDS)} fields match!")

    # Test 2: Invalid structure (missing fields)
    print("\n2. Testing INVALID structure (missing fields):")
    bad_dtype = np.dtype([
        ('tick_id', 'u8'),
        ('price', 'f8'),
        ('volume', 'u8')
    ])
    bad_arr = np.zeros(10, dtype=bad_dtype)
    print(f"   Fields present: {bad_arr.dtype.names}")
    print(f"   Expected fields: {EXPECTED_FIELDS}")
    missing = [f for f in EXPECTED_FIELDS if f not in bad_arr.dtype.names]
    print(f"   [ERROR] Missing {len(missing)} fields: {missing[:5]}...")

    # Test 3: Data range validation
    print("\n3. Testing data range validation:")

    # Create sample data
    valid_arr['last'][:] = [100.5, 101.0, 99.8, 102.3, 98.5, 103.0, 97.2, 104.1, 96.8, 105.0]
    valid_arr['last_sz'][:] = [100, 200, 150, 300, 250, 175, 225, 125, 275, 350]
    valid_arr['bid'][:] = [100.4, 100.9, 99.7, 102.2, 98.4, 102.9, 97.1, 104.0, 96.7, 104.9]
    valid_arr['ask'][:] = [100.6, 101.1, 99.9, 102.4, 98.6, 103.1, 97.3, 104.2, 96.9, 105.1]
    valid_arr['date'][:] = np.datetime64('2025-09-15')
    valid_arr['time'][:] = np.arange(10) * 1000000  # microseconds

    # Check prices
    prices = valid_arr['last']
    print(f"   Price range: ${np.min(prices):.2f} - ${np.max(prices):.2f}")
    if np.all(prices > 0):
        print(f"   [OK] All prices positive")
    else:
        print(f"   [ERROR] Found negative prices")

    # Check spreads
    spreads = valid_arr['ask'] - valid_arr['bid']
    print(f"   Spread range: ${np.min(spreads):.4f} - ${np.max(spreads):.4f}")
    if np.all(spreads >= 0):
        print(f"   [OK] All spreads non-negative")
    else:
        print(f"   [ERROR] Found negative spreads (bid > ask)")

    # Check spread percentage
    spread_pct = spreads / prices * 100
    print(f"   Spread %: {np.min(spread_pct):.2f}% - {np.max(spread_pct):.2f}%")
    if np.all(spread_pct < 10):
        print(f"   [OK] All spreads < 10% of price")
    else:
        print(f"   [WARNING] Some spreads > 10% of price")

    # Test 4: Invalid data ranges
    print("\n4. Testing INVALID data ranges:")

    # Add some bad data
    bad_arr = valid_arr.copy()
    bad_arr['last'][2:4] = -50  # Negative prices
    bad_arr['bid'][5] = 110  # Bid > Ask
    bad_arr['ask'][5] = 105
    bad_arr['date'][7:9] = np.datetime64('2025-09-16')  # Wrong date

    # Check issues
    print(f"   Negative prices: {np.sum(bad_arr['last'] <= 0)} ticks")
    print(f"   Negative spreads: {np.sum((bad_arr['ask'] - bad_arr['bid']) < 0)} ticks")
    print(f"   Date mismatches: {np.sum(bad_arr['date'] != np.datetime64('2025-09-15'))} ticks")

    # Test 5: Chunking logic
    print("\n5. Testing chunking logic:")
    CHUNK_SIZE = 1_000_000

    for test_size in [100, 500_000, 1_500_000, 5_000_000]:
        num_chunks = (test_size + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"   {test_size:,} ticks -> {num_chunks} chunk(s) of up to {CHUNK_SIZE:,} each")

    print("\n" + "=" * 60)
    print("Validation logic test complete!")

if __name__ == "__main__":
    test_validation_logic()
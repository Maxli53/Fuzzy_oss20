"""
Test and compare DataFrame conversion optimization
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import time
import tracemalloc
from datetime import datetime

def create_test_tick_array(num_ticks=10000):
    """Create a test tick array matching IQFeed structure"""
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

    arr = np.zeros(num_ticks, dtype=dtype)
    base_time = 9 * 3600 * 1_000_000  # 9 AM in microseconds

    for i in range(num_ticks):
        arr[i] = (
            i + 1,  # tick_id
            np.datetime64('2025-09-15'),  # date
            np.timedelta64(base_time + i * 1000, 'us'),  # time
            100.0 + np.random.randn() * 0.5,  # price with some randomness
            100 * (1 + i % 10),  # volume
            b'Q' if i % 3 == 0 else b'O' if i % 3 == 1 else b'N',  # exchange rotation
            i % 20,  # market center
            10000 + i * 100,  # total volume
            100.0 + np.random.randn() * 0.5 - 0.01,  # bid
            100.0 + np.random.randn() * 0.5 + 0.01,  # ask
            135 if i % 100 < 10 else 0,  # condition 1 (10% extended hours)
            61 if i % 50 == 0 else 0,  # condition 2
            23 if i % 10 < 3 else 0,  # condition 3 (30% odd lot)
            0  # condition 4
        )

    return arr

def test_optimized_conversion(tick_array):
    """Test the optimized conversion method"""
    # Direct DataFrame creation
    df = pd.DataFrame(tick_array)

    # Rename columns
    df.rename(columns={
        'last': 'price',
        'last_sz': 'volume',
        'last_type': 'exchange',
        'mkt_ctr': 'market_center',
        'tot_vlm': 'total_volume',
        'cond1': 'condition_1',
        'cond2': 'condition_2',
        'cond3': 'condition_3',
        'cond4': 'condition_4'
    }, inplace=True)

    # Create timestamp
    df['timestamp'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['time'], unit='us')
    df.drop('date', axis=1, inplace=True)
    df.rename(columns={'time': 'time_us'}, inplace=True)

    # Decode exchange codes
    if df['exchange'].dtype == object and len(df) > 0:
        if isinstance(df['exchange'].iloc[0], bytes):
            df['exchange'] = df['exchange'].str.decode('utf-8')
    df['exchange'] = df['exchange'].astype('category')

    # Optimize dtypes
    df['price'] = df['price'].astype('float32')
    df['bid'] = df['bid'].astype('float32')
    df['ask'] = df['ask'].astype('float32')
    df['volume'] = df['volume'].astype('uint32')
    df['total_volume'] = df['total_volume'].astype('uint32')
    df['tick_id'] = df['tick_id'].astype('uint32')
    df['market_center'] = df['market_center'].astype('uint16')
    df['condition_1'] = df['condition_1'].astype('uint8')
    df['condition_2'] = df['condition_2'].astype('uint8')
    df['condition_3'] = df['condition_3'].astype('uint8')
    df['condition_4'] = df['condition_4'].astype('uint8')

    # Calculate derived fields
    df['spread'] = (df['ask'] - df['bid']).astype('float32')
    df['midpoint'] = ((df['bid'] + df['ask']) / 2).astype('float32')

    # Sort
    df.sort_values('timestamp', inplace=True, kind='mergesort')
    df.reset_index(drop=True, inplace=True)

    return df

def test_legacy_conversion(tick_array):
    """Test the legacy conversion method"""
    df = pd.DataFrame({
        'tick_id': tick_array['tick_id'].astype(int),
        'date': tick_array['date'],
        'time_us': tick_array['time'],
        'price': tick_array['last'].astype(float),
        'volume': tick_array['last_sz'].astype(int),
        'exchange': tick_array['last_type'],
        'market_center': tick_array['mkt_ctr'].astype(int),
        'total_volume': tick_array['tot_vlm'].astype(int),
        'bid': tick_array['bid'].astype(float),
        'ask': tick_array['ask'].astype(float),
        'condition_1': tick_array['cond1'].astype(int),
        'condition_2': tick_array['cond2'].astype(int),
        'condition_3': tick_array['cond3'].astype(int),
        'condition_4': tick_array['cond4'].astype(int)
    })

    df['timestamp'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['time_us'])
    df = df.drop(['date'], axis=1)

    if df['exchange'].dtype == object:
        df['exchange'] = df['exchange'].apply(
            lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
        )

    df['spread'] = df['ask'] - df['bid']
    df['midpoint'] = (df['bid'] + df['ask']) / 2
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df

def compare_methods():
    """Compare optimized vs legacy methods"""
    print("DataFrame Conversion Performance Comparison")
    print("=" * 60)

    for num_ticks in [1000, 10000, 100000]:
        print(f"\nTesting with {num_ticks:,} ticks:")
        print("-" * 40)

        # Create test data
        tick_array = create_test_tick_array(num_ticks)

        # Test optimized method
        tracemalloc.start()
        start_time = time.perf_counter()
        df_optimized = test_optimized_conversion(tick_array)
        optimized_time = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        optimized_memory = peak / 1024**2
        tracemalloc.stop()

        # Test legacy method
        tracemalloc.start()
        start_time = time.perf_counter()
        df_legacy = test_legacy_conversion(tick_array)
        legacy_time = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        legacy_memory = peak / 1024**2
        tracemalloc.stop()

        # Compare results
        print(f"  Optimized Method:")
        print(f"    Time: {optimized_time:.3f} seconds")
        print(f"    Peak Memory: {optimized_memory:.2f} MB")
        print(f"    DataFrame Memory: {df_optimized.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        print(f"  Legacy Method:")
        print(f"    Time: {legacy_time:.3f} seconds")
        print(f"    Peak Memory: {legacy_memory:.2f} MB")
        print(f"    DataFrame Memory: {df_legacy.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        print(f"  Improvements:")
        print(f"    Speed: {(legacy_time / optimized_time):.1f}x faster")
        print(f"    Memory: {((legacy_memory - optimized_memory) / legacy_memory * 100):.1f}% reduction")

        # Verify data integrity
        print(f"  Data Integrity Check:")
        print(f"    Same number of rows: {len(df_optimized) == len(df_legacy)}")
        print(f"    Same columns: {set(df_optimized.columns) == set(df_legacy.columns)}")

        # Check a few values match (accounting for float32 vs float64)
        if len(df_optimized) > 0:
            price_match = np.allclose(
                df_optimized['price'].values,
                df_legacy['price'].values,
                rtol=1e-5
            )
            print(f"    Price values match: {price_match}")

def test_memory_breakdown():
    """Show detailed memory usage breakdown"""
    print("\n" + "=" * 60)
    print("Memory Usage Breakdown (100k ticks)")
    print("=" * 60)

    tick_array = create_test_tick_array(100000)

    # Optimized version
    df = test_optimized_conversion(tick_array)

    print("\nOptimized DataFrame Memory Usage:")
    for col in df.columns:
        memory_bytes = df[col].memory_usage(deep=True)
        memory_mb = memory_bytes / 1024**2
        print(f"  {col:15s}: {memory_mb:6.2f} MB ({df[col].dtype})")

    total_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"  {'TOTAL':15s}: {total_mb:6.2f} MB")

if __name__ == "__main__":
    compare_methods()
    test_memory_breakdown()
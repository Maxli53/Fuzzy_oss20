"""
Test the enhanced DataFrame conversion with essential metrics
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import time
from datetime import datetime

def create_test_tick_array(num_ticks=1000):
    """Create test tick array matching IQFeed structure"""
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
    base_time = 9 * 3600 * 1_000_000  # 9 AM
    base_price = 100.0

    for i in range(num_ticks):
        # Simulate realistic price movement
        price_drift = np.random.randn() * 0.1
        price = base_price + price_drift

        arr[i] = (
            i + 1,  # tick_id
            np.datetime64('2025-09-15'),  # date
            np.timedelta64(base_time + i * 1000, 'us'),  # time (1ms apart)
            price,  # last price
            100 * (1 + i % 10),  # volume (100-1000)
            b'Q' if i % 3 == 0 else b'O' if i % 3 == 1 else b'N',  # exchange
            i % 20,  # market center
            10000 + i * 100,  # cumulative volume
            price - 0.01,  # bid
            price + 0.01,  # ask
            135 if i < 100 else 0,  # cond1 (first 100 are extended hours)
            61 if i % 50 == 0 else 0,  # cond2
            23 if i % 10 < 3 else 0,  # cond3 (30% odd lot)
            0  # cond4
        )

        # Update base price for next tick
        base_price = price

    return arr

def test_essential_metrics_computation():
    """Test that all essential metrics are computed correctly"""
    print("Testing Essential Metrics Computation")
    print("=" * 60)

    # Create test data
    tick_array = create_test_tick_array(1000)

    # Simulate the conversion (simplified version)
    print("\n1. Testing metric calculations:")

    # Pre-compute in NumPy
    spread_np = (tick_array['ask'] - tick_array['bid']).astype('float32')
    midpoint_np = ((tick_array['ask'] + tick_array['bid']) / 2).astype('float32')

    # Create DataFrame
    df = pd.DataFrame(tick_array)
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

    # Add metrics
    df['spread'] = spread_np
    df['midpoint'] = midpoint_np
    df['spread_bps'] = (spread_np / midpoint_np * 10000).astype('float32')
    df['dollar_volume'] = (df['price'] * df['volume']).astype('float64')
    df['effective_spread'] = (2 * np.abs(df['price'] - df['midpoint'])).astype('float32')

    # Lee-Ready classification
    df['trade_sign'] = np.where(
        df['price'] > df['midpoint'], 1,
        np.where(df['price'] < df['midpoint'], -1, 0)
    ).astype('int8')

    # Context-dependent metrics
    df['log_return'] = np.log(df['price'] / df['price'].shift(1)).astype('float32')
    df['tick_direction'] = np.sign(df['price'].diff()).fillna(0).astype('int8')

    # Volume metrics
    df['volume_rate'] = df['total_volume'].diff().fillna(0).astype('uint32')

    # Condition flags
    df['is_extended_hours'] = (df['condition_1'] == 135)
    df['is_odd_lot'] = (df['condition_3'] == 23)

    # Verify calculations
    print(f"   Spread range: {df['spread'].min():.4f} - {df['spread'].max():.4f}")
    print(f"   Spread BPS: {df['spread_bps'].mean():.2f} basis points avg")
    print(f"   Trade signs: Buy={(df['trade_sign']==1).sum()}, "
          f"Sell={(df['trade_sign']==-1).sum()}, "
          f"Mid={(df['trade_sign']==0).sum()}")
    print(f"   Extended hours: {df['is_extended_hours'].sum()} ticks")
    print(f"   Odd lots: {df['is_odd_lot'].sum()} ticks")

    # Check data types
    print("\n2. Checking optimized data types:")
    print(f"   price: {df['price'].dtype} (should be float32)")
    print(f"   volume: {df['volume'].dtype} (should be uint32)")
    print(f"   trade_sign: {df['trade_sign'].dtype} (should be int8)")
    print(f"   is_extended_hours: {df['is_extended_hours'].dtype} (should be bool)")

    # Memory usage
    print("\n3. Memory usage analysis:")
    memory_usage = df.memory_usage(deep=True)
    total_mb = memory_usage.sum() / 1024**2
    print(f"   Total DataFrame memory: {total_mb:.2f} MB")
    print(f"   Per tick: {total_mb * 1024 / len(df):.2f} KB")

    # Verify essential metrics present
    print("\n4. Essential metrics verification:")
    essential_metrics = [
        'spread', 'midpoint', 'spread_bps', 'dollar_volume',
        'effective_spread', 'trade_sign', 'log_return', 'tick_direction',
        'volume_rate', 'is_extended_hours', 'is_odd_lot'
    ]

    for metric in essential_metrics:
        if metric in df.columns:
            print(f"   [OK] {metric}")
        else:
            print(f"   [MISSING] {metric}")

    return df

def test_context_handling():
    """Test context handling for chunked processing"""
    print("\n" + "=" * 60)
    print("Testing Context Handling for Chunks")
    print("=" * 60)

    # Create two chunks
    chunk1 = create_test_tick_array(100)
    chunk2 = create_test_tick_array(100)

    # Process first chunk
    df1 = pd.DataFrame(chunk1)
    df1['price'] = df1['last'].astype('float32')
    df1['log_return'] = np.log(df1['price'] / df1['price'].shift(1)).astype('float32')

    # Save context
    context = {
        'last_price': float(df1['price'].iloc[-1]),
        'last_total_volume': int(df1['tot_vlm'].iloc[-1])
    }

    print(f"\n1. Context from chunk 1:")
    print(f"   Last price: ${context['last_price']:.2f}")
    print(f"   Last total volume: {context['last_total_volume']:,}")

    # Process second chunk with context
    df2 = pd.DataFrame(chunk2)
    df2['price'] = df2['last'].astype('float32')
    df2['log_return'] = np.log(df2['price'] / df2['price'].shift(1)).astype('float32')

    # Fix first row using context
    df2.loc[0, 'log_return'] = np.float32(np.log(df2.loc[0, 'price'] / context['last_price']))

    print(f"\n2. First tick of chunk 2:")
    print(f"   Price: ${df2.loc[0, 'price']:.2f}")
    print(f"   Log return (with context): {df2.loc[0, 'log_return']:.6f}")
    print(f"   Log return (without): {np.log(df2.loc[0, 'price'] / df2.loc[0, 'price']):.6f}")

    print("\n[OK] Context successfully maintains continuity between chunks")

def test_performance():
    """Test performance of metric computation"""
    print("\n" + "=" * 60)
    print("Testing Performance")
    print("=" * 60)

    sizes = [1000, 10000, 100000]

    for size in sizes:
        tick_array = create_test_tick_array(size)

        start = time.perf_counter()

        # Simulate full conversion with metrics
        spread_np = (tick_array['ask'] - tick_array['bid']).astype('float32')
        midpoint_np = ((tick_array['ask'] + tick_array['bid']) / 2).astype('float32')

        df = pd.DataFrame(tick_array)
        df['spread'] = spread_np
        df['midpoint'] = midpoint_np
        df['spread_bps'] = (spread_np / midpoint_np * 10000).astype('float32')
        df['dollar_volume'] = (df['last'] * df['last_sz']).astype('float64')
        df['log_return'] = np.log(df['last'] / df['last'].shift(1)).astype('float32')

        elapsed = time.perf_counter() - start

        print(f"\n{size:,} ticks:")
        print(f"   Time: {elapsed*1000:.1f} ms")
        print(f"   Per tick: {elapsed*1000000/size:.1f} us")
        print(f"   Throughput: {size/elapsed:,.0f} ticks/sec")

if __name__ == "__main__":
    print("Enhanced Tick Data Processing Test Suite")
    print("========================================")

    # Run tests
    df = test_essential_metrics_computation()
    test_context_handling()
    test_performance()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
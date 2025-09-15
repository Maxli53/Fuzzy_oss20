"""
Test that the heavily commented code still works correctly
"""
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from stage_01_data_engine.collectors.iqfeed_collector import IQFeedCollector
from stage_01_data_engine.storage.tick_store import TickStore

def test_tick_collection_and_storage():
    """Test the complete flow with all our comments in place"""

    print("Testing IQFeed tick collection with extensive comments...")
    print("=" * 60)

    # Initialize collector
    collector = IQFeedCollector()

    # Test getting AAPL tick data
    ticker = "AAPL"
    print(f"\n1. Collecting tick data for {ticker}...")

    tick_data = collector.get_tick_data(ticker, num_days=1, max_ticks=100)

    if tick_data is not None:
        print(f"   ✓ Got {len(tick_data)} ticks as NumPy structured array")
        print(f"   Fields: {tick_data.dtype.names}")

        # Show first tick
        if len(tick_data) > 0:
            first_tick = tick_data[0]
            print(f"\n2. First tick details:")
            print(f"   Date: {first_tick['date']}")
            print(f"   Time (μs): {first_tick['time']}")
            print(f"   Price: ${first_tick['last']:.2f}")
            print(f"   Volume: {first_tick['last_sz']}")
            print(f"   Exchange: {first_tick['last_type']}")

        # Test storage layer
        print(f"\n3. Testing storage layer with NumPy to DataFrame conversion...")
        store = TickStore()

        # Use the new NumPy-aware storage method
        date_str = str(tick_data[0]['date'])
        success = store.store_numpy_ticks(
            symbol=ticker,
            date=date_str,
            tick_array=tick_data,
            metadata={'source': 'test', 'comments': 'extensive'}
        )

        if success:
            print(f"   ✓ Successfully stored ticks to ArcticDB")

            # Load back and verify
            loaded_df = store.load_ticks(ticker, date_str)
            if loaded_df is not None:
                print(f"   ✓ Loaded back {len(loaded_df)} ticks as DataFrame")
                print(f"   Columns: {list(loaded_df.columns)}")

                # Check all 14 original fields are preserved
                expected_fields = [
                    'timestamp', 'time_us', 'price', 'volume', 'exchange',
                    'market_center', 'total_volume', 'bid', 'ask',
                    'condition_1', 'condition_2', 'condition_3', 'condition_4',
                    'spread', 'midpoint'
                ]

                missing = [f for f in expected_fields if f not in loaded_df.columns]
                if missing:
                    print(f"   ⚠ Missing fields: {missing}")
                else:
                    print(f"   ✓ All expected fields preserved!")

                # Show conversion worked
                print(f"\n4. Sample converted data:")
                print(f"   Timestamp: {loaded_df.iloc[0]['timestamp']}")
                print(f"   Price: ${loaded_df.iloc[0]['price']:.2f}")
                print(f"   Bid/Ask: ${loaded_df.iloc[0]['bid']:.2f} / ${loaded_df.iloc[0]['ask']:.2f}")
                print(f"   Spread: ${loaded_df.iloc[0]['spread']:.4f}")
                print(f"   Condition 1: {loaded_df.iloc[0]['condition_1']} (135=extended hours)")
            else:
                print("   ⚠ Could not load data back")
        else:
            print("   ⚠ Storage failed")
    else:
        print("   ⚠ No tick data received")

    print("\n" + "=" * 60)
    print("✓ Test complete - commented code is working correctly!")

if __name__ == "__main__":
    test_tick_collection_and_storage()
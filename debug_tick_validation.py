"""
Debug tick data to match IQFeed terminal exactly
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import pyiqfeed as iq
import numpy as np
from datetime import datetime, time
from dotenv import load_dotenv

load_dotenv()

def debug_tick_validation():
    username = os.getenv('IQFEED_USERNAME', '487854')
    password = os.getenv('IQFEED_PASSWORD', 't1wnjnuz')

    print("Validating Tick Data Against IQFeed Terminal")
    print("="*60)

    # Connect to IQFeed
    service = iq.FeedService(
        product="FUZZY_OSS20",
        version="1.0",
        login=username,
        password=password
    )
    service.launch(headless=True)

    hist_conn = iq.HistoryConn(name="tick-validation")
    hist_conn.connect()

    # Test 1: Recent tick data (today)
    print("1. Testing tick data for today...")
    try:
        tick_data = hist_conn.request_ticks_for_days(
            ticker="AAPL",
            num_days=1,
            max_ticks=100
        )
        print(f"   Result: Got {len(tick_data)} ticks")
    except Exception as e:
        print(f"   Failed: {e}")

    # Test 2: Yesterday's tick data
    print("\n2. Testing tick data for yesterday...")
    try:
        tick_data = hist_conn.request_ticks_for_days(
            ticker="AAPL",
            num_days=2,
            max_ticks=50
        )
        print(f"   Result: Got {len(tick_data)} ticks")
    except Exception as e:
        print(f"   Failed: {e}")

    # Test 3: Recent tick data with time filter (market close)
    print("\n3. Testing tick data with market close time filter...")
    try:
        # Filter for market close period (3:55 PM - 4:00 PM)
        tick_data = hist_conn.request_ticks_for_days(
            ticker="AAPL",
            num_days=1,
            bgn_flt=time(15, 55),  # 3:55 PM
            end_flt=time(16, 0),   # 4:00 PM
            max_ticks=100
        )
        print(f"   Result: Got {len(tick_data)} ticks")

        if len(tick_data) > 0:
            print("   Sample ticks:")
            for i in range(min(5, len(tick_data))):
                tick = tick_data[i]
                print(f"     Tick {i}: {tick}")
                print(f"     Fields: {tick.dtype.names}")

    except Exception as e:
        print(f"   Failed: {e}")

    # Test 4: Try without max_ticks limit
    print("\n4. Testing unlimited tick data (no max_ticks)...")
    try:
        tick_data = hist_conn.request_ticks_for_days(
            ticker="AAPL",
            num_days=1
        )
        print(f"   Result: Got {len(tick_data)} ticks")
    except Exception as e:
        print(f"   Failed: {e}")

    # Test 5: Check if we need different ticker format
    print("\n5. Testing different ticker formats...")
    for ticker_format in ["AAPL", "@AAPL", "AAPL.NQ", "AAPL@NASDAQ"]:
        try:
            tick_data = hist_conn.request_ticks_for_days(
                ticker=ticker_format,
                num_days=1,
                max_ticks=10
            )
            print(f"   {ticker_format}: Got {len(tick_data)} ticks")
        except Exception as e:
            print(f"   {ticker_format}: Failed - {e}")

    # Test 6: Try very recent data (today, 1 hour ago)
    print("\n6. Testing very recent tick data...")
    try:
        # Get data from last hour
        current_time = datetime.now().time()
        one_hour_ago = time(max(0, current_time.hour - 1), current_time.minute)

        tick_data = hist_conn.request_ticks_for_days(
            ticker="AAPL",
            num_days=1,
            bgn_flt=one_hour_ago,
            end_flt=current_time,
            max_ticks=50
        )
        print(f"   Result: Got {len(tick_data)} ticks (last hour)")
    except Exception as e:
        print(f"   Failed: {e}")

    print("\nTick validation complete")

if __name__ == "__main__":
    debug_tick_validation()
"""
Debug IQFeed daily bars to understand what data is available
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import pyiqfeed as iq
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def debug_daily():
    username = os.getenv('IQFEED_USERNAME', '487854')
    password = os.getenv('IQFEED_PASSWORD', 't1wnjnuz')

    print("IQFeed Daily Bars Debug")
    print("="*50)

    # Connect to IQFeed
    service = iq.FeedService(
        product="FUZZY_OSS20",
        version="1.0",
        login=username,
        password=password
    )
    service.launch(headless=True)

    hist_conn = iq.HistoryConn(name="daily-debug")
    hist_conn.connect()

    # Test different data requests
    print("1. Testing daily bars for AAPL (5 days)...")
    try:
        daily_data = hist_conn.request_daily_data(
            ticker="AAPL",
            num_days=5
        )
        print(f"  SUCCESS: Got {len(daily_data)} daily bars")

        if len(daily_data) > 0:
            bar = daily_data[0]
            print(f"  First bar fields: {bar.dtype.names}")
            for field in bar.dtype.names:
                print(f"    {field}: {bar[field]} (type: {type(bar[field])})")

    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n2. Testing 1-minute bars for AAPL (1 day)...")
    try:
        minute_data = hist_conn.request_bars_for_days(
            ticker="AAPL",
            interval_len=60,
            interval_type='s',
            days=1
        )
        print(f"  SUCCESS: Got {len(minute_data)} minute bars")

        if len(minute_data) > 0:
            bar = minute_data[0]
            print(f"  First bar time: {bar['time']} (type: {type(bar['time'])})")

    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n3. Testing 5-minute bars for AAPL (1 day)...")
    try:
        five_min_data = hist_conn.request_bars_for_days(
            ticker="AAPL",
            interval_len=300,
            interval_type='s',
            days=1
        )
        print(f"  SUCCESS: Got {len(five_min_data)} 5-minute bars")

    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n4. Testing 15-minute bars for AAPL (1 day)...")
    try:
        fifteen_min_data = hist_conn.request_bars_for_days(
            ticker="AAPL",
            interval_len=900,
            interval_type='s',
            days=1
        )
        print(f"  SUCCESS: Got {len(fifteen_min_data)} 15-minute bars")

    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n5. Testing tick data for AAPL (1 day)...")
    try:
        tick_data = hist_conn.request_ticks_for_days(
            ticker="AAPL",
            num_days=1,
            max_ticks=100
        )
        print(f"  SUCCESS: Got {len(tick_data)} ticks")

    except Exception as e:
        print(f"  FAILED: {e}")

    print("\nDebug complete - now we know what data is available!")

if __name__ == "__main__":
    debug_daily()
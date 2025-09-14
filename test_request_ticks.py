"""
Test request_ticks method to get recent tick data like terminal
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import pyiqfeed as iq
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def test_recent_ticks():
    username = os.getenv('IQFEED_USERNAME', '487854')
    password = os.getenv('IQFEED_PASSWORD', 't1wnjnuz')

    print("Testing request_ticks for recent AAPL data")
    print("="*50)

    # Connect to IQFeed
    service = iq.FeedService(
        product="FUZZY_OSS20",
        version="1.0",
        login=username,
        password=password
    )
    service.launch(headless=True)

    hist_conn = iq.HistoryConn(name="recent-ticks")
    hist_conn.connect()

    # Test request_ticks (most recent ticks)
    print("Testing request_ticks (most recent ticks)...")
    try:
        tick_data = hist_conn.request_ticks(
            ticker="AAPL",
            max_ticks=50
        )
        print(f"SUCCESS: Got {len(tick_data)} recent ticks!")

        if len(tick_data) > 0:
            print("\nFirst 5 ticks:")
            for i in range(min(5, len(tick_data))):
                tick = tick_data[i]
                print(f"\nTick {i}:")
                print(f"  Type: {type(tick)}")
                print(f"  Fields: {tick.dtype.names}")

                # Show all field values
                for field in tick.dtype.names:
                    value = tick[field]
                    print(f"  {field}: {value} (type: {type(value)})")

    except Exception as e:
        print(f"FAILED: {e}")

    print("\nTest complete")

if __name__ == "__main__":
    test_recent_ticks()
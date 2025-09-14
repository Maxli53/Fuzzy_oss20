"""
Simple IQFeed debug - analyze numpy.timedelta64 issue
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import pyiqfeed as iq
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def debug_simple():
    username = os.getenv('IQFEED_USERNAME', '487854')
    password = os.getenv('IQFEED_PASSWORD', 't1wnjnuz')

    print("Simple IQFeed Debug")
    print("="*50)

    # Connect to IQFeed
    service = iq.FeedService(
        product="FUZZY_OSS20",
        version="1.0",
        login=username,
        password=password
    )
    service.launch(headless=True)

    hist_conn = iq.HistoryConn(name="simple-debug")
    hist_conn.connect()

    # Get just a few bars to analyze
    print("Getting 5-second bars for AAPL (1 day)...")
    bar_data = hist_conn.request_bars_for_days(
        ticker="AAPL",
        interval_len=5,
        interval_type='s',
        days=1
    )

    print(f"Got {len(bar_data)} bars")

    if len(bar_data) > 0:
        # Analyze first bar
        bar = bar_data[0]
        print(f"\nFirst bar analysis:")
        print(f"Type: {type(bar)}")
        print(f"Fields: {bar.dtype.names}")

        print(f"\nField values:")
        for field in bar.dtype.names:
            value = bar[field]
            print(f"  {field}: {value} (type: {type(value)})")

        # Focus on time field
        time_val = bar['time']
        print(f"\nTime field analysis:")
        print(f"Value: {time_val}")
        print(f"Type: {type(time_val)}")
        print(f"Is timedelta64: {isinstance(time_val, np.timedelta64)}")

        if isinstance(time_val, np.timedelta64):
            # Test numpy conversion
            print(f"\nTesting numpy timedelta64 conversion:")
            try:
                # Convert to microseconds
                microseconds = time_val / np.timedelta64(1, 'us')
                print(f"  Microseconds: {microseconds}")

                # Convert to total seconds
                total_seconds = float(microseconds) / 1000000
                print(f"  Total seconds: {total_seconds}")

                # Convert to HH:MM:SS
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                seconds = int(total_seconds % 60)
                microsecs = int((total_seconds % 1) * 1000000)

                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{microsecs:06d}"
                print(f"  Formatted time: {time_str}")

            except Exception as e:
                print(f"  Conversion failed: {e}")

        # Check date range
        print(f"\nDate range analysis:")
        dates = [str(bar['date']) for bar in bar_data]
        unique_dates = sorted(list(set(dates)))
        print(f"Unique dates: {unique_dates}")

        for date in unique_dates:
            count = dates.count(date)
            print(f"  {date}: {count} bars")

    print("\nDebug complete")

if __name__ == "__main__":
    debug_simple()
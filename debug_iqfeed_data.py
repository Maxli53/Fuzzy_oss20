"""
Debug IQFeed data to understand actual format and missing bars
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import pyiqfeed as iq
from datetime import datetime, timedelta
import logging
import pandas as pd
from dotenv import load_dotenv
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def debug_iqfeed_data():
    """Debug actual IQFeed data structure and content"""
    username = os.getenv('IQFEED_USERNAME', '487854')
    password = os.getenv('IQFEED_PASSWORD', 't1wnjnuz')

    print("="*80)
    print("DEBUGGING IQFeed Data Structure and Missing Bars")
    print("="*80)

    try:
        # Connect to IQFeed
        service = iq.FeedService(
            product="FUZZY_OSS20",
            version="1.0",
            login=username,
            password=password
        )
        service.launch(headless=True)

        hist_conn = iq.HistoryConn(name="debug-history")
        hist_conn.connect()
        print("Connected to IQFeed successfully")

        print("\n1. Testing 5-second bars for AAPL (5 days)")

        # Get 5-second bars - no ConnConnector needed since already connected
        bar_data = hist_conn.request_bars_for_days(
            ticker="AAPL",
            interval_len=5,
            interval_type='s',
            days=5
        )

        print(f"Raw data returned: {len(bar_data)} bars")

        if len(bar_data) > 0:
                print("\n2. Analyzing first 10 bars structure:")
                for i in range(min(10, len(bar_data))):
                    bar = bar_data[i]
                    print(f"\nBar {i}:")
                    print(f"  Type: {type(bar)}")
                    print(f"  Fields: {bar.dtype.names}")

                    # Show actual field values
                    for field in bar.dtype.names:
                        value = bar[field]
                        print(f"  {field}: {value} (type: {type(value)})")

                print("\n3. Checking time field conversion methods:")
                sample_bar = bar_data[0]
                time_val = sample_bar['time']

                print(f"Time value: {time_val}")
                print(f"Time type: {type(time_val)}")

                # Test different conversion approaches
                print("\nTesting conversion approaches:")

                # Method 1: Direct int conversion (fails)
                try:
                    int_val = int(time_val)
                    print(f"  int(time_val): {int_val}")
                except Exception as e:
                    print(f"  int(time_val) FAILED: {e}")

                # Method 2: numpy conversion
                try:
                    if isinstance(time_val, np.timedelta64):
                        # Convert to microseconds
                        microseconds = time_val / np.timedelta64(1, 'us')
                        print(f"  numpy conversion to microseconds: {microseconds}")

                        # Convert to seconds for time calculation
                        total_seconds = float(microseconds) / 1000000
                        hours = int(total_seconds // 3600)
                        minutes = int((total_seconds % 3600) // 60)
                        seconds = int(total_seconds % 60)
                        microsecs = int((total_seconds % 1) * 1000000)

                        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{microsecs:06d}"
                        print(f"  Converted time: {time_str}")

                except Exception as e:
                    print(f"  numpy conversion FAILED: {e}")

                # Method 3: Check if it has total_seconds method
                if hasattr(time_val, 'total_seconds'):
                    try:
                        total_secs = time_val.total_seconds()
                        print(f"  total_seconds(): {total_secs}")
                    except Exception as e:
                        print(f"  total_seconds() FAILED: {e}")

                print(f"\n4. Date analysis:")
                date_val = sample_bar['date']
                print(f"Date value: {date_val} (type: {type(date_val)})")

                # Check date range coverage
                print(f"\n5. Date range analysis:")
                dates = [bar['date'] for bar in bar_data]
                unique_dates = list(set(dates))
                unique_dates.sort()
                print(f"Unique dates in data: {len(unique_dates)}")
                for date in unique_dates:
                    count = dates.count(date)
                    print(f"  {date}: {count} bars")

                # Expected bars per day calculation
                market_hours = 6.5  # 9:30 AM to 4:00 PM
                seconds_per_day = market_hours * 3600
                expected_bars_per_day = seconds_per_day / 5  # 5-second intervals
                total_expected = expected_bars_per_day * len(unique_dates)

                print(f"\n6. Expected vs Actual analysis:")
                print(f"Market hours per day: {market_hours}")
                print(f"Expected bars per day: {expected_bars_per_day}")
                print(f"Trading days covered: {len(unique_dates)}")
                print(f"Total expected bars: {total_expected}")
                print(f"Actual bars received: {len(bar_data)}")
                print(f"Missing bars: {total_expected - len(bar_data)}")
                print(f"Data completeness: {len(bar_data)/total_expected*100:.1f}%")

        print("\n" + "="*80)
        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_iqfeed_data()
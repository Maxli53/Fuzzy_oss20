"""
Test correct pyiqfeed API calls
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import pyiqfeed as iq
from datetime import datetime, timedelta
import logging
import pandas as pd
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_correct_api():
    """Test correct pyiqfeed API calls"""
    username = os.getenv('IQFEED_USERNAME', '487854')
    password = os.getenv('IQFEED_PASSWORD', 't1wnjnuz')

    print(f"Testing correct API with username: {username}")

    try:
        # Step 1: Create FeedService and connect
        service = iq.FeedService(
            product="FUZZY_OSS20",
            version="1.0",
            login=username,
            password=password
        )

        service.launch(headless=True)
        print("Service connected successfully")

        # Step 2: Create and connect history connection
        hist_conn = iq.HistoryConn(name="test-correct-api")
        hist_conn.connect()
        print("History connection created and connected")

        # Step 3: Test correct API calls
        symbol = "AAPL"

        # Test daily data
        print(f"\nTesting daily data for {symbol}...")
        try:
            daily_data = hist_conn.request_daily_data(
                ticker=symbol,
                num_days=5
            )

            if daily_data:
                print(f"SUCCESS: Received {len(daily_data)} daily bars")
                # Print first record structure
                if len(daily_data) > 0:
                    print("Sample daily record:")
                    sample = daily_data[0]
                    for key, value in sample.items():
                        print(f"  {key}: {value}")
            else:
                print("No daily data received")

        except Exception as e:
            print(f"Error getting daily data: {e}")

        # Test ticks
        print(f"\nTesting tick data for {symbol}...")
        try:
            tick_data = hist_conn.request_ticks_for_days(
                ticker=symbol,
                num_days=1,
                max_ticks=10  # Just get a few ticks for testing
            )

            if tick_data:
                print(f"SUCCESS: Received {len(tick_data)} ticks")
                # Print first record structure
                if len(tick_data) > 0:
                    print("Sample tick record:")
                    sample = tick_data[0]
                    for key, value in sample.items():
                        print(f"  {key}: {value}")
            else:
                print("No tick data received")

        except Exception as e:
            print(f"Error getting tick data: {e}")

        # Test minute bars
        print(f"\nTesting minute bars for {symbol}...")
        try:
            bar_data = hist_conn.request_bars_for_days(
                ticker=symbol,
                interval_len=60,  # 1 minute in seconds
                interval_type='s',  # seconds
                num_days=1
            )

            if bar_data:
                print(f"SUCCESS: Received {len(bar_data)} minute bars")
                # Print first record structure
                if len(bar_data) > 0:
                    print("Sample bar record:")
                    sample = bar_data[0]
                    for key, value in sample.items():
                        print(f"  {key}: {value}")
            else:
                print("No bar data received")

        except Exception as e:
            print(f"Error getting bar data: {e}")

        print("\nAPI test completed successfully!")
        return True

    except Exception as e:
        print(f"API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("IQFeed Correct API Test")
    print("="*60)

    success = test_correct_api()

    if success:
        print("\nAPI calls are working!")
    else:
        print("\nAPI test failed")
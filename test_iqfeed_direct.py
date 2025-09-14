"""
Direct IQFeed connection test - debug why fetch is failing
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

def test_direct_connection():
    """Test direct connection to IQFeed"""
    username = os.getenv('IQFEED_USERNAME', '487854')
    password = os.getenv('IQFEED_PASSWORD', 't1wnjnuz')

    print(f"Attempting connection with username: {username}")

    try:
        # Step 1: Create FeedService
        print("\n1. Creating FeedService...")
        service = iq.FeedService(
            product="FUZZY_OSS20",
            version="1.0",
            login=username,
            password=password
        )

        # Step 2: Launch/connect
        print("2. Launching/connecting to IQFeed...")
        service.launch(headless=True)
        print("   OK Service launched successfully")

        # Step 3: Create history connection
        print("\n3. Creating history connection...")
        hist_conn = iq.HistoryConn(name="test-history")
        print("   OK History connection created")

        # Step 4: Request historical data for AAPL
        print("\n4. Requesting historical data for AAPL...")

        # Try different time periods based on market status
        now = datetime.now()
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        if now > market_close:
            # After market - can get 180 days
            print("   Market is closed - requesting 30 days of data")
            lookback_days = 30
        else:
            # During market - max 8 days
            print("   Market hours - requesting 5 days of data")
            lookback_days = 5

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        print(f"   Date range: {start_date.date()} to {end_date.date()}")

        # Try getting daily bars first
        print("\n5. Requesting daily bars...")
        try:
            daily_bars = hist_conn.request_daily_data_for_dates(
                ticker="AAPL",
                bgn_dt=start_date,
                end_dt=end_date
            )

            if daily_bars:
                df = pd.DataFrame(daily_bars)
                print(f"   OK Received {len(df)} daily bars")
                print(f"   Columns: {list(df.columns)}")
                print(f"   Date range: {df.index.min()} to {df.index.max()}")
                print("\n   Sample data (first 3 rows):")
                print(df.head(3))
            else:
                print("   ERROR No daily bars received")

        except Exception as e:
            print(f"   ERROR Error getting daily bars: {e}")

        # Try getting minute bars
        print("\n6. Requesting minute bars for today...")
        try:
            minute_bars = hist_conn.request_bars_in_period(
                ticker="AAPL",
                interval_len=60,  # 1 minute
                interval_type='s',  # seconds
                bgn_dt=datetime.now().replace(hour=9, minute=30),
                end_dt=datetime.now()
            )

            if minute_bars:
                df = pd.DataFrame(minute_bars)
                print(f"   OK Received {len(df)} minute bars")
                print(f"   Columns: {list(df.columns)}")
                print("\n   Sample data (first 3 rows):")
                print(df.head(3))
            else:
                print("   ERROR No minute bars received")

        except Exception as e:
            print(f"   ERROR Error getting minute bars: {e}")

        # Try getting ticks
        print("\n7. Requesting recent ticks...")
        try:
            # Request last 100 ticks
            ticks = hist_conn.request_ticks_for_days(
                ticker="AAPL",
                num_days=1,
                max_ticks=100
            )

            if ticks:
                df = pd.DataFrame(ticks)
                print(f"   OK Received {len(df)} ticks")
                print(f"   Columns: {list(df.columns)}")
                print("\n   Sample data (first 3 rows):")
                print(df.head(3))
            else:
                print("   ERROR No ticks received")

        except Exception as e:
            print(f"   ERROR Error getting ticks: {e}")

        print("\nOK Connection test completed")
        return True

    except Exception as e:
        print(f"\nERROR Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("IQFeed Direct Connection Test")
    print("="*60)

    success = test_direct_connection()

    if success:
        print("\nSUCCESS IQFeed is working correctly!")
    else:
        print("\nERROR IQFeed connection has issues")
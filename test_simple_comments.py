"""
Simple test to verify commented code works
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct imports to avoid circular dependency
import pyiqfeed as iq
import numpy as np
from datetime import datetime, timedelta

def test_iqfeed_with_comments():
    """Test that IQFeed still works with all our detailed comments"""

    print("Testing IQFeed with extensive comments...")
    print("=" * 60)

    # Test basic connection
    hist_conn = iq.HistoryConn(name="test-comments")

    with iq.ConnConnector([hist_conn]) as connector:
        # Get some tick data to verify everything works
        ticker = "AAPL"
        print(f"\n1. Testing tick data retrieval for {ticker}...")

        try:
            # Request recent ticks
            tick_data = hist_conn.request_ticks(ticker, 100)

            if tick_data is not None and len(tick_data) > 0:
                print(f"   [OK] Got {len(tick_data)} ticks")
                print(f"   [OK] Data type: {type(tick_data)}")
                print(f"   [OK] Fields: {tick_data.dtype.names}")

                # Show a sample tick
                tick = tick_data[0]
                print(f"\n2. Sample tick:")
                print(f"   Date: {tick['date']}")
                print(f"   Time: {tick['time']} microseconds")
                print(f"   Price: ${tick['last']:.2f}")
                print(f"   Volume: {tick['last_sz']}")
                print(f"   Exchange: {tick['last_type']}")
                print(f"   Bid/Ask: ${tick['bid']:.2f}/${tick['ask']:.2f}")

                # Check weekend advantage timing
                now = datetime.now()
                is_weekend = now.weekday() >= 5
                is_after_hours = now.hour >= 16 or now.hour < 9

                print(f"\n3. Weekend advantage check:")
                print(f"   Current day: {now.strftime('%A')}")
                print(f"   Current time: {now.strftime('%H:%M')}")
                print(f"   Is weekend: {is_weekend}")
                print(f"   Is after hours: {is_after_hours}")

                if is_weekend or is_after_hours:
                    print(f"   -> Can request up to 180 days of tick data!")
                else:
                    print(f"   -> Limited to 8 days during market hours")

                print(f"\n[OK] All systems working with detailed comments!")

            else:
                print("   [WARNING] No tick data received")

        except Exception as e:
            print(f"   Error: {e}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_iqfeed_with_comments()
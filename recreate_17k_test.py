"""
Recreate the exact conditions that gave us 17,563 bars
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from gui.data_interface import GUIDataInterface
import logging

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

def recreate_test():
    """Recreate the exact test that gave us 17,563 AAPL bars"""
    print("Recreating the 17,563 bar test")
    print("="*50)

    # Initialize data interface (same as original test)
    interface = GUIDataInterface()

    # Test the exact same call that gave us 17,563 bars
    print("Fetching AAPL with exact same parameters...")
    fetch_result = interface.fetch_real_data("AAPL", data_type="bars", lookback_days=5)

    if fetch_result['success']:
        data = fetch_result['data']
        print(f"SUCCESS: Got {len(data)} records (same as before: 17,563)")
        print(f"Source: {fetch_result['source']}")

        # Analyze the timestamps to see what went wrong
        print(f"\nTimestamp analysis:")
        print(f"Unique timestamps: {data['timestamp'].nunique()}")
        print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")

        # Check if all timestamps are the same (00:00:00 issue)
        time_strings = data['timestamp'].dt.strftime('%H:%M:%S').unique()
        print(f"Unique time parts: {time_strings}")

        if len(time_strings) == 1 and time_strings[0] == '00:00:00':
            print("🚨 FOUND THE ISSUE: All timestamps have 00:00:00 time!")
            print("   This means timestamp conversion is failing and defaulting to midnight")

            # Check dates
            unique_dates = data['timestamp'].dt.date.unique()
            print(f"Unique dates: {len(unique_dates)}")
            for date in sorted(unique_dates):
                count = (data['timestamp'].dt.date == date).sum()
                print(f"  {date}: {count} bars")

        # Show sample data
        print(f"\nFirst 5 records:")
        print(data.head())

        print(f"\nLast 5 records:")
        print(data.tail())

        return data
    else:
        print(f"FAILED: {fetch_result['error']}")
        return None

if __name__ == "__main__":
    recreate_test()
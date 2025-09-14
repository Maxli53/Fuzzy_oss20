"""
Get Friday 2025-09-12 tick data from API and validate against CSV reference
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from gui.data_interface import GUIDataInterface
import pandas as pd

def validate_friday_api():
    """Get Friday data from API and validate against CSV"""
    print("Getting Friday 2025-09-12 tick data from API")
    print("="*50)

    # Get API data for Friday
    interface = GUIDataInterface()

    print("1. Fetching Friday tick data from API...")
    fetch_result = interface.fetch_real_data("AAPL", data_type="ticks", max_records=5000)

    if not fetch_result['success']:
        print(f"FAILED: {fetch_result['error']}")
        return

    api_data = fetch_result['data']
    print(f"SUCCESS: Got {len(api_data)} tick records from API")

    # Filter for Friday data (2025-09-12)
    friday_data = api_data[api_data['timestamp'].dt.date == pd.to_datetime('2025-09-12').date()]
    print(f"Friday 2025-09-12 data: {len(friday_data)} ticks")

    if len(friday_data) == 0:
        print("No Friday data found in API results")
        return

    # Show last 20 entries
    print("\n2. Last 20 entries from Friday API data:")
    last_20 = friday_data.tail(20)
    for i, (idx, row) in enumerate(last_20.iterrows()):
        print(f"   {i+1:2d}. {row['timestamp'].strftime('%H:%M:%S.%f')} | ${row['price']:.4f} | {row['volume']:3d} | ID:{row['tick_id']}")

    # Search for specific tick IDs
    print(f"\n3. Searching for target tick IDs in Friday data...")
    target_ticks = [157890, 263185, 36925]

    for tick_id in target_ticks:
        matching = friday_data[friday_data['tick_id'] == tick_id]
        if not matching.empty:
            tick = matching.iloc[0]
            print(f"   FOUND {tick_id}: {tick['timestamp'].strftime('%H:%M:%S.%f')} | ${tick['price']:.4f} | {tick['volume']} shares")
        else:
            print(f"   MISSING {tick_id}")

    # Show tick ID range for Friday
    print(f"\n4. Friday tick ID range:")
    print(f"   Min: {friday_data['tick_id'].min()}")
    print(f"   Max: {friday_data['tick_id'].max()}")
    print(f"   Count: {len(friday_data)}")
    print(f"   Time range: {friday_data['timestamp'].min()} to {friday_data['timestamp'].max()}")

    return friday_data

if __name__ == "__main__":
    validate_friday_api()
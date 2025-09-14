"""
Simple check for specific tick IDs: 157890, 263185, 36925
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
from gui.data_interface import GUIDataInterface

def check_tick_ids():
    """Check if our API data contains the specific tick IDs"""
    print("Checking for Tick IDs: 157890, 263185, 36925")
    print("="*50)

    # Get API data
    interface = GUIDataInterface()
    fetch_result = interface.fetch_real_data("AAPL", data_type="ticks", max_records=2000)

    if not fetch_result['success']:
        print(f"FAILED: {fetch_result['error']}")
        return

    api_data = fetch_result['data']
    print(f"Got {len(api_data)} API tick records")

    # Check for each target tick ID
    target_ticks = [157890, 263185, 36925]
    found_count = 0

    for tick_id in target_ticks:
        matching = api_data[api_data['tick_id'] == tick_id]
        if not matching.empty:
            tick = matching.iloc[0]
            print(f"\nTick {tick_id} FOUND:")
            print(f"  Time: {tick['timestamp']}")
            print(f"  Price: ${tick['price']:.4f}")
            print(f"  Volume: {tick['volume']} shares")
            print(f"  Bid/Ask: ${tick['bid']:.4f}/${tick['ask']:.4f}")
            found_count += 1
        else:
            print(f"\nTick {tick_id} NOT FOUND")

    print(f"\nSUMMARY: Found {found_count}/3 target tick IDs")

    if found_count == 3:
        print("SUCCESS: All tick IDs found - API matches CSV exactly!")
    elif found_count > 0:
        print("PARTIAL: Some tick IDs found - API getting overlapping data")
    else:
        print("ISSUE: No tick IDs found - API getting different timeframe")

    # Show tick ID range in our data
    print(f"\nAPI Tick ID Range:")
    print(f"  Min: {api_data['tick_id'].min()}")
    print(f"  Max: {api_data['tick_id'].max()}")
    print(f"  Date range: {api_data['timestamp'].min()} to {api_data['timestamp'].max()}")

if __name__ == "__main__":
    check_tick_ids()
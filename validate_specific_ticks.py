"""
Validate our API data against specific tick IDs from CSV
Target tick IDs: 157890, 263185, 36925
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
from gui.data_interface import GUIDataInterface

def validate_specific_ticks():
    """Validate API data against specific tick IDs from CSV"""
    print("Validating Specific Tick IDs: 157890, 263185, 36925")
    print("="*60)

    # Reference data from CSV
    reference_ticks = {
        157890: {
            'time': '15:59:59.872482',
            'date': '2025-09-12',
            'price': 234.0600,
            'volume': 161,
            'bid': 234.0500,
            'ask': 234.0700,
            'market': 'NASDAQ',
            'condition': 'REGULAR'
        },
        263185: {
            'time': '15:59:59.738525',
            'date': '2025-09-12',
            'price': 234.0550,
            'volume': 200,
            'bid': 234.0500,
            'ask': 234.0600,
            'market': 'NTRF',
            'condition': 'REGULAR'
        },
        36925: {
            'time': '15:59:59.681604',
            'date': '2025-09-12',
            'price': 234.0600,
            'volume': 100,
            'bid': 234.0500,
            'ask': 234.0600,
            'market': 'BATS',
            'condition': 'INTRMRK_SWEEP'
        }
    }

    print("1. Reference data from CSV:")
    for tick_id, data in reference_ticks.items():
        print(f"   Tick {tick_id}: {data['time']} | ${data['price']:.4f} | {data['volume']} shares | {data['market']}")

    # Get API data
    print("\n2. Fetching API data...")
    interface = GUIDataInterface()
    fetch_result = interface.fetch_real_data("AAPL", data_type="ticks", max_records=2000)

    if not fetch_result['success']:
        print(f"   FAILED: {fetch_result['error']}")
        return False

    api_data = fetch_result['data']
    print(f"   SUCCESS: Got {len(api_data)} API tick records")

    # Search for specific tick IDs
    print("\n3. Searching for target tick IDs in API data...")

    found_ticks = {}
    for tick_id in [157890, 263185, 36925]:
        matching_ticks = api_data[api_data['tick_id'] == tick_id]
        if not matching_ticks.empty:
            found_ticks[tick_id] = matching_ticks.iloc[0]
            print(f"   ✓ Found tick ID {tick_id}")
        else:
            print(f"   ✗ Missing tick ID {tick_id}")

    # Detailed comparison
    print(f"\n4. Detailed validation results:")
    print("="*60)

    all_match = True
    for tick_id in [157890, 263185, 36925]:
        print(f"\nTick ID {tick_id}:")

        if tick_id not in found_ticks:
            print(f"   ❌ NOT FOUND in API data")
            all_match = False
            continue

        ref = reference_ticks[tick_id]
        api = found_ticks[tick_id]

        # Compare fields
        price_match = abs(float(api['price']) - ref['price']) < 0.001
        volume_match = int(api['volume']) == ref['volume']
        bid_match = abs(float(api['bid']) - ref['bid']) < 0.001
        ask_match = abs(float(api['ask']) - ref['ask']) < 0.001

        print(f"   CSV:  {ref['time']} | ${ref['price']:.4f} | {ref['volume']:3d} | {ref['bid']:.4f}/{ref['ask']:.4f}")
        print(f"   API:  {api['timestamp'].strftime('%H:%M:%S.%f')} | ${api['price']:.4f} | {api['volume']:3d} | {api['bid']:.4f}/{api['ask']:.4f}")

        print(f"   Price:  {'✓' if price_match else '✗'} {price_match}")
        print(f"   Volume: {'✓' if volume_match else '✗'} {volume_match}")
        print(f"   Bid:    {'✓' if bid_match else '✗'} {bid_match}")
        print(f"   Ask:    {'✓' if ask_match else '✗'} {ask_match}")

        if not all([price_match, volume_match, bid_match, ask_match]):
            all_match = False

    # Summary
    print(f"\n" + "="*60)
    if all_match and len(found_ticks) == 3:
        print("🎉 PERFECT MATCH! Our API data exactly matches your CSV reference.")
        print("   All tick IDs found with matching prices, volumes, and spreads.")
    elif len(found_ticks) > 0:
        print(f"⚠️  PARTIAL MATCH: Found {len(found_ticks)}/3 tick IDs")
        print("   Some validation differences detected.")
    else:
        print("❌ NO MATCH: None of the target tick IDs found in API data")
        print("   This suggests our API is getting different data than your CSV")

    return all_match and len(found_ticks) == 3

if __name__ == "__main__":
    validate_specific_ticks()
"""
Detailed search for tick_id 265596 across all AAPL data
"""
import pandas as pd
from arcticdb import Arctic
import sys

# Fix encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def find_tick_265596_detailed():
    """Find tick_id 265596 with detailed search"""

    print("=" * 80)
    print("DETAILED SEARCH FOR TICK_ID 265596")
    print("=" * 80)

    try:
        # Connect to ArcticDB
        arctic = Arctic("lmdb://./data/arctic_storage")

        # First, check the range we found (58326 to 265309 in iqfeed_base_common_stock)
        print("\n1. Checking iqfeed_base_common_stock library...")
        print("-" * 40)
        lib = arctic['iqfeed_base_common_stock']

        for symbol in lib.list_symbols():
            if 'AAPL' in symbol:
                df = lib.read(symbol).data

                if 'tick_id' in df.columns:
                    # Get the closest values to 265596
                    tick_ids = df['tick_id'].values
                    closest_lower = tick_ids[tick_ids < 265596].max() if any(tick_ids < 265596) else None
                    closest_higher = tick_ids[tick_ids > 265596].min() if any(tick_ids > 265596) else None

                    print(f"Symbol: {symbol}")
                    print(f"  Tick ID range: {tick_ids.min()} to {tick_ids.max()}")
                    print(f"  Total ticks: {len(tick_ids)}")
                    print(f"  Unique tick_ids: {df['tick_id'].nunique()}")

                    # Check if 265596 is in range
                    if tick_ids.min() <= 265596 <= tick_ids.max():
                        print(f"  ✓ 265596 is within range!")

                        # Find exact match or closest
                        exact_match = df[df['tick_id'] == 265596]
                        if not exact_match.empty:
                            print(f"\n  EXACT MATCH FOUND!")
                            print(exact_match.to_string())
                        else:
                            print(f"\n  No exact match, but within range")
                            print(f"  Closest lower tick_id: {closest_lower}")
                            print(f"  Closest higher tick_id: {closest_higher}")

                            # Show the gap
                            if closest_lower and closest_higher:
                                lower_row = df[df['tick_id'] == closest_lower].iloc[0]
                                higher_row = df[df['tick_id'] == closest_higher].iloc[0]

                                print(f"\n  Row before (tick_id={closest_lower}):")
                                print(f"    Timestamp: {lower_row['timestamp']}")
                                print(f"    Price: {lower_row['price']}")
                                print(f"    Volume: {lower_row['volume']}")

                                print(f"\n  Row after (tick_id={closest_higher}):")
                                print(f"    Timestamp: {higher_row['timestamp']}")
                                print(f"    Price: {higher_row['price']}")
                                print(f"    Volume: {higher_row['volume']}")

                                # Check for any specific tick_ids near 265596
                                print(f"\n  Tick IDs near 265596:")
                                near_ticks = df[(df['tick_id'] >= 265590) & (df['tick_id'] <= 265600)]
                                if not near_ticks.empty:
                                    print(near_ticks[['timestamp', 'tick_id', 'price', 'volume']].to_string())
                                else:
                                    # Show wider range
                                    near_ticks = df[(df['tick_id'] >= 265296) & (df['tick_id'] <= 265309)]
                                    if not near_ticks.empty:
                                        print(f"  Showing tick_ids from 265296 to 265309:")
                                        print(near_ticks[['timestamp', 'tick_id', 'price', 'volume']].to_string())

        # Now check tick_data library more thoroughly
        print("\n" + "=" * 80)
        print("2. Checking tick_data library...")
        print("-" * 40)

        lib = arctic['tick_data']
        all_results = []

        for symbol in lib.list_symbols():
            if 'AAPL' in symbol:
                print(f"\nSymbol: {symbol}")
                df = lib.read(symbol).data

                # Check all numeric columns for value 265596
                numeric_cols = df.select_dtypes(include=['int', 'float']).columns

                for col in numeric_cols:
                    unique_count = df[col].nunique()
                    if unique_count > 100:  # Likely an ID column
                        col_min = df[col].min()
                        col_max = df[col].max()

                        # Check if 265596 could be in this column
                        if col_min <= 265596 <= col_max:
                            exact = df[df[col] == 265596]
                            if not exact.empty:
                                print(f"  ✓ FOUND 265596 in column '{col}'!")
                                result = {
                                    'library': 'tick_data',
                                    'symbol': symbol,
                                    'column': col,
                                    'data': exact
                                }
                                all_results.append(result)

                                # Show the data
                                display_cols = ['timestamp', 'price', 'volume', col]
                                display_cols = [c for c in display_cols if c in df.columns]
                                print(exact[display_cols].to_string())
                            elif unique_count > 1000:  # High cardinality column
                                print(f"  Column '{col}': range {col_min:.0f} to {col_max:.0f} (265596 in range but not found)")

                # Also check if there's a tick around 265596 timestamp
                if 'market_center' in df.columns:
                    center_vals = df['market_center'].unique()
                    if 265596 in center_vals:
                        print(f"  Found 265596 in market_center column!")

                if 'total_volume' in df.columns:
                    vol_match = df[df['total_volume'] == 265596]
                    if not vol_match.empty:
                        print(f"  Found 265596 in total_volume column!")
                        print(vol_match[['timestamp', 'price', 'volume', 'total_volume']].head().to_string())

        # Summary
        print("\n" + "=" * 80)
        print("SEARCH SUMMARY")
        print("=" * 80)

        if all_results:
            print(f"\nFound {len(all_results)} matches:")
            for result in all_results:
                print(f"  - Library: {result['library']}, Symbol: {result['symbol']}, Column: {result['column']}")
        else:
            print("\nNo exact matches found for value 265596")
            print("\nPossible reasons:")
            print("1. The tick_id 265596 might have been skipped (gap in sequence)")
            print("2. It might exist in a different date's data not yet loaded")
            print("3. It could be in a different column or transformed")

            # Show what we did find
            print("\nClosest tick_id found in iqfeed_base_common_stock:")
            print("  - Maximum tick_id: 265309 (which is less than 265596)")
            print("  - This suggests tick_id 265596 might be in newer data")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    find_tick_265596_detailed()
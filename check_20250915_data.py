"""
Check for data from September 15, 2025 in ArcticDB
"""
import pandas as pd
from arcticdb import Arctic
from datetime import datetime, date
import sys

# Fix encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_sept15_data():
    """Check for September 15, 2025 data"""

    print("=" * 80)
    print("CHECKING FOR SEPTEMBER 15, 2025 DATA")
    print("=" * 80)

    try:
        # Connect to ArcticDB
        arctic = Arctic("lmdb://./data/arctic_storage")

        # Target date
        target_date = datetime(2025, 9, 15)
        date_str = "2025-09-15"
        date_variations = [
            "2025-09-15",
            "2025/09/15",
            "20250915",
            "09-15-2025",
            "09/15/2025",
            "15.09.2025"
        ]

        found_data = []

        # Check all libraries
        libraries = arctic.list_libraries()

        for lib_name in libraries:
            lib = arctic[lib_name]
            symbols = lib.list_symbols()

            if not symbols:
                continue

            print(f"\nChecking library: {lib_name}")
            print("-" * 40)

            # Check for date in symbol names
            for symbol in symbols:
                # Check if date appears in symbol name
                for date_var in date_variations:
                    if date_var in symbol:
                        print(f"  ✓ Found symbol with date: {symbol}")
                        found_data.append((lib_name, symbol, "symbol_name"))
                        break

                # For AAPL symbols, check the actual data
                if 'AAPL' in symbol:
                    try:
                        df = lib.read(symbol).data

                        # Check if there's a timestamp column
                        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]

                        for ts_col in timestamp_cols:
                            if pd.api.types.is_datetime64_any_dtype(df[ts_col]):
                                # Check for Sept 15 data
                                df_dates = pd.to_datetime(df[ts_col])

                                # Check if Sept 15, 2025 is in the data
                                sept15_mask = (df_dates.dt.date == date(2025, 9, 15))
                                sept15_data = df[sept15_mask]

                                if not sept15_data.empty:
                                    print(f"  ✓ Found Sept 15, 2025 data in {symbol}!")
                                    print(f"    Rows: {len(sept15_data)}")
                                    print(f"    Time range: {df_dates[sept15_mask].min()} to {df_dates[sept15_mask].max()}")

                                    # Check for tick_id in this data
                                    if 'tick_id' in df.columns:
                                        tick_ids_sept15 = sept15_data['tick_id']
                                        print(f"    Tick ID range: {tick_ids_sept15.min()} to {tick_ids_sept15.max()}")

                                        # Check if 265596 is in this range
                                        if 265596 in tick_ids_sept15.values:
                                            print(f"    ✓✓✓ FOUND TICK_ID 265596 on Sept 15!")
                                            tick_265596 = sept15_data[sept15_data['tick_id'] == 265596]
                                            print("\n    TICK 265596 DATA:")
                                            print("    " + "=" * 60)
                                            for col in tick_265596.columns:
                                                val = tick_265596.iloc[0][col]
                                                print(f"    {col}: {val}")
                                        elif tick_ids_sept15.min() <= 265596 <= tick_ids_sept15.max():
                                            print(f"    Note: Tick 265596 is in range but not found")

                                    # Show sample data
                                    print(f"\n    Sample Sept 15 data (first 5 rows):")
                                    display_cols = [col for col in ['timestamp', 'price', 'volume', 'tick_id'] if col in df.columns]
                                    if display_cols:
                                        print(sept15_data[display_cols].head().to_string(index=False))

                                    found_data.append((lib_name, symbol, "data_content"))
                                else:
                                    # Check what dates ARE in the data
                                    unique_dates = pd.to_datetime(df[ts_col]).dt.date.unique()
                                    if date(2025, 9, 14) in unique_dates or date(2025, 9, 16) in unique_dates:
                                        print(f"  Symbol {symbol} has data near Sept 15 but not on that date")
                                        print(f"    Available dates: {sorted(unique_dates)[:5]}...")

                                        # Check if Sept 12 data has our tick_id
                                        if 'tick_id' in df.columns:
                                            sept12_mask = (df_dates.dt.date == date(2025, 9, 12))
                                            if any(sept12_mask):
                                                sept12_data = df[sept12_mask]
                                                if 265596 in sept12_data['tick_id'].values:
                                                    print(f"    ✓ Found tick_id 265596 on Sept 12 instead!")
                                                    tick_data = sept12_data[sept12_data['tick_id'] == 265596]
                                                    print(f"      Time: {tick_data.iloc[0][ts_col]}")
                                                    print(f"      Price: {tick_data.iloc[0]['price']}")
                                                    print(f"      Volume: {tick_data.iloc[0]['volume']}")

                        # For non-datetime index, check if it's RangeIndex and look at metadata
                        try:
                            metadata = lib.read_metadata(symbol).metadata
                            if metadata and 'date' in metadata:
                                if str(metadata['date']) == date_str or metadata['date'] == date_str:
                                    print(f"  ✓ Symbol {symbol} has metadata date: {metadata['date']}")
                                    found_data.append((lib_name, symbol, "metadata"))
                        except:
                            pass

                    except Exception as e:
                        print(f"  Error reading {symbol}: {str(e)[:50]}")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY FOR SEPTEMBER 15, 2025")
        print("=" * 80)

        if found_data:
            print(f"\nFound {len(found_data)} references to Sept 15, 2025:")
            for lib, sym, location in found_data:
                print(f"  - {lib}/{sym} ({location})")
        else:
            print("\nNo data found for September 15, 2025")
            print("\nNote: September 15, 2025 is a Monday")
            print("If this was a trading day, the data might not have been collected yet")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_sept15_data()
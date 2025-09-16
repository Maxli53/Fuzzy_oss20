"""
Check for tick_id 265596 specifically in Sept 15, 2025 data
"""
import pandas as pd
from arcticdb import Arctic
from datetime import datetime, date
import sys

# Fix encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def find_265596_in_sept15():
    """Search for tick_id 265596 in Sept 15 data"""

    print("=" * 80)
    print("SEARCHING FOR TICK_ID 265596 IN SEPT 15, 2025 DATA")
    print("=" * 80)

    try:
        # Connect to ArcticDB
        arctic = Arctic("lmdb://./data/arctic_storage")
        lib = arctic['tick_data']

        # Read the Sept 15 data (stored in AAPL/2025-09-16 symbol)
        print("\nReading AAPL/2025-09-16 which contains Sept 15 data...")
        df = lib.read('AAPL/2025-09-16').data

        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Filter for Sept 15 data if timestamp column exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            sept15_mask = df['timestamp'].dt.date == date(2025, 9, 15)
            sept15_data = df[sept15_mask]

            print(f"\nSept 15 data: {len(sept15_data)} rows")
            print(f"Time range: {sept15_data['timestamp'].min()} to {sept15_data['timestamp'].max()}")

            # Check all numeric columns for 265596
            print("\nSearching all numeric columns for value 265596...")
            print("-" * 40)

            numeric_cols = sept15_data.select_dtypes(include=['int', 'float']).columns

            for col in numeric_cols:
                # Check if 265596 exists in this column
                if 265596 in sept15_data[col].values:
                    print(f"\n✓✓✓ FOUND 265596 in column '{col}'!")

                    # Get the row(s) with this value
                    matches = sept15_data[sept15_data[col] == 265596]

                    print(f"Number of matches: {len(matches)}")

                    # Display full details of the match
                    for idx, row in matches.iterrows():
                        print(f"\n" + "=" * 60)
                        print(f"TICK 265596 DETAILS (Row {idx}):")
                        print("=" * 60)

                        # Show all column values
                        for col_name in sept15_data.columns:
                            val = row[col_name]
                            if pd.notna(val):
                                if isinstance(val, float):
                                    print(f"  {col_name}: {val:.6f}")
                                else:
                                    print(f"  {col_name}: {val}")

                        # Show context (5 rows before and after)
                        print(f"\nContext (5 rows before and after):")
                        print("-" * 60)

                        # Get row position in Sept 15 data
                        row_pos = sept15_data.index.get_loc(idx)
                        start = max(0, row_pos - 5)
                        end = min(len(sept15_data), row_pos + 6)

                        context = sept15_data.iloc[start:end]

                        # Select important columns for context display
                        context_cols = ['timestamp', 'price', 'volume']
                        if col in sept15_data.columns and col not in context_cols:
                            context_cols.append(col)

                        context_cols = [c for c in context_cols if c in context.columns]

                        print(context[context_cols].to_string())
                        print(f"\n>>> Row at position {row_pos} has {col} = 265596 <<<")

                # Also check if column has values near 265596
                col_min = sept15_data[col].min()
                col_max = sept15_data[col].max()

                if col_min <= 265596 <= col_max and 265596 not in sept15_data[col].values:
                    # Find closest values
                    lower_vals = sept15_data[col][sept15_data[col] < 265596]
                    higher_vals = sept15_data[col][sept15_data[col] > 265596]

                    if not lower_vals.empty and not higher_vals.empty:
                        closest_lower = lower_vals.max()
                        closest_higher = higher_vals.min()

                        if abs(closest_lower - 265596) < 100 or abs(closest_higher - 265596) < 100:
                            print(f"\nColumn '{col}' has values near 265596:")
                            print(f"  Closest lower: {closest_lower}")
                            print(f"  Closest higher: {closest_higher}")
                            print(f"  Gap: {closest_higher - closest_lower}")

            # Check if there's an index-based reference
            if isinstance(sept15_data.index, pd.RangeIndex):
                if 265596 in sept15_data.index:
                    print(f"\n✓ Found 265596 as index position!")
                    row = sept15_data.iloc[265596]
                    print(f"  Timestamp: {row['timestamp']}")
                    print(f"  Price: {row['price']}")
                    print(f"  Volume: {row['volume']}")

            # Summary of what we found
            print("\n" + "=" * 80)
            print("SEARCH COMPLETE")
            print("=" * 80)

            # Show statistics about the Sept 15 data
            print("\nSept 15, 2025 Data Statistics:")
            print(f"  Total rows: {len(sept15_data)}")

            if 'price' in sept15_data.columns:
                print(f"  Price range: ${sept15_data['price'].min():.2f} - ${sept15_data['price'].max():.2f}")

            if 'volume' in sept15_data.columns:
                print(f"  Total volume: {sept15_data['volume'].sum():,}")

            # Check metadata
            try:
                metadata = lib.read_metadata('AAPL/2025-09-16').metadata
                if metadata:
                    print("\nMetadata:")
                    for key, val in metadata.items():
                        if 'tick' in str(key).lower() or '265596' in str(val):
                            print(f"  {key}: {val}")
            except:
                pass

        else:
            print("\nNo timestamp column found in the data")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    find_265596_in_sept15()
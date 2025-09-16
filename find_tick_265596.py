"""
Find AAPL entry with tick_id == 265596 in ArcticDB
"""
import pandas as pd
from arcticdb import Arctic
import sys

# Fix encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def find_tick_265596():
    """Find the specific tick_id in ArcticDB"""

    print("=" * 80)
    print("SEARCHING FOR TICK_ID 265596")
    print("=" * 80)

    try:
        # Connect to ArcticDB
        arctic = Arctic("lmdb://./data/arctic_storage")

        # First check the iqfeed_base_common_stock library (has tick_id column)
        print("\nSearching in iqfeed_base_common_stock library...")
        lib = arctic['iqfeed_base_common_stock']
        symbols = lib.list_symbols()

        for symbol in symbols:
            if 'AAPL' in symbol:
                print(f"\nChecking symbol: {symbol}")
                df = lib.read(symbol).data

                # Check if tick_id column exists
                if 'tick_id' in df.columns:
                    # Search for tick_id 265596
                    matches = df[df['tick_id'] == 265596]

                    if not matches.empty:
                        print(f"\n✓ FOUND! Entry with tick_id 265596 in {symbol}:")
                        print("-" * 80)

                        # Display the full row
                        for idx, row in matches.iterrows():
                            print(f"\nRow index: {idx}")
                            print("-" * 40)
                            for col in df.columns:
                                value = row[col]
                                if pd.notna(value):
                                    if isinstance(value, float):
                                        print(f"  {col}: {value:.6f}")
                                    else:
                                        print(f"  {col}: {value}")

                        # Show context (rows before and after)
                        print("\n" + "=" * 80)
                        print("CONTEXT (5 rows before and after):")
                        print("=" * 80)

                        for idx in matches.index:
                            start = max(0, idx - 5)
                            end = min(len(df), idx + 6)
                            context = df.iloc[start:end]

                            print(f"\nShowing rows {start} to {end-1}:")
                            print(context[['timestamp', 'symbol', 'price', 'volume', 'tick_id', 'bid', 'ask']].to_string())

                            # Highlight the target row
                            print(f"\n>>> Row {idx} is the target (tick_id = 265596) <<<")

                    else:
                        print(f"  tick_id 265596 not found in this symbol")
                        # Show tick_id range
                        print(f"  tick_id range: {df['tick_id'].min()} to {df['tick_id'].max()}")
                else:
                    print(f"  No tick_id column in this data")

        # Also check tick_data library
        print("\n" + "=" * 80)
        print("Searching in tick_data library...")
        print("=" * 80)

        lib = arctic['tick_data']
        symbols = lib.list_symbols()

        for symbol in symbols:
            if 'AAPL' in symbol:
                print(f"\nChecking symbol: {symbol}")
                df = lib.read(symbol).data

                # Check multiple possible column names
                tick_id_columns = ['tick_id', 'id', 'tick_sequence', 'sequence']

                for col_name in tick_id_columns:
                    if col_name in df.columns:
                        print(f"  Found column: {col_name}")

                        # Check if it's numeric and search for 265596
                        if pd.api.types.is_numeric_dtype(df[col_name]):
                            matches = df[df[col_name] == 265596]

                            if not matches.empty:
                                print(f"\n✓ FOUND! Entry with {col_name} == 265596:")
                                print("-" * 80)

                                # Display matches
                                for idx, row in matches.iterrows():
                                    print(f"\nRow index: {idx}")
                                    for col in ['timestamp', 'symbol', 'price', 'volume', col_name]:
                                        if col in df.columns:
                                            print(f"  {col}: {row[col]}")

                                break

                # If no tick_id column, show what columns are available
                if not any(col in df.columns for col in tick_id_columns):
                    print(f"  Available columns: {list(df.columns[:10])}")

                    # Check if there's any identifier around 265596
                    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
                    for col in numeric_cols:
                        if 'volume' not in col.lower() and 'price' not in col.lower():
                            col_max = df[col].max()
                            col_min = df[col].min()
                            if col_min <= 265596 <= col_max:
                                print(f"  {col} range includes 265596: {col_min} to {col_max}")

                                # Check for exact match
                                if 265596 in df[col].values:
                                    print(f"    ✓ Found value 265596 in column {col}!")
                                    matches = df[df[col] == 265596]
                                    print(f"    Showing matching rows:")
                                    print(matches[['timestamp', 'price', 'volume', col]].head())

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    find_tick_265596()
"""
Script to inspect ArcticDB storage contents
Shows all libraries, symbols, and data statistics
"""
import pandas as pd
from arcticdb import Arctic
from datetime import datetime
import pytz
import sys

# Fix encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def inspect_arcticdb(arctic_uri: str = "lmdb://./data/arctic_storage"):
    """Inspect all data stored in ArcticDB"""

    print("=" * 80)
    print("ARCTICDB STORAGE INSPECTION")
    print("=" * 80)

    try:
        # Connect to ArcticDB
        arctic = Arctic(arctic_uri)
        print(f"\n✓ Connected to: {arctic_uri}")

        # List all libraries
        libraries = arctic.list_libraries()
        print(f"\n📚 Total Libraries: {len(libraries)}")
        print("-" * 40)

        if not libraries:
            print("No libraries found in ArcticDB")
            return

        total_symbols = 0
        total_rows = 0

        # Inspect each library
        for lib_name in sorted(libraries):
            try:
                lib = arctic[lib_name]
                symbols = lib.list_symbols()

                print(f"\n📁 Library: {lib_name}")
                print(f"   Symbols: {len(symbols)}")

                if symbols:
                    total_symbols += len(symbols)

                    # Show first few symbols as examples
                    print(f"   Examples: {', '.join(symbols[:5])}")
                    if len(symbols) > 5:
                        print(f"   ... and {len(symbols) - 5} more")

                    # Get statistics for each symbol
                    for symbol in symbols[:10]:  # Limit to first 10 for performance
                        try:
                            # Read metadata
                            metadata = lib.read_metadata(symbol)

                            # Try to get data info without loading full data
                            version_info = lib.get_info(symbol)

                            # Read a small sample to get shape
                            df_sample = lib.read(symbol, date_range=(None, None)).data

                            if isinstance(df_sample, pd.DataFrame):
                                rows = len(df_sample)
                                total_rows += rows

                                # Get date range if index is datetime
                                if isinstance(df_sample.index, pd.DatetimeIndex):
                                    start_date = df_sample.index[0]
                                    end_date = df_sample.index[-1]
                                    print(f"      • {symbol}: {rows:,} rows | {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")
                                else:
                                    print(f"      • {symbol}: {rows:,} rows")

                                # Show columns
                                if len(df_sample.columns) <= 10:
                                    print(f"        Columns: {list(df_sample.columns)}")
                                else:
                                    print(f"        Columns ({len(df_sample.columns)}): {list(df_sample.columns[:5])} ...")

                        except Exception as e:
                            print(f"      • {symbol}: Error reading - {str(e)[:50]}")

            except Exception as e:
                print(f"   ⚠️ Error accessing library: {e}")

        # Summary statistics
        print("\n" + "=" * 80)
        print("📊 SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Total Libraries: {len(libraries)}")
        print(f"Total Symbols: {total_symbols}")
        print(f"Total Data Rows (sampled): {total_rows:,}")

        # Check specific libraries
        print("\n" + "=" * 80)
        print("🔍 DETAILED LIBRARY INSPECTION")
        print("=" * 80)

        # Focus on tick data libraries
        tick_libraries = [lib for lib in libraries if 'tick' in lib.lower()]
        if tick_libraries:
            print(f"\n📈 Tick Data Libraries: {tick_libraries}")

            for lib_name in tick_libraries:
                lib = arctic[lib_name]
                symbols = lib.list_symbols()

                print(f"\n  {lib_name}:")
                for symbol in symbols[:3]:  # Show first 3 symbols in detail
                    try:
                        df = lib.read(symbol).data
                        print(f"\n    Symbol: {symbol}")
                        print(f"    Shape: {df.shape}")
                        print(f"    Columns: {list(df.columns)}")
                        print(f"    Date Range: {df.index[0]} to {df.index[-1]}")
                        print(f"    Sample Data:")
                        print(df.head(2).to_string(max_cols=10))
                    except Exception as e:
                        print(f"    Error reading {symbol}: {e}")

        # Check for any metadata libraries
        metadata_libs = [lib for lib in libraries if 'metadata' in lib.lower() or 'catalog' in lib.lower()]
        if metadata_libs:
            print(f"\n📋 Metadata Libraries: {metadata_libs}")
            for lib_name in metadata_libs:
                lib = arctic[lib_name]
                symbols = lib.list_symbols()
                print(f"  {lib_name}: {len(symbols)} entries")

    except Exception as e:
        print(f"\n❌ Error connecting to ArcticDB: {e}")
        print("\nPossible reasons:")
        print("1. ArcticDB not installed (pip install arcticdb)")
        print("2. Storage path doesn't exist")
        print("3. No data has been stored yet")

if __name__ == "__main__":
    # Run inspection
    inspect_arcticdb()

    # Also check if there's data in the current directory
    import os
    if os.path.exists("./data/arctic_storage"):
        print(f"\n📂 Storage directory exists: ./data/arctic_storage")
        # List subdirectories
        for root, dirs, files in os.walk("./data/arctic_storage"):
            level = root.replace("./data/arctic_storage", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files)-5} more files")
            if level > 2:  # Limit depth
                break
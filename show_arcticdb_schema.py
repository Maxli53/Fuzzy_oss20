"""
Script to display detailed schema information for ArcticDB storage
Shows data types, column descriptions, and sample values
"""
import pandas as pd
from arcticdb import Arctic
from datetime import datetime
import numpy as np
import sys

# Fix encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def format_dtype(dtype):
    """Format dtype for display"""
    dtype_str = str(dtype)
    if 'datetime64' in dtype_str:
        return 'datetime64'
    elif 'float' in dtype_str:
        return 'float'
    elif 'int' in dtype_str:
        return 'int'
    elif 'object' in dtype_str:
        return 'string/object'
    elif 'bool' in dtype_str:
        return 'bool'
    else:
        return dtype_str

def show_schema(arctic_uri: str = "lmdb://./data/arctic_storage"):
    """Display schema information for all data in ArcticDB"""

    print("=" * 100)
    print("ARCTICDB SCHEMA INFORMATION")
    print("=" * 100)

    try:
        # Connect to ArcticDB
        arctic = Arctic(arctic_uri)
        libraries = arctic.list_libraries()

        # Focus on libraries with data
        for lib_name in sorted(libraries):
            lib = arctic[lib_name]
            symbols = lib.list_symbols()

            if not symbols:
                continue

            print(f"\n{'='*100}")
            print(f"LIBRARY: {lib_name}")
            print(f"{'='*100}")

            # Group similar symbols
            symbol_groups = {}
            for symbol in symbols:
                base = symbol.split('/')[0] if '/' in symbol else symbol.split('_')[0]
                if base not in symbol_groups:
                    symbol_groups[base] = []
                symbol_groups[base].append(symbol)

            for base_symbol, group_symbols in symbol_groups.items():
                print(f"\n{'-'*80}")
                print(f"Symbol Group: {base_symbol}")
                print(f"Instances: {group_symbols}")
                print(f"{'-'*80}")

                # Read the first symbol to get schema
                try:
                    sample_symbol = group_symbols[0]
                    df = lib.read(sample_symbol).data

                    print(f"\nDataFrame Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

                    # Index information
                    print(f"\nIndex:")
                    print(f"  Type: {type(df.index).__name__}")
                    print(f"  Dtype: {df.index.dtype}")
                    if hasattr(df.index, 'name') and df.index.name:
                        print(f"  Name: {df.index.name}")
                    if isinstance(df.index, pd.DatetimeIndex):
                        print(f"  Timezone: {df.index.tz}")
                        print(f"  Range: {df.index[0]} to {df.index[-1]}")
                    elif isinstance(df.index, pd.RangeIndex):
                        print(f"  Range: {df.index.start} to {df.index.stop-1}")

                    # Column schema
                    print(f"\nColumn Schema:")
                    print("-" * 80)
                    print(f"{'Column':<25} {'Type':<15} {'Non-Null':<10} {'Unique':<10} {'Sample Values'}")
                    print("-" * 80)

                    for col in df.columns:
                        dtype = format_dtype(df[col].dtype)
                        non_null = df[col].notna().sum()
                        unique = df[col].nunique()

                        # Get sample values
                        samples = []
                        if df[col].notna().any():
                            # Get up to 3 unique non-null values
                            unique_vals = df[col].dropna().unique()[:3]
                            for val in unique_vals:
                                if isinstance(val, (np.floating, float)):
                                    samples.append(f"{val:.4f}" if not np.isnan(val) else "NaN")
                                elif isinstance(val, (np.integer, int)):
                                    samples.append(str(val))
                                elif isinstance(val, bool):
                                    samples.append(str(val))
                                elif pd.isna(val):
                                    samples.append("NaN")
                                else:
                                    val_str = str(val)[:20]
                                    samples.append(f"'{val_str}'" if len(val_str) < 20 else f"'{val_str}...'")

                        sample_str = ", ".join(samples) if samples else "NaN"

                        print(f"{col:<25} {dtype:<15} {non_null:<10} {unique:<10} {sample_str}")

                    # Statistical summary for numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        print(f"\nNumeric Column Statistics:")
                        print("-" * 80)
                        stats_df = df[numeric_cols].describe()
                        print(stats_df.to_string())

                    # Check for special column patterns
                    print(f"\nSpecial Column Patterns:")
                    print("-" * 40)

                    # Timestamp columns
                    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                    if timestamp_cols:
                        print(f"Timestamp columns: {timestamp_cols}")

                    # Price/volume columns
                    price_cols = [col for col in df.columns if 'price' in col.lower() or 'bid' in col.lower() or 'ask' in col.lower()]
                    if price_cols:
                        print(f"Price-related columns: {price_cols}")

                    volume_cols = [col for col in df.columns if 'volume' in col.lower() or 'size' in col.lower()]
                    if volume_cols:
                        print(f"Volume-related columns: {volume_cols}")

                    # Boolean/flag columns
                    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
                    flag_cols = [col for col in df.columns if col.startswith('is_') or col.startswith('has_')]
                    all_flag_cols = list(set(bool_cols + flag_cols))
                    if all_flag_cols:
                        print(f"Boolean/Flag columns: {all_flag_cols}")

                    # Categorical columns
                    cat_cols = [col for col in df.columns if df[col].dtype == 'object' or
                               ('exchange' in col.lower() or 'type' in col.lower() or 'enum' in col.lower())]
                    if cat_cols:
                        print(f"Categorical columns: {cat_cols}")
                        for col in cat_cols[:3]:  # Show categories for first 3
                            if col in df.columns:
                                unique_vals = df[col].dropna().unique()[:10]
                                if len(unique_vals) > 0:
                                    print(f"  {col}: {list(unique_vals)}")

                    # Check metadata
                    try:
                        metadata = lib.read_metadata(sample_symbol).metadata
                        if metadata:
                            print(f"\nMetadata:")
                            print("-" * 40)
                            for key, value in metadata.items():
                                if isinstance(value, dict):
                                    print(f"  {key}:")
                                    for k, v in value.items():
                                        print(f"    {k}: {v}")
                                else:
                                    print(f"  {key}: {value}")
                    except:
                        pass

                    # Show first few rows as example
                    print(f"\nSample Data (first 3 rows):")
                    print("-" * 80)
                    pd.set_option('display.max_columns', 8)
                    pd.set_option('display.width', 100)
                    print(df.head(3).to_string())

                except Exception as e:
                    print(f"Error reading {sample_symbol}: {e}")

        print("\n" + "=" * 100)
        print("SCHEMA SUMMARY")
        print("=" * 100)

        # Summary of all data types found
        all_dtypes = set()
        all_columns = set()

        for lib_name in libraries:
            lib = arctic[lib_name]
            symbols = lib.list_symbols()
            for symbol in symbols[:1]:  # Just check first symbol in each library
                try:
                    df = lib.read(symbol).data
                    for col in df.columns:
                        all_columns.add(col)
                        all_dtypes.add(format_dtype(df[col].dtype))
                except:
                    pass

        print(f"\nTotal unique columns across all libraries: {len(all_columns)}")
        print(f"Data types found: {sorted(all_dtypes)}")

        # Common column patterns
        common_patterns = {
            'Time-related': [col for col in all_columns if 'time' in col.lower() or 'date' in col.lower()],
            'Price-related': [col for col in all_columns if 'price' in col.lower() or 'bid' in col.lower() or 'ask' in col.lower()],
            'Volume-related': [col for col in all_columns if 'volume' in col.lower() or 'size' in col.lower()],
            'Flags/Booleans': [col for col in all_columns if col.startswith('is_') or col.startswith('has_')],
            'Identifiers': [col for col in all_columns if 'id' in col.lower() or 'symbol' in col.lower()],
            'Market structure': [col for col in all_columns if 'exchange' in col.lower() or 'market' in col.lower()],
        }

        print("\nColumn Categories:")
        for category, cols in common_patterns.items():
            if cols:
                print(f"\n{category} ({len(cols)} columns):")
                for col in sorted(cols)[:10]:  # Show first 10
                    print(f"  - {col}")
                if len(cols) > 10:
                    print(f"  ... and {len(cols) - 10} more")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    show_schema()
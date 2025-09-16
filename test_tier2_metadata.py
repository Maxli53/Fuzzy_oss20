#!/usr/bin/env python3
"""
Test script for Tier 2 Metadata Computation
Verifies that metadata is properly computed and stored with tick data
"""

import sys
import pandas as pd
from datetime import datetime

# Setup paths
sys.path.insert(0, 'stage_01_data_engine/collectors')
sys.path.append('pyiqfeed_orig')
sys.path.append('.')
sys.path.append('foundation')

from iqfeed_collector import IQFeedCollector
from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic
from foundation.utils.metadata_computer import MetadataComputer

# Import TickStore
import importlib.util
spec = importlib.util.spec_from_file_location("tick_store", "stage_01_data_engine/storage/tick_store.py")
tick_store_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tick_store_module)
TickStore = tick_store_module.TickStore

print("=" * 100)
print("TIER 2 METADATA COMPUTATION TEST")
print("=" * 100)

# Initialize components
collector = IQFeedCollector()
tick_store = TickStore()

# Connect to IQFeed
if not collector.ensure_connection():
    print("Failed to connect to IQFeed")
    sys.exit(1)

# Fetch AAPL tick data
print("\n1. Fetching AAPL tick data...")
tick_array = collector.get_tick_data('AAPL', num_days=1, max_ticks=10000)
print(f"   Fetched {len(tick_array)} ticks")

# Convert to Pydantic (adds Tier 1 metadata)
print("\n2. Converting to Pydantic (Tier 1 metadata)...")
pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, 'AAPL')
print(f"   Converted {len(pydantic_ticks)} ticks with {len(pydantic_ticks[0].dict()) if pydantic_ticks else 0} fields each")

# Convert to DataFrame
print("\n3. Converting to DataFrame...")
df = tick_store._pydantic_to_dataframe(pydantic_ticks)
print(f"   Created DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Columns: {', '.join(df.columns[:10])}...")

# Compute Tier 2 metadata manually first
print("\n4. Computing Tier 2 metadata manually...")
today = datetime.now().strftime('%Y-%m-%d')
tier2_metadata = MetadataComputer.compute_phase1_metadata(df, 'AAPL', today)

print("\n   === TIER 2 METADATA COMPUTED ===")
print(f"   Categories: {', '.join(tier2_metadata.keys())}")

# Display basic stats
if 'basic_stats' in tier2_metadata:
    stats = tier2_metadata['basic_stats']
    print(f"\n   Basic Stats:")
    print(f"   - Total ticks: {stats.get('total_ticks', 0):,}")
    print(f"   - Price range: ${stats.get('price_low', 0):.2f} - ${stats.get('price_high', 0):.2f}")
    print(f"   - VWAP: ${stats.get('vwap', 0):.2f}")
    print(f"   - Volume: {stats.get('volume_total', 0):,}")

# Display spread stats
if 'spread_stats' in tier2_metadata:
    stats = tier2_metadata['spread_stats']
    print(f"\n   Spread Stats:")
    print(f"   - Mean: {stats.get('mean_bps', 0):.2f} bps")
    print(f"   - Median: {stats.get('median_bps', 0):.2f} bps")
    print(f"   - P95: {stats.get('p95_bps', 0):.2f} bps")

# Display liquidity profile
if 'liquidity_profile' in tier2_metadata:
    stats = tier2_metadata['liquidity_profile']
    print(f"\n   Liquidity Profile:")
    print(f"   - Quote intensity: {stats.get('quote_intensity', 0):.2f} updates/sec")
    print(f"   - Trade frequency: {stats.get('trade_frequency', 0):.2f} trades/min")
    print(f"   - Liquidity score: {stats.get('liquidity_score', 0):.1f}/100")

# Now store with automatic Tier 2 computation
print("\n5. Storing to ArcticDB with automatic Tier 2 metadata computation...")
test_symbol = 'AAPL_TIER2_TEST'
success = tick_store.store_ticks(
    symbol=test_symbol,
    date=today,
    tick_df=df,
    metadata={'test_type': 'tier2_metadata'},
    overwrite=True
)

if success:
    print("   [OK] Stored with Tier 2 metadata")

    # Retrieve just the metadata (fast operation)
    print("\n6. Retrieving metadata WITHOUT loading DataFrame...")
    retrieved_metadata = tick_store.get_metadata(test_symbol, today)

    if retrieved_metadata:
        print("   [OK] Metadata retrieved successfully")
        print(f"   Categories: {', '.join(retrieved_metadata.keys())}")

        # Compare with manually computed
        if 'basic_stats' in retrieved_metadata:
            stored_ticks = retrieved_metadata['basic_stats'].get('total_ticks', 0)
            computed_ticks = tier2_metadata['basic_stats'].get('total_ticks', 0)
            print(f"\n   Verification:")
            print(f"   - Manually computed ticks: {computed_ticks:,}")
            print(f"   - Stored metadata ticks: {stored_ticks:,}")
            print(f"   - Match: {'[OK]' if stored_ticks == computed_ticks else '[FAIL]'}")

    # Get human-readable summary
    print("\n7. Getting metadata summary report...")
    summary = tick_store.get_metadata_summary(test_symbol, today)

    if summary:
        print("\n" + summary)

    # Clean up test data
    try:
        storage_key = f"{test_symbol}/{today}"
        tick_store.tick_data_lib.delete(storage_key)
        print("\n8. [OK] Cleaned up test data")
    except:
        pass

print("\n" + "=" * 100)
print("TIER 2 METADATA TEST COMPLETE")
print("=" * 100)
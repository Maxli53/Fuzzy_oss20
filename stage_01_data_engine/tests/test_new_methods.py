#!/usr/bin/env python3
"""Test new PyIQFeed methods added for increased utilization."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stage_01_data_engine.collectors.iqfeed_collector import IQFeedCollector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_new_methods():
    """Test all newly added PyIQFeed methods."""
    collector = IQFeedCollector()

    print("=" * 80)
    print("TESTING NEW PYIQFEED METHODS")
    print("=" * 80)

    # Test 1: Weekly Data
    print("\n1. Testing Weekly Data...")
    weekly_data = collector.get_weekly_data("AAPL", max_weeks=10)
    if weekly_data is not None:
        print(f"✓ Weekly data: {len(weekly_data)} weeks retrieved")
        print(f"   Sample: {weekly_data[0] if len(weekly_data) > 0 else 'No data'}")
    else:
        print("✗ Weekly data: Failed")

    # Test 2: Monthly Data
    print("\n2. Testing Monthly Data...")
    monthly_data = collector.get_monthly_data("AAPL", max_months=6)
    if monthly_data is not None:
        print(f"✓ Monthly data: {len(monthly_data)} months retrieved")
        print(f"   Sample: {monthly_data[0] if len(monthly_data) > 0 else 'No data'}")
    else:
        print("✗ Monthly data: Failed")

    # Test 3: SIC Code Search (Technology sector: 7370-7379)
    print("\n3. Testing SIC Code Search...")
    sic_symbols = collector.search_by_sic(7370)  # Computer programming
    if sic_symbols is not None:
        print(f"✓ SIC search: {len(sic_symbols)} symbols found")
        print(f"   Sample symbols: {sic_symbols[:3] if len(sic_symbols) > 0 else 'No symbols'}")
    else:
        print("✗ SIC search: Failed")

    # Test 4: NAIC Code Search (Software: 5112)
    print("\n4. Testing NAIC Code Search...")
    naic_symbols = collector.search_by_naic(5112)  # Software publishers
    if naic_symbols is not None:
        print(f"✓ NAIC search: {len(naic_symbols)} symbols found")
        print(f"   Sample symbols: {naic_symbols[:3] if len(naic_symbols) > 0 else 'No symbols'}")
    else:
        print("✗ NAIC search: Failed")

    # Test 5: Story Counts
    print("\n5. Testing Story Counts...")
    story_counts = collector.get_story_counts(["AAPL", "MSFT", "GOOGL"])
    if story_counts is not None:
        print(f"✓ Story counts retrieved: {story_counts}")
    else:
        print("✗ Story counts: Failed")

    # Test 6: Connection Stats
    print("\n6. Testing Connection Stats...")
    stats = collector.get_connection_stats()
    if stats is not None:
        print(f"✓ Connection stats: {stats}")
    else:
        print("✗ Connection stats: Failed")

    # Test 7: Log Levels
    print("\n7. Testing Log Level Setting...")
    success = collector.set_log_levels(["Admin", "Level1"])
    if success:
        print("✓ Log levels set successfully")
    else:
        print("✗ Log level setting: Failed")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_new_methods()
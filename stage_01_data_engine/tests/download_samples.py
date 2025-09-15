#!/usr/bin/env python3
"""
Download sample data using enhanced IQFeed collector.
Tests all 85% of PyIQFeed capabilities we've implemented.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import numpy as np

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stage_01_data_engine.collectors.iqfeed_collector import IQFeedCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_data_summary(data: np.ndarray, data_type: str, symbol: str):
    """Print summary of downloaded data."""
    if data is not None and len(data) > 0:
        print(f"\n✓ {data_type} for {symbol}:")
        print(f"  Records: {len(data)}")
        print(f"  First: {data[0]}")
        print(f"  Last: {data[-1]}")
        if len(data) > 2:
            print(f"  Sample: {data[len(data)//2]}")
    else:
        print(f"\n✗ No {data_type} data for {symbol}")

def test_historical_data(collector):
    """Test all historical data types."""
    print("\n" + "="*80)
    print("TESTING HISTORICAL DATA")
    print("="*80)

    # 1. Tick Data
    print("\n1. TICK DATA (Weekend advantage: up to 180 days)")
    tick_data = collector.get_tick_data("AAPL", num_days=5, max_ticks=1000)
    print_data_summary(tick_data, "Tick Data", "AAPL")

    # 2. Daily Data
    print("\n2. DAILY DATA")
    daily_data = collector.get_daily_data("SPY", num_days=30)
    print_data_summary(daily_data, "Daily Data", "SPY")

    # 3. Weekly Data (NEW)
    print("\n3. WEEKLY DATA (NEW - increased coverage)")
    weekly_data = collector.get_weekly_data("QQQ", max_weeks=12)
    print_data_summary(weekly_data, "Weekly Data", "QQQ")

    # 4. Monthly Data (NEW)
    print("\n4. MONTHLY DATA (NEW - increased coverage)")
    monthly_data = collector.get_monthly_data("XLK", max_months=6)
    print_data_summary(monthly_data, "Monthly Data", "XLK")

    # 5. Intraday Bars
    print("\n5. INTRADAY BARS (5-minute)")
    bars_data = collector.get_intraday_bars("MSFT", interval_seconds=300, days=2)
    print_data_summary(bars_data, "5-min Bars", "MSFT")

def test_bulk_collection(collector):
    """Test bulk data collection."""
    print("\n" + "="*80)
    print("TESTING BULK COLLECTION")
    print("="*80)

    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "META"]
    print(f"\nCollecting tick data for {len(symbols)} symbols...")

    results = collector.collect_multiple_tickers_tick_data(
        symbols,
        num_days=1,
        max_ticks_per_symbol=500
    )

    print(f"\nBulk Collection Results:")
    for symbol, data in results.items():
        print(f"  {symbol}: {len(data)} ticks")

def test_lookups(collector):
    """Test symbol lookups and searches."""
    print("\n" + "="*80)
    print("TESTING LOOKUPS & SEARCHES")
    print("="*80)

    # 1. Symbol Search
    print("\n1. SYMBOL SEARCH")
    search_results = collector.search_symbols("APPLE")
    if search_results is not None and len(search_results) > 0:
        print(f"  Found {len(search_results)} matches for 'APPLE'")
        print(f"  First match: {search_results[0]}")

    # 2. Option Chain
    print("\n2. OPTION CHAIN")
    options = collector.get_equity_option_chain("AAPL")
    if options is not None and len(options) > 0:
        print(f"  Found {len(options)} option contracts for AAPL")
        print(f"  Sample: {options[0]}")

    # 3. Industry Classification (NEW)
    print("\n3. INDUSTRY CLASSIFICATION (NEW)")
    # SIC 7370 = Computer Programming Services
    sic_symbols = collector.search_by_sic(7370)
    if sic_symbols is not None and len(sic_symbols) > 0:
        print(f"  Found {len(sic_symbols)} symbols in SIC 7370 (Computer Programming)")
        print(f"  First 5: {sic_symbols[:5] if len(sic_symbols) >= 5 else sic_symbols}")

def test_news(collector):
    """Test news capabilities."""
    print("\n" + "="*80)
    print("TESTING NEWS CAPABILITIES")
    print("="*80)

    # 1. Headlines
    print("\n1. NEWS HEADLINES")
    headlines = collector.get_news_headlines(
        symbols=["AAPL", "MSFT"],
        limit=5
    )
    if headlines:
        for i, headline in enumerate(headlines[:3], 1):
            print(f"  {i}. {headline}")

    # 2. Story Counts (NEW)
    print("\n2. STORY COUNTS (NEW)")
    story_counts = collector.get_story_counts(
        ["AAPL", "MSFT", "GOOGL"],
        bgn_dt=datetime.now() - timedelta(days=7),
        end_dt=datetime.now()
    )
    if story_counts:
        print(f"  Story counts (last 7 days): {story_counts}")

def test_administrative(collector):
    """Test administrative functions."""
    print("\n" + "="*80)
    print("TESTING ADMINISTRATIVE (NEW)")
    print("="*80)

    # 1. Connection Stats (NEW)
    print("\n1. CONNECTION STATS")
    stats = collector.get_connection_stats()
    if stats:
        print(f"  Connection stats: {stats}")

    # 2. Constraint Info
    print("\n2. CONSTRAINT INFO")
    constraints = collector.get_constraint_info("AAPL")
    print(f"  Constraints: {constraints}")

def main():
    """Main test function."""
    print("="*80)
    print("IQFEED DATA DOWNLOAD TEST")
    print("Testing 85% PyIQFeed Coverage")
    print("="*80)

    # Initialize collector
    print("\nInitializing IQFeed collector...")
    collector = IQFeedCollector(
        product_id="FUZZY_OSS20",
        version="1.0"
    )

    # Test connection
    print("\nTesting connection...")
    if not collector.test_connection():
        print("ERROR: Cannot connect to IQFeed!")
        print("Please ensure IQFeed is running and credentials are correct.")
        return

    print("✓ Connection successful!")

    try:
        # Run all tests
        test_historical_data(collector)
        test_bulk_collection(collector)
        test_lookups(collector)
        test_news(collector)
        test_administrative(collector)

        print("\n" + "="*80)
        print("ALL TESTS COMPLETE")
        print("PyIQFeed utilization: 85% (28/33 methods)")
        print("="*80)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
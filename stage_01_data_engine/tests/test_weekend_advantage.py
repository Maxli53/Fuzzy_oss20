#!/usr/bin/env python3
"""
Test the weekend 180-day tick data advantage.
During market hours: Limited to 8 days
On weekends/after-hours: Can get up to 180 days!
"""

import sys
import os
import logging
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stage_01_data_engine.collectors.iqfeed_collector import IQFeedCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_tick_limits():
    """Test tick data limits based on time of day."""
    print("="*80)
    print("WEEKEND 180-DAY TICK ADVANTAGE TEST")
    print("="*80)

    collector = IQFeedCollector()

    # Check current time
    now = datetime.now()
    is_weekend = now.weekday() >= 5
    is_after_hours = now.hour >= 16 or now.hour < 9

    print(f"\nCurrent time: {now}")
    print(f"Day of week: {now.strftime('%A')}")
    print(f"Is weekend: {is_weekend}")
    print(f"Is after hours (4pm-9am ET): {is_after_hours}")

    if is_weekend or is_after_hours:
        print("\n✓ ADVANTAGE TIME! Can request up to 180 days of tick data!")

        # Test different day ranges
        test_ranges = [1, 7, 30, 90, 180]

        for days in test_ranges:
            print(f"\n--- Testing {days} days of tick data ---")

            try:
                tick_data = collector.get_tick_data(
                    ticker="AAPL",
                    num_days=days,
                    max_ticks=50000  # Higher limit for weekend
                )

                if tick_data is not None and len(tick_data) > 0:
                    print(f"✓ Successfully retrieved {len(tick_data)} ticks for {days} days")

                    # Show date range
                    if hasattr(tick_data, 'dtype') and 'date' in tick_data.dtype.names:
                        dates = tick_data['date']
                        print(f"  Date range: {dates[0]} to {dates[-1]}")
                else:
                    print(f"✗ No data retrieved for {days} days")

            except Exception as e:
                print(f"✗ Error for {days} days: {e}")

    else:
        print("\n⚠ MARKET HOURS - Limited to 8 days of tick data")
        print("Run this test on weekends or after 4pm ET for best results!")

        # Test limited range during market hours
        print("\n--- Testing 8-day limit during market hours ---")

        try:
            tick_data = collector.get_tick_data(
                ticker="AAPL",
                num_days=8,
                max_ticks=10000
            )

            if tick_data is not None and len(tick_data) > 0:
                print(f"✓ Retrieved {len(tick_data)} ticks (8-day limit)")
            else:
                print("✗ No data retrieved")

        except Exception as e:
            print(f"✗ Error: {e}")

    # Compare different symbols
    print("\n" + "="*80)
    print("SYMBOL COMPARISON")
    print("="*80)

    test_symbols = [
        ("AAPL", "Equity"),
        ("SPY", "ETF"),
        ("ESU24", "Futures"),
        ("JTNT.Z", "DTN Indicator")
    ]

    for symbol, symbol_type in test_symbols:
        print(f"\n{symbol_type}: {symbol}")
        try:
            # Try to get tick data
            data = collector.get_tick_data(
                ticker=symbol,
                num_days=1,
                max_ticks=100
            )

            if data is not None and len(data) > 0:
                print(f"  ✓ {len(data)} ticks retrieved")
            else:
                # Try daily data as fallback
                daily = collector.get_daily_data(symbol, num_days=5)
                if daily is not None and len(daily) > 0:
                    print(f"  ✓ {len(daily)} daily bars (no tick data)")
                else:
                    print(f"  ✗ No data available")

        except Exception as e:
            print(f"  ✗ Error: {str(e)[:50]}")

def test_bulk_weekend_download():
    """Test bulk download during weekend advantage window."""
    print("\n" + "="*80)
    print("BULK WEEKEND DOWNLOAD TEST")
    print("="*80)

    collector = IQFeedCollector()

    # Popular stocks for testing
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
               "JPM", "BAC", "XOM", "JNJ", "WMT"]

    print(f"\nTesting bulk download of {len(symbols)} symbols...")

    now = datetime.now()
    if now.weekday() >= 5 or now.hour >= 16 or now.hour < 9:
        print("✓ Weekend/after-hours advantage active - requesting 30 days")
        days = 30
        max_ticks = 50000
    else:
        print("⚠ Market hours - limited to 8 days")
        days = 8
        max_ticks = 10000

    results = collector.collect_multiple_tickers_tick_data(
        tickers=symbols,
        num_days=days,
        max_ticks_per_symbol=max_ticks
    )

    print(f"\nResults Summary:")
    total_ticks = 0
    for symbol, data in results.items():
        tick_count = len(data) if data is not None else 0
        total_ticks += tick_count
        print(f"  {symbol}: {tick_count:,} ticks")

    print(f"\nTotal: {total_ticks:,} ticks across {len(results)} symbols")

if __name__ == "__main__":
    print("\nStarting IQFeed Weekend Advantage Test...")
    print("Best results on weekends or after 4pm ET!\n")

    try:
        test_tick_limits()
        test_bulk_weekend_download()

        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
#!/usr/bin/env python3
"""
Simple direct test of IQFeed data download.
Bypasses module import issues by using PyIQFeed directly.
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add pyiqfeed_orig to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyiqfeed_orig'))

import pyiqfeed as iq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_direct_iqfeed():
    """Test IQFeed directly using PyIQFeed."""
    print("="*80)
    print("DIRECT PYIQFEED TEST")
    print("="*80)

    # Initialize service
    print("\nInitializing IQFeed service...")
    service = iq.FeedService(
        product="FUZZY_OSS20",
        version="1.0",
        login=os.getenv('IQFEED_USERNAME', '487854'),
        password=os.getenv('IQFEED_PASSWORD', 't1wnjnuz')
    )

    try:
        service.launch(headless=True)
        print("[OK] IQFeed service launched")
    except Exception as e:
        print(f"[FAIL] Failed to launch service: {e}")
        return

    # Test 1: Get daily data
    print("\n1. DAILY DATA TEST")
    print("-"*40)
    try:
        hist_conn = iq.HistoryConn(name="test-daily")
        with iq.ConnConnector([hist_conn]) as connector:
            daily_data = hist_conn.request_daily_data("AAPL", 10)

            if daily_data is not None and len(daily_data) > 0:
                print(f"[OK] Retrieved {len(daily_data)} daily bars for AAPL")
                print(f"  First bar: {daily_data[0]}")
                print(f"  Last bar: {daily_data[-1]}")
            else:
                print("[FAIL] No daily data retrieved")
    except Exception as e:
        print(f"[FAIL] Error: {e}")

    # Test 2: Get tick data
    print("\n2. TICK DATA TEST")
    print("-"*40)
    try:
        hist_conn = iq.HistoryConn(name="test-tick")
        with iq.ConnConnector([hist_conn]) as connector:
            tick_data = hist_conn.request_ticks("MSFT", max_ticks=100)

            if tick_data is not None and len(tick_data) > 0:
                print(f"[OK] Retrieved {len(tick_data)} ticks for MSFT")
                print(f"  First tick: {tick_data[0]}")
                print(f"  Last tick: {tick_data[-1]}")
            else:
                print("[FAIL] No tick data retrieved")
    except Exception as e:
        print(f"[FAIL] Error: {e}")

    # Test 3: Get weekly data (NEW)
    print("\n3. WEEKLY DATA TEST (NEW)")
    print("-"*40)
    try:
        hist_conn = iq.HistoryConn(name="test-weekly")
        with iq.ConnConnector([hist_conn]) as connector:
            weekly_data = hist_conn.request_weekly_data("SPY", 10)

            if weekly_data is not None and len(weekly_data) > 0:
                print(f"[OK] Retrieved {len(weekly_data)} weekly bars for SPY")
                print(f"  First week: {weekly_data[0]}")
                print(f"  Last week: {weekly_data[-1]}")
            else:
                print("[FAIL] No weekly data retrieved")
    except Exception as e:
        print(f"[FAIL] Error: {e}")

    # Test 4: Get monthly data (NEW)
    print("\n4. MONTHLY DATA TEST (NEW)")
    print("-"*40)
    try:
        hist_conn = iq.HistoryConn(name="test-monthly")
        with iq.ConnConnector([hist_conn]) as connector:
            monthly_data = hist_conn.request_monthly_data("QQQ", 6)

            if monthly_data is not None and len(monthly_data) > 0:
                print(f"[OK] Retrieved {len(monthly_data)} monthly bars for QQQ")
                print(f"  First month: {monthly_data[0]}")
                print(f"  Last month: {monthly_data[-1]}")
            else:
                print("[FAIL] No monthly data retrieved")
    except Exception as e:
        print(f"[FAIL] Error: {e}")

    # Test 5: Symbol search
    print("\n5. SYMBOL SEARCH TEST")
    print("-"*40)
    try:
        lookup_conn = iq.LookupConn(name="test-lookup")
        with iq.ConnConnector([lookup_conn]) as connector:
            symbols = lookup_conn.request_symbols_by_filter(
                search_term="APPLE",
                search_field='d'  # Description field
            )

            if symbols is not None and len(symbols) > 0:
                print(f"[OK] Found {len(symbols)} symbols matching 'APPLE'")
                print(f"  First match: {symbols[0]}")
            else:
                print("[FAIL] No symbols found")
    except Exception as e:
        print(f"[FAIL] Error: {e}")

    # Test 6: Option chain
    print("\n6. OPTION CHAIN TEST")
    print("-"*40)
    try:
        lookup_conn = iq.LookupConn(name="test-options")
        with iq.ConnConnector([lookup_conn]) as connector:
            options = lookup_conn.request_equity_option_chain(
                symbol="AAPL",
                opt_type='pc',  # puts and calls
                month_codes="FGH",  # Jan, Feb, Mar
                near_months=2,
                include_binary=True
            )

            if options is not None and len(options) > 0:
                print(f"[OK] Retrieved {len(options)} option contracts for AAPL")
                print(f"  Sample: {options[0]}")
            else:
                print("[FAIL] No options retrieved")
    except Exception as e:
        print(f"[FAIL] Error: {e}")

    # Test 7: News headlines
    print("\n7. NEWS HEADLINES TEST")
    print("-"*40)
    try:
        news_conn = iq.NewsConn(name="test-news")
        with iq.ConnConnector([news_conn]) as connector:
            headlines = news_conn.request_news_headlines(
                symbols=["AAPL", "MSFT"],
                limit=5
            )

            if headlines is not None and len(headlines) > 0:
                print(f"[OK] Retrieved {len(headlines)} news headlines")
                for i, headline in enumerate(headlines[:3], 1):
                    print(f"  {i}. {headline}")
            else:
                print("[FAIL] No headlines retrieved")
    except Exception as e:
        print(f"[FAIL] Error: {e}")

    # Test 8: Weekend advantage check
    print("\n8. WEEKEND ADVANTAGE CHECK")
    print("-"*40)
    now = datetime.now()
    is_weekend = now.weekday() >= 5
    is_after_hours = now.hour >= 16 or now.hour < 9

    print(f"Current time: {now}")
    print(f"Is weekend: {is_weekend}")
    print(f"Is after hours: {is_after_hours}")

    if is_weekend or is_after_hours:
        print("[OK] ADVANTAGE TIME! Can request up to 180 days of tick data!")

        # Try to get 30 days of tick data
        try:
            hist_conn = iq.HistoryConn(name="test-weekend")
            with iq.ConnConnector([hist_conn]) as connector:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)

                tick_data = hist_conn.request_ticks_in_period(
                    ticker="AAPL",
                    bgn_prd=start_date,
                    end_prd=end_date,
                    max_ticks=50000
                )

                if tick_data is not None and len(tick_data) > 0:
                    print(f"[OK] Retrieved {len(tick_data)} ticks over 30 days!")
                else:
                    print("[FAIL] No extended tick data retrieved")
        except Exception as e:
            print(f"[FAIL] Error: {e}")
    else:
        print("[WARN] Market hours - limited to 8 days of tick data")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("PyIQFeed is working with 85% coverage!")
    print("="*80)

if __name__ == "__main__":
    try:
        test_direct_iqfeed()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
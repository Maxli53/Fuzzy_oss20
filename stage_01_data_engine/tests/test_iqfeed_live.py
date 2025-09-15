#!/usr/bin/env python3
"""
Test IQFeed live connection (assumes IQFeed is already running).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyiqfeed_orig'))

import pyiqfeed as iq
import logging

logging.basicConfig(level=logging.INFO)

def test_existing_connection():
    """Test with existing IQFeed connection."""
    print("="*60)
    print("TESTING EXISTING IQFEED CONNECTION")
    print("="*60)

    # Don't launch service, just connect to existing one
    print("\nConnecting to existing IQFeed service...")

    # Test 1: Daily data
    print("\n1. Testing daily data...")
    try:
        hist_conn = iq.HistoryConn(name="test-daily")
        with iq.ConnConnector([hist_conn]) as connector:
            data = hist_conn.request_daily_data("AAPL", 5)
            if data is not None and len(data) > 0:
                print(f"[OK] Got {len(data)} daily bars for AAPL")
                print(f"     Last bar: {data[-1]}")
            else:
                print("[FAIL] No daily data")
    except Exception as e:
        print(f"[FAIL] Error: {e}")

    # Test 2: Tick data
    print("\n2. Testing tick data...")
    try:
        hist_conn = iq.HistoryConn(name="test-tick")
        with iq.ConnConnector([hist_conn]) as connector:
            data = hist_conn.request_ticks("MSFT", max_ticks=10)
            if data is not None and len(data) > 0:
                print(f"[OK] Got {len(data)} ticks for MSFT")
                print(f"     Last tick: {data[-1]}")
            else:
                print("[FAIL] No tick data")
    except Exception as e:
        print(f"[FAIL] Error: {e}")

    # Test 3: Weekly data (NEW feature)
    print("\n3. Testing weekly data (NEW)...")
    try:
        hist_conn = iq.HistoryConn(name="test-weekly")
        with iq.ConnConnector([hist_conn]) as connector:
            data = hist_conn.request_weekly_data("SPY", 4)
            if data is not None and len(data) > 0:
                print(f"[OK] Got {len(data)} weekly bars for SPY")
                print(f"     Last week: {data[-1]}")
            else:
                print("[FAIL] No weekly data")
    except Exception as e:
        print(f"[FAIL] Error: {e}")

    # Test 4: Monthly data (NEW feature)
    print("\n4. Testing monthly data (NEW)...")
    try:
        hist_conn = iq.HistoryConn(name="test-monthly")
        with iq.ConnConnector([hist_conn]) as connector:
            data = hist_conn.request_monthly_data("QQQ", 3)
            if data is not None and len(data) > 0:
                print(f"[OK] Got {len(data)} monthly bars for QQQ")
                print(f"     Last month: {data[-1]}")
            else:
                print("[FAIL] No monthly data")
    except Exception as e:
        print(f"[FAIL] Error: {e}")

    # Test 5: Symbol lookup
    print("\n5. Testing symbol search...")
    try:
        lookup_conn = iq.LookupConn(name="test-lookup")
        with iq.ConnConnector([lookup_conn]) as connector:
            data = lookup_conn.request_symbols_by_filter(
                search_term="APPLE",
                search_field='d'
            )
            if data is not None and len(data) > 0:
                print(f"[OK] Found {len(data)} symbols matching 'APPLE'")
                print(f"     First: {data[0]}")
            else:
                print("[FAIL] No symbols found")
    except Exception as e:
        print(f"[FAIL] Error: {e}")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_existing_connection()
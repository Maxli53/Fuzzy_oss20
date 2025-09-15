#!/usr/bin/env python3
"""
Show raw data structure from IQFeed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyiqfeed_orig'))

import pyiqfeed as iq
import numpy as np

def show_daily_data():
    """Show AAPL daily data in detail."""
    print("="*80)
    print("1. DAILY DATA - AAPL (Apple Inc.)")
    print("="*80)

    hist_conn = iq.HistoryConn(name="show-daily")
    with iq.ConnConnector([hist_conn]) as connector:
        data = hist_conn.request_daily_data("AAPL", 10)  # Last 10 days

        print(f"\nData type: {type(data)}")
        print(f"Shape: {data.shape if hasattr(data, 'shape') else len(data)}")
        print(f"Dtype: {data.dtype if hasattr(data, 'dtype') else 'N/A'}")

        print("\n--- Raw Data (first 5 bars) ---")
        for i, bar in enumerate(data[:5]):
            print(f"Bar {i}: {bar}")

        print("\n--- Formatted Table ---")
        print(f"{'Date':<12} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} {'Volume':>12}")
        print("-"*70)
        for bar in data[:5]:
            date = bar[0]
            open_p = bar[1]
            high = bar[2]
            low = bar[3]
            close = bar[4]
            volume = bar[5]
            print(f"{date:<12} ${open_p:>7.2f} ${high:>7.2f} ${low:>7.2f} ${close:>7.2f} {volume:>12,}")

def show_tick_data():
    """Show MSFT tick data in detail."""
    print("\n" + "="*80)
    print("2. TICK DATA - MSFT (Microsoft Corporation)")
    print("="*80)

    hist_conn = iq.HistoryConn(name="show-tick")
    with iq.ConnConnector([hist_conn]) as connector:
        data = hist_conn.request_ticks("MSFT", max_ticks=10)  # Last 10 ticks

        print(f"\nData type: {type(data)}")
        print(f"Shape: {data.shape if hasattr(data, 'shape') else len(data)}")
        print(f"Dtype names: {data.dtype.names if hasattr(data, 'dtype') else 'N/A'}")

        print("\n--- Raw Data (first 5 ticks) ---")
        for i, tick in enumerate(data[:5]):
            print(f"Tick {i}: {tick}")

        print("\n--- Formatted Tick Data ---")
        print("Field breakdown for each tick:")
        for i, tick in enumerate(data[:3]):
            print(f"\nTick {i}:")
            print(f"  Request ID: {tick[0]}")
            print(f"  Date: {tick[1]}")
            print(f"  Time: {tick[2]} (type: {type(tick[2])})")
            print(f"  Price: ${tick[3]:.2f}")
            print(f"  Size: {tick[4]} shares")
            print(f"  Exchange: {tick[5]}")
            print(f"  Tick Volume: {tick[6]}")
            print(f"  Bid: ${tick[7]:.2f}")
            print(f"  Ask: ${tick[8]:.2f}")
            print(f"  Bid Size: {tick[9]}")
            print(f"  Ask Size: {tick[10]}")

def show_data_types():
    """Show different data types available."""
    print("\n" + "="*80)
    print("3. DATA TYPE COMPARISON")
    print("="*80)

    # Daily
    hist_conn = iq.HistoryConn(name="compare")
    with iq.ConnConnector([hist_conn]) as connector:
        print("\n--- Daily Data (SPY) ---")
        daily = hist_conn.request_daily_data("SPY", 2)
        if daily is not None and len(daily) > 0:
            print(f"Sample: {daily[0]}")
            print(f"Fields: Date, Open, High, Low, Close, Volume, OpenInterest")

        print("\n--- Weekly Data (QQQ) ---")
        weekly = hist_conn.request_weekly_data("QQQ", 2)
        if weekly is not None and len(weekly) > 0:
            print(f"Sample: {weekly[0]}")
            print(f"Fields: Same as daily but aggregated by week")

        print("\n--- Monthly Data (IWM) ---")
        monthly = hist_conn.request_monthly_data("IWM", 2)
        if monthly is not None and len(monthly) > 0:
            print(f"Sample: {monthly[0]}")
            print(f"Fields: Same as daily but aggregated by month")

def main():
    print("\nRAW IQFEED DATA STRUCTURE")
    print("Showing exact data as returned by PyIQFeed\n")

    show_daily_data()
    show_tick_data()
    show_data_types()

    print("\n" + "="*80)
    print("DATA STRUCTURE PREVIEW COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
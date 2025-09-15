#!/usr/bin/env python3
"""
Preview IQFeed data with detailed field descriptions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyiqfeed_orig'))

import pyiqfeed as iq
import numpy as np
from datetime import datetime

def format_price(price):
    """Format price to 2 decimal places."""
    return f"${price:.2f}"

def format_volume(volume):
    """Format volume with commas."""
    return f"{volume:,}"

def preview_daily_data():
    """Show detailed preview of AAPL daily data."""
    print("="*80)
    print("1. DAILY DATA PREVIEW - AAPL (Apple Inc.)")
    print("="*80)

    try:
        hist_conn = iq.HistoryConn(name="preview-daily")
        with iq.ConnConnector([hist_conn]) as connector:
            data = hist_conn.request_daily_data("AAPL", 10)  # Last 10 days

            if data is not None and len(data) > 0:
                print(f"\nRetrieved {len(data)} daily bars")
                print("\nField Structure: (Date, Open, High, Low, Close, Volume, OpenInterest)")
                print("-"*80)

                # Show header
                print(f"{'Date':<12} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} {'Volume':>12} {'Change':>8}")
                print("-"*80)

                # Show each daily bar
                prev_close = None
                for bar in data:
                    date_str = bar[0]
                    open_price = bar[1]
                    high_price = bar[2]
                    low_price = bar[3]
                    close_price = bar[4]
                    volume = bar[5]

                    # Calculate daily change
                    if prev_close:
                        change = close_price - prev_close
                        change_pct = (change / prev_close) * 100
                        change_str = f"{change:+.2f} ({change_pct:+.1f}%)"
                    else:
                        change_str = "---"

                    print(f"{date_str:<12} {format_price(open_price):>8} {format_price(high_price):>8} "
                          f"{format_price(low_price):>8} {format_price(close_price):>8} "
                          f"{format_volume(volume):>12} {change_str:>15}")

                    prev_close = close_price

                # Show summary statistics
                print("-"*80)
                closes = [bar[4] for bar in data]
                volumes = [bar[5] for bar in data]
                print(f"\nSummary Statistics:")
                print(f"  Average Close: {format_price(np.mean(closes))}")
                print(f"  High (period): {format_price(max(closes))}")
                print(f"  Low (period):  {format_price(min(closes))}")
                print(f"  Avg Volume:    {format_volume(int(np.mean(volumes)))}")
                print(f"  Total Volume:  {format_volume(sum(volumes))}")

    except Exception as e:
        print(f"Error: {e}")

def preview_tick_data():
    """Show detailed preview of MSFT tick data."""
    print("\n" + "="*80)
    print("2. TICK DATA PREVIEW - MSFT (Microsoft Corporation)")
    print("="*80)

    try:
        hist_conn = iq.HistoryConn(name="preview-tick")
        with iq.ConnConnector([hist_conn]) as connector:
            data = hist_conn.request_ticks("MSFT", max_ticks=20)  # Last 20 ticks

            if data is not None and len(data) > 0:
                print(f"\nRetrieved {len(data)} ticks")
                print("\nField Structure:")
                print("  [0] Request ID")
                print("  [1] Date (YYYY-MM-DD)")
                print("  [2] Time (ticks since midnight in microseconds)")
                print("  [3] Last/Trade Price")
                print("  [4] Last/Trade Size")
                print("  [5] Tick Volume")
                print("  [6] Bid Price")
                print("  [7] Ask Price")
                print("  [8] Bid Size")
                print("  [9] Ask Size")
                print("  [10] Tick ID")
                print("  [11] Basis")
                print("  [12] Trade Market Center")
                print("  [13] Trade Conditions")
                print("-"*80)

                # Show header for tick data
                print(f"{'Time':<20} {'Price':>8} {'Size':>6} {'Bid':>8} {'Ask':>8} {'Spread':>8} {'Exchange'}")
                print("-"*80)

                # Show each tick
                for tick in data:
                    # Parse fields
                    date = tick[1]
                    time_field = tick[2]
                    price = tick[3]
                    size = tick[4]
                    exchange = tick[5].decode() if isinstance(tick[5], bytes) else tick[5]
                    tick_vol = tick[6]
                    bid = tick[7]
                    ask = tick[8]
                    bid_size = tick[9]
                    ask_size = tick[10]

                    # Convert time to readable format
                    # Check if time_field is a timedelta or integer
                    if hasattr(time_field, 'total_seconds'):
                        # It's a timedelta
                        total_seconds = time_field.total_seconds()
                        hours = int(total_seconds // 3600)
                        minutes = int((total_seconds % 3600) // 60)
                        seconds = int(total_seconds % 60)
                        microseconds = int((total_seconds * 1_000_000) % 1_000_000 / 1000)
                    else:
                        # It's microseconds since midnight
                        seconds_since_midnight = int(time_field) / 1_000_000
                        hours = int(seconds_since_midnight // 3600)
                        minutes = int((seconds_since_midnight % 3600) // 60)
                        seconds = int(seconds_since_midnight % 60)
                        microseconds = int((int(time_field) % 1_000_000) / 1000)

                    time_str = f"{date} {hours:02d}:{minutes:02d}:{seconds:02d}.{microseconds:03d}"

                    # Calculate spread
                    spread = ask - bid

                    print(f"{time_str:<20} {format_price(price):>8} {size:>6} "
                          f"{format_price(bid):>8} {format_price(ask):>8} "
                          f"{format_price(spread):>8} {exchange:>8}")

                # Show tick summary
                print("-"*80)
                prices = [tick[3] for tick in data]
                sizes = [tick[4] for tick in data]
                spreads = [tick[8] - tick[7] for tick in data]

                print(f"\nTick Summary:")
                print(f"  Price Range: {format_price(min(prices))} - {format_price(max(prices))}")
                print(f"  Average Price: {format_price(np.mean(prices))}")
                print(f"  Total Volume: {format_volume(sum(sizes))} shares")
                print(f"  Average Size: {int(np.mean(sizes))} shares")
                print(f"  Average Spread: {format_price(np.mean(spreads))}")
                print(f"  VWAP: {format_price(np.average(prices, weights=sizes))}")

    except Exception as e:
        print(f"Error: {e}")

def preview_additional_data():
    """Show preview of other data types."""
    print("\n" + "="*80)
    print("3. ADDITIONAL DATA TYPES")
    print("="*80)

    # Intraday bars
    print("\n--- 5-Minute Bars (SPY) ---")
    try:
        hist_conn = iq.HistoryConn(name="preview-bars")
        with iq.ConnConnector([hist_conn]) as connector:
            data = hist_conn.request_bars(
                ticker="SPY",
                interval_len=5,  # 5 minutes
                interval_type='m',  # minutes
                max_bars=10
            )

            if data is not None and len(data) > 0:
                print(f"Retrieved {len(data)} 5-minute bars")
                print(f"Latest bar: Time={data[-1][1]}, O={format_price(data[-1][2])}, "
                      f"H={format_price(data[-1][3])}, L={format_price(data[-1][4])}, "
                      f"C={format_price(data[-1][5])}, Vol={format_volume(data[-1][6])}")
    except Exception as e:
        print(f"Error: {e}")

    # Option chain preview
    print("\n--- Option Chain (AAPL) ---")
    try:
        lookup_conn = iq.LookupConn(name="preview-options")
        with iq.ConnConnector([lookup_conn]) as connector:
            data = lookup_conn.request_equity_option_chain(
                symbol="AAPL",
                opt_type='pc',
                month_codes="FG",  # Next 2 months
                near_months=1,
                include_binary=True
            )

            if data is not None and len(data) > 0:
                print(f"Retrieved {len(data)} option contracts")
                # Show first few options
                for i, option in enumerate(data[:3]):
                    print(f"  Option {i+1}: {option}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run all previews."""
    print("\nIQFEED DATA PREVIEW")
    print("Showing detailed field-by-field breakdown")
    print()

    preview_daily_data()
    preview_tick_data()
    preview_additional_data()

    print("\n" + "="*80)
    print("PREVIEW COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
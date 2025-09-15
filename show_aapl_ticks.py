#!/usr/bin/env python3
"""
Show detailed AAPL tick data preview.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyiqfeed_orig'))

import pyiqfeed as iq
import numpy as np
from datetime import datetime, timedelta

def show_aapl_ticks():
    """Show AAPL tick data in full detail."""
    print("="*100)
    print("AAPL (Apple Inc.) - TICK BY TICK DATA")
    print("="*100)

    hist_conn = iq.HistoryConn(name="aapl-ticks")
    with iq.ConnConnector([hist_conn]) as connector:
        # Get more ticks for better analysis
        data = hist_conn.request_ticks("AAPL", max_ticks=50)  # Last 50 ticks

        print(f"\nData type: {type(data)}")
        print(f"Total ticks retrieved: {len(data)}")
        print(f"Dtype fields: {data.dtype.names}")

        print("\n" + "="*100)
        print("RAW DATA STRUCTURE (First 5 ticks)")
        print("="*100)

        for i, tick in enumerate(data[:5]):
            print(f"\nTick {i}: {tick}")

        print("\n" + "="*100)
        print("FIELD-BY-FIELD BREAKDOWN")
        print("="*100)

        for i, tick in enumerate(data[:5]):
            print(f"\n--- Tick {i} ---")
            print(f"  [0]  Tick ID:        {tick[0]}")
            print(f"  [1]  Date:           {tick[1]}")
            print(f"  [2]  Time:           {tick[2]} (timedelta64[us])")
            print(f"  [3]  Price:          ${tick[3]:.2f}")
            print(f"  [4]  Size:           {tick[4]} shares")
            print(f"  [5]  Exchange:       {tick[5]} ({decode_exchange(tick[5])})")
            print(f"  [6]  Market Center:  {tick[6]}")
            print(f"  [7]  Total Volume:   {tick[7]:,}")
            print(f"  [8]  Bid Price:      ${tick[8]:.2f}")
            print(f"  [9]  Ask Price:      ${tick[9]:.2f}")
            print(f"  [10] Condition 1:    {tick[10]} ({decode_condition(tick[10])})")
            print(f"  [11] Condition 2:    {tick[11]}")
            print(f"  [12] Condition 3:    {tick[12]}")
            print(f"  [13] Condition 4:    {tick[13]}")

        print("\n" + "="*100)
        print("FORMATTED TICK TABLE")
        print("="*100)

        # Convert timedelta to readable time
        print(f"\n{'Time':<22} {'Price':>8} {'Size':>6} {'Bid':>8} {'Ask':>8} {'Spread':>7} {'Exchange':<10} {'Volume':>12}")
        print("-"*100)

        for tick in data[:20]:  # Show first 20 ticks
            # Parse time
            time_td = tick[2]
            if hasattr(time_td, 'astype'):
                # Convert timedelta64 to seconds
                total_microseconds = time_td.astype('timedelta64[us]').astype(np.int64)
                total_seconds = total_microseconds / 1_000_000
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                seconds = int(total_seconds % 60)
                microseconds = int(total_microseconds % 1_000_000)
                time_str = f"{tick[1]} {hours:02d}:{minutes:02d}:{seconds:02d}.{microseconds:06d}"
            else:
                time_str = f"{tick[1]} {tick[2]}"

            price = tick[3]
            size = tick[4]
            exchange = decode_exchange(tick[5])
            bid = tick[8]
            ask = tick[9]
            spread = ask - bid
            volume = tick[7]

            print(f"{time_str:<22} ${price:>7.2f} {size:>6} ${bid:>7.2f} ${ask:>7.2f} ${spread:>6.3f} {exchange:<10} {volume:>12,}")

        print("\n" + "="*100)
        print("TICK DATA ANALYSIS")
        print("="*100)

        # Analyze the ticks
        prices = [tick[3] for tick in data]
        sizes = [tick[4] for tick in data]
        bids = [tick[8] for tick in data]
        asks = [tick[9] for tick in data]
        spreads = [ask - bid for bid, ask in zip(bids, asks)]

        print(f"\nPrice Statistics:")
        print(f"  Current Price:    ${prices[-1]:.2f}")
        print(f"  High (session):   ${max(prices):.2f}")
        print(f"  Low (session):    ${min(prices):.2f}")
        print(f"  Average:          ${np.mean(prices):.2f}")
        print(f"  Std Dev:          ${np.std(prices):.3f}")

        print(f"\nVolume Statistics:")
        print(f"  Total Volume:     {sum(sizes):,} shares")
        print(f"  Average Size:     {int(np.mean(sizes))} shares")
        print(f"  Max Trade:        {max(sizes)} shares")
        print(f"  Min Trade:        {min(sizes)} shares")

        print(f"\nSpread Statistics:")
        print(f"  Current Spread:   ${spreads[-1]:.3f}")
        print(f"  Average Spread:   ${np.mean(spreads):.3f}")
        print(f"  Max Spread:       ${max(spreads):.3f}")
        print(f"  Min Spread:       ${min(spreads):.3f}")

        # VWAP calculation
        vwap = np.average(prices, weights=sizes)
        print(f"\nVWAP (Volume Weighted Average Price): ${vwap:.2f}")

        # Exchange distribution
        print(f"\nExchange Distribution:")
        exchanges = {}
        for tick in data:
            ex = decode_exchange(tick[5])
            exchanges[ex] = exchanges.get(ex, 0) + 1

        for ex, count in sorted(exchanges.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(data)) * 100
            print(f"  {ex:<20} {count:>3} trades ({pct:>5.1f}%)")

        # Time analysis
        print(f"\nTime Analysis:")
        first_tick_time = data[0][2]
        last_tick_time = data[-1][2]

        if hasattr(first_tick_time, 'astype'):
            time_diff = (last_tick_time - first_tick_time).astype('timedelta64[s]').astype(np.float64)
            print(f"  Time span: {time_diff:.1f} seconds")
            print(f"  Tick rate: {len(data) / time_diff:.1f} ticks/second")

def decode_exchange(exchange_code):
    """Decode exchange codes."""
    if isinstance(exchange_code, bytes):
        exchange_code = exchange_code.decode()

    exchanges = {
        'A': 'NYSE American',
        'B': 'NASDAQ BX',
        'C': 'NYSE National',
        'D': 'FINRA ADF',
        'E': 'Market Independent',
        'H': 'MIAX',
        'I': 'ISE',
        'J': 'EDGA',
        'K': 'EDGX',
        'L': 'LTSE',
        'M': 'NYSE Chicago',
        'N': 'NYSE',
        'O': 'NYSE Arca',
        'P': 'NYSE Arca',
        'Q': 'NASDAQ',
        'S': 'NASDAQ Small Cap',
        'T': 'NASDAQ Int',
        'U': 'Members Exchange',
        'V': 'IEX',
        'W': 'CBOE',
        'X': 'NASDAQ PSX',
        'Y': 'CBOE BYX',
        'Z': 'CBOE BZX'
    }

    return exchanges.get(exchange_code, f'Unknown ({exchange_code})')

def decode_condition(condition_code):
    """Decode common condition codes."""
    conditions = {
        0: 'Regular',
        1: 'Cash',
        2: 'NextDay',
        3: 'Seller',
        4: 'YellowFlag',
        5: 'Intermarket',
        10: 'OfficialClose',
        11: 'Crossed',
        12: 'DerivativelyPriced',
        13: 'ReOpeningPrints',
        14: 'ClosingPrints',
        15: 'QualifiedContingentTrade',
        16: 'AveragePriceTrade',
        17: 'PriceVariationTrade',
        23: 'OddLot',
        29: 'Rule155Trade',
        30: 'ContingentTrade',
        135: 'ExtendedHours',
        155: 'SoldLast'
    }

    return conditions.get(condition_code, f'Code {condition_code}')

if __name__ == "__main__":
    show_aapl_ticks()
"""
Analyze why we got 5000 ticks for Friday 2025-09-12
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from gui.data_interface import GUIDataInterface
import pandas as pd

def analyze_friday_volume():
    """Analyze Friday tick volume and time distribution"""
    print("Analyzing Friday 2025-09-12 Tick Volume")
    print("="*50)

    # Get API data
    interface = GUIDataInterface()
    fetch_result = interface.fetch_real_data("AAPL", data_type="ticks", max_records=5000)

    if not fetch_result['success']:
        print(f"FAILED: {fetch_result['error']}")
        return

    api_data = fetch_result['data']
    friday_data = api_data[api_data['timestamp'].dt.date == pd.to_datetime('2025-09-12').date()]

    print(f"Total Friday ticks: {len(friday_data)}")

    # Time distribution analysis
    print(f"\n1. Time Distribution Analysis:")
    friday_data['hour'] = friday_data['timestamp'].dt.hour
    friday_data['minute'] = friday_data['timestamp'].dt.minute

    # Group by hour
    hourly_counts = friday_data['hour'].value_counts().sort_index()
    print(f"   Hourly tick distribution:")
    for hour, count in hourly_counts.items():
        hour_label = f"{hour:02d}:00-{hour:02d}:59"
        if 9 <= hour <= 16:
            session = "REGULAR"
        elif 4 <= hour <= 9:
            session = "PRE-MARKET"
        elif 16 <= hour <= 20:
            session = "AFTER-HOURS"
        else:
            session = "EXTENDED"
        print(f"     {hour_label}: {count:4d} ticks ({session})")

    # Market session breakdown
    regular_hours = friday_data[(friday_data['hour'] >= 9) & (friday_data['hour'] < 16)]
    premarket = friday_data[(friday_data['hour'] >= 4) & (friday_data['hour'] < 9)]
    afterhours = friday_data[(friday_data['hour'] >= 16) & (friday_data['hour'] <= 20)]

    print(f"\n2. Market Session Breakdown:")
    print(f"   Pre-market (04:00-09:30): {len(premarket):4d} ticks")
    print(f"   Regular hours (09:30-16:00): {len(regular_hours):4d} ticks")
    print(f"   After-hours (16:00-20:00): {len(afterhours):4d} ticks")

    # Calculate tick rate
    if len(regular_hours) > 0:
        regular_duration = 6.5 * 3600  # 6.5 hours in seconds
        tick_rate = len(regular_hours) / regular_duration
        print(f"   Regular hours tick rate: {tick_rate:.2f} ticks/second")

    # Volume analysis
    print(f"\n3. Volume Analysis:")
    total_volume = friday_data['volume'].sum()
    avg_trade_size = friday_data['volume'].mean()
    max_trade_size = friday_data['volume'].max()
    min_trade_size = friday_data['volume'].min()

    print(f"   Total volume: {total_volume:,} shares")
    print(f"   Average trade size: {avg_trade_size:.1f} shares")
    print(f"   Largest trade: {max_trade_size:,} shares")
    print(f"   Smallest trade: {min_trade_size} shares")

    # Compare with expected
    print(f"\n4. Expected vs Actual:")
    expected_daily_volume = 50_000_000  # Typical AAPL daily volume
    expected_trades_per_day = expected_daily_volume / avg_trade_size

    print(f"   Expected daily volume: ~{expected_daily_volume:,} shares")
    print(f"   Expected trades/day: ~{expected_trades_per_day:,.0f} trades")
    print(f"   Actual Friday trades: {len(friday_data):,} trades")
    print(f"   Coverage: {len(friday_data)/expected_trades_per_day*100:.1f}% of expected")

    # Time range coverage
    print(f"\n5. Time Coverage:")
    time_start = friday_data['timestamp'].min()
    time_end = friday_data['timestamp'].max()
    duration_hours = (time_end - time_start).total_seconds() / 3600

    print(f"   Start: {time_start}")
    print(f"   End: {time_end}")
    print(f"   Duration: {duration_hours:.1f} hours")

    # Market close focus
    market_close = friday_data[friday_data['timestamp'].dt.hour == 15]
    print(f"   Market close hour (15:xx): {len(market_close)} ticks")

    if len(market_close) > 0:
        close_minutes = market_close['timestamp'].dt.minute.value_counts().sort_index()
        print(f"   Market close minutes:")
        for minute, count in close_minutes.items():
            print(f"     15:{minute:02d}: {count} ticks")

if __name__ == "__main__":
    analyze_friday_volume()
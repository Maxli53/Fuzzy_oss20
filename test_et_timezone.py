#!/usr/bin/env python3
"""
Test that ET timezone is used everywhere
"""

import sys
import pytz
from datetime import datetime

# Setup paths
sys.path.insert(0, 'stage_01_data_engine/collectors')
sys.path.append('pyiqfeed_orig')
sys.path.append('.')

print("="*80)
print("TIMEZONE TEST - Confirming ET is used everywhere")
print("="*80)

# 1. Check current times
et_tz = pytz.timezone('America/New_York')
print("\n1. TIME COMPARISON:")
print("-"*40)
print(f"System local time: {datetime.now()}")
print(f"ET time: {datetime.now(et_tz)}")
print(f"System timezone: {datetime.now().tzname()}")
print(f"ET timezone: {datetime.now(et_tz).tzname()}")

# 2. Check market hours detection
now_et = datetime.now(et_tz)
is_weekend = now_et.weekday() >= 5
is_after_hours = now_et.hour >= 16 or now_et.hour < 9

print("\n2. MARKET HOURS DETECTION (in ET):")
print("-"*40)
print(f"Current ET time: {now_et}")
print(f"Hour (ET): {now_et.hour}")
print(f"Weekday: {now_et.weekday()} (0=Mon, 6=Sun)")
print(f"Is weekend: {is_weekend}")
print(f"Is after hours: {is_after_hours}")
print(f"Market status: {'CLOSED' if is_weekend or is_after_hours else 'OPEN'}")

# 3. Test IQFeed collector
print("\n3. TESTING IQFEED COLLECTOR:")
print("-"*40)

from iqfeed_collector import IQFeedCollector

collector = IQFeedCollector()
if collector.ensure_connection():
    print("Connected to IQFeed")

    # This should now use the correct after-hours detection
    print("\nFetching 100 ticks to test market hours detection...")
    ticks = collector.get_tick_data('AAPL', num_days=1, max_ticks=100)
    print(f"Retrieved {len(ticks)} ticks")

    # The log should show "After-hours/weekend advantage" if it's after hours
else:
    print("Failed to connect to IQFeed")

print("\n" + "="*80)
print("TIMEZONE TEST COMPLETE")
print("All times should be in ET (Eastern Time)")
print("="*80)
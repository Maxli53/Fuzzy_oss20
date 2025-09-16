#!/usr/bin/env python3
"""
Verify a tick that's visible in the IQFeed screenshot
"""

import sys
import numpy as np
from datetime import datetime, timedelta

# Setup paths
sys.path.insert(0, 'stage_01_data_engine/collectors')
sys.path.append('.')

from iqfeed_collector import IQFeedCollector

# Initialize collector
collector = IQFeedCollector()

if not collector.ensure_connection():
    print("Failed to connect to IQFeed")
    sys.exit(1)

print("Fetching AAPL ticks...")
tick_array = collector.get_tick_data('AAPL', num_days=1, max_ticks=100000)

print(f"Total ticks fetched: {len(tick_array)}")

# From screenshot: tick at 15:59:59.903912 with price $238.1000, size 360
# Let's find ticks around 15:59:59 (which is 57599 seconds = 57599000000 microseconds)

target_time_start = 57599000000  # 15:59:59.000000
target_time_end = 57600000000    # 16:00:00.000000

# Find ticks in this time range
matching_ticks = []
for i, tick in enumerate(tick_array):
    time_us = int(tick['time'] / np.timedelta64(1, 'us')) if isinstance(tick['time'], np.timedelta64) else int(tick['time'])
    if target_time_start <= time_us <= target_time_end:
        matching_ticks.append((i, tick, time_us))

print(f"\nFound {len(matching_ticks)} ticks between 15:59:59 and 16:00:00")

# Show the last few before 16:00:00
print("\nLast 10 ticks before market close (16:00:00):")
for idx, tick, time_us in matching_ticks[-10:]:
    time_seconds = time_us / 1_000_000
    hours = int(time_seconds // 3600)
    minutes = int((time_seconds % 3600) // 60)
    seconds = time_seconds % 60
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    print(f"Index {idx:5d}: {time_str} | ${tick['last']:7.4f} | {tick['last_sz']:4d} shares | {tick['last_type']} | TickID: {tick['tick_id']}")

# Now show what's AFTER 16:00:00
print("\nFirst 5 ticks at/after 16:00:00:")
after_close = []
for i, tick in enumerate(tick_array):
    time_us = int(tick['time'] / np.timedelta64(1, 'us')) if isinstance(tick['time'], np.timedelta64) else int(tick['time'])
    if time_us >= 57600000000:  # 16:00:00 or later
        after_close.append((i, tick, time_us))
        if len(after_close) >= 5:
            break

for idx, tick, time_us in after_close:
    time_seconds = time_us / 1_000_000
    hours = int(time_seconds // 3600)
    minutes = int((time_seconds % 3600) // 60)
    seconds = time_seconds % 60
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    print(f"Index {idx:5d}: {time_str} | ${tick['last']:7.4f} | {tick['last_sz']:4d} shares | {tick['last_type']} | TickID: {tick['tick_id']}")

# Let's specifically look for the visible tick from screenshot: 15:59:59.903912, price 238.10, size 360
print("\nSearching for specific tick from screenshot (15:59:59.903912, $238.10, 360 shares):")
target_price = 238.10
target_size = 360
target_time_approx = 57599903912  # 15:59:59.903912 in microseconds

for i, tick in enumerate(tick_array[:1000]):  # Check first 1000
    if abs(tick['last'] - target_price) < 0.01 and tick['last_sz'] == target_size:
        time_us = int(tick['time'] / np.timedelta64(1, 'us')) if isinstance(tick['time'], np.timedelta64) else int(tick['time'])
        if abs(time_us - target_time_approx) < 1000000:  # Within 1 second
            print(f"FOUND at index {i}: TickID {tick['tick_id']}, exact match!")